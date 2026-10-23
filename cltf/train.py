"""Train one transcoder variant on GPT-2 small, online from a shuffle buffer.

The sparsity coefficient is adapted so every variant lands at the same
target total L0 (active features per token, summed over source layers), so
variants are compared at matched sparsity.

    python -m cltf.train --name clt16 --kind clt --l0 192
"""
import argparse
import json
import math
import os
import time

import torch

from .data import ActivationBuffer, Tokens
from .gpt2 import load_gpt2
from .transcoder import Transcoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="clt", choices=["plt", "clt", "window"])
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--norm", default="concat", choices=["concat", "sum"])
    ap.add_argument("--F", type=int, default=8192)
    ap.add_argument("--l0", type=float, default=192.0, help="target total L0 per token")
    ap.add_argument("--tokens", type=float, default=3e8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--c", type=float, default=4.0, help="tanh sharpness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam0", type=float, default=0.1)
    ap.add_argument("--data", default="/scratch/tok/fwe.bin")
    ap.add_argument("--out", default=os.path.expanduser("~/runs"))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)

    model, _ = load_gpt2()
    buf = ActivationBuffer(model, Tokens(a.data, seed=a.seed))
    L, d = model.n_layer, model.d

    # per-layer scalar normalisation: E||x|| = E||y|| = sqrt(d)
    with torch.no_grad():
        X, Y = buf.X[:, :65536].float(), buf.Y[:, :65536].float()
        sx = (d ** 0.5) / X.norm(dim=-1).mean(-1)
        sy = (d ** 0.5) / Y.norm(dim=-1).mean(-1)
    torch.save({"sx": sx.cpu(), "sy": sy.cpu()}, os.path.join(out, "scales.pt"))

    tc = Transcoder(L, d, a.F, a.kind, a.window, a.norm).cuda()
    opt = torch.optim.Adam(tc.parameters(), lr=a.lr, betas=(0.9, 0.999))
    steps = int(a.tokens // a.batch)
    warm = 1000
    sched = lambda i: min(1.0, (i + 1) / warm) * (1.0 if i < 0.8 * steps else
                                                   max(0.0, (steps - i) / (0.2 * steps)))
    log_lam = math.log(a.lam0)
    l0_ema = None
    lam_p = 3e-6
    ramp = 5000
    logf = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    for i in range(steps):
        x, y = buf.next(a.batch)
        x = x * sx[:, None, None]
        y = y * sy[:, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yhat, acts, pres = tc(x)
        yhat = yhat.float()
        mse = (yhat - y).pow(2).sum(-1).mean()  # summed over layers & dims, mean over tokens
        spars = 0.0
        pre_pen = 0.0
        l0_t = 0.0
        for s in range(L):
            f = acts[s].float()
            w = tc.dec_norm(s)
            spars = spars + torch.tanh(a.c * f * w).sum(-1).mean()
            theta = tc.log_theta[s].exp()
            pre_pen = pre_pen + (torch.relu(theta - pres[s].float()) * w).sum(-1).mean()
            l0_t = l0_t + (f > 0).float().sum(-1).mean()
        l0 = l0_t.item()
        lam_eff = math.exp(log_lam) * min(1.0, (i + 1) / ramp)
        loss = mse / L + lam_eff * spars + lam_p * pre_pen
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(i)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tc.parameters(), 1.0)
        opt.step()

        # sparsity controller: steer the coefficient toward the target total L0
        # slow integral controller, only after the lr warmup; lambda is also
        # ramped in linearly over the first `ramp` steps
        l0_ema = l0 if l0_ema is None else 0.99 * l0_ema + 0.01 * l0
        if i > 3000:
            step = 0.002 * math.log(max(l0_ema, 1e-3) / a.l0)
            log_lam += max(-0.002, min(0.002, step))
            log_lam = min(max(log_lam, math.log(1e-5)), math.log(30.0))

        if i % 200 == 0 or i == steps - 1:
            with torch.no_grad():
                fvu = ((yhat - y).pow(2).sum(-1).mean(-1) / (y - y.mean(1, keepdim=True)).pow(2).sum(-1).mean(-1))
                l0_layer = [(acts[s] > 0).float().sum(-1).mean().item() for s in range(L)]
            rec = {"step": i, "tok": (i + 1) * a.batch, "mse": mse.item(), "l0": l0, "l0_ema": l0_ema,
                   "lam": math.exp(log_lam), "fvu": [round(v, 4) for v in fvu.tolist()],
                   "l0_layer": [round(v, 2) for v in l0_layer], "sec": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if i % 2000 == 0:
                print(json.dumps({k: rec[k] for k in ["step", "mse", "l0", "lam", "sec"]}), flush=True)
        if (i + 1) % 10000 == 0 or i == steps - 1:
            torch.save({"state": tc.state_dict(), "cfg": vars(a), "step": i, "log_lam": log_lam},
                       os.path.join(out, "ckpt.pt"))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
