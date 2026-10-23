"""The instrument: does a replacement model's downstream computation respond to
a mid-layer intervention the way the real model's does?

Intervention: resample-ablate the MLP output of source layer s (replace it with
its value on a different, held-out text at the same positions). Then measure,
for every later layer t, how the MLP output y_t changes, and how the logits
change, in

  real      the actual GPT-2
  frozen    a "hybrid" where every MLP is replaced by the transcoder's
            reconstruction PLUS the clean-run error term held constant
            (circuit-tracer's local replacement model). It reproduces the
            real model exactly on the clean run, so any disagreement under
            intervention comes from the transcoder's features alone.
  full      the hybrid with no error term (the pure replacement model).

Faithfulness of the downstream response at (s, t):
  explained = 1 - E||dy_rep - dy_real||^2 / E||dy_real||^2
  ratio     = E||dy_rep|| / E||dy_real||
  cos       = mean token cosine(dy_rep, dy_real)

Controls (computed every run, must pass):
  POS  "replacement" = the real MLPs           -> explained == 1
  NEG  every MLP frozen at its clean output   -> explained == 0, ratio == 0

    python -m cltf.evaluate --runs plt16 clt16 ... --out results/gpt2_interventions.json
"""
import argparse
import json
import os

import torch

from .data import Tokens
from .gpt2 import load_gpt2
from .transcoder import Transcoder


def load_tc(run_dir):
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cuda")
    c = ck["cfg"]
    tc = Transcoder(12, 768, c["F"], c["kind"], c["window"], c["norm"]).cuda()
    tc.load_state_dict(ck["state"])
    sc = torch.load(os.path.join(run_dir, "scales.pt"))
    return tc.eval(), sc["sx"].cuda(), sc["sy"].cuda(), c


@torch.no_grad()
def run_hybrid(model, tc, sx, sy, ids, error=None, override=None, collect_acts=False, src=None):
    """Replacement-model forward. error: list of per-layer tensors added to the
    reconstruction (frozen-error mode) or None (full replacement)."""
    h = model.embed(ids)
    B, T, d = h.shape
    L = model.n_layer
    pending = [torch.zeros(B * T, d, device=ids.device) for _ in range(L)]
    xs, ys, acts = [], [], []
    for l in range(L):
        h = h + model.attn(l, h)
        x = model.mlp_in(l, h)
        _, f = tc.encode(l, x.reshape(-1, d) * sx[l])
        if src is not None and l in src:
            f = src[l]  # swap this source layer's features (their cross-layer writes too)
        out = tc.decode_from(l, f)
        for j, t in enumerate(tc.targets[l]):
            pending[t] = pending[t] + out[:, j]
        yhat = ((pending[l] + tc.b_dec[l]) / sy[l]).view(B, T, d)
        # error=(real_ys, hat): real + (recon(h) - recon(real x)), bit-exact on clean
        y = yhat if error is None else error[0][l] + (yhat - error[1][l])
        if override is not None and l in override:
            y = override[l]
        xs.append(x)
        ys.append(y)
        if collect_acts:
            acts.append(f.view(B, T, -1))
        h = h + y
    return model.logits(h), xs, ys, acts


@torch.no_grad()
@torch.no_grad()
def recon_from_xs(tc, sx, sy, xs):
    """Transcoder reconstruction of every MLP output given the REAL MLP inputs."""
    B, T, d = xs[0].shape
    L = len(xs)
    pending = [torch.zeros(B * T, d, device=xs[0].device) for _ in range(L)]
    for l in range(L):
        _, f = tc.encode(l, xs[l].reshape(-1, d) * sx[l])
        out = tc.decode_from(l, f)
        for j, t in enumerate(tc.targets[l]):
            pending[t] = pending[t] + out[:, j]
    return [((pending[l] + tc.b_dec[l]) / sy[l]).view(B, T, d) for l in range(L)]


def run_frozen_mlps(model, ids, clean_ys, override=None):
    """NEG control: every MLP output pinned to its clean value (except overrides)."""
    h = model.embed(ids)
    ys = []
    for l in range(model.n_layer):
        h = h + model.attn(l, h)
        y = clean_ys[l]
        if override is not None and l in override:
            y = override[l]
        ys.append(y)
        h = h + y
    return model.logits(h), None, ys


def kl(p_logits, q_logits):
    """KL(p || q) per token, positions 1.. only."""
    lp = torch.log_softmax(p_logits[:, 1:].float(), -1)
    lq = torch.log_softmax(q_logits[:, 1:].float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def ce(logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    return -lp.gather(-1, ids[:, 1:, None]).squeeze(-1)[:, 1:].mean().item()


def compare(d_real, d_rep):
    """d_*: (B, T, d) downstream deltas, position 0 excluded by caller."""
    a = d_real[:, 1:].reshape(-1, d_real.shape[-1]).float()
    b = d_rep[:, 1:].reshape(-1, d_rep.shape[-1]).float()
    num = (b - a).pow(2).sum(-1).mean()
    den = a.pow(2).sum(-1).mean()
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    m = (na > 1e-6) & (nb > 1e-6)
    cos = (a[m] * b[m]).sum(-1) / (na[m] * nb[m])
    return {"explained": (1 - num / den).item(),
            "ratio": (nb.mean() / na.mean()).item(),
            "cos": cos.mean().item() if m.any() else 0.0,
            "real_mag": na.mean().item()}


@torch.no_grad()
def evaluate_one(model, name, ids, donor, tc=None, sx=None, sy=None, kind="frozen", alpha=1.0):
    """Returns per-(s,t) comparisons for one replacement against the real model."""
    L = model.n_layer
    real_logits, _, real_ys = model.run(ids)
    _, donor_xs, donor_ys = model.run(donor)
    if name == "POS":
        rep = lambda ov=None: model.run(ids, mlp_override=ov)
    elif name == "NEG":
        rep = lambda ov=None: run_frozen_mlps(model, ids, real_ys, ov)
    else:
        if kind == "frozen":
            _, real_xs, _ = model.run(ids)
            hat_ys = recon_from_xs(tc, sx, sy, real_xs)
            err = (real_ys, hat_ys)
        else:
            err = None
        rep = lambda ov=None, src=None: run_hybrid(model, tc, sx, sy, ids, error=err, override=ov, src=src)[:3]
    rep_logits, _, rep_ys = rep()
    if kind == "frozen" or name in ("POS", "NEG"):
        # the frozen-error hybrid must reproduce the real model on the clean run
        assert kl(real_logits, rep_logits).mean().item() < 1e-3, f"{name}: frozen hybrid != real on clean"
    out = {"clean_ce_real": ce(real_logits, ids), "clean_ce_rep": ce(rep_logits, ids),
           "clean_kl": kl(real_logits, rep_logits).mean().item(), "cells": [], "cells_src": []}
    for s in range(L - 1):
        # alpha=1: full resample ablation; alpha<1: partial move toward the donor value
        ov = {s: real_ys[s] + alpha * (donor_ys[s] - real_ys[s])}
        ri_logits, _, ri_ys = model.run(ids, mlp_override=ov)
        pi_logits, _, pi_ys = rep(ov)
        for t in range(s + 1, L):
            c = compare(ri_ys[t] - real_ys[t], pi_ys[t] - rep_ys[t])
            c.update({"s": s, "t": t})
            out["cells"].append(c)
        # logit-level response
        dl_real = (torch.log_softmax(ri_logits.float(), -1) - torch.log_softmax(real_logits.float(), -1))
        dl_rep = (torch.log_softmax(pi_logits.float(), -1) - torch.log_softmax(rep_logits.float(), -1))
        c = compare(dl_real, dl_rep)
        c.update({"s": s, "t": "logits",
                  "kl_real": kl(real_logits, ri_logits).mean().item(),
                  "kl_rep": kl(rep_logits, pi_logits).mean().item()})
        out["cells"].append(c)
        if tc is not None and tc.kind != "plt" and alpha == 1.0:
            d = donor_xs[s].shape[-1]
            _, f_don = tc.encode(s, donor_xs[s].reshape(-1, d) * sx[s])
            _, _, si_ys = rep(ov, {s: f_don})
            for t in range(s + 1, L):
                c = compare(ri_ys[t] - real_ys[t], si_ys[t] - rep_ys[t])
                c.update({"s": s, "t": t})
                out["cells_src"].append(c)
    return out


def summarize(res, L=12):
    cells = [c for c in res["cells"] if c["t"] != "logits"]
    lg = [c for c in res["cells"] if c["t"] == "logits"]
    w = sum(c["real_mag"] ** 2 for c in cells)
    return {
        "mlp_explained_mean": sum(c["explained"] for c in cells) / len(cells),
        "mlp_explained_weighted": sum(c["explained"] * c["real_mag"] ** 2 for c in cells) / w,
        "mlp_ratio_mean": sum(c["ratio"] for c in cells) / len(cells),
        "logit_explained_mean": sum(c["explained"] for c in lg) / len(lg),
        "logit_kl_ratio_mean": sum(c["kl_rep"] / max(c["kl_real"], 1e-9) for c in lg) / len(lg),
        "clean_ce_real": res["clean_ce_real"], "clean_ce_rep": res["clean_ce_rep"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--runs_dir", default=os.path.expanduser("~/runs"))
    ap.add_argument("--data", default="/scratch/tok/fwe.bin")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="results/gpt2_interventions.json")
    a = ap.parse_args()
    model, _ = load_gpt2()
    tok = Tokens(a.data, seed=123, split="val")
    batches = [(tok.batch(a.n), tok.batch(a.n)) for _ in range(a.reps)]
    results = {}

    def run(name, **kw):
        per = [evaluate_one(model, name, ids, donor, alpha=a.alpha, **kw) for ids, donor in batches]
        # average cells across reps
        merged = per[0]
        for k in ["clean_ce_real", "clean_ce_rep", "clean_kl"]:
            merged[k] = sum(p[k] for p in per) / len(per)
        for key in ["cells", "cells_src"]:
            for i, c in enumerate(merged[key]):
                for k in ["explained", "ratio", "cos", "real_mag"] + (["kl_real", "kl_rep"] if c["t"] == "logits" else []):
                    c[k] = sum(p[key][i][k] for p in per) / len(per)
        merged["summary"] = summarize(merged)
        print(name, json.dumps({k: round(v, 4) for k, v in merged["summary"].items()}), flush=True)
        return merged

    results["POS"] = run("POS")
    results["NEG"] = run("NEG")
    pos, neg = results["POS"]["summary"], results["NEG"]["summary"]
    assert pos["mlp_explained_mean"] > 0.999, "POS control failed"
    assert abs(neg["mlp_ratio_mean"]) < 1e-6, "NEG control failed"
    for r in a.runs:
        tc, sx, sy, cfg = load_tc(os.path.join(a.runs_dir, r))
        for kind in ["frozen", "full"]:
            results[f"{r}/{kind}"] = run(f"{r}/{kind}", tc=tc, sx=sx, sy=sy, kind=kind)
            results[f"{r}/{kind}"]["cfg"] = cfg
        del tc
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
