"""Write-share by distance + per-layer L0 for our own GPT-2 transcoders (held-out data).

    python -m cltf.writeshare --runs clt_l16 ... --out results/writeshare_<x>.json
"""
import argparse
import json
import os

import torch

from .data import Tokens
from .evaluate import load_tc
from .gpt2 import load_gpt2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--runs_dir", default=os.path.expanduser("~/runs"))
    ap.add_argument("--data", default="/scratch/tok/fwe.bin")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, _ = load_gpt2(name=a.model)
    tok = Tokens(a.data, seed=123, split="val")
    batches = [tok.batch(64) for _ in range(4)]
    out = {}
    for r in a.runs:
        tc, sx, sy, cfg = load_tc(os.path.join(a.runs_dir, r), model.n_layer, model.d)
        L = model.n_layer
        l0 = torch.zeros(L)
        share = torch.zeros(L)
        for ids in batches:
            _, xs, _ = model.run(ids)
            for s in range(L):
                x = xs[s][:, 1:].reshape(-1, model.d) * sx[s]
                _, f = tc.encode(s, x)
                l0[s] += (f > 0).float().sum(-1).mean().item()
                norms = tc.W_dec[s].float().norm(dim=-1)  # (F, n_t)
                e = f.float() @ norms.pow(2)
                share[: e.shape[1]] += e.mean(0).cpu()
        out[r] = {"l0": (l0 / len(batches)).tolist(), "write_by_distance": (share / share.sum()).tolist()}
        print(r, "l0", [round(v, 1) for v in out[r]["l0"]], "share", [round(v, 3) for v in out[r]["write_by_distance"]], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
