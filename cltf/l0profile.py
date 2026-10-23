"""Per-layer L0 and decoder write-mass-by-distance for the public transcoders.

For a CLT, `write_share[k]` = fraction of total reconstruction energy written
from a source layer to a target k layers later (activation-weighted decoder
norms), i.e. how much of the work it routes through cross-layer writes.

    python -m cltf.l0profile --model llama|gemma --out results/l0_<model>.json
"""
import argparse
import json
import os

import torch

from . import llama as LL


@torch.no_grad()
def profile(model, tc, batches):
    L = model.n_layer
    l0 = torch.zeros(L)
    share = torch.zeros(L)  # by distance
    n = 0
    for ids, _ in batches:
        _, xs, _ = model.run(ids)
        d = xs[0].shape[-1]
        for s in range(L):
            x = tc.read(model, s, xs[s])[:, 1:].reshape(-1, d)
            f = tc.encode(s, x)
            if f is None:
                continue
            l0[s] += (f > 0).float().sum(-1).mean().item()
            if tc.kind == "clt":
                W = tc.dec[s]  # (F, n_t, d)
                norms = W.float().norm(dim=-1)  # (F, n_t)
                e = (f.float() @ norms.pow(2))  # (N, n_t) energy-ish per target
                share[: e.shape[1]] += e.mean(0).cpu()
        n += 1
    return {"l0": (l0 / n).tolist(), "write_by_distance": (share / share.sum()).tolist() if share.sum() > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    if a.model == "llama":
        tok = AutoTokenizer.from_pretrained(LL.MIRROR)
        model = LL.Llama()
        tcs = {"clt": LL.PublicCLT, "plt": LL.PublicPLT}
    else:
        from . import gemma as G
        tok = AutoTokenizer.from_pretrained(G.MIRROR)
        model = G.Gemma()
        tcs = {"clt": G.GemmaCLT, "plt": G.GemmaScopePLT}
    batches = LL.batches_from_fineweb(tok, 16, 2)
    out = {}
    for k, cls in tcs.items():
        tc = cls()
        out[k] = profile(model, tc, batches)
        print(a.model, k, "l0", [round(v, 1) for v in out[k]["l0"]], flush=True)
        if out[k]["write_by_distance"]:
            print(a.model, k, "write share by distance", [round(v, 3) for v in out[k]["write_by_distance"]], flush=True)
        del tc
        torch.cuda.empty_cache()
    path = a.out or f"results/l0_{a.model}.json"
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(path, "w"))


if __name__ == "__main__":
    main()
