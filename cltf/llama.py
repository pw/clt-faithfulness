"""The same intervention instrument, on Llama-3.2-1B with the PUBLIC transcoders
people use for attribution graphs (circuit-tracer's defaults):

  CLT  mntss/clt-llama-3.2-1b-524k   32,768 feats/layer, JumpReLU, reads resid_mid
  PLT  mntss/transcoder-Llama-3.2-1B 131,072 feats/layer, ReLU + linear skip, reads resid_mid

Both write hook_mlp_out. Weights are loaded straight from the safetensors.
Model weights come from an ungated mirror of meta-llama/Llama-3.2-1B.

    python -m cltf.llama --which clt plt --out results/llama_interventions.json
"""
import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .evaluate import compare, kl, ce, summarize

MIRROR = "unsloth/Llama-3.2-1B"


class Llama:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForCausalLM
        self.hf = AutoModelForCausalLM.from_pretrained(MIRROR, torch_dtype=torch.float32).to(device).eval()
        m = self.hf.model
        self.m = m
        self.n_layer = len(m.layers)
        self.d = m.config.hidden_size

    def _pe(self, h, T):
        pos = torch.arange(T, device=h.device)[None]
        return self.m.rotary_emb(h, pos)

    def attn(self, l, h, pe):
        layer = self.m.layers[l]
        return layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=pe,
                               attention_mask=None)[0]

    def mlp(self, l, h):
        layer = self.m.layers[l]
        return layer.mlp(layer.post_attention_layernorm(h))

    def logits(self, h):
        return self.hf.lm_head(self.m.norm(h))

    @torch.no_grad()
    def run(self, ids, mlp_override=None):
        h = self.m.embed_tokens(ids)
        pe = self._pe(h, ids.shape[1])
        xs, ys = [], []
        for l in range(self.n_layer):
            h = h + self.attn(l, h, pe)
            x = h  # resid_mid: what both transcoders read
            y = self.mlp(l, h)
            if mlp_override is not None and l in mlp_override:
                y = mlp_override[l]
            xs.append(x)
            ys.append(y)
            h = h + y
        return self.logits(h), xs, ys


class PublicCLT:
    def __init__(self, repo="mntss/clt-llama-3.2-1b-524k", L=16, device="cuda"):
        self.kind = "clt"
        self.L = L
        self.enc, self.dec = [], []
        self.b_dec = []
        for i in range(L):
            e = load_file(hf_hub_download(repo, f"W_enc_{i}.safetensors"), device=device)
            self.enc.append((e[f"W_enc_{i}"], e[f"b_enc_{i}"], e[f"threshold_{i}"]))
            self.b_dec.append(e[f"b_dec_{i}"])
            self.dec.append(load_file(hf_hub_download(repo, f"W_dec_{i}.safetensors"), device=device)[f"W_dec_{i}"])
        self.targets = [list(range(s, L)) for s in range(L)]

    def encode(self, l, x):
        W, b, thr = self.enc[l]
        z = x.to(W.dtype) @ W.T + b
        return z * (z > thr)

    def decode_from(self, l, f, x):
        W = self.dec[l]
        return (f @ W.reshape(W.shape[0], -1)).view(f.shape[0], W.shape[1], -1).float()


class PublicPLT:
    def __init__(self, repo="mntss/transcoder-Llama-3.2-1B", L=16, device="cuda"):
        self.kind = "plt"
        self.L = L
        self.p = [load_file(hf_hub_download(repo, f"layer_{i}.safetensors"), device=device) for i in range(L)]
        self.b_dec = [p["b_dec"] for p in self.p]
        self.targets = [[s] for s in range(L)]
        self.skip_only = False

    def encode(self, l, x):
        p = self.p[l]
        if self.skip_only:
            return None
        return torch.relu(x.to(p["W_enc"].dtype) @ p["W_enc"].T + p["b_enc"])

    def decode_from(self, l, f, x):
        p = self.p[l]
        out = (x.to(p["W_skip"].dtype) @ p["W_skip"].T).float()
        if f is not None:
            out = out + (f @ p["W_dec"]).float()
        return out[:, None, :]


@torch.no_grad()
def recon_from_xs(tc, xs):
    B, T, d = xs[0].shape
    L = len(xs)
    pending = [torch.zeros(B * T, d, device=xs[0].device) for _ in range(L)]
    for l in range(L):
        x = xs[l].reshape(-1, d)
        out = tc.decode_from(l, tc.encode(l, x), x)
        for j, t in enumerate(tc.targets[l]):
            pending[t] = pending[t] + out[:, j]
    return [(pending[l] + tc.b_dec[l].float()).view(B, T, d) for l in range(L)]


@torch.no_grad()
def run_hybrid(model, tc, ids, error=None, override=None, real_ys0=None):
    h = model.m.embed_tokens(ids)
    B, T, d = h.shape
    pe = model._pe(h, T)
    pending = [torch.zeros(B * T, d, device=ids.device) for _ in range(model.n_layer)]
    ys = []
    for l in range(model.n_layer):
        h = h + model.attn(l, h, pe)
        x = h.reshape(-1, d)
        out = tc.decode_from(l, tc.encode(l, x), x)
        for j, t in enumerate(tc.targets[l]):
            pending[t] = pending[t] + out[:, j]
        y = (pending[l] + tc.b_dec[l].float()).view(B, T, d)
        if error is not None:
            y = y + error[l]
        elif real_ys0 is not None:
            y = y.clone()
            y[:, 0] = real_ys0[l][:, 0]  # BOS position pinned (transcoders zero it)
        if override is not None and l in override:
            y = override[l]
        ys.append(y)
        h = h + y
    return model.logits(h), None, ys


@torch.no_grad()
def evaluate_one(model, name, ids, donor, tc=None, kind="frozen", alpha=1.0):
    L = model.n_layer
    real_logits, real_xs, real_ys = model.run(ids)
    _, _, donor_ys = model.run(donor)
    if name == "POS":
        rep = lambda ov=None: model.run(ids, mlp_override=ov)
        fvu = None
    else:
        hat = recon_from_xs(tc, real_xs)
        fvu = []
        for l in range(L):
            a, b = real_ys[l][:, 1:].float(), hat[l][:, 1:]
            fvu.append(((a - b).pow(2).sum(-1).mean() / (a - a.mean((0, 1))).pow(2).sum(-1).mean()).item())
        err = [real_ys[l] - hat[l] for l in range(L)] if kind == "frozen" else None
        rep = lambda ov=None: run_hybrid(model, tc, ids, error=err, override=ov,
                                         real_ys0=None if kind == "frozen" else real_ys)
    rep_logits, _, rep_ys = rep()
    if kind == "frozen" or name == "POS":
        assert kl(real_logits, rep_logits).mean().item() < 1e-2, f"{name}: frozen hybrid != real on clean"
    out = {"clean_ce_real": ce(real_logits, ids), "clean_ce_rep": ce(rep_logits, ids),
           "clean_kl": kl(real_logits, rep_logits).mean().item(), "fvu": fvu, "cells": []}
    for s in range(L - 1):
        ov = {s: real_ys[s] + alpha * (donor_ys[s] - real_ys[s])}
        ri_logits, _, ri_ys = model.run(ids, mlp_override=ov)
        pi_logits, _, pi_ys = rep(ov)
        for t in range(s + 1, L):
            c = compare(ri_ys[t] - real_ys[t], pi_ys[t] - rep_ys[t])
            c.update({"s": s, "t": t})
            out["cells"].append(c)
        dl_real = torch.log_softmax(ri_logits.float(), -1) - torch.log_softmax(real_logits.float(), -1)
        dl_rep = torch.log_softmax(pi_logits.float(), -1) - torch.log_softmax(rep_logits.float(), -1)
        c = compare(dl_real, dl_rep)
        c.update({"s": s, "t": "logits", "kl_real": kl(real_logits, ri_logits).mean().item(),
                  "kl_rep": kl(rep_logits, pi_logits).mean().item()})
        out["cells"].append(c)
        del dl_real, dl_rep
    return out


def batches_from_fineweb(tok, n, reps, seq=128, seed=7):
    import glob
    import random
    import pyarrow.parquet as pq
    f = sorted(glob.glob("/scratch/tok/fwe/sample/10BT/*.parquet"))[0]
    texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()[-20000:]  # tail = held out
    random.Random(seed).shuffle(texts)
    rows = []
    for t in texts:
        ids = tok(t, add_special_tokens=True)["input_ids"]
        if len(ids) >= seq:
            rows.append(ids[:seq])
        if len(rows) >= 2 * n * reps:
            break
    x = torch.tensor(rows, device="cuda")
    return [(x[2 * i * n:(2 * i + 1) * n], x[(2 * i + 1) * n:(2 * i + 2) * n]) for i in range(reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=["clt", "plt"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="results/llama_interventions.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MIRROR)
    model = Llama()
    # sanity: our layer loop == HF forward
    ids = torch.randint(0, 128000, (2, 32), device="cuda")
    ref = model.hf(ids).logits
    ours, _, _ = model.run(ids)
    err = (ours - ref).abs().max().item()
    print("max |logit diff| vs HF:", err, flush=True)
    assert err < 1e-2
    batches = batches_from_fineweb(tok, a.n, a.reps)
    results = {}

    def run(name, **kw):
        per = [evaluate_one(model, name, ids, donor, alpha=a.alpha, **kw) for ids, donor in batches]
        merged = per[0]
        for k in ["clean_ce_real", "clean_ce_rep", "clean_kl"]:
            merged[k] = sum(p[k] for p in per) / len(per)
        if merged["fvu"] is not None:
            merged["fvu"] = [sum(p["fvu"][l] for p in per) / len(per) for l in range(len(merged["fvu"]))]
        for i, c in enumerate(merged["cells"]):
            for k in ["explained", "ratio", "cos", "real_mag"] + (["kl_real", "kl_rep"] if c["t"] == "logits" else []):
                c[k] = sum(p["cells"][i][k] for p in per) / len(per)
        merged["summary"] = summarize(merged)
        print(name, json.dumps({k: round(v, 4) for k, v in merged["summary"].items()}),
              "fvu", None if merged["fvu"] is None else [round(v, 3) for v in merged["fvu"]], flush=True)
        return merged

    results["POS"] = run("POS")
    for w in a.which:
        tc = PublicCLT() if w == "clt" else PublicPLT()
        for kind in ["frozen", "full"]:
            results[f"{w}/{kind}"] = run(f"{w}/{kind}", tc=tc, kind=kind)
        if w == "plt":
            # how much of the PLT's response is just its linear skip path?
            tc.skip_only = True
            results["plt_skiponly/frozen"] = run("plt_skiponly/frozen", tc=tc, kind="frozen")
        del tc
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
