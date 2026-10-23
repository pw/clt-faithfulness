"""Gemma-2-2B with its public transcoders, through the llama.py instrument.

  CLT  mntss/clt-gemma-2-2b-426k   16,384 feats/layer x 26, ReLU, reads resid_mid
  PLT  GemmaScope transcoders      16,384 feats/layer, JumpReLU, NO skip path,
       reads ln2 (the pre-feedforward RMSNorm output) -- which exact tensor is
       settled empirically: we load both candidate inputs and report the FVU of
       each, keeping the one the transcoder actually reconstructs from.

Both write hook_mlp_out = post_feedforward_layernorm(mlp(...)), the vector added
to the residual stream. Same width per layer, no skip: a cleaner pair than Llama's.

    python -m cltf.gemma --out results/gemma_interventions.json
"""
import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from . import llama as LL
from .evaluate import summarize

MIRROR = "unsloth/gemma-2-2b"

# the GemmaScope set circuit-tracer uses (mntss/gemma-scope-transcoders config.yaml)
GS_L0 = [76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6, 8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41]


class Gemma:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForCausalLM
        self.hf = AutoModelForCausalLM.from_pretrained(MIRROR, dtype=torch.float32,
                                                       attn_implementation="eager").to(device).eval()
        self.m = self.hf.model
        self.cfg = self.hf.config
        self.n_layer = len(self.m.layers)
        self.d = self.cfg.hidden_size

    def embed(self, ids):
        return self.m.embed_tokens(ids) * torch.tensor(self.d ** 0.5, dtype=torch.float32)

    def _pe(self, h, T):
        pos = torch.arange(T, device=h.device)[None]
        self._mask = torch.full((T, T), torch.finfo(torch.float32).min, device=h.device).triu(1)[None, None]
        return self.m.rotary_emb(h, pos)

    def attn(self, l, h, pe):
        layer = self.m.layers[l]
        a = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=pe,
                            attention_mask=self._mask)[0]
        return layer.post_attention_layernorm(a)

    def mlp(self, l, h):
        layer = self.m.layers[l]
        return layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))

    def logits(self, h):
        z = self.hf.lm_head(self.m.norm(h))
        c = self.cfg.final_logit_softcapping
        return torch.tanh(z / c) * c if c else z

    @torch.no_grad()
    def run(self, ids, mlp_override=None):
        h = self.embed(ids)
        pe = self._pe(h, ids.shape[1])
        xs, ys = [], []
        for l in range(self.n_layer):
            h = h + self.attn(l, h, pe)
            xs.append(h)
            y = self.mlp(l, h)
            if mlp_override is not None and l in mlp_override:
                y = mlp_override[l]
            ys.append(y)
            h = h + y
        return self.logits(h), xs, ys


class GemmaCLT(LL.PublicCLT):
    def __init__(self, repo="mntss/clt-gemma-2-2b-426k", L=26, device="cuda"):
        self.kind = "clt"
        self.L = L
        self.enc, self.dec, self.b_dec = [], [], []
        for i in range(L):
            e = load_file(hf_hub_download(repo, f"W_enc_{i}.safetensors"), device=device)
            self.enc.append((e[f"W_enc_{i}"], e[f"b_enc_{i}"], e.get(f"threshold_{i}")))
            self.b_dec.append(e[f"b_dec_{i}"])
            self.dec.append(load_file(hf_hub_download(repo, f"W_dec_{i}.safetensors"), device=device)[f"W_dec_{i}"])
        self.targets = [list(range(s, L)) for s in range(L)]

    def encode(self, l, x):
        W, b, thr = self.enc[l]
        z = x.to(W.dtype) @ W.T + b
        return torch.relu(z) if thr is None else z * (z > thr)


class GemmaScopePLT:
    """read: "norm" = h / rms(h) (TransformerLens ln2.hook_normalized with folded weights)
             "weighted" = the model's pre_feedforward_layernorm(h)"""
    def __init__(self, L=26, device="cuda", read="norm"):
        self.kind = "plt"
        self.L = L
        self.mode = read
        self.p = []
        for i in range(L):
            f = hf_hub_download("google/gemma-scope-2b-pt-transcoders",
                                f"layer_{i}/width_16k/average_l0_{GS_L0[i]}/params.npz")
            z = np.load(f)
            self.p.append({k: torch.tensor(z[k], device=device, dtype=torch.float32) for k in z.files})
        self.b_dec = [p["b_dec"] for p in self.p]
        self.targets = [[s] for s in range(L)]

    def read(self, model, l, h):
        if self.mode == "weighted":
            return model.m.layers[l].pre_feedforward_layernorm(h)
        eps = model.cfg.rms_norm_eps
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)

    def encode(self, l, x):
        p = self.p[l]
        z = x @ p["W_enc"] + p["b_enc"]
        return z * (z > p["threshold"])

    def decode_from(self, l, f, x):
        return (f @ self.p[l]["W_dec"])[:, None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="results/gemma_interventions.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MIRROR)
    model = Gemma()
    ids = torch.randint(0, 250000, (2, 32), device="cuda")
    ids[:, 0] = tok.bos_token_id
    ref = model.hf(ids).logits
    ours, _, _ = model.run(ids)
    err = (ours - ref).abs().max().item()
    print("max |logit diff| vs HF:", err, flush=True)
    assert err < 5e-2
    batches = LL.batches_from_fineweb(tok, a.n, a.reps)
    results = {}

    def run(name, **kw):
        per = [LL.evaluate_one(model, name, ids, donor, alpha=a.alpha, **kw) for ids, donor in batches]
        merged = per[0]
        for k in ["clean_ce_real", "clean_ce_rep", "clean_kl"]:
            merged[k] = sum(p[k] for p in per) / len(per)
        if merged["fvu"] is not None:
            merged["fvu"] = [sum(p["fvu"][l] for p in per) / len(per) for l in range(len(merged["fvu"]))]
        for key in ["cells", "cells_src"]:
            for i, c in enumerate(merged[key]):
                for k in ["explained", "ratio", "cos", "real_mag"] + (["kl_real", "kl_rep"] if c["t"] == "logits" else []):
                    c[k] = sum(p[key][i][k] for p in per) / len(per)
        merged["summary"] = summarize(merged)
        print(name, json.dumps({k: round(v, 4) for k, v in merged["summary"].items()}),
              "fvu", None if merged["fvu"] is None else [round(v, 3) for v in merged["fvu"]], flush=True)
        return merged

    results["POS"] = run("POS")
    # pick the PLT input convention by reconstruction quality on real activations
    plt_tc = GemmaScopePLT(read="norm")
    _, xs, ys = model.run(batches[0][0])
    best = None
    for mode in ["norm", "weighted"]:
        plt_tc.mode = mode
        hat = LL.recon_from_xs(plt_tc, xs, model)
        fvu = np.mean([((ys[l][:, 1:] - hat[l][:, 1:]).pow(2).sum(-1).mean() /
                        (ys[l][:, 1:] - ys[l][:, 1:].mean((0, 1))).pow(2).sum(-1).mean()).item()
                       for l in range(model.n_layer)])
        print("PLT input", mode, "mean FVU", round(fvu, 3), flush=True)
        if best is None or fvu < best[1]:
            best = (mode, fvu)
    plt_tc.mode = best[0]
    assert best[1] < 0.6, "no input convention reconstructs: hook mismatch"
    for kind in ["frozen", "full"]:
        results[f"plt/{kind}"] = run(f"plt/{kind}", tc=plt_tc, kind=kind)
    del plt_tc
    torch.cuda.empty_cache()
    clt = GemmaCLT()
    for kind in ["frozen", "full"]:
        results[f"clt/{kind}"] = run(f"clt/{kind}", tc=clt, kind=kind)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
