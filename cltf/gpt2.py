"""A plain GPT-2 forward pass we control layer by layer.

Every MLP's input (post-ln_2) and output can be read, replaced, or
intervened on. Weights come from HuggingFace `gpt2`; `check_against_hf`
proves the logits match the reference implementation.
"""
import math
import torch
import torch.nn.functional as F


class GPT2:
    def __init__(self, sd, device="cuda", dtype=torch.float32, n_layer=12, n_head=12):
        g = lambda k: sd[k].to(device=device, dtype=dtype)
        self.wte = g("transformer.wte.weight")
        self.wpe = g("transformer.wpe.weight")
        self.n_layer = n_layer
        self.n_head = n_head
        self.d = self.wte.shape[1]
        self.L = []
        for l in range(self.n_layer):
            p = f"transformer.h.{l}."
            self.L.append({k: g(p + k) for k in [
                "ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias",
                "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]})
        self.lnf_w = g("transformer.ln_f.weight")
        self.lnf_b = g("transformer.ln_f.bias")

    def ln(self, x, w, b):
        return F.layer_norm(x, (self.d,), w, b, 1e-5)

    def attn(self, l, h):
        P = self.L[l]
        x = self.ln(h, P["ln_1.weight"], P["ln_1.bias"])
        qkv = x @ P["attn.c_attn.weight"] + P["attn.c_attn.bias"]
        B, T, _ = qkv.shape
        q, k, v = qkv.split(self.d, dim=-1)
        sh = lambda t: t.view(B, T, self.n_head, self.d // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, self.d)
        return y @ P["attn.c_proj.weight"] + P["attn.c_proj.bias"]

    def mlp_in(self, l, h):
        P = self.L[l]
        return self.ln(h, P["ln_2.weight"], P["ln_2.bias"])

    def mlp(self, l, x):
        P = self.L[l]
        a = F.gelu(x @ P["mlp.c_fc.weight"] + P["mlp.c_fc.bias"], approximate="tanh")
        return a @ P["mlp.c_proj.weight"] + P["mlp.c_proj.bias"]

    def embed(self, ids):
        T = ids.shape[1]
        return self.wte[ids] + self.wpe[torch.arange(T, device=ids.device)]

    def logits(self, h):
        return self.ln(h, self.lnf_w, self.lnf_b) @ self.wte.T

    @torch.no_grad()
    def run(self, ids, mlp_override=None):
        """Real model. Returns (logits, xs, ys): per-layer MLP inputs and outputs.

        mlp_override: dict layer -> tensor; that layer's MLP output is replaced.
        """
        h = self.embed(ids)
        xs, ys = [], []
        for l in range(self.n_layer):
            h = h + self.attn(l, h)
            x = self.mlp_in(l, h)
            y = self.mlp(l, x)
            if mlp_override is not None and l in mlp_override:
                y = mlp_override[l]
            xs.append(x)
            ys.append(y)
            h = h + y
        return self.logits(h), xs, ys


def load_gpt2(device="cuda", dtype=torch.float32, name="gpt2"):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(name)
    c = hf.config
    return GPT2(hf.state_dict(), device, dtype, c.n_layer, c.n_head), hf


@torch.no_grad()
def check_against_hf(model, hf, device="cuda"):
    hf = hf.to(device).eval()
    ids = torch.randint(0, 50257, (2, 64), device=device)
    ref = hf(ids).logits.float()
    ours, _, _ = model.run(ids)
    err = (ours.float() - ref).abs().max().item()
    return err
