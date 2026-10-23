"""One transcoder class covering every variant we compare.

A feature living at source layer s reads the MLP input x_s and writes into the
MLP outputs of the layers in `targets(s)`:
    PLT       targets(s) = {s}
    CLT       targets(s) = {s, ..., L-1}
    window k  targets(s) = {s, ..., s+k-1}

Sparsity penalty  lam * sum_i tanh(c * f_i * w_i)  (Anthropic's CLT recipe),
where w_i is the feature's decoder norm, either
    "concat": sqrt(sum_t ||W_dec[s->t]_i||^2)   (standard)
    "sum":    sum_t ||W_dec[s->t]_i||           (the proposed fix: writing to
              more layers costs linearly, not sublinearly)
Activation: JumpReLU with straight-through threshold gradients.
"""
import torch
import torch.nn as nn


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, log_theta, eps):
        theta = log_theta.exp()
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return z * (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, g):
        z, theta = ctx.saved_tensors
        eps = ctx.eps
        active = (z > theta).to(z.dtype)
        gz = g * active
        # d/dtheta of z*H(z-theta) ~= -theta * (1/eps) * rect((z-theta)/eps)
        rect = (((z - theta) / eps).abs() < 0.5).to(z.dtype)
        gtheta = (g * (-theta / eps) * rect).sum(0)
        # chain rule through theta = exp(log_theta)
        return gz, gtheta * theta, None


def jumprelu(z, log_theta, eps=0.1):
    return _JumpReLU.apply(z, log_theta, eps)


class Transcoder(nn.Module):
    def __init__(self, n_layer=12, d=768, F=8192, kind="clt", window=None, norm="concat"):
        super().__init__()
        self.n_layer, self.d, self.F = n_layer, d, F
        self.kind, self.window, self.norm = kind, window, norm
        self.targets = [self._targets(s) for s in range(n_layer)]
        self.W_enc = nn.Parameter(torch.empty(n_layer, d, F))
        self.b_enc = nn.Parameter(torch.zeros(n_layer, F))
        self.log_theta = nn.Parameter(torch.full((n_layer, F), 0.0))  # theta = 1: ~4% active at init
        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.empty(F, len(self.targets[s]), d)) for s in range(n_layer)])
        self.b_dec = nn.Parameter(torch.zeros(n_layer, d))
        bound_e = 1.0 / d ** 0.5
        nn.init.uniform_(self.W_enc, -bound_e, bound_e)
        for s in range(n_layer):
            bound_d = 0.1 / d ** 0.5  # per-target decoder rows of norm ~0.06
            nn.init.uniform_(self.W_dec[s], -bound_d, bound_d)

    def _targets(self, s):
        if self.kind == "plt":
            return [s]
        if self.kind == "clt":
            return list(range(s, self.n_layer))
        if self.kind == "window":
            return list(range(s, min(s + self.window, self.n_layer)))
        raise ValueError(self.kind)

    def dec_norm(self, s):
        per_target = self.W_dec[s].norm(dim=-1)  # (F, n_t)
        if self.norm == "concat":
            return per_target.pow(2).sum(-1).sqrt()
        if self.norm == "sum":
            return per_target.sum(-1)
        raise ValueError(self.norm)

    def encode(self, s, x):
        z = x @ self.W_enc[s] + self.b_enc[s]
        return z, jumprelu(z, self.log_theta[s])

    def decode_from(self, s, f):
        """f: (N, F) acts at source s -> (N, n_targets, d)."""
        W = self.W_dec[s]
        return (f @ W.reshape(self.F, -1)).view(f.shape[0], W.shape[1], self.d)

    def forward(self, xs):
        """xs: (L, N, d) normalised MLP inputs -> yhat (L, N, d), acts list, pre list."""
        N = xs.shape[1]
        yhat = self.b_dec[:, None, :].expand(self.n_layer, N, self.d).clone()
        acts, pres = [], []
        for s in range(self.n_layer):
            z, f = self.encode(s, xs[s])
            acts.append(f)
            pres.append(z)
            out = self.decode_from(s, f)
            for j, t in enumerate(self.targets[s]):
                yhat[t] = yhat[t] + out[:, j]
        return yhat, acts, pres
