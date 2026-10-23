"""Tokens and an activation shuffle buffer for GPT-2 small.

Tokens: fineweb-edu sample-10BT, GPT-2 BPE, uint16 memmap (prep_tokens).
Activations: per-layer MLP input x and output y, generated online from GPT-2,
position 0 dropped (GPT-2's first position is a massive-norm outlier).
"""
import os
import numpy as np
import torch

SEQ = 128


def prep_tokens(out_path, n_files=1, workers=16):
    import glob
    import tiktoken
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from multiprocessing import Pool

    d = snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset",
                          allow_patterns=[f"sample/10BT/00{i}_00000.parquet" for i in range(n_files)],
                          local_dir=os.path.dirname(out_path) + "/fwe")
    files = sorted(glob.glob(d + "/sample/10BT/*.parquet"))
    enc = tiktoken.get_encoding("gpt2")
    texts = []
    for f in files:
        texts += pq.read_table(f, columns=["text"]).column("text").to_pylist()
    eot = enc.eot_token

    def chunks(xs, n):
        k = len(xs) // n + 1
        return [xs[i:i + k] for i in range(0, len(xs), k)]

    with Pool(workers) as p:
        parts = p.map(_encode_batch, chunks(texts, workers * 8))
    toks = np.concatenate(parts).astype(np.uint16)
    toks.tofile(out_path)
    return len(toks)


def _encode_batch(texts):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    out = []
    for t in texts:
        out.extend(enc.encode_ordinary(t))
        out.append(enc.eot_token)
    return np.array(out, dtype=np.uint16)


class Tokens:
    def __init__(self, path, seed=0, split="train"):
        self.t = np.memmap(path, dtype=np.uint16, mode="r")
        n_seq = len(self.t) // SEQ
        # last 2% of sequences are held out for evaluation
        cut = int(n_seq * 0.98)
        self.idx = np.arange(0, cut) if split == "train" else np.arange(cut, n_seq)
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.idx)
        self.p = 0

    def batch(self, n, device="cuda"):
        if self.p + n > len(self.idx):
            self.rng.shuffle(self.idx)
            self.p = 0
        rows = self.idx[self.p:self.p + n]
        self.p += n
        arr = np.stack([self.t[r * SEQ:(r + 1) * SEQ] for r in rows]).astype(np.int64)
        return torch.from_numpy(arr).to(device)


class ActivationBuffer:
    def __init__(self, model, tokens, size=262144, gen_seqs=128, device="cuda"):
        self.model, self.tokens = model, tokens
        self.size, self.gen_seqs, self.device = size, gen_seqs, device
        L, d = model.n_layer, model.d
        self.X = torch.empty(L, size, d, dtype=torch.bfloat16, device=device)
        self.Y = torch.empty(L, size, d, dtype=torch.bfloat16, device=device)
        self.filled = 0
        self._fill(0, size)
        self.order = torch.randperm(size, device=device)
        self.p = 0

    @torch.no_grad()
    def _gen(self):
        ids = self.tokens.batch(self.gen_seqs, self.device)
        _, xs, ys = self.model.run(ids)
        x = torch.stack(xs)[:, :, 1:].reshape(self.model.n_layer, -1, self.model.d)
        y = torch.stack(ys)[:, :, 1:].reshape(self.model.n_layer, -1, self.model.d)
        return x, y

    def _fill(self, start, end):
        p = start
        while p < end:
            x, y = self._gen()
            n = min(x.shape[1], end - p)
            self.X[:, p:p + n] = x[:, :n].to(torch.bfloat16)
            self.Y[:, p:p + n] = y[:, :n].to(torch.bfloat16)
            p += n

    def next(self, n):
        if self.p + n > self.size:
            half = self.size // 2
            # refresh a random half of the buffer, then reshuffle
            perm = torch.randperm(self.size, device=self.device)
            keep, fresh = perm[:half], perm[half:]
            Xn = torch.empty(self.model.n_layer, self.size - half, self.model.d,
                             dtype=torch.bfloat16, device=self.device)
            Yn = torch.empty_like(Xn)
            p = 0
            while p < Xn.shape[1]:
                x, y = self._gen()
                k = min(x.shape[1], Xn.shape[1] - p)
                Xn[:, p:p + k] = x[:, :k].to(torch.bfloat16)
                Yn[:, p:p + k] = y[:, :k].to(torch.bfloat16)
                p += k
            self.X[:, fresh] = Xn
            self.Y[:, fresh] = Yn
            self.order = torch.randperm(self.size, device=self.device)
            self.p = 0
        i = self.order[self.p:self.p + n]
        self.p += n
        return self.X[:, i].float(), self.Y[:, i].float()
