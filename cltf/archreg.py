"""Is there an architecture effect once sparsity is accounted for? (public models)

For each adjacent pair (s, t=s+1), regress the response cosine on ln(L0) plus a CLT indicator.
L0 pairing: CLT -> L0 of source layer t (its features that can respond); PLT -> L0 of layer t.

    python -m cltf.archreg
"""
import json, math
import numpy as np


def fit(cells_clt, cells_plt, l0c, l0p, fixed_effects=False):
    rows = []
    for s, v in cells_clt.items():
        rows.append((math.log(l0c[s + 1]), 1, v, s))
    for s, v in cells_plt.items():
        rows.append((math.log(l0p[s + 1]), 0, v, s))
    y = np.array([r[2] for r in rows])
    if fixed_effects:
        S = sorted(set(r[3] for r in rows))
        X = np.array([[r[0], r[1]] + [1.0 if r[3] == k else 0.0 for k in S] for r in rows]); j = 1
    else:
        X = np.array([[1.0, r[0], r[1]] for r in rows]); j = 2
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    cov = (r @ r / (len(y) - X.shape[1])) * np.linalg.pinv(X.T @ X)
    return b[j], math.sqrt(cov[j, j])


def d1(R, name):
    return {c["s"]: c["cos"] for c in R[name]["cells"] if c["t"] != "logits" and c["t"] - c["s"] == 1}


if __name__ == "__main__":
    L = json.load(open("results/l0_profiles.json"))
    for model, f in [("llama", "results/llama_interventions.json"), ("gemma", "results/gemma_interventions.json")]:
        R = json.load(open(f))
        c, p = d1(R, "clt/frozen"), d1(R, "plt/frozen")
        for fe in (False, True):
            est, se = fit(c, p, L[model + "_clt"], L[model + "_plt"], fe)
            print(f"{model:6} {'layer FE' if fe else 'pooled  '}  CLT term {est:+.3f} (SE {se:.3f})")
