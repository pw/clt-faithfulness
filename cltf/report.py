"""Turn an intervention-results JSON into readable tables.

    python -m cltf.report results/gpt2_interventions.json
"""
import json
import sys


def by_distance(cells, key="explained"):
    """Average a metric over (s,t) cells grouped by layer distance t-s."""
    out = {}
    for c in cells:
        if c["t"] == "logits":
            continue
        k = c["t"] - c["s"]
        out.setdefault(k, []).append(c[key])
    return {k: sum(v) / len(v) for k, v in sorted(out.items())}


def main(path):
    R = json.load(open(path))
    names = list(R)
    print("| model | clean CE (real→rep) | MLP explained | MLP ratio | logit explained | KL ratio |")
    print("|---|---|---|---|---|---|")
    for n in names:
        s = R[n]["summary"]
        print(f"| {n} | {s['clean_ce_real']:.3f}→{s['clean_ce_rep']:.3f} | {s['mlp_explained_mean']:.3f} "
              f"| {s['mlp_ratio_mean']:.3f} | {s['logit_explained_mean']:.3f} | {s['logit_kl_ratio_mean']:.3f} |")
    print()
    for key in ["explained", "ratio"]:
        dists = sorted(by_distance(R[names[0]]["cells"], key))
        print(f"### downstream MLP {key} by layer distance (t - s)")
        print("| model | " + " | ".join(str(d) for d in dists) + " |")
        print("|---|" + "---|" * len(dists))
        for n in names:
            bd = by_distance(R[n]["cells"], key)
            print(f"| {n} | " + " | ".join(f"{bd[d]:.2f}" for d in dists) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
