"""Print cos / ratio by layer distance for every model in a results JSON.

    python -m cltf.bydist results/x.json [cells|cells_src]
"""
import collections
import json
import sys


def table(R, key="cells"):
    for n, r in R.items():
        cells = [c for c in r.get(key, []) if c["t"] != "logits"]
        if not cells:
            continue
        bd = collections.defaultdict(list)
        for c in cells:
            bd[c["t"] - c["s"]].append(c)
        ks = sorted(bd)
        print(f"{n:<22} cos   " + " ".join(f"{sum(c['cos'] for c in bd[k]) / len(bd[k]):.2f}" for k in ks))
        print(f"{'':<22} ratio " + " ".join(f"{sum(c['ratio'] for c in bd[k]) / len(bd[k]):.2f}" for k in ks))


if __name__ == "__main__":
    R = json.load(open(sys.argv[1]))
    table(R, sys.argv[2] if len(sys.argv) > 2 else "cells")
