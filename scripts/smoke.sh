#!/bin/bash
# Correctness + speed smoke test on one GPU.
set -e
cd ~/clt-faithfulness
python3 -c "
from cltf.gpt2 import load_gpt2, check_against_hf
m, hf = load_gpt2(); e = check_against_hf(m, hf); print('max |logit diff| vs HF:', e); assert e < 1e-2
"
for v in "plt --kind plt" "clt --kind clt" "clts --kind clt --norm sum" "win --kind window --window 3"; do
  set -- $v; n=smoke_$1; shift
  python3 -m cltf.train --name $n "$@" --tokens 1.2e6 --out ~/runs
  tail -1 ~/runs/$n/log.jsonl
done
python3 -m cltf.evaluate --runs smoke_plt smoke_clt --n 16 --reps 1 --out results/smoke.json
