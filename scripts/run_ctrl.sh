#!/bin/bash
# controller-L0 run, then evaluate + write share. usage: run_ctrl.sh NAME [train args...]
N=$1; shift
cd ~/clt-faithfulness && git stash -q; git pull -q
bash scripts/setup.sh > /dev/null 2>&1
python3 -m cltf.train --name $N --model ${MODEL:-gpt2} "$@" || exit 1
python3 -m cltf.evaluate --model ${MODEL:-gpt2} --runs $N --out results/gpt2_$N.json 2>&1 | grep -v -E "Loading|Warning"
python3 -m cltf.writeshare --model ${MODEL:-gpt2} --runs $N --out results/ws_$N.json 2>&1 | grep -v -E "Loading|Warning"
python3 -m cltf.bydist results/gpt2_$N.json | grep -A1 -E "LIN/frozen|$N/frozen"
python3 scripts/status.py | grep $N
