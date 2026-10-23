#!/bin/bash
# Deps + tokens on a fresh node (code already cloned to ~/clt-faithfulness).
set -e
cd ~/clt-faithfulness
pip3 install -q --user --break-system-packages transformers tiktoken pyarrow huggingface_hub 2>&1 | grep -v WARNING | tail -2 || true
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ ! -s /scratch/tok/fwe.bin ]; then
  mkdir -p /scratch/tok
  python3 -c "from cltf.data import prep_tokens; print('tokens', prep_tokens('/scratch/tok/fwe.bin', n_files=1, workers=24))"
fi
echo SETUP_OK
