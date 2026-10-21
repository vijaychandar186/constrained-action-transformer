#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
if ! command -v stockfish >/dev/null && [[ ! -x /usr/games/stockfish ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y stockfish
fi
python -c 'import torch, chess, zstandard; print("Training dependencies ready")'

python -m pip install -q -r lightningai/requirements.txt
