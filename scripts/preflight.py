"""Check prerequisites before a long experiment consumes compute."""
import argparse
import platform
import subprocess
from pathlib import Path
import chess.engine
import torch
from src.pipeline import choose_device, file_hash, stockfish_path, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    device = choose_device(a.require_cuda)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path()) as engine:
        engine_id = dict(engine.id)
    files = sorted(Path("src").glob("*.py")) + sorted(Path("scripts").glob("*.py"))
    write_json(a.output, {"python": platform.python_version(), "torch": torch.__version__,
                          "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                          "stockfish": engine_id, "source_sha256": {str(path): file_hash(path) for path in files}})
    print("Preflight passed:", device, engine_id, flush=True)

if __name__ == "__main__":
    main()
