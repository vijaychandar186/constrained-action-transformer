"""Run the sequential queue in a Studio and stop compute on exit."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from lightningai.config import ROOT, initialize, teamspace
from src.pipeline import write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-probe", action="store_true")
    parser.add_argument('--plan')
    parser.add_argument('--run-dir')
    parser.add_argument('--max-runtime', type=int)
    args = parser.parse_args()
    settings = initialize()
    if args.plan:
        settings['plan'] = args.plan
    if args.run_dir:
        settings['run_dir'] = args.run_dir
    if args.max_runtime:
        settings['max_runtime_seconds'] = args.max_runtime
    from lightning_sdk import Studio
    studio = Studio(settings["studio"], teamspace=teamspace(), create_ok=False)
    directory = ROOT / settings["run_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    if args.stop_probe:
        write_json(directory / "native_stop_probe.json", {"state": "stop requested"})
        print("Requesting CPU Studio shutdown.", flush=True)
        studio.stop()
        return
    if not shutil.which("timeout"):
        raise RuntimeError("GNU timeout is required before launching GPU training.")
    started = time.monotonic()
    code = 1
    write_json(directory / "wrapper_status.json", {"state": "running", "runtime_limit_seconds": settings["max_runtime_seconds"]})
    try:
        command = ["timeout", "--signal=TERM", "--kill-after=60s",
                   str(int(settings["max_runtime_seconds"])) + "s",
                   sys.executable, str(ROOT / "scripts/run_experiment_queue.py"),
                   "--plan", settings["plan"], "--run-dir", settings["run_dir"], "--resume"]
        environment = dict(__import__("os").environ)
        environment["PATH"] = "/usr/games:" + environment.get("PATH", "")
        code = subprocess.run(command, cwd=ROOT, env=environment).returncode
    finally:
        write_json(directory / "wrapper_status.json",
                   {"state": "completed" if code == 0 else "failed", "return_code": code,
                    "elapsed_seconds": time.monotonic() - started, "timed_out": code == 124})
        print("Stopping Studio compute.", flush=True)
        studio.stop()

if __name__ == "__main__":
    main()
