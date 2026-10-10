"""Run a sequence of long-lived experiments with durable stage status.

The queue is intentionally provider-agnostic: run it in a Lightning Studio,
on a workstation, or in any Linux job runner. A failed stage stops the queue;
``--resume`` continues from the first incomplete stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable experiment queue.")
    parser.add_argument("--plan", type=Path, required=True, help="JSON experiment plan")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory for logs and status")
    parser.add_argument("--resume", action="store_true", help="Skip previously successful stages")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    plan = read_json(args.plan)
    stages = plan.get("stages", [])
    if not stages:
        raise ValueError("The plan must contain at least one stage.")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    write_json(args.run_dir / "plan.json", plan)
    status_path = args.run_dir / "status.json"
    status = read_json(status_path) if args.resume and status_path.exists() else {"stages": {}}

    for stage in stages:
        name = stage["name"]
        command = stage["command"]
        previous = status["stages"].get(name, {})
        if args.resume and previous.get("state") == "completed":
            print(f"Skipping completed stage: {name}")
            continue

        print(f"\n==> {name}: {' '.join(command)}")
        if args.dry_run:
            continue

        log_path = args.run_dir / "logs" / f"{name}.log"
        status["stages"][name] = {"state": "running", "started_at": utc_now(), "command": command}
        write_json(status_path, status)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)

        state = "completed" if result.returncode == 0 else "failed"
        status["stages"][name] = {
            "state": state,
            "started_at": status["stages"][name]["started_at"],
            "finished_at": utc_now(),
            "command": command,
            "return_code": result.returncode,
            "log": str(log_path),
        }
        write_json(status_path, status)
        if result.returncode:
            raise SystemExit(f"Stage '{name}' failed; see {log_path}")


if __name__ == "__main__":
    main()
