"""Run stages in order, persist their status, and resume completed work."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def validate_plan(plan):
    names = set()
    for stage in plan.get("stages", []):
        name = stage["name"]
        command = stage["command"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ValueError("Stage names must be unique simple filenames.")
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError("Stage commands must be nonempty lists of strings.")
        if stage.get("timeout_seconds", 1) <= 0:
            raise ValueError("Stage timeouts must be positive.")
        if len(command) >= 3 and command[0] in ("python", "python3") and command[1] == "-m":
            module = ROOT.joinpath(*command[2].split(".")).with_suffix(".py")
            if not module.exists():
                raise ValueError("Stage module does not exist: " + command[2])
        names.add(name)
    if not names:
        raise ValueError("The plan must contain at least one stage.")

def process_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    # os.kill(pid, 0) can terminate a process on Windows; query it instead.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)

def run_command(command, log, timeout):
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"},
                               start_new_session=os.name != "nt")
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    plan = read_json(a.plan)
    validate_plan(plan)
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if a.dry_run:
        for stage in plan["stages"]:
            print(stage["name"], ":", subprocess.list2cmdline(stage["command"]))
        return
    a.run_dir = a.run_dir.resolve()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    (a.run_dir / "logs").mkdir(exist_ok=True)
    status_path = a.run_dir / "status.json"
    if status_path.exists() and not a.resume:
        raise ValueError("Existing run directory; use --resume or choose a new directory.")
    status = read_json(status_path) if status_path.exists() else {"plan_sha256": digest, "stages": {}}
    if status.get("plan_sha256") != digest:
        raise ValueError("The saved plan differs; use a new run directory.")
    lock = a.run_dir / ".queue.lock"
    if lock.exists():
        owner = read_json(lock)
        alive = owner["host"] == platform.node() and process_alive(owner["pid"])
        if alive:
            raise RuntimeError("Another queue owns this run directory.")
        lock.unlink()
    handle = lock.open("x", encoding="utf-8")
    json.dump({"pid": os.getpid(), "host": platform.node()}, handle)
    handle.close()
    try:
        status["state"] = "running"
        write_json(a.run_dir / "plan.json", plan)
        for stage in plan["stages"]:
            name = stage["name"]
            artifacts = [ROOT / artifact for artifact in stage.get("artifacts", [])]
            previous = status["stages"].get(name, {})
            if a.resume and previous.get("state") == "completed":
                if not all(path.exists() for path in artifacts):
                    raise RuntimeError("Completed stage has missing artifacts: " + name)
                print("Skipping completed stage:", name, flush=True)
                continue
            command = list(stage["command"])
            if command[0] in ("python", "python3"):
                command[0] = sys.executable
            log_path = a.run_dir / "logs" / (name + ".log")
            record = {"state": "running", "started_at": utc_now(), "command": command, "log": str(log_path)}
            status["stages"][name] = record
            write_json(status_path, status)
            print("Starting:", name, flush=True)
            code, error = 1, None
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    code = run_command(command, log, stage.get("timeout_seconds"))
                if code == 0 and not all(path.exists() for path in artifacts):
                    code, error = 1, "Required output artifact missing."
            except subprocess.TimeoutExpired:
                error = "Stage exceeded timeout."
            except KeyboardInterrupt:
                error = "Queue interrupted."
            except OSError as exception:
                error = str(exception)
            record.update(state="completed" if code == 0 else "failed", finished_at=utc_now(), return_code=code)
            if error:
                record["error"] = error
            write_json(status_path, status)
            if code:
                status["state"] = "failed"
                write_json(status_path, status)
                raise SystemExit("Stage failed: " + name + "; see " + str(log_path))
            print("Completed:", name, flush=True)
        status["state"] = "completed"
        status["finished_at"] = utc_now()
        write_json(status_path, status)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
