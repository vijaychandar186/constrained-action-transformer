"""Upload code without secrets, then start the sequential queue with a runtime limit."""
from __future__ import annotations
import argparse
import datetime
import os
import shlex
import zipfile
from pathlib import Path
from lightningai.config import ROOT, initialize, teamspace
from src.pipeline import write_json

def package(destination):
    files = []
    for folder in ("src", "scripts", "experiments", "lightningai", "ui"):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in (".py", ".json", ".sh", ".txt", ".md", ".html", ".css", ".js", ".LICENSE") and "__pycache__" not in path.parts and path.name not in ("launch.json", "large_launch.json", "comparison_launch.json", "large_rl_launch.json", "serving.json"))
    files.extend(ROOT / name for name in ("config.py", "requirements.txt", "requirements-serving.txt", "README.md", "TODO.md"))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.name == ".env":
                raise RuntimeError("Credential file must never be uploaded.")
            name = path.relative_to(ROOT).as_posix()
            if path.suffix == ".sh":
                archive.writestr(name, path.read_bytes().replace(b"\r\n", b"\n"))
            else:
                archive.write(path, name)
    return files

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    settings = initialize()
    launch_path = ROOT / "lightningai" / "launch.json"
    if launch_path.exists():
        previous = __import__("json").loads(launch_path.read_text())
        if previous.get("state") == "submitted":
            raise RuntimeError("A queue was already submitted; inspect it before launching another.")
    archive = ROOT / "lightningai" / "package.zip"
    files = package(archive)
    if a.dry_run:
        print("Package files:", len(files), "| Secrets excluded | Machine:", settings["machine"],
              "| Runtime cap (seconds):", settings["max_runtime_seconds"], "| Execution:", settings.get("execution", "studio"))
        return
    from lightning_sdk import Job, Studio, Machine
    studio = None
    detached = False
    started_here = False
    try:
        space = teamspace()
        studio = Studio(settings["studio"], teamspace=space, create_ok=True)
        running = str(studio.status).split(".")[-1].lower() == "running"
        if not running:
            print("Starting build Studio on CPU.", flush=True)
            studio.start(machine=Machine.CPU)
            started_here = True
        remote = settings["remote_dir"]
        if not remote or any(part in ("..", "") for part in remote.split("/")):
            raise ValueError("remote_dir must be a simple relative path.")
        remote_archive = "queens-gambit-package.zip"
        studio.upload_file(str(archive), remote_path=remote_archive)
        directory = shlex.quote(remote)
        archive_path = shlex.quote(remote_archive)
        studio.run(f"mkdir -p ~/{directory} && python -m zipfile -e ~/{archive_path} ~/{directory} && cd ~/{directory} && bash lightningai/bootstrap.sh")
        plan, run_dir = shlex.quote(settings["plan"]), shlex.quote(settings["run_dir"])
        seconds = int(settings["max_runtime_seconds"])
        execution = settings.get("execution", "studio")
        if execution == "studio":
            # Native Studio credentials handle shutdown; the local key is never copied.
            studio.run(f"cd ~/{directory} && python -m pip install -q -r lightningai/requirements.txt && command -v timeout")
            print("Switching Studio to", settings["machine"], flush=True)
            studio.switch_machine(Machine.from_str(settings["machine"]))
            command = f"cd ~/{directory} && mkdir -p {run_dir} && python -u -m lightningai.run_studio_queue > {run_dir}/studio_queue.log 2>&1"
            output, code = studio.run_and_detach(command, timeout=15)
            if code is not None:
                raise RuntimeError("Queue process exited during startup; inspect studio_queue.log.")
            record = {"state": "submitted", "execution": "studio", "name": studio.name,
                      "teamspace": space.name, "org": settings.get("org"),
                      "studio": studio.name, "machine": settings["machine"],
                      "max_runtime_seconds": seconds, "runtime_limit": "GNU timeout with Studio shutdown"}
            write_json(launch_path, record)
            detached = True
            print("Queue started in Studio:", studio.name, flush=True)
        else:
            command = f"cd ~/{directory} && export PATH=/usr/games:$PATH && exec timeout --signal=TERM --kill-after=60s {seconds}s python scripts/run_experiment_queue.py --plan {plan} --run-dir {run_dir} --resume"
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            name = "queens-gambit-" + stamp
            job = Job.run(name=name, teamspace=space, studio=studio, machine=Machine.from_str(settings["machine"]),
                          command=command, keep_minutes=0, reuse_snapshot=False)
            write_json(launch_path, {"state": "submitted", "execution": "job", "name": name, "url": job.link,
                                     "teamspace": space.name, "studio": studio.name,
                                     "machine": settings["machine"], "max_runtime_seconds": seconds})
            print("Queue submitted:", job.link, flush=True)
    except Exception as error:
        message = str(getattr(error, "body", None) or error).replace(os.environ.get("LIGHTNING_API_KEY", "<unused>"), "[redacted]")
        write_json(launch_path, {"state": "blocked", "error_type": type(error).__name__, "reason": message[:1000]})
        print("Queue launch blocked:", type(error).__name__, message[:300], flush=True)
        raise SystemExit(1) from None
    finally:
        if not detached and started_here and studio is not None:
            studio.stop()

if __name__ == "__main__":
    main()
