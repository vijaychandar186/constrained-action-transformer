"""Load Lightning credentials locally and construct authenticated SDK objects."""
from __future__ import annotations
import json
import http.client
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Path(__file__).with_name("settings.json")

def initialize():
    os.environ["LIGHTNING_DEBUG"] = "0"
    os.environ["DEBUG"] = "0"
    # The SDK can be imported before this function, so reset its cached flag too.
    from lightning_sdk.lightning_cloud import env as cloud_env
    cloud_env.DEBUG = False
    http.client.HTTPConnection.debuglevel = 0
    http.client.HTTPSConnection.debuglevel = 0
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if not os.environ.get("LIGHTNING_API_KEY"):
        raise RuntimeError("Set LIGHTNING_API_KEY in the repository .env.")
    if not SETTINGS.exists():
        raise RuntimeError(
            "Create lightningai/settings.json from lightningai/settings.example.json before using Lightning."
        )
    return json.loads(SETTINGS.read_text())

def account():
    initialize()
    from lightning_sdk import User
    from lightning_sdk.lightning_cloud import env
    env.DEBUG = False
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    identity = LightningClient().auth_service_whoami()
    user = User(identity.username)
    os.environ.setdefault("LIGHTNING_USERNAME", user.name)
    return user

def teamspace():
    settings = initialize()
    from lightning_sdk import Teamspace
    if settings.get("teamspace"):
        return Teamspace(settings["teamspace"], org=settings.get("org"), user=settings.get("user"))
    spaces = account().teamspaces
    if len(spaces) != 1:
        raise RuntimeError("Set teamspace in lightningai/settings.json to choose an account workspace.")
    return spaces[0]

if __name__ == "__main__":
    try:
        user = account()
        print("Authenticated user:", user.name)
        print("Teamspaces:", [{"name": ts.name, "owner_type": str(ts._teamspace.owner_type), "owner_id": ts._teamspace.owner_id} for ts in user.teamspaces])
    except Exception as error:
        print("Lightning authentication failed:", type(error).__name__)
        raise SystemExit(1) from None
