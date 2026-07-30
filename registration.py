"""
Self-registration + heartbeat client. Calls POST /dimensioners/register on the WMS backend using
this device's warehouse/zone identity (baked in via provisioning/set-warehouse-identity.sh into
.env) and its own device_id (device_config.py). First call mints a device token from the
warehouse's shared provisioning secret; every later call re-authenticates with that own token
instead (so rotating the warehouse secret never breaks an already-registered device).

Persists the minted token + resolved endpointUrl to registration_state.json so api.py's
_check_auth can pick up a freshly-rotated token on the very next request with no service
restart -- see load_registration_state()'s mtime-cached read in api.py.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from device_config import load_device_config

STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "registration_state.json")

WMS_BACKEND_URL = os.environ.get("WMS_BACKEND_URL", "https://api.uniewms.com/api/v1")
WAREHOUSE_CODE = os.environ.get("DIMENSIONER_WAREHOUSE_CODE", "")
ZONE_CODE = os.environ.get("DIMENSIONER_ZONE_CODE", "")
PROVISIONING_SECRET = os.environ.get("DIMENSIONER_PROVISIONING_SECRET", "")
CAMERA_MODEL = os.environ.get("DIMENSIONER_CAMERA_MODEL", "")
DEVICE_NAME = os.environ.get("DIMENSIONER_DEVICE_NAME", "")

REQUEST_TIMEOUT_SEC = 10


@dataclass
class RegistrationState:
    auth_token: Optional[str] = None
    endpoint_url: Optional[str] = None
    registered_at: Optional[str] = None


def load_registration_state() -> RegistrationState:
    if not os.path.exists(STATE_PATH):
        return RegistrationState()
    with open(STATE_PATH, "r") as f:
        data = json.load(f)
    return RegistrationState(
        auth_token=data.get("auth_token"),
        endpoint_url=data.get("endpoint_url"),
        registered_at=data.get("registered_at"),
    )


def save_registration_state(state: RegistrationState) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(asdict(state), f)


def _git_sha() -> Optional[str]:
    """Short commit SHA of whatever's actually checked out, read live rather than baked into a
    version file -- always current, and a device that isn't a git checkout at all (never
    bootstrapped onto Pi_System) simply reports none instead of erroring."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _lan_ip() -> str:
    """Best-effort LAN IP -- opens a UDP socket to a public address without actually sending
    anything, which is the standard no-network-call way to ask the OS which local interface/IP
    it would route through. Falls back to localhost if genuinely offline."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def register_or_heartbeat(force_rotate_token: bool = False) -> dict:
    """Calls POST /dimensioners/register. Returns the parsed response body. Raises
    urllib.error.HTTPError/URLError on a non-2xx response or network failure."""
    if not WAREHOUSE_CODE or not ZONE_CODE:
        raise RuntimeError(
            "DIMENSIONER_WAREHOUSE_CODE/DIMENSIONER_ZONE_CODE not set -- run "
            "provisioning/set-warehouse-identity.sh first."
        )

    device_id = load_device_config().device_id
    state = load_registration_state()

    # First call ever: no own token yet, must present the warehouse's shared provisioning
    # secret. Every later call presents the device's own token instead (see
    # dimensioner-registration.controller.ts's two-credential auth model) -- this is what lets
    # a warehouse rotate its provisioning secret without breaking already-registered devices.
    credential = state.auth_token or PROVISIONING_SECRET
    if not credential:
        raise RuntimeError(
            "No device token yet and DIMENSIONER_PROVISIONING_SECRET not set -- "
            "run provisioning/set-warehouse-identity.sh first."
        )

    body = {
        "warehouseCode": WAREHOUSE_CODE,
        "zoneCode": ZONE_CODE,
        "deviceId": device_id,
        "lanIp": _lan_ip(),
        "forceRotateToken": force_rotate_token,
    }
    if CAMERA_MODEL:
        body["cameraModel"] = CAMERA_MODEL
    if DEVICE_NAME:
        body["name"] = DEVICE_NAME
    git_sha = _git_sha()
    if git_sha:
        body["gitSha"] = git_sha

    request = urllib.request.Request(
        f"{WMS_BACKEND_URL}/dimensioners/register",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        result = json.loads(response.read().decode("utf-8"))

    if result.get("authToken"):
        state.auth_token = result["authToken"]
        state.endpoint_url = result.get("endpointUrl")
        state.registered_at = datetime.now(timezone.utc).isoformat()
        save_registration_state(state)

    return result


def fetch_route53_credentials() -> dict:
    """Calls POST /dimensioners/route53-credentials -- short-lived (1hr) AWS credentials this
    device's Caddy instance uses for its own Route53 DNS-01 ACME challenge, replacing the old
    flow where a human manually copied the backend's own long-lived Route53 IAM key onto the
    Pi (see provisioning/provision-pi.sh). Requires the device to have already completed
    self-registration (state.auth_token set) -- run set-warehouse-identity.sh first, same
    precondition register_or_heartbeat() has for every call after the first."""
    if not WAREHOUSE_CODE or not ZONE_CODE:
        raise RuntimeError(
            "DIMENSIONER_WAREHOUSE_CODE/DIMENSIONER_ZONE_CODE not set -- run "
            "provisioning/set-warehouse-identity.sh first."
        )

    state = load_registration_state()
    credential = state.auth_token or PROVISIONING_SECRET
    if not credential:
        raise RuntimeError(
            "No device token yet -- run set-warehouse-identity.sh (which self-registers) "
            "before requesting Route53 credentials."
        )

    body = {"warehouseCode": WAREHOUSE_CODE, "zoneCode": ZONE_CODE, "deviceId": load_device_config().device_id}
    request = urllib.request.Request(
        f"{WMS_BACKEND_URL}/dimensioners/route53-credentials",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_vendor_driver_download_url() -> dict:
    """Calls POST /dimensioners/vendor-driver-download-url -- a short-lived (10min) presigned S3
    URL for the licensed Deptrum camera driver tarball, used once during initial golden-image
    setup (see provisioning/setup-golden-image.sh). Same auth precondition as
    fetch_route53_credentials(): requires self-registration to have already happened."""
    if not WAREHOUSE_CODE or not ZONE_CODE:
        raise RuntimeError(
            "DIMENSIONER_WAREHOUSE_CODE/DIMENSIONER_ZONE_CODE not set -- run "
            "provisioning/set-warehouse-identity.sh first."
        )

    state = load_registration_state()
    credential = state.auth_token or PROVISIONING_SECRET
    if not credential:
        raise RuntimeError(
            "No device token yet -- run set-warehouse-identity.sh (which self-registers) "
            "before requesting the vendor driver download URL."
        )

    body = {"warehouseCode": WAREHOUSE_CODE, "zoneCode": ZONE_CODE, "deviceId": load_device_config().device_id}
    request = urllib.request.Request(
        f"{WMS_BACKEND_URL}/dimensioners/vendor-driver-download-url",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    import sys

    if "--route53-credentials" in sys.argv:
        print(json.dumps(fetch_route53_credentials()))
    elif "--vendor-driver-download-url" in sys.argv:
        print(json.dumps(fetch_vendor_driver_download_url()))
    else:
        force_rotate = "--force-rotate-token" in sys.argv
        outcome = register_or_heartbeat(force_rotate_token=force_rotate)
        print(json.dumps(outcome))
