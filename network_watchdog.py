"""
Network-reachability watchdog: detects the Pi losing network connectivity (wired OR WiFi -- this
fleet has both) and forces a reconnect on whichever interface actually carries the default route.
Runs as a periodic oneshot (dimensioner-network-watchdog.timer, every 2 minutes) -- tighter than
dimensioner-heartbeat.timer's 5-minute cadence, since losing the network is exactly the thing that
would also silently break the heartbeat/DNS-sync loop, so this needs to notice and recover faster
than that dependent loop's own cadence.

Deliberately does NOT reuse registration.py's _lan_ip() as the reachability check: that function
opens a UDP socket and only raises if there's no route at all (interface down / no gateway
configured) -- it happily reports "fine" even when the interface is up but the gateway or the
wider internet is actually unreachable, exactly the failure mode this watchdog exists to catch
(e.g. associated but the router rebooted, or the link's fine but upstream internet is down).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from registration import WMS_BACKEND_URL

STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "network_watchdog_state.json")

# Require this many CONSECUTIVE failed checks before reconnecting -- a single transient blip
# (e.g. one dropped ping) should never trigger a reconnect action on its own.
CONSECUTIVE_FAILURES_BEFORE_ACTION = 2

PING_TIMEOUT_SEC = 2
PING_COUNT = 2
BACKEND_CHECK_TIMEOUT_SEC = 5

# Fallback ONLY for the rare case where the default route has vanished entirely (so there's no
# interface name to read off the routing table) AND this is the very first check ever (no
# last-known interface persisted in state either). Real fleet devices are a mix of wired (eth0)
# and WiFi (wlan0) -- confirmed WH-007's ED1 is wired -- so this is deliberately just a
# last-resort guess, not the primary way the interface gets chosen (see _default_route()).
FALLBACK_INTERFACE = os.environ.get("DIMENSIONER_NETWORK_INTERFACE", "wlan0")


@dataclass
class WatchdogState:
    consecutive_failures: int = 0
    last_check_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_action_at: Optional[str] = None
    last_known_interface: Optional[str] = None


def load_state() -> WatchdogState:
    if not os.path.exists(STATE_PATH):
        return WatchdogState()
    with open(STATE_PATH, "r") as f:
        data = json.load(f)
    return WatchdogState(
        consecutive_failures=data.get("consecutive_failures", 0),
        last_check_at=data.get("last_check_at"),
        last_success_at=data.get("last_success_at"),
        last_action_at=data.get("last_action_at"),
        last_known_interface=data.get("last_known_interface"),
    )


def save_state(state: WatchdogState) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(asdict(state), f)


def _default_route() -> "tuple[Optional[str], Optional[str]]":
    """Parses `ip route show default` for the gateway IP AND the interface actually carrying it
    (e.g. `eth0` on a wired device, `wlan0` on a WiFi one -- this fleet has both kinds, so the
    reconnect action below must target whichever interface is actually in use, not a hardcoded
    one). Returns (None, None) if no default route exists at all (every interface down)."""
    try:
        output = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    match = re.search(r"default via (\d+\.\d+\.\d+\.\d+) dev (\S+)", output)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _ping(host: str) -> bool:
    try:
        result = subprocess.run(
            ["ping", "-c", str(PING_COUNT), "-W", str(PING_TIMEOUT_SEC), host],
            capture_output=True, timeout=(PING_TIMEOUT_SEC * PING_COUNT) + 5,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _backend_reachable() -> bool:
    """Informational only, not the primary signal -- distinguishes "WiFi dropped entirely" from
    "gateway's fine but upstream internet/the WMS backend specifically is down." The recovery
    action (WiFi reconnect) is the same either way, so this doesn't gate anything, it's just
    logged for whoever reads journalctl later trying to understand what actually happened."""
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(f"{WMS_BACKEND_URL}/health", method="GET")
        with urllib.request.urlopen(request, timeout=BACKEND_CHECK_TIMEOUT_SEC):
            return True
    except (urllib.error.URLError, OSError):
        return False


def check_reachability() -> "tuple[bool, Optional[str]]":
    """Returns (healthy, interface). `interface` is whichever device currently carries the
    default route (e.g. 'eth0' or 'wlan0') when known -- None only when there's no default route
    at all to read it from. Never raises."""
    gateway, interface = _default_route()
    if not gateway:
        print("[network_watchdog] No default route -- interface likely down or never associated.")
        return False, interface
    if _ping(gateway):
        return True, interface
    print(f"[network_watchdog] Gateway {gateway} (dev {interface}) did not respond to ping.")
    return False, interface


def reconnect_interface(interface: str) -> None:
    """Forces a disconnect/reconnect cycle on the given interface via nmcli -- Raspberry Pi OS's
    NetworkManager-based network stack (Bookworm and later), not the older wpa_supplicant/dhcpcd
    stack earlier releases used. Works the same way for a wired (eth0) or WiFi (wlan0) interface --
    this fleet has both kinds, confirmed via WH-007's ED1 being wired. Guards on NetworkManager
    actually being the active stack first, in case a golden image ever deviates from README.md's
    documented default."""
    try:
        nm_active = subprocess.run(
            ["systemctl", "is-active", "NetworkManager"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        nm_active = ""

    if nm_active != "active":
        print(
            "[network_watchdog] NetworkManager is not the active network stack "
            f"(is-active reported '{nm_active}') -- skipping reconnect action. "
            "This Pi may have been built from a non-standard image; see README.md Section 1."
        )
        return

    print(f"[network_watchdog] Reconnecting {interface} via nmcli...")
    subprocess.run(
        ["sudo", "-n", "nmcli", "device", "disconnect", interface],
        capture_output=True, timeout=15,
    )
    subprocess.run(
        ["sudo", "-n", "nmcli", "device", "connect", interface],
        capture_output=True, timeout=30,
    )


def main() -> int:
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    state.last_check_at = now

    healthy, interface = check_reachability()
    if interface:
        state.last_known_interface = interface

    if healthy:
        if state.consecutive_failures > 0:
            print(f"[network_watchdog] Recovered after {state.consecutive_failures} failed check(s).")
        state.consecutive_failures = 0
        state.last_success_at = now
        save_state(state)
        return 0

    state.consecutive_failures += 1
    backend_ok = _backend_reachable()
    print(
        f"[network_watchdog] Check failed (consecutive={state.consecutive_failures}, "
        f"backend_reachable={backend_ok})."
    )

    if state.consecutive_failures >= CONSECUTIVE_FAILURES_BEFORE_ACTION:
        # Prefer the interface the routing table just reported; if the default route has
        # vanished entirely (interface is fully down, nothing to read), fall back to whichever
        # interface last carried it, and only as a last resort the configured guess.
        target_interface = interface or state.last_known_interface or FALLBACK_INTERFACE
        reconnect_interface(target_interface)
        state.last_action_at = now
        state.consecutive_failures = 0  # give the reconnect a fresh window to prove itself

    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
