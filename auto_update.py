"""
Self-update: pulls the latest dimensioner code from unielogics/Pi_System (public repo, no
credentials needed) and restarts the app services if a new commit was applied. Runs as a
periodic oneshot (dimensioner-auto-update.timer, every 30 minutes) -- much less time-critical
than the heartbeat/network-watchdog timers, since a code update is never an emergency the way
"is the camera dead right now" is.

Deliberately uses `git pull --ff-only`, never a merge/rebase: if this device's working tree
has ever been hand-edited (e.g. during a live debugging SSH session), a fast-forward simply
fails loudly instead of silently discarding or auto-merging over an in-progress fix. On any
failure this script logs clearly and exits non-zero without touching the running services --
leaving a broken pull for a human to resolve is much safer than restarting into a half-applied
state.
"""
from __future__ import annotations

import subprocess
import sys

REPO_DIR = __file__.rsplit("/", 1)[0] or "."

# Files whose changes need a follow-up action beyond a plain restart. Kept as a short, explicit
# list rather than trying to generically infer intent from a diff.
REQUIREMENTS_FILE = "requirements.txt"
UNIT_FILE_SUFFIXES = (".service", ".timer")

SERVICES_TO_RESTART = ["dimensioner-api.service", "dimensioner-ros.service"]


def _run(args: list, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=REPO_DIR, capture_output=True, text=True, timeout=timeout
    )


def _rev_parse(ref: str) -> str:
    result = _run(["git", "rev-parse", ref])
    return result.stdout.strip()


def main() -> int:
    fetch = _run(["git", "fetch", "origin"], timeout=60)
    if fetch.returncode != 0:
        print(f"[auto_update] git fetch failed: {fetch.stderr.strip()}")
        return 1

    old_sha = _rev_parse("HEAD")
    new_sha = _rev_parse("origin/main")
    if not old_sha or not new_sha:
        print("[auto_update] Could not resolve HEAD/origin/main -- skipping.")
        return 1

    if old_sha == new_sha:
        # Normal, silent case -- runs every 30 min, most ticks are no-ops.
        return 0

    print(f"[auto_update] New commit(s) available: {old_sha[:7]} -> {new_sha[:7]}")

    pull = _run(["git", "pull", "--ff-only", "origin", "main"], timeout=60)
    if pull.returncode != 0:
        print(
            "[auto_update] git pull --ff-only failed -- likely local edits diverged from "
            f"origin/main. Left untouched for manual resolution. stderr: {pull.stderr.strip()}"
        )
        return 1

    changed_files_result = _run(["git", "diff", "--name-only", old_sha, new_sha])
    changed_files = [f for f in changed_files_result.stdout.splitlines() if f]

    if any(f == REQUIREMENTS_FILE for f in changed_files):
        print("[auto_update] requirements.txt changed -- reinstalling dependencies.")
        pip_install = _run(
            ["/home/franco/micromamba/bin/micromamba", "run", "-n", "ros2",
             "pip", "install", "-r", REQUIREMENTS_FILE],
            timeout=180,
        )
        if pip_install.returncode != 0:
            print(f"[auto_update] pip install failed: {pip_install.stderr.strip()}")

    if any(f.endswith(UNIT_FILE_SUFFIXES) for f in changed_files):
        print("[auto_update] systemd unit file(s) changed -- reloading + re-enabling.")
        _run(["sudo", "-n", "cp"] + [f for f in changed_files if f.endswith(UNIT_FILE_SUFFIXES)]
             + ["/etc/systemd/system/"], timeout=15)
        _run(["sudo", "-n", "systemctl", "daemon-reload"], timeout=15)
        for f in changed_files:
            if f.endswith(UNIT_FILE_SUFFIXES):
                _run(["sudo", "-n", "systemctl", "enable", f.rsplit("/", 1)[-1]], timeout=15)

    print(f"[auto_update] Restarting: {', '.join(SERVICES_TO_RESTART)}")
    restart = _run(["sudo", "-n", "systemctl", "restart"] + SERVICES_TO_RESTART, timeout=30)
    if restart.returncode != 0:
        print(f"[auto_update] Service restart failed: {restart.stderr.strip()}")
        return 1

    log = _run(["git", "log", "--oneline", f"{old_sha}..{new_sha}"])
    print(f"[auto_update] Applied. Changes:\n{log.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
