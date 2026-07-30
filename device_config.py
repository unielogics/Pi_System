"""
Persisted device-level settings: the camera's mount height, which calibration mode was last
used, and this device's own identity. Same local-JSON-file pattern as calibration.npz -- state
lives on the Pi, not in the WMS backend.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "device_config.json")

# Guards every read-modify-write cycle against device_config.json. Without this, two overlapping
# requests (e.g. two browser tabs both calling POST /detections/select/correct, or an Accuracy
# Check's Apply racing a PUT /config) each do their own load->mutate->save with no synchronization
# -- both reads see the same stale baseline, and whichever save() lands last silently overwrites
# the other's change (a classic lost-update), since save_device_config always writes the WHOLE
# dataclass, not just the field one caller meant to change. FastAPI dispatches these plain `def`
# route handlers to separate threadpool threads (not coroutines on one event-loop thread), so this
# is a real, reachable race, not a theoretical one -- confirmed via code audit of every setter
# below plus every route that calls one.
_CONFIG_LOCK = threading.Lock()


@dataclass
class DeviceConfig:
    camera_height_mm: Optional[float] = None
    calibration_mode: Optional[str] = None  # 'height' | 'live'
    calibrated_at: Optional[str] = None  # ISO timestamp, set alongside calibration.npz
    # This device's own stable identity, used by registration.py to self-register/heartbeat with
    # the backend (see dimensioner-registration.service.ts). Generated once, on first load.
    device_id: Optional[str] = None
    # Persistent (non-TTL) pixel-space rectangles always subtracted from the foreground mask --
    # e.g. a fixed shadow or shelf edge that always misreads as an item. Each: {"x","y","w","h"}
    # in the RGB/depth frame's native pixel coordinates. Contrast with api.py's in-memory,
    # short-TTL one-shot overrides, which cover a single capture rather than every future frame.
    permanent_exclude_regions: list[dict] = field(default_factory=list)
    # Ground-truth accuracy calibration (see measure.py's _apply_system_scale_factor): a
    # multiplicative per-axis factor (true_size / measured_size) derived from measuring a
    # certified-size reference object via /calibrate/accuracy-check/apply. None/empty until an
    # operator has run that flow at least once -- until then this is a no-op in measure().
    system_scale_factor: Optional[dict] = None  # {"length": float, "width": float, "height": float}
    # Auto-absorb static clutter into the background reference (api.py's _auto_absorb_loop): a
    # foreground pixel that's held the same depth (within jitter tolerance) for this many seconds
    # gets folded into background_mm and silently stops being flagged. Default ON, 2 minutes --
    # long enough that no normal capture workflow (seconds, not minutes) is ever affected.
    auto_absorb_enabled: bool = True
    auto_absorb_timeout_sec: float = 120.0


def _cpu_serial() -> Optional[str]:
    """Reads the Pi's hardware serial from /proc/cpuinfo. Used as part of device_id generation
    so that cloning an SD card to provision a second physical Pi doesn't silently produce two
    devices with the same device_id (which would make them overwrite each other's endpointUrl/
    ipAddress on every heartbeat) -- a pure random ID would collide invisibly in that case since
    both cards start with an identical device_config.json."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            match = re.search(r"^Serial\s*:\s*([0-9a-fA-F]+)$", f.read(), re.MULTILINE)
        return match.group(1) if match else None
    except OSError:
        return None


def _generate_device_id() -> str:
    serial = _cpu_serial()
    if serial:
        return f"dev_{serial.lower()}"
    # No /proc/cpuinfo serial available (e.g. non-Pi dev box) -- fall back to a random id. This
    # path IS clone-vulnerable, but only on hardware where the serial-based path isn't possible.
    return f"dev_{secrets.token_hex(8)}"


def _read_config_unlocked() -> DeviceConfig:
    """Raw read, assumes the caller already holds _CONFIG_LOCK (or is the one-time bootstrap
    before any lock is needed). Never call directly outside this module."""
    if not os.path.exists(CONFIG_PATH):
        config = DeviceConfig(device_id=_generate_device_id())
        _write_config_unlocked(config)
        return config
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    device_id = data.get("device_id")
    config = DeviceConfig(
        camera_height_mm=data.get("camera_height_mm"),
        calibration_mode=data.get("calibration_mode"),
        calibrated_at=data.get("calibrated_at"),
        device_id=device_id,
        permanent_exclude_regions=data.get("permanent_exclude_regions", []),
        system_scale_factor=data.get("system_scale_factor"),
        auto_absorb_enabled=data.get("auto_absorb_enabled", True),
        auto_absorb_timeout_sec=data.get("auto_absorb_timeout_sec", 120.0),
    )
    if not device_id:
        # Upgrading a device_config.json written before device_id existed -- generate once and
        # persist immediately so it's stable from here on.
        config.device_id = _generate_device_id()
        _write_config_unlocked(config)
    return config


def _write_config_unlocked(config: DeviceConfig) -> None:
    """Raw write, assumes the caller already holds _CONFIG_LOCK. Never call directly outside
    this module. Writes to a temp file + os.replace (atomic on both POSIX and Windows) instead of
    truncating CONFIG_PATH in place, so a crash/power-loss mid-write can never leave behind a
    truncated or interleaved-partial-write JSON file that bricks every future load_device_config()
    call until fixed by hand."""
    directory = os.path.dirname(CONFIG_PATH)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{CONFIG_PATH}.tmp-{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(asdict(config), f)
    os.replace(tmp_path, CONFIG_PATH)


def load_device_config() -> DeviceConfig:
    with _CONFIG_LOCK:
        return _read_config_unlocked()


def save_device_config(config: DeviceConfig) -> None:
    """Overwrites the ENTIRE stored config with exactly what's passed in. Prefer
    update_device_config() for any read-modify-write (i.e. every setter below) -- this raw
    save is for the rare case of committing an already-complete config object with no separate
    prior read of "current" state to race against."""
    with _CONFIG_LOCK:
        _write_config_unlocked(config)


def update_device_config(mutate: "Callable[[DeviceConfig], None]") -> DeviceConfig:
    """Atomic read-modify-write: holds _CONFIG_LOCK for the ENTIRE cycle so `mutate` always sees
    the truly-current on-disk state and no concurrent writer's change can be silently lost between
    this read and this write. `mutate` receives the freshly-loaded config and modifies it in place
    (return value ignored) -- e.g. `lambda c: setattr(c, "camera_height_mm", 900.0)`, or something
    that reads c.system_scale_factor to compute a new one before assigning it back onto the SAME
    object, closing the gap where a caller used to read the old factor via a separate
    load_device_config() call before this write, which a concurrent writer could invalidate in
    between (see api.py's _compound_scale_factor, which now runs its read-and-compute inside this
    mutator instead of before it)."""
    with _CONFIG_LOCK:
        config = _read_config_unlocked()
        mutate(config)
        _write_config_unlocked(config)
        return config


def set_camera_height_mm(height_mm: float) -> DeviceConfig:
    return update_device_config(lambda c: setattr(c, "camera_height_mm", height_mm))


def set_calibration_state(mode: str, calibrated_at: str) -> DeviceConfig:
    def mutate(c: DeviceConfig) -> None:
        c.calibration_mode = mode
        c.calibrated_at = calibrated_at

    return update_device_config(mutate)


def set_permanent_exclude_regions(regions: list[dict]) -> DeviceConfig:
    return update_device_config(lambda c: setattr(c, "permanent_exclude_regions", regions))


def set_system_scale_factor(scale_factor: Optional[dict]) -> DeviceConfig:
    return update_device_config(lambda c: setattr(c, "system_scale_factor", scale_factor))


def set_auto_absorb_config(enabled: bool, timeout_sec: float) -> DeviceConfig:
    def mutate(c: DeviceConfig) -> None:
        c.auto_absorb_enabled = enabled
        c.auto_absorb_timeout_sec = timeout_sec

    return update_device_config(mutate)
