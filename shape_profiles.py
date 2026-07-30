"""
Per-shape-type correction profiles -- the actual "training" feedback loop.

Each time an operator corrects a capture that was tagged with a shape type (box, bag, can,
irregular, or a custom label), this updates that shape's running correction factor: the mean
(corrected - measured) per axis, across every correction ever submitted for that shape. measure()
then applies the active shape's factor to its raw output whenever a capture is tagged with a
known shape type -- so a correction visibly changes future scans of that same shape, without a
full ML pipeline.

Same local-JSON-file pattern as products.py/device_config.py -- intentionally local to the Pi,
not wired into the WMS backend (that stays a passive log of corrections; the LIVE feedback loop
lives here, closest to the camera).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Callable, Optional

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "data", "shape_profiles.json")

BUILTIN_SHAPE_TYPES = ["box", "bag", "can", "irregular"]

# Guards every read-modify-write cycle against shape_profiles.json -- same reasoning as
# device_config.py's _CONFIG_LOCK (added when an adversarial audit found that file's setters were
# unsynchronized and could lose a concurrent write). This file had no lock at all until Training
# Mode needed to atomically replace a shape profile as part of a larger batch-calibration write.
_PROFILES_LOCK = threading.Lock()


@dataclass
class ShapeProfile:
    shape_type: str
    correction_count: int = 0
    # Running mean of (corrected_in - measured_in) per axis, in inches. Added directly to a raw
    # measurement's length/width/height before reporting, when this shape type is active.
    length_offset_in: float = 0.0
    width_offset_in: float = 0.0
    height_offset_in: float = 0.0


def _read_all_unlocked() -> dict[str, ShapeProfile]:
    """Raw read, assumes the caller already holds _PROFILES_LOCK. Never call directly outside
    this module."""
    if not os.path.exists(PROFILES_PATH):
        return {}
    with open(PROFILES_PATH, "r") as f:
        data = json.load(f)
    return {k: ShapeProfile(**v) for k, v in data.items()}


def _write_all_unlocked(profiles: dict[str, ShapeProfile]) -> None:
    """Raw write, assumes the caller already holds _PROFILES_LOCK. Never call directly outside
    this module. Writes to a temp file + os.replace (atomic on both POSIX and Windows) instead of
    truncating PROFILES_PATH in place, matching device_config.py's write pattern -- a crash/
    power-loss mid-write can never leave behind a malformed JSON file this way."""
    directory = os.path.dirname(PROFILES_PATH)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{PROFILES_PATH}.tmp-{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump({k: asdict(v) for k, v in profiles.items()}, f)
    os.replace(tmp_path, PROFILES_PATH)


def _load_all() -> dict[str, ShapeProfile]:
    with _PROFILES_LOCK:
        return _read_all_unlocked()


def _save_all(profiles: dict[str, ShapeProfile]) -> None:
    with _PROFILES_LOCK:
        _write_all_unlocked(profiles)


def update_shape_profiles(mutate: "Callable[[dict[str, ShapeProfile]], None]") -> dict[str, ShapeProfile]:
    """Atomic read-modify-write: holds _PROFILES_LOCK for the ENTIRE cycle, same pattern as
    device_config.py's update_device_config -- `mutate` receives the freshly-loaded profiles dict
    and modifies it in place (return value ignored). Needed by Training Mode, which must replace
    several shape types' profiles as part of one atomic batch-calibration step; a caller that
    read via _load_all() and wrote back via _save_all() separately could lose a concurrent
    writer's change in between (the exact TOCTOU class device_config.py's lock was added to
    close)."""
    with _PROFILES_LOCK:
        profiles = _read_all_unlocked()
        mutate(profiles)
        _write_all_unlocked(profiles)
        return profiles


def list_shape_profiles() -> list[ShapeProfile]:
    return list(_load_all().values())


def get_shape_profile(shape_type: str) -> Optional[ShapeProfile]:
    return _load_all().get(shape_type)


def record_correction(
    shape_type: str,
    measured_length_in: float,
    measured_width_in: float,
    measured_height_in: float,
    corrected_length_in: float,
    corrected_width_in: float,
    corrected_height_in: float,
) -> ShapeProfile:
    """Incorporates one new correction into this shape type's running mean offset.

    Uses an incremental mean update (new_mean = old_mean + (sample - old_mean) / new_count) so
    the whole correction history never needs to be replayed -- this file only ever stores the
    current running mean + count, not every past correction (that full history lives in the WMS
    backend's DimensionCaptureLog, which this Pi doesn't need to duplicate).
    """
    result: dict[str, ShapeProfile] = {}

    def mutate(profiles: dict[str, ShapeProfile]) -> None:
        profile = profiles.get(shape_type) or ShapeProfile(shape_type=shape_type)

        new_count = profile.correction_count + 1
        length_delta = corrected_length_in - measured_length_in
        width_delta = corrected_width_in - measured_width_in
        height_delta = corrected_height_in - measured_height_in

        profile.length_offset_in += (length_delta - profile.length_offset_in) / new_count
        profile.width_offset_in += (width_delta - profile.width_offset_in) / new_count
        profile.height_offset_in += (height_delta - profile.height_offset_in) / new_count
        profile.correction_count = new_count

        profiles[shape_type] = profile
        result["profile"] = profile

    update_shape_profiles(mutate)
    return result["profile"]


def replace_shape_profile(
    shape_type: str, correction_count: int, length_offset_in: float, width_offset_in: float, height_offset_in: float
) -> ShapeProfile:
    """Overwrites a shape type's profile outright, rather than blending onto its existing running
    mean (contrast with record_correction's incremental blend). Used by Training Mode: a batch
    calibration computes offsets from `raw` extents and a freshly-fit `system_scale_factor`
    together in one step (see api.py's /training/compute), so any PRIOR profile for a shape type
    in this batch was necessarily derived under a different, now-superseded scale factor --
    blending onto it would launder stale, possibly rank-crossed corrections into the new number
    instead of cleanly superseding them. Shape types NOT present in a training batch are
    untouched by this function; only replace_shape_profile'd types change."""
    result: dict[str, ShapeProfile] = {}

    def mutate(profiles: dict[str, ShapeProfile]) -> None:
        profile = ShapeProfile(
            shape_type=shape_type,
            correction_count=correction_count,
            length_offset_in=length_offset_in,
            width_offset_in=width_offset_in,
            height_offset_in=height_offset_in,
        )
        profiles[shape_type] = profile
        result["profile"] = profile

    update_shape_profiles(mutate)
    return result["profile"]


def reset_shape_profile(shape_type: str) -> None:
    """Deletes a shape type's learned correction profile entirely -- e.g. after a bad/mistaken
    correction skewed its running offset far enough that it's actively making future measurements
    of that shape worse instead of better. The next capture tagged with this shape type starts
    fresh (apply_shape_correction is a no-op until at least one new correction is recorded)."""
    update_shape_profiles(lambda profiles: profiles.pop(shape_type, None))


def apply_shape_correction(
    shape_type: Optional[str], length_in: float, width_in: float, height_in: float
) -> tuple[float, float, float]:
    """Applies the shape's learned offset to a raw measurement, if a profile exists for it.
    No-op (returns the input unchanged) for an unknown shape type or one with no corrections yet.
    """
    if not shape_type:
        return length_in, width_in, height_in
    profile = get_shape_profile(shape_type)
    if profile is None or profile.correction_count == 0:
        return length_in, width_in, height_in
    return (
        round(length_in + profile.length_offset_in, 2),
        round(width_in + profile.width_offset_in, 2),
        round(height_in + profile.height_offset_in, 2),
    )
