"""
Camera-agnostic dimensioning core.

Everything here operates only on numpy arrays (a depth frame in millimeters, an RGB frame,
and camera intrinsics) plus a stored calibration reference. It has zero knowledge of which
camera produced the frames -- any CameraAdapter that supplies correctly-scaled depth + correct
intrinsics works with this module unchanged. Swapping cameras means writing a new adapter,
never touching this file.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shape_profiles import apply_shape_correction
from device_config import load_device_config

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "data", "calibration.npz")

# Guards calibration.npz writes and the in-memory generation counter below -- see
# save_calibration_if_current's docstring for the specific race this closes (a slow background
# computation built from a stale snapshot silently reverting a fresher operator-triggered write).
_calibration_lock = threading.Lock()
_calibration_generation = 0

# Depth pixels closer to the camera than the calibrated background by at least this many mm
# are considered "object", not noise/background. Structured-light sensors have several mm of
# frame-to-frame jitter even on a static scene.
FOREGROUND_THRESHOLD_MM = 15.0

# Pixels within this many rows/columns of the frame's edge are NEVER considered foreground,
# regardless of what the depth threshold test says -- see _segment_foreground_mask's docstring.
# Structured-light/stereo depth sensors are structurally noisiest at the edge of their field of
# view; live-diagnosed on the real device (a border-pixel scan during a reported "measuring the
# frame border" incident) as the mechanism behind that report. An operator should never need to
# place an item flush against the very edge of the frame to measure it, so this costs nothing in
# practice.
BORDER_MARGIN_PX = 8

# A point cloud with fewer than this many foreground points is treated as "nothing placed".
MIN_OBJECT_POINTS = 200

# Fraction of foreground points that must lie within FIT_TOLERANCE_MM of the fitted cuboid's
# faces for the object to be classified as a regular box. Below this, it's an irregular item --
# still measured, but flagged for operator review rather than trusted as an exact box size.
BOX_FIT_RATIO_THRESHOLD = 0.85
FIT_TOLERANCE_MM = 6.0

# Splits measure()'s existing "irregular-item" bucket (BOX_FIT_RATIO_THRESHOLD, above) further,
# for Training Mode's suggested_shape_type (see _footprint_circularity/_suggest_shape_type).
# _footprint_circularity uses the standard 4*pi*Area/Perimeter^2 metric -- exactly 1.0 for a
# perfect circle, ~0.785 for a square, lower for elongated shapes. CIRCULARITY_CAN_THRESHOLD sits
# between a real can's expected value and a square's ~0.785 so a boxy footprint that slipped past
# BOX_FIT_RATIO_THRESHOLD doesn't get misread as a can; CIRCULARITY_BAG_THRESHOLD is low enough
# that only a genuinely elongated/irregular footprint falls below it. The band between the two
# thresholds falls back to plain "irregular" (can't confidently call it either way) rather than
# forcing a guess. NEEDS LIVE TUNING: these are reasonable starting guesses, not validated against
# real scans of known can/bag items on the actual device -- confirm/adjust against real readings
# before trusting the suggestion (see this feature's verification step).
CIRCULARITY_CAN_THRESHOLD = 0.85
CIRCULARITY_BAG_THRESHOLD = 0.55

MM_PER_INCH = 25.4


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


IN3_PER_FT3 = 1728.0  # matches UnieBackend's calculateCubicFeet() / UnieDashboard's inchesToCubicFeet()


@dataclass
class MeasurementResult:
    length_in: float
    width_in: float
    height_in: float
    cubic_feet: float
    classification: str  # 'box' | 'irregular-item'
    confidence: float  # fraction of points hugging the fitted cuboid's faces, 0-1
    point_count: int
    # 8 oriented-bbox corners in camera-space mm (x, y, z) -- the caller (api.py) projects these
    # into 2D pixel space using the camera intrinsics to draw a bounding-box overlay on the RGB
    # image, so the live picture visibly matches the reported numbers.
    corners_mm: list[tuple[float, float, float]]
    # Sorted-descending extents in inches BEFORE _apply_corrections (shape offset + system scale
    # factor) -- i.e. what the sensor/geometry alone produced. Training Mode's batch calibration
    # (api.py's /training/compute) needs this: it fits a fresh system_scale_factor from
    # true_size/raw_size ratios across a whole batch of samples, which only makes sense against
    # the UNCORRECTED geometry -- fitting against already-corrected values would be circular
    # (correcting the correction). Everywhere else (measure()'s own return value, detect_live())
    # continues to report length_in/width_in/height_in as the fully-corrected numbers; these
    # raw_* fields are additive, not a behavior change to any existing caller.
    raw_length_in: float = 0.0
    raw_width_in: float = 0.0
    raw_height_in: float = 0.0
    # Which of shape_profiles.py's BUILTIN_SHAPE_TYPES this scan's depth geometry suggests --
    # 'box'/'can'/'bag'/'irregular', see _suggest_shape_type. Purely advisory: derived entirely
    # from geometry already computed above (classification/confidence + 2D footprint circularity),
    # no camera image or ML model involved. Training Mode surfaces this to pre-select the
    # shape-type dropdown instead of always defaulting to "box," while leaving the operator free to
    # override it.
    suggested_shape_type: str = "irregular"


@dataclass
class LiveDetection:
    """Lightweight per-frame detection for the continuous live-view overlay -- oriented extents
    + corners only, no cuboid-fit-ratio/classification pass (that full analysis is reserved for
    an explicit /capture, since running it on every preview frame for every blob would be wasted
    CPU on a Pi4). Multiple items in the same frame each get their own LiveDetection.

    length_in/width_in/height_in are the SAME sorted-oriented-extent values measure() would
    report for this blob alone -- NOT recoverable by taking max/min of corners_mm (those are in
    the blob's own oriented axes, not globally axis-aligned, so an axis-aligned max/min over them
    would overestimate size). Callers needing dimensions (e.g. the /detections endpoint) must use
    these fields, not derive them from corners_mm.
    """
    id: int
    length_in: float
    width_in: float
    height_in: float
    cubic_feet: float
    corners_mm: list[tuple[float, float, float]]
    point_count: int
    # Pixel-space bounding box {x, y, w, h} in the native depth/RGB frame's own resolution --
    # straight from cv2.connectedComponentsWithStats, already computed for free in detect_live().
    # Lets a click on the RGB preview be matched to "which detection is this" (see api.py's
    # POST /detections/select) without any new geometry: just a point-in-rect test.
    bbox: tuple[int, int, int, int]


class NoObjectDetectedError(Exception):
    """Raised when the current frame has no foreground object relative to the calibration."""


class NotCalibratedError(Exception):
    """Raised when /capture is called before /calibrate has ever run."""


def load_calibration() -> Optional[np.ndarray]:
    if not os.path.exists(CALIBRATION_PATH):
        return None
    data = np.load(CALIBRATION_PATH)
    return data["background_depth_mm"]


def load_calibration_with_generation() -> tuple[Optional[np.ndarray], int]:
    """Same as load_calibration(), plus the in-memory generation counter at the moment of the
    read -- lets a caller that does a slow computation off this snapshot (see
    save_calibration_if_current) later detect whether a newer calibration was written in the
    meantime, without needing to diff the actual array contents."""
    with _calibration_lock:
        generation = _calibration_generation
    return load_calibration(), generation


def save_calibration(background_depth_mm: np.ndarray) -> None:
    """Unconditional write + generation bump -- the normal path, used by an explicit operator
    /calibrate call. Any concurrent save_calibration_if_current() built from an older generation
    will now correctly see itself as stale and skip its own write (see that function)."""
    with _calibration_lock:
        global _calibration_generation
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        np.savez(CALIBRATION_PATH, background_depth_mm=background_depth_mm)
        _calibration_generation += 1


def save_calibration_if_current(background_depth_mm: np.ndarray, expected_generation: int) -> bool:
    """Compare-and-swap write: only persists if the generation counter is still exactly what it
    was when the caller's `expected_generation` snapshot was read (via
    load_calibration_with_generation). Returns True if written, False if skipped as stale.

    This closes a real race an adversarial audit confirmed: _auto_absorb_loop reads
    background_mm once per tick, then does substantial work (a full detect_live() pass plus large
    numpy ops) before eventually calling save_calibration() with a copy DERIVED from that
    old snapshot. If an operator's POST /calibrate (running on a separate threadpool thread) writes
    a brand-new background snapshot while that tick is still mid-flight, the tick's later,
    unconditional save used to silently overwrite the operator's fresh Quick/Precise Calibrate
    with stale, pre-recalibration data -- with no error, no log, nothing to indicate the operator's
    action was just reverted a few seconds later. Now the tick's write is conditional on nothing
    else having changed calibration.npz since the tick started; if something did, the tick simply
    discards its own (now-stale-by-definition) derived result and the next tick recomputes fresh
    against the new background.
    """
    with _calibration_lock:
        global _calibration_generation
        if _calibration_generation != expected_generation:
            return False
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        np.savez(CALIBRATION_PATH, background_depth_mm=background_depth_mm)
        _calibration_generation += 1
        return True


def synthesize_flat_background(shape: tuple[int, int], camera_height_mm: float) -> np.ndarray:
    """A computed (not captured) background reference: a flat surface perpendicular to the
    camera at a known distance.

    Depth cameras report per-pixel Z-depth (perpendicular distance to the camera's image plane),
    not radial distance -- the same convention _deproject() already assumes. That means a flat
    floor/table directly below the camera has the SAME depth value at every pixel, equal to the
    mount height. This lets calibration be instant (no live empty-scene capture needed, and no
    "item fills the whole frame" edge case) whenever the camera is mounted at a known, fixed
    height looking straight down.
    """
    return np.full(shape, round(camera_height_mm), dtype=np.uint16)


def _segment_foreground_mask(
    depth_mm: np.ndarray,
    background_mm: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    include_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pixels where the current depth is meaningfully closer than the calibrated background.

    exclude_mask/include_mask are optional pixel-space boolean masks (same shape as depth_mm) an
    operator can supply to manually correct segmentation -- e.g. suppressing a shadow/reflection
    the depth threshold alone can't distinguish from a real item, or forcing a very low-profile
    item that fails FOREGROUND_THRESHOLD_MM into consideration. Excluded pixels are removed after
    the normal threshold test; included pixels are added regardless of what the threshold test
    said, since the operator is asserting ground truth for that region.

    Unconditionally zeroes a thin margin at the frame's edge (BORDER_MARGIN_PX), overriding even
    an explicit include_mask there. Live-confirmed why: depth sensors are structurally noisiest
    at the edge of their field of view (lens falloff, incomplete stereo/structured-light coverage
    right at the frame boundary) -- an operator diagnosed the system "measuring the frame border"
    instead of the actual item, and a live diagnostic on the real device caught the mechanism:
    border pixels can occasionally cross FOREGROUND_THRESHOLD_MM on their own noise, and a real
    item's blob can be adjacent to (or clipped by) that noisy strip, pulling the reported bounding
    box outward to include it. No correctly-placed item for measurement should ever need pixels
    in the outermost few rows/columns of the frame to be part of its footprint, so this margin is
    a pure loss-nothing safety net, not a usable-area tradeoff.
    """
    valid = (depth_mm > 0) & (background_mm > 0)
    closer_by = background_mm.astype(np.float32) - depth_mm.astype(np.float32)
    mask = valid & (closer_by > FOREGROUND_THRESHOLD_MM)
    if exclude_mask is not None:
        mask = mask & ~exclude_mask
    if include_mask is not None:
        mask = mask | include_mask
    if BORDER_MARGIN_PX > 0:
        mask[:BORDER_MARGIN_PX, :] = False
        mask[-BORDER_MARGIN_PX:, :] = False
        mask[:, :BORDER_MARGIN_PX] = False
        mask[:, -BORDER_MARGIN_PX:] = False
    return mask


def _deproject(depth_mm: np.ndarray, mask: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Turn masked depth pixels into an (N, 3) point cloud in camera-space millimeters.

    This is the one step where camera differences (resolution, FOV, sensor placement) get
    absorbed -- any adapter that hands us depth-in-mm + matching intrinsics produces an
    equivalent point cloud here, regardless of the underlying sensor technology.
    """
    ys, xs = np.nonzero(mask)
    z = depth_mm[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)


VOXEL_SIZE_MM = 10.0
MIN_VOXEL_NEIGHBOR_COUNT = 3


def _remove_outliers(points: np.ndarray) -> np.ndarray:
    """Voxel-density outlier removal: bin points into a coarse 3D grid and drop only points
    sitting in near-empty cells (isolated sensor speckle noise).

    Deliberately NOT distance-from-centroid trimming (the prior approach): that method is biased
    against elongated/irregular shapes -- it can shave off legitimate extremities of a long or
    lumpy item while leaving a compact box untouched, which fights against handling irregular
    items. Voxel density only reacts to LOCAL sparsity, so it treats every shape the same way:
    real surfaces are locally dense regardless of overall shape, isolated noise points aren't.
    """
    if len(points) < MIN_VOXEL_NEIGHBOR_COUNT * 4:
        return points
    voxel_ids = np.floor(points / VOXEL_SIZE_MM).astype(np.int64)
    _, inverse, counts = np.unique(voxel_ids, axis=0, return_inverse=True, return_counts=True)
    point_voxel_counts = counts[inverse]
    keep = point_voxel_counts >= MIN_VOXEL_NEIGHBOR_COUNT
    return points[keep]


def _oriented_bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-based oriented bounding box. Returns (extents_mm, principal_axes, centroid) -- NOT
    sorted by extent here (that sorting happens later, only for the reported L/W/H numbers) so
    axes/extents/centroid stay consistent with each other for corner reconstruction.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = centered @ axes
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = maxs - mins
    return extents, axes, centroid


def _bounding_box_corners(axes: np.ndarray, centroid: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """The 8 corners of the oriented bounding box, in camera-space mm."""
    half = extents / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local_corners = signs * half[None, :]
    return centroid[None, :] + local_corners @ axes.T


def _cuboid_fit_ratio(points: np.ndarray, axes: np.ndarray, centroid: np.ndarray, extents: np.ndarray) -> float:
    """Fraction of points lying within FIT_TOLERANCE_MM of the nearest face of the fitted cuboid.

    A regular box's surface points all sit close to one of six faces; an irregular/soft item
    (e.g. a balled-up garment) has points scattered through the interior of its bounding box,
    far from any face -- this ratio distinguishes the two without any object-specific model.
    """
    centered = points - centroid
    projected = centered @ axes
    half_extents = extents / 2.0
    # Distance from each point to the nearest face along each axis, then take the min across axes
    # (distance to the closest of the six faces).
    dist_to_face = half_extents[None, :] - np.abs(projected)
    min_dist_to_any_face = dist_to_face.min(axis=1)
    within_tolerance = np.abs(min_dist_to_any_face) <= FIT_TOLERANCE_MM
    return float(within_tolerance.mean())


def _footprint_circularity(points: np.ndarray, axes: np.ndarray) -> float:
    """Standard 4*pi*Area/Perimeter^2 circularity of the item's 2D footprint on its own two
    LONGEST principal axes (i.e. looking straight down its shortest/height axis) -- 1.0 for a
    perfect circle, ~0.785 for a square, lower for anything elongated or irregular. Distinguishes
    a can's round cross-section from a box's rectangular one purely from point-cloud geometry, no
    camera image involved -- see _suggest_shape_type.

    Uses cv2's convex hull (already a transitive dependency via cv_bridge -- see detect_live's own
    "already installed" note; deliberately NOT scipy, which is NOT installed in this device's
    ros2 env, confirmed live) over the axis-projected points -- cheap and robust to noisy interior
    points, since only the hull boundary matters for this metric.
    """
    import cv2

    order = np.argsort(np.ptp(points @ axes, axis=0))[::-1]  # longest-extent axes first
    footprint_axes = axes[:, order[:2]]
    projected_2d = (points @ footprint_axes).astype(np.float32)
    if len(projected_2d) < 3:
        return 0.0  # too few points for a hull -- no signal either way, falls through to "irregular"
    hull = cv2.convexHull(projected_2d)
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, closed=True)
    if perimeter <= 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))


def _suggest_shape_type(classification: str, points: np.ndarray, axes: np.ndarray) -> str:
    """Suggests which BUILTIN_SHAPE_TYPE (box/bag/can/irregular -- see shape_profiles.py) an
    operator should pick for this scan, derived PURELY from depth geometry already computed by
    measure() -- no camera image, no ML model, nothing to train or retrain. `classification` is
    measure()'s own existing box/irregular-item split (BOX_FIT_RATIO_THRESHOLD); this only needs
    to further split the "irregular-item" bucket into can/bag/irregular via footprint circularity.
    Purely advisory -- the operator can always override it (see api.py's shape_type_overridden
    tracking on TrainingSample)."""
    if classification == "box":
        return "box"
    circularity = _footprint_circularity(points, axes)
    if circularity >= CIRCULARITY_CAN_THRESHOLD:
        return "can"
    if circularity < CIRCULARITY_BAG_THRESHOLD:
        return "bag"
    return "irregular"


def _background_relative_height_mm(depth_mm: np.ndarray, background_mm: np.ndarray, mask: np.ndarray) -> float:
    """Height of a foreground blob's top surface above the calibrated background, derived
    directly from depth (background_mm - depth_mm over the blob's own footprint) rather than
    from the oriented bounding box's third principal axis.

    For a flat-topped item, that third PCA axis's spread mostly reflects a few mm of
    structured-light sensor jitter across the (very flat) top surface, not the true elevation
    above the table -- confirmed live on the real device: a real item's PCA-derived height read
    0.28in while the SAME blob's background-relative depth read ~0.9in (median over its
    footprint). Length/width don't have this problem since they're measured as in-plane spread,
    where there's plenty of real geometric signal to average over; height needs this more direct
    measurement instead.

    Median (not mean/max) because it's robust to a handful of noisy dip/spike pixels at the
    item's edges while still tracking the top surface's real elevation across its whole footprint
    (typically hundreds of pixels for even a small item).
    """
    depth_vals = depth_mm[mask].astype(np.float64)
    bg_vals = background_mm[mask].astype(np.float64)
    valid = (depth_vals > 0) & (bg_vals > 0)
    if not np.any(valid):
        return 0.0
    closer_by = bg_vals[valid] - depth_vals[valid]
    return float(np.median(closer_by))


def _refine_height(
    depth_mm: np.ndarray,
    background_mm: np.ndarray,
    mask: np.ndarray,
    extents_mm: np.ndarray,
    axes: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replaces the PCA-derived height axis's extent with a direct background-relative depth
    measurement (see _background_relative_height_mm above) and rebuilds the box corners to
    match -- length/width extents and the axes/centroid are left untouched. Returns
    (corrected_extents_mm, corners_mm) so the drawn wireframe box and its dimension labels stay
    visually consistent with the reported number; a mismatch there would defeat the entire point
    of labeling the box directly in the picture.

    The height axis is whichever of the 3 PCA axes has the SMALLEST raw extent (argmin), matching
    the existing sort-descending convention every caller already uses to assign length/width/
    height (largest=length, smallest=height) regardless of how the item happens to be rotated
    under the camera -- this stays consistent with that convention rather than assuming a fixed
    axis index.
    """
    height_axis = int(np.argmin(extents_mm))
    height_mm = _background_relative_height_mm(depth_mm, background_mm, mask)
    corrected_extents = extents_mm.copy()
    corrected_extents[height_axis] = max(height_mm, 0.0)
    corners_mm = _bounding_box_corners(axes, centroid, corrected_extents)
    return corrected_extents, corners_mm


def _apply_system_scale_factor(length_in: float, width_in: float, height_in: float) -> tuple[float, float, float]:
    """Final ground-truth accuracy correction: a multiplicative per-axis scale factor derived from
    measuring a certified-true-size reference object (see the /calibrate/accuracy-check/apply
    endpoint in api.py). Distinct from shape_profiles.py's additive per-shape-category offset --
    that corrects "this shape type tends to read N inches off" (a fixed band-aid); this corrects
    "the whole system's scale is off by X%" (a proportional lens/depth-calibration error), and
    must run AFTER shape correction so shape profiles are learning against an already-true-scaled
    measurement rather than compensating for both errors conflated together.
    """
    device_config = load_device_config()
    scale = device_config.system_scale_factor
    if not scale:
        return length_in, width_in, height_in
    return (
        length_in * scale.get("length", 1.0),
        width_in * scale.get("width", 1.0),
        height_in * scale.get("height", 1.0),
    )


def _apply_corrections(
    length_in: float, width_in: float, height_in: float, shape_type: Optional[str] = None
) -> tuple[float, float, float]:
    """The full correction chain, applied identically everywhere a measurement is reported --
    shared by measure() (an explicit /capture) and detect_live() (the continuous live overlay +
    /detections + /detections/select). Before this existed, detect_live() only had the
    background-relative height fix and NEVER applied shape correction or system_scale_factor,
    while measure() applied both -- so the live cyan-bordered box (and its on-image L/W/H labels)
    reported a DIFFERENT number than the Capture result for the exact same physical object.
    Live-verified: a selection read 8.82x7.18x4.76in while /capture on the same item read
    4.21x8.2x9.91in -- multiplying the selection's numbers by the device's stored
    system_scale_factor reproduces the capture numbers almost exactly, proving the two paths were
    never applying the same corrections. shape_type is None for detect_live() (it's an untagged
    background poll, not a capture an operator tagged), which makes apply_shape_correction a
    no-op there exactly as before -- only system_scale_factor's behavior actually changes.
    """
    length_in, width_in, height_in = apply_shape_correction(shape_type, length_in, width_in, height_in)
    length_in, width_in, height_in = _apply_system_scale_factor(length_in, width_in, height_in)
    return length_in, width_in, height_in


def _corrected_measurement(
    extents_mm: np.ndarray, axes: np.ndarray, centroid: np.ndarray, shape_type: Optional[str] = None
) -> tuple[float, float, float, np.ndarray, float, float, float]:
    """Sorts extents into reported L>=W>=H order, applies the correction chain, and rebuilds
    corners_mm from the CORRECTED extents -- not the raw ones. Without this, corners_mm (what the
    drawn wireframe box and its on-image dimension labels come from, in api.py) would keep
    reflecting pre-correction geometry even after _apply_corrections fixed the reported L/W/H
    numbers, since the box shape and the scalar numbers were built from two different sources
    that only the numbers side was being corrected. This is the second half of the same root-cause
    fix as _apply_corrections: the picture and the numbers must be built from the SAME corrected
    values, not just have the same correction CALLED on both.

    extents_mm/axes are in PCA-axis order (not sorted by magnitude); the sort permutation is
    tracked (`order`) so the corrected sorted-descending inches values can be placed back onto
    their original physical axes before rebuilding the box -- correcting length doesn't mean
    "whichever axis happens to be first in PCA order," it means "whichever axis is currently the
    longest," which can differ capture to capture as an item rotates under the camera.

    Also returns the RAW sorted-descending extents (inches, pre-correction) -- Training Mode's
    batch calibration (api.py's /training/compute) fits a fresh system_scale_factor from
    true_size/raw_size ratios, which must be computed against uncorrected geometry or the fit
    would be circular (correcting an already-corrected number). Every other existing caller
    ignores these last 3 return values; this is purely additive.
    """
    order = np.argsort(extents_mm)[::-1]
    sorted_extents_mm = extents_mm[order]
    raw_length_in, raw_width_in, raw_height_in = (float(e) / MM_PER_INCH for e in sorted_extents_mm)
    length_in, width_in, height_in = _apply_corrections(raw_length_in, raw_width_in, raw_height_in, shape_type)

    corrected_sorted_mm = np.array([length_in, width_in, height_in]) * MM_PER_INCH
    corrected_extents_mm = np.empty_like(extents_mm)
    corrected_extents_mm[order] = corrected_sorted_mm
    corners_mm = _bounding_box_corners(axes, centroid, corrected_extents_mm)
    return length_in, width_in, height_in, corners_mm, raw_length_in, raw_width_in, raw_height_in


def measure(
    depth_mm: np.ndarray,
    intrinsics: Intrinsics,
    shape_type: Optional[str] = None,
    exclude_mask: Optional[np.ndarray] = None,
    include_mask: Optional[np.ndarray] = None,
) -> MeasurementResult:
    background_mm = load_calibration()
    if background_mm is None:
        raise NotCalibratedError("Not calibrated yet -- set a camera height and Quick Calibrate, or Precise Calibrate with an empty surface.")
    if background_mm.shape != depth_mm.shape:
        raise NotCalibratedError("Calibration reference resolution does not match the current frame.")

    mask = _segment_foreground_mask(depth_mm, background_mm, exclude_mask=exclude_mask, include_mask=include_mask)
    points = _deproject(depth_mm, mask, intrinsics)
    points = _remove_outliers(points)

    if len(points) < MIN_OBJECT_POINTS:
        raise NoObjectDetectedError("No object detected under the camera.")

    extents_mm, axes, centroid = _oriented_bounding_box(points)
    fit_ratio = _cuboid_fit_ratio(points, axes, centroid, extents_mm)
    classification = "box" if fit_ratio >= BOX_FIT_RATIO_THRESHOLD else "irregular-item"

    # Replace the PCA-noisy height axis with a direct background-relative depth measurement --
    # see _refine_height's docstring for why. Length/width extents are untouched.
    extents_mm, _height_refined_corners_mm = _refine_height(depth_mm, background_mm, mask, extents_mm, axes, centroid)

    # Sort L >= W >= H, apply shape-profile offset + system_scale_factor, and rebuild corners_mm
    # from the CORRECTED extents (not the raw ones) -- see _corrected_measurement's docstring for
    # why this second half of the fix matters just as much as calling the same correction chain
    # detect_live() does: the drawn wireframe box and its on-image labels read directly off
    # corners_mm in api.py, so if corners_mm stayed uncorrected the picture would keep showing
    # pre-correction geometry even with the reported numbers now fixed.
    length_in, width_in, height_in, corners_mm, raw_length_in, raw_width_in, raw_height_in = _corrected_measurement(
        extents_mm, axes, centroid, shape_type
    )

    cubic_feet = (length_in * width_in * height_in) / IN3_PER_FT3
    suggested_shape_type = _suggest_shape_type(classification, points, axes)

    return MeasurementResult(
        length_in=round(length_in, 2),
        width_in=round(width_in, 2),
        height_in=round(height_in, 2),
        cubic_feet=round(cubic_feet, 3),
        classification=classification,
        confidence=round(fit_ratio, 3),
        point_count=len(points),
        corners_mm=[tuple(c) for c in corners_mm.tolist()],
        raw_length_in=round(raw_length_in, 2),
        raw_width_in=round(raw_width_in, 2),
        raw_height_in=round(raw_height_in, 2),
        suggested_shape_type=suggested_shape_type,
    )


# Live-overlay blob segmentation runs much more often than an explicit /capture (every preview
# frame vs. on demand), so its floor is meant to be higher than MIN_OBJECT_POINTS to avoid
# flickering boxes around small noise clusters that a one-shot /capture would correctly reject
# via voxel filtering but that would be distracting flashing on every single frame.
#
# This value was actually LOWER than MIN_OBJECT_POINTS (150 vs 200) -- backwards from the stated
# intent above, and confirmed live to cause a real "picture and result don't match" bug: /capture
# now scopes measurement to exactly the blob detect_live() drew a cyan border around (see
# _capture's single-detection scoping), so a blob that clears THIS floor but not
# MIN_OBJECT_POINTS's floor shows confidently on screen yet Capture rejects it with "No object
# detected." Set with margin above MIN_OBJECT_POINTS (not just +1) because deprojecting a raw
# pixel-area blob into 3D points and then removing outliers (_remove_outliers) always loses some
# fraction of points -- a live floor equal to the post-outlier-removal floor would still leave a
# gap where borderline blobs pass one filter but not the other.
LIVE_MIN_BLOB_PIXELS = 260

# No item placed under the camera is realistically going to occupy more than this fraction of the
# frame -- past this size a "detection" is almost always background bleed: a stale/drifted
# calibration reference, glare/reflection off a shiny floor or wallpaper, or a shadow that reads
# as a large uniform depth-shift region. Without this ceiling, detect_live() will happily draw a
# bounding box around most of a table or wall and report it as a real item (the exact "the system
# is registering wallpaper/wood grain" failure mode this guards against). Only caps the LIVE
# overlay, which runs on every preview frame and has no per-blob confidence check -- an explicit
# /capture still measures whatever it's given, since a genuinely large real item is possible there.
LIVE_MAX_BLOB_AREA_FRACTION = 0.30

# Frame-to-frame nearest-centroid tracking so a blob's `id` stays stable across polls (needed for
# "click detection id 7 to exclude it" to mean anything -- without this, id would just be that
# frame's list position, which reshuffles as items are added/removed). A real tracker (Kalman/
# SORT) is overkill given the low poll rate and mostly-static scene; greedy nearest-centroid
# matching within a distance threshold is enough. State lives at module scope, same pattern as
# api.py's _last_bbox_overlay -- there's only ever one active camera/scene being tracked.
TRACK_MATCH_DISTANCE_MM = 150.0
_track_state = {"next_id": 1, "prev": []}  # prev: list of {"id": int, "centroid": np.ndarray}


# Exponential-moving-average smoothing of a tracked item's reported L/W/H, keyed by tracking id.
# Live-confirmed the problem this fixes: polling /detections repeatedly against a genuinely
# stationary item (same tracked id, same bbox, camera not moving) still showed the reported
# length/width/height drift by ~0.05-0.08in poll to poll -- normal structured-light sensor noise
# in the raw point cloud feeding a fresh PCA fit every single frame, with no memory of what this
# same physical item measured a moment ago. A real, unmoving object does not actually change
# size between two consecutive frames; only the noisy MEASUREMENT of it does. EMA_ALPHA=0.3 (30%
# weight on the newest sample) converges to a stable reading within a few frames of an item being
# placed while still responding quickly to it actually being picked up and swapped for a
# different item (a new tracking id starts fresh with no smoothing history to fight against).
SMOOTHING_EMA_ALPHA = 0.3
_smoothing_state: dict[int, dict] = {}  # {tracking_id: {"length": float, "width": float, "height": float}}


def _smoothed_extents_in(tracking_id: int, length_in: float, width_in: float, height_in: float) -> tuple[float, float, float]:
    prev = _smoothing_state.get(tracking_id)
    if prev is None:
        smoothed = {"length": length_in, "width": width_in, "height": height_in}
    else:
        a = SMOOTHING_EMA_ALPHA
        smoothed = {
            "length": a * length_in + (1 - a) * prev["length"],
            "width": a * width_in + (1 - a) * prev["width"],
            "height": a * height_in + (1 - a) * prev["height"],
        }
    _smoothing_state[tracking_id] = smoothed
    return smoothed["length"], smoothed["width"], smoothed["height"]


def _prune_smoothing_state(active_ids: set) -> None:
    """Drops smoothing history for any tracking id no longer present in the current frame --
    otherwise this dict would grow unboundedly over a long-running session as items come and go,
    and a stale entry could theoretically collide if _track_state's id counter ever wrapped
    (it won't in practice, but pruning is free and keeps the invariant simple: an id present in
    _smoothing_state is always an id detect_live() saw in its most recent call)."""
    stale = [tid for tid in _smoothing_state if tid not in active_ids]
    for tid in stale:
        del _smoothing_state[tid]


def _assign_tracking_ids(centroids: list[np.ndarray]) -> list[int]:
    prev = _track_state["prev"]
    assigned_ids: list[Optional[int]] = [None] * len(centroids)
    used_prev = set()

    if prev and centroids:
        # Greedy nearest-centroid matching: repeatedly pick the closest remaining (current, prev)
        # pair under the distance threshold, so a large jump doesn't steal a nearby item's id.
        pairs = []
        for ci, c in enumerate(centroids):
            for pi, p in enumerate(prev):
                dist = float(np.linalg.norm(c - p["centroid"]))
                if dist <= TRACK_MATCH_DISTANCE_MM:
                    pairs.append((dist, ci, pi))
        pairs.sort(key=lambda t: t[0])
        used_current = set()
        for dist, ci, pi in pairs:
            if ci in used_current or pi in used_prev:
                continue
            assigned_ids[ci] = prev[pi]["id"]
            used_current.add(ci)
            used_prev.add(pi)

    for i, assigned in enumerate(assigned_ids):
        if assigned is None:
            assigned_ids[i] = _track_state["next_id"]
            _track_state["next_id"] += 1

    _track_state["prev"] = [{"id": assigned_ids[i], "centroid": centroids[i]} for i in range(len(centroids))]
    return assigned_ids  # type: ignore[return-value]


def detect_live(
    depth_mm: np.ndarray,
    intrinsics: Intrinsics,
    exclude_mask: Optional[np.ndarray] = None,
    include_mask: Optional[np.ndarray] = None,
) -> list[LiveDetection]:
    """Continuous, lightweight multi-item detection for the live-view overlay.

    Unlike measure(), this segments the foreground mask into SEPARATE connected components
    first (via cv2.connectedComponentsWithStats -- already installed transitively through
    cv_bridge, confirmed present in the ros2 env) so multiple distinct items each get their own
    bounding box, rather than one box spanning everything. Each blob then gets the same
    deproject -> voxel-filter -> PCA-bbox treatment measure() uses for its single object, just
    without the cuboid-fit-ratio/classification pass (not needed for a live overlay border).
    """
    import cv2

    background_mm = load_calibration()
    if background_mm is None or background_mm.shape != depth_mm.shape:
        return []

    mask = _segment_foreground_mask(depth_mm, background_mm, exclude_mask=exclude_mask, include_mask=include_mask)
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    max_blob_area = depth_mm.size * LIVE_MAX_BLOB_AREA_FRACTION

    blobs = []  # (points, corners_mm, length_in, width_in, height_in, centroid, bbox)
    for label in range(1, num_labels):  # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < LIVE_MIN_BLOB_PIXELS or area > max_blob_area:
            continue
        blob_mask = labels == label
        points = _deproject(depth_mm, blob_mask, intrinsics)
        points = _remove_outliers(points)
        if len(points) < MIN_VOXEL_NEIGHBOR_COUNT * 4:
            continue
        extents_mm, axes, centroid = _oriented_bounding_box(points)
        # Same background-relative height correction measure() applies -- see _refine_height.
        extents_mm, _height_refined_corners_mm = _refine_height(depth_mm, background_mm, blob_mask, extents_mm, axes, centroid)
        # Same correction chain + corners_mm rebuild measure() applies (shape_type=None here --
        # see _apply_corrections'/_corrected_measurement's docstrings for why this call being
        # MISSING was the actual root cause of the live view reporting a different number, AND
        # drawing a different-shaped box, than /capture for the same object).
        length_in, width_in, height_in, corners_mm, _raw_l, _raw_w, _raw_h = _corrected_measurement(extents_mm, axes, centroid)
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        # extents_mm/axes/centroid carried through alongside the already-corrected L/W/H so the
        # smoothing step below can rebuild corners_mm in the correct PCA-axis order (see that
        # step's comment for why a naive re-sort would silently produce a wrong/rotated box).
        blobs.append((points, corners_mm, length_in, width_in, height_in, extents_mm, axes, centroid, bbox))

    ids = _assign_tracking_ids([b[7] for b in blobs])
    _prune_smoothing_state(set(ids))

    detections: list[LiveDetection] = []
    for (points, corners_mm, length_in, width_in, height_in, extents_mm, axes, centroid, bbox), blob_id in zip(blobs, ids):
        # Smooth by tracking id, not by raw per-frame PCA output -- see SMOOTHING_EMA_ALPHA's
        # docstring for the live-confirmed jitter this fixes. Rebuild corners_mm from the SMOOTHED
        # extents (same "picture must match the number" principle _corrected_measurement already
        # applies for shape-offset/scale-factor correction) so the drawn live-overlay box doesn't
        # visibly disagree with the smoothed L/W/H shown next to it. _bounding_box_corners needs
        # extents in PCA-axis order (matching `axes`' columns), not sorted L>=W>=H order, so the
        # sort permutation is recomputed here exactly as _corrected_measurement does internally --
        # extents_mm hasn't changed since that call, so np.argsort(extents_mm) reproduces the
        # identical mapping deterministically.
        length_in, width_in, height_in = _smoothed_extents_in(blob_id, length_in, width_in, height_in)
        order = np.argsort(extents_mm)[::-1]
        smoothed_sorted_mm = np.array([length_in, width_in, height_in]) * MM_PER_INCH
        smoothed_extents_mm = np.empty_like(extents_mm)
        smoothed_extents_mm[order] = smoothed_sorted_mm
        corners_mm = _bounding_box_corners(axes, centroid, smoothed_extents_mm)
        detections.append(
            LiveDetection(
                id=blob_id,
                length_in=round(length_in, 2),
                width_in=round(width_in, 2),
                height_in=round(height_in, 2),
                cubic_feet=round((length_in * width_in * height_in) / IN3_PER_FT3, 3),
                corners_mm=[tuple(c) for c in corners_mm.tolist()],
                point_count=len(points),
                bbox=bbox,
            )
        )
    return detections
