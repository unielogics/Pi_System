"""
Local capture API. Binds to 127.0.0.1:8090 -- only reachable via the Caddy HTTPS reverse proxy
in front of it, never directly. Bearer-token guarded; the token is set via DIMENSIONER_AUTH_TOKEN
and matched against whatever the WMS dashboard stores (encrypted) for this device.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np

from adapters.aurora_adapter import get_aurora_adapter
from device_config import (
    DeviceConfig,
    load_device_config,
    set_auto_absorb_config,
    set_calibration_state,
    set_camera_height_mm,
    set_permanent_exclude_regions,
    set_system_scale_factor,
    update_device_config,
)
from measure import (
    FOREGROUND_THRESHOLD_MM,
    MM_PER_INCH,
    Intrinsics,
    MeasurementResult,
    NoObjectDetectedError,
    NotCalibratedError,
    detect_live,
    load_calibration,
    load_calibration_with_generation,
    measure,
    save_calibration,
    save_calibration_if_current,
    synthesize_flat_background,
)
from products import create_product, delete_product, get_product, list_products
from shape_profiles import (
    BUILTIN_SHAPE_TYPES,
    list_shape_profiles,
    record_correction,
    replace_shape_profile,
    reset_shape_profile,
)
from local_capture_log import add_entry, correct_entry, list_entries
from registration import STATE_PATH as REGISTRATION_STATE_PATH
from training import (
    IDLE_ABANDON_SEC,
    MAX_CONSECUTIVE_AUTO_RECALIBRATIONS,
    MIN_REQUIRED_ACCURACY_PCT,
    MIN_TRAINING_SAMPLES,
    VERIFICATION_ROUND_SIZE,
    TrainingSample,
    TrainingSession,
    load_training_session,
    new_session_id,
    update_training_session,
)

app = FastAPI()

# v5: the dashboard now calls this device directly from the browser (the backend can't reach a
# device's LAN-private endpointUrl at all), so this origin is genuinely cross-origin from the
# browser's perspective and needs CORS headers -- bearer-token auth on every route is what
# actually gates access, this just lets the browser's preflight through.
#
# The kiosk app ALSO calls this device directly from the browser (see Kiosk/lib/api/kiosk-
# client.ts's fetchDimensionerDirect), from an origin like https://kiosk.wh-007.uniewms.com --
# one per warehouse. allow_origins only takes exact literal strings, which can't express "any
# warehouse's kiosk subdomain," so allow_origin_regex covers that -- same pattern UnieBackend's
# own CORS config already uses for the identical subdomain shape (see server.ts's
# warehouseKioskPattern), kept in sync with it deliberately.
#
# allow_private_network=True is REQUIRED on top of the origin allowlist above -- Chrome's Private
# Network Access policy adds a SEPARATE preflight check whenever a public HTTPS page (the
# dashboard or kiosk) calls a hostname that resolves to a private LAN address (this device,
# always). Chrome sends `Access-Control-Request-Private-Network: true` on the OPTIONS preflight
# and requires the server to answer `Access-Control-Allow-Private-Network: true` or it blocks the
# real request client-side with "Permission was denied for this request to access the `local`
# address space" -- confirmed live: curl'ing the preflight without this flag returns HTTP 400
# "Disallowed CORS private-network" from Starlette's own CORSMiddleware. This is independent of
# (and in addition to) the allow_origins/allow_origin_regex check; both must pass.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.warehouse-admin.uniewms.com",
        "https://warehouse-admin.uniewms.com",
    ],
    allow_origin_regex=r"^https?://kiosk\.wh-[a-z0-9-]+\.(uniewms|unielogics)\.com$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

AUTH_TOKEN = os.environ.get("DIMENSIONER_AUTH_TOKEN", "")

# Self-registered devices get their token minted by the backend at runtime (see registration.py),
# long after this module's AUTH_TOKEN constant was already read from the env at import time --
# so _check_auth also consults registration_state.json on every request, re-read only when its
# mtime changes (a heartbeat-driven token rotation should take effect on the very next request,
# with no service restart). Manually configured devices (DIMENSIONER_AUTH_TOKEN set, no
# registration_state.json) fall through to that env var unchanged.
_registration_token_cache: dict = {"mtime": None, "token": None}


def _registered_auth_token() -> str | None:
    try:
        mtime = os.path.getmtime(REGISTRATION_STATE_PATH)
    except OSError:
        return None
    if _registration_token_cache["mtime"] != mtime:
        try:
            with open(REGISTRATION_STATE_PATH, "r") as f:
                data = json.load(f)
            _registration_token_cache["token"] = data.get("auth_token")
            _registration_token_cache["mtime"] = mtime
        except (OSError, ValueError):
            return None
    return _registration_token_cache["token"]

# Live preview runs at a fixed, modest rate independent of the camera's own publish rate --
# this Pi's camera is on a USB 2.0 port and is bandwidth-capped (see driver_launcher.py), so the
# preview just re-serves whatever the latest frame is rather than trying to keep up 1:1.
PREVIEW_INTERVAL_SEC = 0.3

# A frame older than this is considered stale -- the camera/driver has hung or disconnected.
# Matches the observed disconnect pattern (driver keeps running but stops receiving frames).
STALE_FRAME_THRESHOLD_SEC = 3.0

# If frames stay stale this long, the watchdog restarts dimensioner-ros.service. Set well above
# STALE_FRAME_THRESHOLD_SEC so a normal brief hiccup doesn't trigger a restart.
WATCHDOG_RESTART_AFTER_SEC = 12.0
WATCHDOG_POLL_INTERVAL_SEC = 2.0
WATCHDOG_COOLDOWN_AFTER_RESTART_SEC = 30.0
# If the driver has published exactly ZERO frames this long after this process started (e.g.
# WORKSPACE_SETUP missing, rclpy init/subscription setup failed), also restart it -- previously
# this case was detected (never_received) but never actually acted on. Longer than
# WATCHDOG_RESTART_AFTER_SEC to give the driver's own normal startup sequence (ROS2 node init,
# camera handshake) room to finish before treating "no frames yet" as a real failure.
WATCHDOG_NEVER_RECEIVED_GRACE_SEC = 30.0

_last_restart_at: float = 0.0
_process_started_at: float = time.monotonic()
_last_bbox_overlay: dict | None = None  # set by /capture, drawn on the next /stream/rgb frame


def _check_auth(authorization: str | None, token_param: str | None = None) -> None:
    registered_token = _registered_auth_token()
    valid_token = registered_token or AUTH_TOKEN
    if not valid_token:
        # No token configured on this device yet -- refuse to serve rather than run open.
        raise HTTPException(status_code=503, detail="Device has no auth token configured.")
    presented = None
    if authorization and authorization.startswith("Bearer "):
        presented = authorization[len("Bearer ") :]
    elif token_param:
        presented = token_param
    if presented != valid_token:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


# ---- Manual region correction (v7): one-shot live overrides + persistent exclude zones ----

# One-shot overrides expire quickly so a stale override from a closed browser tab doesn't silently
# keep suppressing/forcing a region forever. Keyed by nothing (single active camera/operator) --
# same simplifying assumption _last_bbox_overlay already makes.
DETECTION_OVERRIDE_TTL_SEC = 60.0
_detection_overrides: dict = {"exclude": [], "include": [], "at": 0.0}


def _rects_to_mask(rects: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    """Rasterizes a list of {"x","y","w","h"} pixel rectangles (native frame pixel coordinates)
    into a boolean mask of the given (height, width) shape. Returns None for an empty list so
    callers can cheaply skip mask-modification work when there's nothing to apply."""
    if not rects:
        return None
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for rect in rects:
        x0 = max(0, int(rect.get("x", 0)))
        y0 = max(0, int(rect.get("y", 0)))
        x1 = min(width, x0 + max(0, int(rect.get("w", 0))))
        y1 = min(height, y0 + max(0, int(rect.get("h", 0))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _active_override_masks(shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Combines the persistent (device_config) exclude zones with any still-live one-shot
    overrides into the exclude/include masks measure()/detect_live() expect. Both sources use the
    same pixel-rect format, so persistent + one-shot excludes are just concatenated."""
    persistent_exclude = load_device_config().permanent_exclude_regions
    one_shot_expired = (time.monotonic() - _detection_overrides["at"]) > DETECTION_OVERRIDE_TTL_SEC
    one_shot_exclude = [] if one_shot_expired else _detection_overrides["exclude"]
    one_shot_include = [] if one_shot_expired else _detection_overrides["include"]
    exclude_mask = _rects_to_mask(persistent_exclude + one_shot_exclude, shape)
    include_mask = _rects_to_mask(one_shot_include, shape)
    return exclude_mask, include_mask


# Click-to-select: an operator clicks one of the live cyan-bordered detections in the preview to
# say "measure THIS one" when several items are visible. This is deliberately NOT the same
# mechanism as includeRegions above -- forcing an arbitrary rectangle into the foreground mask
# (what Include does) trusts whatever raw pixels are under that rectangle regardless of whether
# they belong to a real object, which live-tested into a false 45x8x5in detection over a random
# patch of floor. Selecting an EXISTING tracked detection instead only ever scopes measurement to
# pixels detect_live() already independently decided were a real object.
#
# Longer TTL (5 min) than the one-shot region override's 60s -- an operator needs time to select,
# possibly reframe/nudge the item, and then press the separate Capture button, vs. a region
# correction that's meant to affect the very next capture immediately.
SELECTED_DETECTION_TTL_SEC = 300.0
_selected_detection: dict = {"id": None, "at": 0.0}


def _selected_detection_id() -> int | None:
    if _selected_detection["id"] is None:
        return None
    if (time.monotonic() - _selected_detection["at"]) > SELECTED_DETECTION_TTL_SEC:
        return None
    return _selected_detection["id"]


def _bbox_to_dict(bbox: tuple[int, int, int, int]) -> dict:
    return {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}


@app.post("/detections/select")
def select_detection(body: dict, authorization: str | None = Header(default=None)):
    """Click-to-select: body { x, y } in native frame pixel coordinates (same coordinate space
    the region-override click mapping already produces). Finds whichever currently-live
    detection's bbox contains that point and remembers it as the active selection for the next
    /capture to scope to -- see /capture's disambiguation logic below.

    Also runs one scoped measure() against just the clicked item to return suggestedShapeType
    (see measure.py's _suggest_shape_type) -- detect_live()'s own continuous per-frame pass
    deliberately skips the cuboid-fit-ratio/classification work this needs (see its docstring:
    running that on every live preview frame for every blob would waste CPU on a Pi4), so this is
    the one place a fuller measurement is affordable: a single click, not a polling loop. Training
    Mode uses this to pre-select the shape-type dropdown instead of always defaulting to "box,"
    while leaving the operator free to override it. Failure to compute a suggestion (stale
    calibration, a transient NoObjectDetectedError) is swallowed -- suggestedShapeType is simply
    omitted from the response rather than failing the whole selection, since the selection itself
    already succeeded independently."""
    _check_auth(authorization)
    try:
        x = float(body["x"])
        y = float(body["y"])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid selection payload: {exc}")

    adapter = get_aurora_adapter()
    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        raise HTTPException(status_code=503, detail="Camera feed is stale -- device may be disconnected.")

    depth_frame = adapter.get_depth_frame()
    intrinsics = adapter.get_intrinsics()
    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
    live_detections = detect_live(depth_frame, intrinsics, exclude_mask=exclude_mask, include_mask=include_mask)

    for detection in live_detections:
        bx, by, bw, bh = detection.bbox
        if bx <= x <= bx + bw and by <= y <= by + bh:
            _selected_detection["id"] = detection.id
            _selected_detection["at"] = time.monotonic()
            response = {
                "selectedId": detection.id,
                "length": detection.length_in,
                "width": detection.width_in,
                "height": detection.height_in,
                "bbox": _bbox_to_dict(detection.bbox),
                "expiresInSec": SELECTED_DETECTION_TTL_SEC,
            }
            try:
                scope_exclude = _rects_to_mask([_bbox_to_dict(detection.bbox)], depth_frame.shape)
                outside_scope = ~scope_exclude if scope_exclude is not None else None
                suggestion_exclude = exclude_mask if outside_scope is None else (outside_scope if exclude_mask is None else (exclude_mask | outside_scope))
                suggestion_result = measure(depth_frame, intrinsics, exclude_mask=suggestion_exclude, include_mask=include_mask)
                response["suggestedShapeType"] = suggestion_result.suggested_shape_type
            except (NotCalibratedError, NoObjectDetectedError):
                pass
            return response

    raise HTTPException(status_code=404, detail="No detected item at that point -- click inside a cyan-bordered box.")


def _scope_exclude_to_single_detection(
    depth_frame: np.ndarray,
    intrinsics: Intrinsics,
    exclude_mask: np.ndarray | None,
    include_mask: np.ndarray | None,
    allow_selection_fallback: bool = True,
) -> np.ndarray | None:
    """Scopes a measurement to exactly the one real, blob-filtered object detect_live() finds --
    NEVER the raw whole-frame foreground mask. Shared by /capture, the accuracy-check endpoints,
    and anything else that calls measure() directly on a live frame, so every measurement path
    gets the same guarantee instead of drifting apart (which is exactly how this bug family keeps
    recurring -- see _apply_corrections' docstring for the same lesson applied to the correction
    chain instead of the segmentation mask).

    Root cause this closes: /calibrate/accuracy-check(/apply) called measure() with only the
    exclude/include overrides applied, no detection-scoping -- the exact unscoped bug /capture had
    before this session's Part 1 fix. A raw mask can merge a shadow, a hand, or any other object
    left in frame into the "reference object" measurement, producing a wildly inflated number.
    Live-confirmed the actual damage: an 8.75x7x4.75in reference card produced a stored
    system_scale_factor of ~0.33/0.55/0.97 -- solving true/measured=factor backwards shows the
    unscoped "measured" value was ~26in, roughly 3x the card's real length. Because
    system_scale_factor is GLOBAL and multiplicative, that one corrupted Accuracy Check silently
    made every future measurement of every future object wrong, not just that one reading.

    `allow_selection_fallback` (default True): whether 2+ detections may be disambiguated by
    reusing whatever POST /detections/select last recorded, rather than always requiring a fresh
    selection right now. /capture passes True -- the Measure tab's click-to-select IS its
    disambiguation UI, by design. The accuracy-check endpoints pass False: their own tab (Calibrate)
    has no click-to-select affordance of its own at all, so a selection made earlier on the
    unrelated Measure tab is not something the operator running an Accuracy Check has any reason
    to know exists. An adversarial audit confirmed this was reachable and silent: select item A on
    Measure, switch to Calibrate without clicking Clear (nothing auto-deselects on tab switch,
    selection TTL is 300s), place reference object B without removing A, run Measure & Compare --
    with both A and B detected, the OLD code silently scoped to the stale selection A instead of
    raising the disambiguation error, and neither the Check nor Apply response echoes an id/bbox
    that would let the operator notice a different object than the one in front of them was used.
    With allow_selection_fallback=False, 2+ detections during an accuracy check ALWAYS 409s
    (asking the operator to physically clear clutter -- there's nothing to "click" on this tab),
    which can never silently pick the wrong one.

    Raises HTTPException(409) if 2+ detections are visible and (with fallback allowed) none is
    selected, or (with fallback disallowed) unconditionally -- silently merging multiple real
    objects together would repeat the exact mistake this function exists to prevent.
    """
    live_detections = detect_live(depth_frame, intrinsics, exclude_mask=exclude_mask, include_mask=include_mask)

    if len(live_detections) == 1:
        only = live_detections[0]
        scope_exclude = _rects_to_mask([_bbox_to_dict(only.bbox)], depth_frame.shape)
        outside_scope = ~scope_exclude if scope_exclude is not None else None
        if outside_scope is not None:
            exclude_mask = outside_scope if exclude_mask is None else (exclude_mask | outside_scope)
    elif len(live_detections) >= 2:
        selected_id = _selected_detection_id() if allow_selection_fallback else None
        if selected_id is None:
            detail = (
                "Multiple items visible; click one in the preview first."
                if allow_selection_fallback
                else "Multiple items visible under the camera -- remove all but the reference object before running Measure & Compare."
            )
            raise HTTPException(status_code=409, detail=detail)
        selected = next((d for d in live_detections if d.id == selected_id), None)
        if selected is None:
            _selected_detection["id"] = None
            raise HTTPException(status_code=409, detail="Selected item is no longer visible; click it again.")
        selection_exclude = _rects_to_mask([_bbox_to_dict(selected.bbox)], depth_frame.shape)
        outside_selection = ~selection_exclude if selection_exclude is not None else None
        if outside_selection is not None:
            exclude_mask = outside_selection if exclude_mask is None else (exclude_mask | outside_selection)
    # else: 0 detections passed detect_live()'s blob filters -- fall through to the legacy
    # unscoped path. Covers a genuinely large item that exceeds detect_live()'s max-blob-area
    # ceiling (a real object, not noise) as well as the true "nothing here" case, which measure()
    # itself correctly rejects via NoObjectDetectedError (MIN_OBJECT_POINTS).
    return exclude_mask


@app.post("/detections/deselect")
def deselect_detection(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    _selected_detection["id"] = None
    _selected_detection["at"] = 0.0
    return {"selectedId": None}


def _rgb_to_jpeg_bytes(rgb_frame: np.ndarray, quality: int = 80) -> bytes:
    image = PILImage.fromarray(rgb_frame, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _rgb_to_jpeg_base64(rgb_frame: np.ndarray) -> str:
    return base64.b64encode(_rgb_to_jpeg_bytes(rgb_frame, quality=85)).decode("ascii")


def _project_point(point_mm: tuple[float, float, float], intrinsics: Intrinsics) -> tuple[int, int]:
    x, y, z = point_mm
    if z <= 0:
        z = 1.0
    px = intrinsics.cx + (x * intrinsics.fx) / z
    py = intrinsics.cy + (y * intrinsics.fy) / z
    return int(round(px)), int(round(py))


# The 12 edges of a cube given corner indices from the (-1,1)^3 ordering in
# _bounding_box_corners: sx in (-1,1) outer, sy middle, sz inner -> index = sx_bit*4 + sy_bit*2 + sz_bit.
_BBOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]

# Which of the 12 edges (indexing _BBOX_EDGES) runs along each of the box's own local axes --
# _bounding_box_corners' sign ordering is sx outer/sy middle/sz inner, so edges that flip ONLY
# the sx bit vary along axis 0, only sy vary along axis 1, only sz vary along axis 2. One
# representative edge per axis is enough to place a label; using the same edge every frame (not
# picking whichever happens to be shortest on screen) keeps the label position stable as the
# camera's view of the item doesn't change between frames.
_AXIS_REPRESENTATIVE_EDGE = {0: (0, 4), 1: (0, 2), 2: (0, 1)}

_LABEL_FONT_CACHE: dict = {}


def _label_font(size: int = 14) -> "ImageFont.FreeTypeFont":
    if size not in _LABEL_FONT_CACHE:
        _LABEL_FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _LABEL_FONT_CACHE[size]


def _draw_single_bbox(draw: ImageDraw.ImageDraw, corners_mm: list, intrinsics: Intrinsics, color: tuple, width: int) -> None:
    points_2d = [_project_point(c, intrinsics) for c in corners_mm]
    for a, b in _BBOX_EDGES:
        draw.line([points_2d[a], points_2d[b]], fill=color, width=width)


# How far outward (px) each label is pushed away from the box's screen-space center, along the
# direction from that center to its edge's midpoint. A fixed pixel offset (not proportional to
# edge length) because axis edges vary wildly in on-screen length -- height's edge in particular
# can project to only a few pixels for a flat item, and a proportional offset would put that
# label right on top of the box outline. Confirmed by rendering a real frame during development:
# without this offset, all 3 labels cluster at/inside the box center and overlap illegibly,
# especially on small detections.
_LABEL_OUTWARD_OFFSET_PX = 16.0


# Fixed per-axis colors, independent of the box outline's own color (cyan for live, yellow for
# capture -- that signal is "is this live or a captured result," separate from "which axis is
# which"). Chosen to stay visually distinct from both outline colors so labels don't blend into
# the border they sit next to, and to give operators one consistent visual language between the
# on-image labels and the matching L/W/H input fields in the dashboard/Pi-viewer UI.
_AXIS_LABEL_COLORS = {0: (0, 220, 255), 1: (255, 60, 220), 2: (255, 210, 0)}  # L=cyan, W=magenta, H=yellow


def _draw_dimension_labels(draw: ImageDraw.ImageDraw, corners_mm: list, intrinsics: Intrinsics) -> None:
    """Labels each of the box's 3 real-world axes with its length, positioned outward from that
    edge's midpoint (away from the box's own screen-space center) -- e.g. "L 12.3in" printed just
    outside the long side of the box in the image, not in a disconnected numbers list elsewhere on
    the page and not overlapping the box outline or the other two labels. This is the actual point
    of the overlay: an operator correcting a measurement needs to see, on the picture, which
    physical edge is "length" before they can know which number to change. Each axis is drawn in
    its own fixed color (see _AXIS_LABEL_COLORS) regardless of the box outline's color.
    """
    points_2d = [_project_point(c, intrinsics) for c in corners_mm]
    corners_arr = np.array(corners_mm)
    font = _label_font()
    axis_letters = {0: "L", 1: "W", 2: "H"}  # matches measure()'s sort order: longest=L, then W, shortest=H

    center_x = sum(p[0] for p in points_2d) / len(points_2d)
    center_y = sum(p[1] for p in points_2d) / len(points_2d)

    for axis, (a, b) in _AXIS_REPRESENTATIVE_EDGE.items():
        edge_length_in = float(np.linalg.norm(corners_arr[a] - corners_arr[b])) / MM_PER_INCH
        mid_x = (points_2d[a][0] + points_2d[b][0]) / 2
        mid_y = (points_2d[a][1] + points_2d[b][1]) / 2

        # Direction from the box's screen-space center out through this edge's midpoint --
        # pushing the label further along that same direction moves it away from the box (and
        # away from the other two labels, since each edge midpoint points a different direction)
        # instead of leaving it sitting on the outline where all three midpoints visually cluster.
        dir_x, dir_y = mid_x - center_x, mid_y - center_y
        dir_len = (dir_x ** 2 + dir_y ** 2) ** 0.5
        if dir_len < 1e-3:
            dir_x, dir_y, dir_len = 0.0, -1.0, 1.0  # degenerate (edge midpoint == center): push up
        label_x = mid_x + (dir_x / dir_len) * _LABEL_OUTWARD_OFFSET_PX
        label_y = mid_y + (dir_y / dir_len) * _LABEL_OUTWARD_OFFSET_PX

        label = f"{axis_letters[axis]} {edge_length_in:.1f}in"
        draw.text((label_x, label_y), label, font=font, fill=_AXIS_LABEL_COLORS[axis], stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")


def _draw_bbox_overlay(rgb_frame: np.ndarray, corners_mm: list, adapter) -> np.ndarray:
    """Draws the measured 3D oriented bounding box from an explicit /capture, projected into 2D
    image space, onto the RGB frame -- so the live picture visibly matches the reported
    measurement instead of the operator having to trust a disconnected set of numbers. Also
    labels each axis's real length directly on its edge (see _draw_dimension_labels) so an
    operator can see exactly which edge is length/width/height before correcting a number.

    corners_mm is in DEPTH-camera space (how measure()/detect_live() produce it); this converts
    to RGB-camera space via adapter.depth_to_rgb_mm() before projecting with the RGB camera's OWN
    intrinsics (adapter.get_rgb_intrinsics()) -- see CameraAdapter.get_rgb_intrinsics()'s
    docstring for why projecting a depth-space point with depth intrinsics onto the RGB image
    (the original bug here) visibly offsets the drawn box from the real item: depth and RGB are
    two physically separate lenses on the Aurora 930, confirmed live via distinct camera_info
    topics and a real ~10mm /tf_static baseline between them. Live-confirmed the fix: rendering
    both methods against the same real detection and comparing pixel-for-pixel against the actual
    item in the image showed the depth-intrinsics method's box visibly shifted off the item's
    edges while this method's box sits on them.
    """
    corners_rgb_mm = [adapter.depth_to_rgb_mm(c) for c in corners_mm]
    rgb_intrinsics = adapter.get_rgb_intrinsics()
    image = PILImage.fromarray(rgb_frame, mode="RGB").convert("RGB")
    draw = ImageDraw.Draw(image)
    _draw_single_bbox(draw, corners_rgb_mm, rgb_intrinsics, color=(255, 255, 0), width=3)
    _draw_dimension_labels(draw, corners_rgb_mm, rgb_intrinsics)
    return np.array(image)


def _draw_live_detections_overlay(rgb_frame: np.ndarray, detections: list, adapter) -> np.ndarray:
    """Draws a border around EVERY item currently detected in the frame, continuously (every
    preview frame, not just after a /capture) -- cyan, thinner than the post-capture highlight
    color so the two are visually distinguishable if a capture overlay happens to be showing too.
    Also labels each item's length/width/height directly on the corresponding edge (see
    _draw_dimension_labels) -- this is what lets an operator SEE which physical edge is "length"
    before correcting a number, instead of matching a box in the picture to a number in a
    disconnected list purely by guessing.

    Each detection's corners_mm is in DEPTH-camera space -- see _draw_bbox_overlay's docstring for
    why this must be converted into RGB-camera space (adapter.depth_to_rgb_mm) and projected with
    the RGB camera's own intrinsics (adapter.get_rgb_intrinsics), not the depth camera's, before
    drawing onto the RGB frame.
    """
    if not detections:
        return rgb_frame
    rgb_intrinsics = adapter.get_rgb_intrinsics()
    image = PILImage.fromarray(rgb_frame, mode="RGB").convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        corners_rgb_mm = [adapter.depth_to_rgb_mm(c) for c in detection.corners_mm]
        _draw_single_bbox(draw, corners_rgb_mm, rgb_intrinsics, color=(0, 220, 255), width=2)
        _draw_dimension_labels(draw, corners_rgb_mm, rgb_intrinsics)
    return np.array(image)


DEPTH_VIEW_MODES = ("colorized", "mask", "raw")


def _depth_to_visual_jpeg_bytes(depth_mm: np.ndarray, mode: str = "colorized") -> bytes:
    """Renders a raw depth frame for human viewing in one of three modes:
      - colorized: near=warm, far=cool, invalid=black, detected foreground overlaid in green
                   (the original/default view -- good for general monitoring).
      - mask: ONLY the foreground/background split as solid black/white -- easiest to judge
              exactly what the segmentation step sees, useful while tuning calibration.
      - raw: plain grayscale depth, no color/overlay -- useful for spotting sensor noise or
             artifacts that a colorized view can hide.
    """
    valid = depth_mm > 0
    if not valid.any():
        rgb = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
        return _rgb_to_jpeg_bytes(rgb, quality=70)

    background_mm = load_calibration()
    foreground_mask = None
    if background_mm is not None and background_mm.shape == depth_mm.shape:
        closer_by = background_mm.astype(np.float32) - depth_mm.astype(np.float32)
        foreground_mask = valid & (background_mm > 0) & (closer_by > FOREGROUND_THRESHOLD_MM)

    if mode == "mask":
        rgb = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
        if foreground_mask is not None:
            rgb[foreground_mask] = [255, 255, 255]
        return _rgb_to_jpeg_bytes(rgb, quality=70)

    near_mm, far_mm = 300.0, 3000.0  # matches the vendor's documented 30cm-300cm working range
    clipped = np.clip(depth_mm.astype(np.float32), near_mm, far_mm)
    normalized = 1.0 - (clipped - near_mm) / (far_mm - near_mm)  # near=1 (bright), far=0

    if mode == "raw":
        gray = (normalized * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[~valid] = 0
        return _rgb_to_jpeg_bytes(rgb, quality=75)

    # mode == "colorized" (default)
    r = (normalized * 255).astype(np.uint8)
    b = ((1 - normalized) * 255).astype(np.uint8)
    g = np.zeros_like(r)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    if foreground_mask is not None:
        rgb[foreground_mask] = [0, 255, 0]

    return _rgb_to_jpeg_bytes(rgb, quality=75)


_OFFLINE_PLACEHOLDER_JPEG_CACHE: dict[tuple[int, int], bytes] = {}


def _offline_placeholder_jpeg(size: tuple[int, int] = (640, 400)) -> bytes:
    cached = _OFFLINE_PLACEHOLDER_JPEG_CACHE.get(size)
    if cached is not None:
        return cached
    image = PILImage.new("RGB", size, (30, 10, 10))
    draw = ImageDraw.Draw(image)
    text = "CAMERA OFFLINE"
    draw.text((size[0] // 2 - 70, size[1] // 2 - 10), text, fill=(255, 80, 80))
    jpeg_bytes = _rgb_to_jpeg_bytes(np.array(image), quality=70)
    _OFFLINE_PLACEHOLDER_JPEG_CACHE[size] = jpeg_bytes
    return jpeg_bytes


def _mjpeg_stream(frame_fn, is_stale_fn):
    boundary = "dimensionerframe"
    while True:
        try:
            if is_stale_fn():
                jpeg_bytes = _offline_placeholder_jpeg()
            else:
                jpeg_bytes = frame_fn()
        except Exception:
            jpeg_bytes = _offline_placeholder_jpeg()
        yield (
            f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg_bytes)}\r\n\r\n"
        ).encode("ascii") + jpeg_bytes + b"\r\n"
        time.sleep(PREVIEW_INTERVAL_SEC)


def _camera_status() -> dict:
    try:
        adapter = get_aurora_adapter()
        age = adapter.get_frame_age_sec()
    except Exception:
        age = None
    live = age is not None and age < STALE_FRAME_THRESHOLD_SEC
    return {"cameraLive": live, "frameAgeSec": round(age, 1) if age is not None else None}


async def _watchdog_loop() -> None:
    global _last_restart_at
    while True:
        await asyncio.sleep(WATCHDOG_POLL_INTERVAL_SEC)
        try:
            adapter = get_aurora_adapter()
            age = adapter.get_frame_age_sec()
        except Exception:
            age = None

        now = time.monotonic()
        since_last_restart = now - _last_restart_at
        stale_too_long = age is not None and age > WATCHDOG_RESTART_AFTER_SEC
        never_received = age is None and (now - _process_started_at) > WATCHDOG_NEVER_RECEIVED_GRACE_SEC

        if (stale_too_long or never_received) and since_last_restart > WATCHDOG_COOLDOWN_AFTER_RESTART_SEC:
            _last_restart_at = now
            subprocess.run(
                ["sudo", "-n", "systemctl", "restart", "dimensioner-ros.service"],
                capture_output=True,
            )


# Auto-absorb static clutter into the background reference: a Precise Calibrate clears the
# surface at that moment, but anything left in frame afterward (or never actually cleared before
# calibrating) stays flagged as a detection forever -- _segment_foreground_mask only ever compares
# against the one-time calibration snapshot, with no concept of "this has been sitting here
# unchanged for minutes, stop flagging it." This loop watches for exactly that: foreground pixels
# whose depth hasn't meaningfully changed in a while get folded directly into background_mm, so
# the very next measure()/detect_live() call silently stops seeing them -- no separate "ignore
# these detections" list needed, since the fix is applied at the actual source of truth.
AUTO_ABSORB_POLL_INTERVAL_SEC = 8.0
# Same jitter tolerance _segment_foreground_mask already uses for "did this pixel really move" --
# a structured-light sensor has a few mm of frame-to-frame noise even on a completely static scene,
# so anything under this isn't a real change, just sensor jitter.
AUTO_ABSORB_JITTER_TOLERANCE_MM = FOREGROUND_THRESHOLD_MM

_static_last_depth: np.ndarray | None = None  # last-seen depth per pixel, for change detection
_static_since: np.ndarray | None = None  # monotonic timestamp each pixel's depth last changed


def _reset_static_tracker() -> None:
    """Called whenever /calibrate runs -- a fresh background reference invalidates every prior
    per-pixel timer (comparing pre-recalibration timestamps against the new background could
    absorb pixels that never actually sat still against THIS background)."""
    global _static_last_depth, _static_since
    _static_last_depth = None
    _static_since = None


async def _auto_absorb_loop() -> None:
    global _static_last_depth, _static_since
    while True:
        await asyncio.sleep(AUTO_ABSORB_POLL_INTERVAL_SEC)
        try:
            config = load_device_config()
            if not config.auto_absorb_enabled:
                continue

            background_mm, calibration_generation = load_calibration_with_generation()
            if background_mm is None:
                continue

            adapter = get_aurora_adapter()
            age = adapter.get_frame_age_sec()
            if age is None or age > STALE_FRAME_THRESHOLD_SEC:
                continue
            depth_frame = adapter.get_depth_frame()
            if depth_frame.shape != background_mm.shape:
                continue

            foreground_mask = _segment_foreground_mask(depth_frame, background_mm)
            now = time.monotonic()

            if _static_last_depth is None or _static_last_depth.shape != depth_frame.shape:
                # First tick since startup/calibration -- nothing to compare against yet, just
                # seed the trackers so the NEXT tick can start measuring elapsed time.
                _static_last_depth = depth_frame.copy()
                _static_since = np.full(depth_frame.shape, now, dtype=np.float64)
                continue

            changed = np.abs(depth_frame.astype(np.float64) - _static_last_depth.astype(np.float64)) > AUTO_ABSORB_JITTER_TOLERANCE_MM
            _static_since[changed] = now
            _static_last_depth = depth_frame.copy()

            # Never absorb a pixel inside a CURRENTLY live-detected object's bbox, no matter how
            # long it's sat still -- a real object being actively worked with (mid Accuracy Check,
            # mid click-to-correct, an operator reading a Capture result before moving it) is
            # exactly the scenario that used to erode: it can easily sit still for the full 120s
            # timeout while someone types 3 numbers and reads a result. Confirmed live as the
            # mechanism behind a real corruption incident (see _scope_exclude_to_single_detection's
            # docstring for the accuracy-check-side half of the same root cause). This is a live
            # re-check every tick, not a one-time snapshot, so it stays correct even if the object
            # is swapped out or moved mid-session.
            try:
                intrinsics = adapter.get_intrinsics()
                live_now = detect_live(depth_frame, intrinsics)
                for detection in live_now:
                    bx, by, bw, bh = detection.bbox
                    _static_since[by : by + bh, bx : bx + bw] = now
            except Exception:
                pass

            elapsed = now - _static_since
            absorb_mask = foreground_mask & (elapsed >= config.auto_absorb_timeout_sec)
            if np.any(absorb_mask):
                new_background = background_mm.copy()
                new_background[absorb_mask] = depth_frame[absorb_mask]
                # Compare-and-swap on the generation this tick's background_mm was actually read
                # at: if an operator's /calibrate landed a NEWER background while this tick was
                # doing its (now substantial, detect_live()-including) work, this tick's
                # derived-from-stale-data result is discarded instead of silently reverting the
                # operator's fresh calibration. See save_calibration_if_current's docstring.
                if not save_calibration_if_current(new_background, calibration_generation):
                    # Someone recalibrated mid-tick -- our derived background is stale-by-
                    # definition now. Re-seed the static trackers against the shape we still have
                    # (harmless either way; the very next tick reloads background_mm fresh) so we
                    # don't compare future frames against a depth array that predates the new
                    # calibration.
                    _static_last_depth = None
                    _static_since = None
        except Exception:
            # Never let a transient camera hiccup kill this background loop permanently.
            continue


@app.on_event("startup")
async def _start_watchdog() -> None:
    asyncio.create_task(_watchdog_loop())
    asyncio.create_task(_auto_absorb_loop())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    config = load_device_config()
    return {
        **_camera_status(),
        "cameraHeightMm": config.camera_height_mm,
        "calibration": {
            "mode": config.calibration_mode,
            "calibratedAt": config.calibrated_at,
            "isCalibrated": load_calibration() is not None,
        },
    }


@app.get("/detections")
def detections(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    """Structured (non-image) view of what's currently under the camera -- one entry per
    detected item, with real dimensions computed from its corners. This is what the dashboard's
    "detected items" list polls; the live RGB stream's border overlay is the visual counterpart
    of the exact same detect_live() call.
    """
    _check_auth(authorization, token)
    adapter = get_aurora_adapter()

    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        return {"detections": [], "cameraLive": False}

    depth_frame = adapter.get_depth_frame()
    intrinsics = adapter.get_intrinsics()
    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
    live_detections = detect_live(depth_frame, intrinsics, exclude_mask=exclude_mask, include_mask=include_mask)

    items = [
        {
            "id": detection.id,
            "length": detection.length_in,
            "width": detection.width_in,
            "height": detection.height_in,
            "cubicFeet": detection.cubic_feet,
            "pointCount": detection.point_count,
            "bbox": {"x": detection.bbox[0], "y": detection.bbox[1], "w": detection.bbox[2], "h": detection.bbox[3]},
        }
        for detection in live_detections
    ]
    return {"detections": items, "cameraLive": True}


@app.post("/detection-overrides")
def post_detection_overrides(body: dict, authorization: str | None = Header(default=None)):
    """One-shot manual correction from the live-view click-to-include/exclude canvas overlay.
    Body: { excludeRegions?: [{x,y,w,h}], includeRegions?: [{x,y,w,h}] } in native frame pixel
    coordinates. Replaces whatever the previous override was (not additive) and resets the TTL --
    an operator drawing a second correction almost always means "here's the current full set of
    corrections," not "add to whatever was there before."
    """
    _check_auth(authorization)
    _detection_overrides["exclude"] = body.get("excludeRegions", []) or []
    _detection_overrides["include"] = body.get("includeRegions", []) or []
    _detection_overrides["at"] = time.monotonic()
    return {
        "excludeRegions": _detection_overrides["exclude"],
        "includeRegions": _detection_overrides["include"],
        "expiresInSec": DETECTION_OVERRIDE_TTL_SEC,
    }


@app.get("/permanent-exclude-regions")
def get_permanent_exclude_regions(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    return {"regions": load_device_config().permanent_exclude_regions}


@app.put("/permanent-exclude-regions")
def put_permanent_exclude_regions(body: dict, authorization: str | None = Header(default=None)):
    """Saves the full set of persistent exclude zones (e.g. a fixed shadow or shelf edge) --
    replaces the whole list, same "operator sends the current full set" contract as
    /detection-overrides. Body: { regions: [{x,y,w,h}] }."""
    _check_auth(authorization)
    regions = body.get("regions", [])
    if not isinstance(regions, list):
        raise HTTPException(status_code=400, detail="regions must be a list.")
    config = set_permanent_exclude_regions(regions)
    return {"regions": config.permanent_exclude_regions}


@app.put("/config")
def update_config(
    body: dict,
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    height_mm = body.get("cameraHeightMm")
    if height_mm is None or float(height_mm) <= 0:
        raise HTTPException(status_code=400, detail="cameraHeightMm must be a positive number.")
    config = set_camera_height_mm(float(height_mm))
    return {"cameraHeightMm": config.camera_height_mm}


@app.get("/config/auto-absorb")
def get_auto_absorb_config(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    config = load_device_config()
    return {"enabled": config.auto_absorb_enabled, "timeoutSec": config.auto_absorb_timeout_sec}


@app.put("/config/auto-absorb")
def put_auto_absorb_config(body: dict, authorization: str | None = Header(default=None)):
    """Enable/disable + set the timeout for auto-absorbing static clutter into the background
    reference -- see _auto_absorb_loop. Body: { enabled: bool, timeoutSec: number }."""
    _check_auth(authorization)
    enabled = bool(body.get("enabled", True))
    try:
        timeout_sec = float(body.get("timeoutSec", 120.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timeoutSec: {exc}")
    if timeout_sec <= 0:
        raise HTTPException(status_code=400, detail="timeoutSec must be positive.")
    config = set_auto_absorb_config(enabled, timeout_sec)
    return {"enabled": config.auto_absorb_enabled, "timeoutSec": config.auto_absorb_timeout_sec}


@app.get("/stream/rgb")
def stream_rgb(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    adapter = get_aurora_adapter()

    def get_frame() -> bytes:
        frame = adapter.get_rgb_frame()
        intrinsics = adapter.get_intrinsics()

        # Continuous live border around every currently-detected item -- runs every served
        # preview frame (not just after an explicit /capture), so multiple items on the table
        # are each outlined in real time.
        try:
            depth_frame = adapter.get_depth_frame()
            exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
            live_detections = detect_live(depth_frame, intrinsics, exclude_mask=exclude_mask, include_mask=include_mask)
            frame = _draw_live_detections_overlay(frame, live_detections, adapter)
        except Exception:
            pass

        # The brighter yellow single-item highlight from the last explicit /capture, shown for a
        # few seconds afterward so the operator can visually confirm what was just measured.
        if _last_bbox_overlay is not None and (time.monotonic() - _last_bbox_overlay["at"]) < 5.0:
            try:
                frame = _draw_bbox_overlay(frame, _last_bbox_overlay["corners_mm"], adapter)
            except Exception:
                pass
        return _rgb_to_jpeg_bytes(frame)

    def is_stale() -> bool:
        age = adapter.get_frame_age_sec()
        return age is None or age > STALE_FRAME_THRESHOLD_SEC

    return StreamingResponse(
        _mjpeg_stream(get_frame, is_stale), media_type="multipart/x-mixed-replace; boundary=dimensionerframe"
    )


@app.get("/stream/depth")
def stream_depth(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
    mode: str = Query(default="colorized"),
):
    _check_auth(authorization, token)
    if mode not in DEPTH_VIEW_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {DEPTH_VIEW_MODES}")
    adapter = get_aurora_adapter()

    def get_frame() -> bytes:
        return _depth_to_visual_jpeg_bytes(adapter.get_depth_frame(), mode=mode)

    def is_stale() -> bool:
        age = adapter.get_frame_age_sec()
        return age is None or age > STALE_FRAME_THRESHOLD_SEC

    return StreamingResponse(
        _mjpeg_stream(get_frame, is_stale), media_type="multipart/x-mixed-replace; boundary=dimensionerframe"
    )


@app.post("/calibrate")
def calibrate(
    mode: str = Query(default="height"),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    adapter = get_aurora_adapter()
    now_iso = datetime.now(timezone.utc).isoformat()

    if mode == "height":
        config = load_device_config()
        if not config.camera_height_mm:
            raise HTTPException(
                status_code=400,
                detail="Set a camera height first (PUT /config) before Quick Calibrate.",
            )
        depth_frame = adapter.get_depth_frame()
        background = synthesize_flat_background(depth_frame.shape, config.camera_height_mm)
        save_calibration(background)
        set_calibration_state("height", now_iso)
        _reset_static_tracker()
        return {"calibrated": True, "mode": "height", "capturedAt": now_iso}

    if mode == "live":
        depth_frame = adapter.get_depth_frame()
        save_calibration(depth_frame)
        set_calibration_state("live", now_iso)
        _reset_static_tracker()
        return {"calibrated": True, "mode": "live", "capturedAt": now_iso}

    raise HTTPException(status_code=400, detail="mode must be 'height' or 'live'.")


def _accuracy_check_measure(body: dict) -> tuple[MeasurementResult, list[float], list[float]]:
    """Shared logic for /calibrate/accuracy-check and its /apply sibling: measure the current
    frame (through active overrides + optional shape correction, same pipeline a real capture
    uses) and pair it against the operator-entered true dimensions.

    Scopes to exactly one real detection via _scope_exclude_to_single_detection, same as /capture
    -- this used to run measure() unscoped against the raw whole-frame mask, which is the actual
    root cause of a real, reproducible corruption bug: anything else in frame (a shadow, a hand,
    leftover clutter) silently merged into "the reference object," and because the resulting
    system_scale_factor is GLOBAL and multiplicative, one bad Accuracy Check poisoned every future
    measurement of every future object system-wide. Confirmed live: a stored factor of
    ~0.33/0.55/0.97 backed out to an unscoped "measured" length of ~26in against an 8.75in
    reference card -- a ~3x inflation consistent with a second object merging into the mask, not
    gradual drift.

    Both measured and true dimensions are sorted descending before pairing (largest-to-largest,
    same convention measure() itself uses for L/W/H) rather than trusting the operator's
    length/width/height labels to match the camera's arbitrary rotation of the reference object on
    the table -- an operator has no way to know which axis the camera will call "length".
    """
    try:
        true_length = float(body["trueLength"])
        true_width = float(body["trueWidth"])
        true_height = float(body["trueHeight"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid reference dimensions: {exc}")

    adapter = get_aurora_adapter()
    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        raise HTTPException(status_code=503, detail="Camera feed is stale -- device may be disconnected.")

    depth_frame = adapter.get_depth_frame()
    intrinsics = adapter.get_intrinsics()
    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
    # allow_selection_fallback=False: the Calibrate tab has no click-to-select affordance of its
    # own, so a selection made earlier on the unrelated Measure tab must never be silently reused
    # here -- see _scope_exclude_to_single_detection's docstring for the cross-tab hijack this
    # closes (an audit-confirmed bug distinct from the unscoped-merge bug this function's scoping
    # already fixes).
    exclude_mask = _scope_exclude_to_single_detection(
        depth_frame, intrinsics, exclude_mask, include_mask, allow_selection_fallback=False
    )
    shape_type = body.get("shapeType") or None
    try:
        result = measure(depth_frame, intrinsics, shape_type=shape_type, exclude_mask=exclude_mask, include_mask=include_mask)
    except NotCalibratedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NoObjectDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    measured_sorted = sorted([result.length_in, result.width_in, result.height_in], reverse=True)
    true_sorted = sorted([true_length, true_width, true_height], reverse=True)
    return result, measured_sorted, true_sorted


@app.post("/calibrate/accuracy-check")
def accuracy_check(body: dict, authorization: str | None = Header(default=None)):
    """Report-only: measures the current frame and compares it against a certified-true-size
    reference object's dimensions, WITHOUT changing any stored calibration. Body:
    { trueLength, trueWidth, trueHeight, shapeType? } (inches, any axis order)."""
    _check_auth(authorization)
    _, measured_sorted, true_sorted = _accuracy_check_measure(body)
    deltas = [round(m - t, 3) for m, t in zip(measured_sorted, true_sorted)]
    percent_errors = [round((d / t) * 100, 2) if t else 0.0 for d, t in zip(deltas, true_sorted)]
    return {
        "measured": measured_sorted,
        "true": true_sorted,
        "deltaIn": deltas,
        "percentError": percent_errors,
    }


def _compound_scale_factor(existing: Optional[dict], measured_sorted: list[float], true_sorted: list[float]) -> dict:
    """Computes the NEW system_scale_factor from a true-vs-measured pair, correctly compounding
    on top of `existing` (whatever scale factor the caller already has in hand) -- NOT just
    true/measured in isolation, and NOT re-reading device_config itself (see
    _apply_compound_scale_factor, the only caller, for why that read must happen INSIDE the same
    lock as the eventual write rather than as a separate earlier call).

    `measured_sorted` is the value AFTER any already-applied scale factor (that's what measure()/
    detect_live() actually report), so a naive true/measured would silently discard the existing
    correction rather than refine it: e.g. an old scale of 1.5x turns a 10in raw item into a
    measured 15in; correcting that to a true 12in with the naive formula computes 12/15=0.8x,
    which applied to the RAW 10in gives 8in -- not the requested 12in. The fix multiplies the
    correction ratio onto the EXISTING factor (old_scale * (true/measured_with_old_scale)) so
    repeated corrections keep converging toward true size instead of overshooting. Confirmed by
    direct calculation before writing this: buggy formula on a second correction produced 8in
    against a requested 12in; this compounding formula produces exactly 12in.
    """
    existing = existing or {"length": 1.0, "width": 1.0, "height": 1.0}
    existing_sorted = [existing.get("length", 1.0), existing.get("width", 1.0), existing.get("height", 1.0)]
    scale_sorted = [
        round(old * (t / m), 4) for old, t, m in zip(existing_sorted, true_sorted, measured_sorted)
    ]
    # measured_sorted is length>=width>=height by construction (see _accuracy_check_measure), so
    # the sorted scale factors map directly onto the length/width/height axes measure() sorts
    # into that same order every time.
    return {"length": scale_sorted[0], "width": scale_sorted[1], "height": scale_sorted[2]}


def _apply_compound_scale_factor(measured_sorted: list[float], true_sorted: list[float]) -> dict:
    """Reads the CURRENT system_scale_factor and writes the newly-compounded one as a single
    atomic operation (one device_config.py lock acquisition covering both), via
    update_device_config's mutator callback -- not two separate load_device_config() /
    set_system_scale_factor() calls.

    This closes a real lost-update race an adversarial audit confirmed against the earlier
    version of this code: that version called load_device_config() once (inside the old
    _compound_scale_factor) to read the existing factor, THEN set_system_scale_factor() again
    later, which did its OWN independent load_device_config() + save. Two concurrent callers (two
    browser tabs both correcting near-simultaneously, or an Accuracy Check apply overlapping a
    click-to-correct) could both read the same stale "existing" factor, compute two DIFFERENT
    compounded results, and whichever save landed last would silently overwrite the other's
    change -- the operator would see a 200 response with an appliedScaleFactor that was never
    durably the one actually stored. Routing the read+compute+write through one
    update_device_config() call means the mutator sees the truly-current on-disk state, and no
    concurrent writer's change can vanish in the gap between two separate calls.
    """
    result_holder: dict = {}

    def mutate(c: DeviceConfig) -> None:
        new_factor = _compound_scale_factor(c.system_scale_factor, measured_sorted, true_sorted)
        c.system_scale_factor = new_factor
        result_holder["scale_factor"] = new_factor

    update_device_config(mutate)
    return result_holder["scale_factor"]


@app.post("/calibrate/accuracy-check/apply")
def accuracy_check_apply(body: dict, authorization: str | None = Header(default=None)):
    """Computes and stores a multiplicative per-axis system_scale_factor (compounded on top of
    whatever's already stored -- see _compound_scale_factor) from a certified reference object --
    the ground-truth accuracy calibration step. Distinct from shape_profiles.py's additive
    per-shape offset (see measure.py's _apply_system_scale_factor); this is the root-cause scale
    correction, applied as the final step of every future measure()/detect_live() call.

    Body accepts an optional `measured: [l, w, h]` -- the EXACT sorted-descending values a prior
    POST /calibrate/accuracy-check call already returned to the operator. When present, this
    commits THOSE values verbatim instead of taking a brand-new live measurement, closing a real
    "review vs. commit" gap: without this, Check and Apply were two independent point-in-time
    measurements of a live scene, so what an operator reviewed on screen was never guaranteed to
    be what actually got stored a few seconds later (the object could shift, a shadow could move,
    auto-absorb could nibble a pixel). `measured` is optional (not required) so a client that
    skips the Check step can still Apply directly off one fresh scoped measurement -- but the
    normal, safe flow now passes it through unchanged from Check's own response.
    """
    _check_auth(authorization)
    reviewed_measured = body.get("measured")
    if reviewed_measured is not None:
        try:
            measured_sorted = sorted([float(v) for v in reviewed_measured], reverse=True)
            true_sorted = sorted(
                [float(body["trueLength"]), float(body["trueWidth"]), float(body["trueHeight"])], reverse=True
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid correction payload: {exc}")
    else:
        _, measured_sorted, true_sorted = _accuracy_check_measure(body)
    if any(m <= 0 for m in measured_sorted):
        raise HTTPException(status_code=422, detail="Measured dimensions must be positive to compute a scale factor.")
    applied_scale_factor = _apply_compound_scale_factor(measured_sorted, true_sorted)
    return {
        "measured": measured_sorted,
        "true": true_sorted,
        "appliedScaleFactor": applied_scale_factor,
    }


@app.post("/detections/select/correct")
def correct_selected_detection(body: dict, authorization: str | None = Header(default=None)):
    """Click-item-then-correct, the fast path to the SAME ground-truth calibration mechanism as
    /calibrate/accuracy-check/apply -- true L/W/H in, system_scale_factor out, compounded on top
    of whatever was already stored (see _apply_compound_scale_factor). The difference is WHAT
    gets measured: instead of requiring a trip to the Calibrate tab to re-measure a fresh
    reference object, this operates on whichever detection the operator already selected via
    POST /detections/select (click-to-select in the live view).

    This closes the "two different numbers" loop by design, not just by coincidence: because
    Part 1 made detect_live() and measure() apply system_scale_factor identically, correcting it
    HERE immediately changes what the live view, /detections, AND /capture all report for every
    future item of any shape -- there is no separate "before" and "after" number for the operator
    to compare, since every read from this point on already reflects the correction. Body:
    { trueLength, trueWidth, trueHeight, measured?: [l, w, h] } (inches, any axis order -- same
    convention as accuracy-check, sorted descending before pairing so the operator doesn't need
    to know which physical axis the camera happened to call "length").

    `measured` is the SAME review-vs-commit passthrough accuracy_check_apply already has: when
    present, this is the EXACT selectedDetection snapshot the operator saw on screen (frozen at
    the moment they clicked "Correct", from POST /detections/select's own response) and this
    commits it verbatim, instead of re-measuring the selected item live at commit time. Without
    this, an adversarial audit confirmed a real gap: the operator could sit reading the "Selected:
    LxWxH" line and typing a true size for several seconds (well within the 300s selection TTL)
    before clicking "Recalibrate" -- if the item shifted, was nudged, or the live number drifted
    even slightly in that window, the OLD code re-ran detect_live() fresh at commit time and
    silently computed the scale factor against numbers the operator never actually reviewed.
    `measured` is optional so an older client (or a fresh /detections/select immediately followed
    by a correct, with no intervening delay) still works via one fresh live re-measurement.
    """
    _check_auth(authorization)
    try:
        true_length = float(body["trueLength"])
        true_width = float(body["trueWidth"])
        true_height = float(body["trueHeight"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid correction payload: {exc}")

    reviewed_measured = body.get("measured")
    if reviewed_measured is not None:
        try:
            measured_sorted = sorted([float(v) for v in reviewed_measured], reverse=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid measured payload: {exc}")
    else:
        selected_id = _selected_detection_id()
        if selected_id is None:
            raise HTTPException(status_code=409, detail="No item currently selected -- click one in the preview first.")

        adapter = get_aurora_adapter()
        age = adapter.get_frame_age_sec()
        if age is None or age > STALE_FRAME_THRESHOLD_SEC:
            raise HTTPException(status_code=503, detail="Camera feed is stale -- device may be disconnected.")

        depth_frame = adapter.get_depth_frame()
        intrinsics = adapter.get_intrinsics()
        exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
        live_detections = detect_live(depth_frame, intrinsics, exclude_mask=exclude_mask, include_mask=include_mask)
        selected = next((d for d in live_detections if d.id == selected_id), None)
        if selected is None:
            _selected_detection["id"] = None
            raise HTTPException(status_code=409, detail="Selected item is no longer visible; click it again.")
        measured_sorted = sorted([selected.length_in, selected.width_in, selected.height_in], reverse=True)

    true_sorted = sorted([true_length, true_width, true_height], reverse=True)
    if any(m <= 0 for m in measured_sorted):
        raise HTTPException(status_code=422, detail="Measured dimensions must be positive to compute a scale factor.")

    applied_scale_factor = _apply_compound_scale_factor(measured_sorted, true_sorted)
    return {
        "measured": measured_sorted,
        "true": true_sorted,
        "appliedScaleFactor": applied_scale_factor,
    }


# ── Training Mode: guided, open-ended calibration loop ───────────────────────────────────────
# Replaces the two ad-hoc correction mechanisms above (per-shape "Save Correction" and
# click-to-select "Recalibrate") -- both are retired from every shipped UI once this ships. See
# training.py's module docstring for the full root-cause writeup of why those two mechanisms
# conflicted and why sequential single-item calibration never converged. The short version: an
# additive shape offset computed from an already-scaled number gets silently re-multiplied on
# every future read, and length/width/height are per-item RANK labels (not fixed camera axes),
# so calibrating against different items one at a time can map "length" onto a different
# physical sensor axis each time and never settle. This fixes both by fitting ONE calibration
# from the WHOLE accumulated sample set at once (median-based, robust to a rank-crossed outlier)
# instead of compounding single-item corrections sequentially -- and, per explicit user design,
# there is no fixed "collect then verify" phase split: an operator scans an item, marks it
# accurate or not (entering the true size if not), and that becomes one more sample immediately
# available to the next Recalibrate; Recalibrate can be re-run as many times as wanted against
# the ever-growing sample set, and the operator decides when the numbers look good enough to
# Finish.


def _training_session_dict(session: Optional[TrainingSession]) -> dict:
    return {"session": asdict(session) if session is not None else None}


def _require_active_training_session() -> TrainingSession:
    session = load_training_session()
    if session is None or session.status != "active":
        raise HTTPException(status_code=404, detail="No active training session. Start one first.")
    return session


def _snapshot_shape_profiles() -> dict:
    return {p.shape_type: asdict(p) for p in list_shape_profiles()}


@app.post("/training/start")
def start_training(authorization: str | None = Header(default=None)):
    """Starts a new Training Mode session -- the ONLY supported way to calibrate this device
    going forward. 409s if a session is already active and recently touched; a session that's
    been idle past IDLE_ABANDON_SEC is treated as abandoned and silently rolled back (to ITS OWN
    initial snapshot, i.e. exactly how the device was before that abandoned session started)
    before the new one begins, so an abandoned session never leaves a half-applied calibration
    stuck in place. Snapshots the device's CURRENT scale factor + shape profiles so
    /training/cancel can restore this exact starting point regardless of how many times
    /training/recalibrate runs in between."""
    _check_auth(authorization)
    now_iso = datetime.now(timezone.utc).isoformat()

    existing = load_training_session()
    if existing is not None and existing.status == "active":
        last_activity = datetime.fromisoformat(existing.last_activity_at)
        idle_sec = (datetime.now(timezone.utc) - last_activity).total_seconds()
        if idle_sec <= IDLE_ABANDON_SEC:
            raise HTTPException(
                status_code=409,
                detail="A training session is already in progress. Finish or cancel it before starting a new one.",
            )
        _rollback_training_session(existing)

    initial_scale_factor = load_device_config().system_scale_factor
    initial_shape_profiles = _snapshot_shape_profiles()

    def mutate(_current: Optional[TrainingSession]) -> TrainingSession:
        return TrainingSession(
            id=new_session_id(),
            status="active",
            started_at=now_iso,
            last_activity_at=now_iso,
            initial_scale_factor=initial_scale_factor,
            initial_shape_profiles=initial_shape_profiles,
        )

    session = update_training_session(mutate)
    return {"sessionId": session.id, "status": session.status, "minSamples": MIN_TRAINING_SAMPLES}


@app.get("/training/session")
def get_training_session(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    return _training_session_dict(load_training_session())


@app.get("/training/legacy-check")
def check_legacy_calibration(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    """True if this device has pre-Training-Mode calibration data (a system_scale_factor or any
    shape-profile offsets from the old ad-hoc mechanisms) but has never completed a Training run.
    Drives the "legacy calibration data" banner in both UI surfaces -- see the training-mode
    plan's migration section: existing data is left in place as a fallback rather than auto-
    wiped, but operators should be nudged toward replacing it with a verified batch calibration."""
    _check_auth(authorization, token)
    session = load_training_session()
    if session is not None and session.status == "completed":
        return {"hasLegacyData": False}
    config = load_device_config()
    has_legacy_scale_factor = bool(config.system_scale_factor)
    has_legacy_shape_profiles = len(list_shape_profiles()) > 0
    return {"hasLegacyData": has_legacy_scale_factor or has_legacy_shape_profiles}


@app.post("/training/sample")
def submit_training_sample(body: dict, authorization: str | None = Header(default=None)):
    """Records one item's outcome into the session's ever-growing sample set. Scopes to a single
    real detection (same guarantee /capture already has, via _scope_exclude_to_single_detection),
    measures it, and records its RAW (pre-correction) extents alongside the ground truth for this
    sample. Body: { shapeType, verdict: 'accurate'|'not_accurate', trueLength?, trueWidth?,
    trueHeight? } -- true* fields are REQUIRED when verdict is 'not_accurate' (the operator's
    corrected ground truth) and IGNORED when verdict is 'accurate' (the system's own current
    fully-corrected reading becomes the ground truth for that sample, confirming it was right).

    raw_sorted is captured before ANY correction, so it stays valid no matter how many times
    /training/recalibrate has already run this session -- recalibrating never invalidates a
    previously-recorded sample, which is what makes the open-ended "recalibrate, then keep
    scanning, then recalibrate again" loop safe to repeat indefinitely.
    """
    _check_auth(authorization)
    session = _require_active_training_session()

    try:
        shape_type = str(body["shapeType"])
        verdict = str(body["verdict"])
        if verdict not in ("accurate", "not_accurate"):
            raise ValueError("verdict must be 'accurate' or 'not_accurate'")
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sample payload: {exc}")

    adapter = get_aurora_adapter()
    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        raise HTTPException(status_code=503, detail="Camera feed is stale -- device may be disconnected.")

    depth_frame = adapter.get_depth_frame()
    intrinsics = adapter.get_intrinsics()
    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
    exclude_mask = _scope_exclude_to_single_detection(depth_frame, intrinsics, exclude_mask, include_mask)

    try:
        result = measure(depth_frame, intrinsics, shape_type=shape_type, exclude_mask=exclude_mask, include_mask=include_mask)
    except NotCalibratedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NoObjectDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    measured_sorted = sorted([result.length_in, result.width_in, result.height_in], reverse=True)
    if verdict == "accurate":
        true_sorted = measured_sorted
    else:
        try:
            true_length = float(body["trueLength"])
            true_width = float(body["trueWidth"])
            true_height = float(body["trueHeight"])
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid true-size payload: {exc}")
        true_sorted = sorted([true_length, true_width, true_height], reverse=True)

    selected_id = _selected_detection_id()
    raw_sorted = sorted([result.raw_length_in, result.raw_width_in, result.raw_height_in], reverse=True)
    delta_in = [round(t - m, 3) for t, m in zip(true_sorted, measured_sorted)]
    now_iso = datetime.now(timezone.utc).isoformat()
    # Purely a tracking signal (see TrainingSample.shape_type_overridden's docstring) -- doesn't
    # affect measurement or correction in any way. shape_type here is what the operator picked in
    # the request body BEFORE this scan ran; result.suggested_shape_type is what the same scan's
    # depth geometry would have suggested (see measure.py's _suggest_shape_type).
    shape_type_overridden = shape_type != result.suggested_shape_type

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None or current.status != "active":
            raise HTTPException(status_code=409, detail="Training session is no longer active.")
        sample = TrainingSample(
            index=len(current.samples),
            shape_type=shape_type,
            verdict=verdict,
            true_sorted=true_sorted,
            measured_sorted=measured_sorted,
            raw_sorted=raw_sorted,
            delta_in=delta_in,
            detection_id=selected_id if selected_id is not None else -1,
            captured_at=now_iso,
            shape_type_overridden=shape_type_overridden,
        )
        current.samples.append(sample)
        current.last_activity_at = now_iso
        return current

    session = update_training_session(mutate)
    return {"session": asdict(session), "sample": asdict(session.samples[-1]), "suggestedShapeType": result.suggested_shape_type}


@app.delete("/training/samples/{sample_index}")
def delete_training_sample(sample_index: int, authorization: str | None = Header(default=None)):
    """Undo one bad sample without restarting the session."""
    _check_auth(authorization)
    now_iso = datetime.now(timezone.utc).isoformat()

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None or current.status != "active":
            raise HTTPException(status_code=409, detail="No active training session.")
        current.samples = [s for s in current.samples if s.index != sample_index]
        # Re-index sequentially so the on-screen sample list stays a clean 0..N-1 sequence.
        for i, s in enumerate(current.samples):
            s.index = i
        current.last_activity_at = now_iso
        return current

    session = update_training_session(mutate)
    return {"session": asdict(session)}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fit_recalibration(samples: list[TrainingSample]) -> tuple[dict, dict]:
    """Pure computation, extracted from the original single-endpoint /training/recalibrate so it
    can be reused by both that endpoint AND the auto-recalibrate-on-failed-round trigger in
    /training/verify (see _recalibrate_and_open_round). Fits ONE global system_scale_factor from
    the ENTIRE accumulated sample set (median of per-sample true/raw ratios, independently per
    rank -- robust to 1-2 bad/rank-crossed samples once len(samples) >= MIN_TRAINING_SAMPLES) and
    derives fresh per-shape offsets from that SAME new factor in the same step. This is a full
    RESET of system_scale_factor (not a compound onto whatever was previously stored) -- see the
    module's prior docstrings for why compounding onto a possibly-rank-crossed old factor has no
    sound basis here.

    Raises plain ValueError (not HTTPException) on a non-positive raw_sorted value -- callers
    choose how to react: the manual endpoint re-raises as a 422 the operator sees; the auto-trigger
    catches it broadly and silently falls back to leaving the round for a manual Recalibrate,
    since a successful verify action must never itself 500."""
    ratios = [[], [], []]  # per rank: length, width, height
    for sample in samples:
        for k in range(3):
            if sample.raw_sorted[k] <= 0:
                raise ValueError("A recorded sample has a non-positive raw measurement; cannot fit a scale factor.")
            ratios[k].append(sample.true_sorted[k] / sample.raw_sorted[k])

    new_scale_sorted = [round(_median(ratios[k]), 4) for k in range(3)]
    new_scale_factor = {"length": new_scale_sorted[0], "width": new_scale_sorted[1], "height": new_scale_sorted[2]}

    shape_types = sorted({s.shape_type for s in samples})
    offsets_by_shape: dict[str, list[float]] = {t: [[], [], []] for t in shape_types}
    for sample in samples:
        for k in range(3):
            scaled_raw = sample.raw_sorted[k] * new_scale_sorted[k]
            offsets_by_shape[sample.shape_type][k].append(sample.true_sorted[k] - scaled_raw)

    shape_offsets_result: dict[str, dict] = {}
    for shape_type, per_rank in offsets_by_shape.items():
        # MEDIAN, not mean -- a "verdict: accurate" sample self-confirms the live (pre-correction)
        # reading as its own ground truth, so it always has offset ~0 under the OLD calibration;
        # once a fresh scale factor is fit from the whole batch, that same sample's offset under
        # the NEW factor can be a large outlier relative to every sample that had a real typed
        # true size. A mean lets that one outlier drag the fitted offset far from what the actual
        # corrections agree on (confirmed live: one such sample pulled a length offset from
        # ~-0.03..-0.19in, per every other sample, to +0.171in) -- exactly the "recalibrates to the
        # wrong size" a user reported. Median is consistent with new_scale_sorted's OWN aggregation
        # above, which already uses median for the identical reason.
        length_offset = round(_median(per_rank[0]), 3)
        width_offset = round(_median(per_rank[1]), 3)
        height_offset = round(_median(per_rank[2]), 3)
        shape_offsets_result[shape_type] = {
            "lengthOffsetIn": length_offset,
            "widthOffsetIn": width_offset,
            "heightOffsetIn": height_offset,
            "sampleCount": len(per_rank[0]),
        }

    return new_scale_factor, shape_offsets_result


def _apply_recalibration(new_scale_factor: dict, shape_offsets_result: dict) -> None:
    """Device-state writes only -- set_system_scale_factor then replace_shape_profile per shape
    type, each acquiring/releasing its own lock (device_config.py's _CONFIG_LOCK, shape_profiles.py's
    _PROFILES_LOCK) independently, exactly as this endpoint always has. NEVER call this from inside
    a training.py update_training_session mutate closure -- doing so would nest _CONFIG_LOCK/
    _PROFILES_LOCK acquisition inside _SESSION_LOCK, a new invariant nothing else in this file
    enforces. Every caller of this function does its own separate update_training_session call
    AFTER this returns, matching this endpoint's own long-standing ordering (compute -> apply
    device writes -> THEN stamp session metadata)."""
    set_system_scale_factor(new_scale_factor)
    for shape_type, offsets in shape_offsets_result.items():
        replace_shape_profile(
            shape_type,
            offsets["sampleCount"],
            offsets["lengthOffsetIn"],
            offsets["widthOffsetIn"],
            offsets["heightOffsetIn"],
        )


def _select_verification_queue(samples: list[TrainingSample]) -> list[int]:
    """Bounds a verification round to the VERIFICATION_ROUND_SIZE MOST RECENTLY captured samples
    (highest .index = most recent, since index is assigned sequentially at append time in both
    /training/sample and /training/verify) -- NOT the whole accumulated sample set, which keeps
    growing every round. Fitting (_fit_recalibration) still uses every sample; only the physical
    re-scanning workload is bounded. Simple recency selection -- a shape type absent from the most
    recent VERIFICATION_ROUND_SIZE samples isn't re-verified that round even if its offset changed;
    accepted tradeoff for v1, not implementing stratify-by-shape-type."""
    ordered = sorted(samples, key=lambda s: s.index, reverse=True)
    return [s.index for s in ordered[:VERIFICATION_ROUND_SIZE]]


def _recalibrate_and_open_round(session: TrainingSession) -> dict:
    """Shared core of /training/recalibrate and the auto-recalibrate trigger in /training/verify:
    fits from ALL of session.samples, applies the device writes, and returns
    {"computed_result": {...}, "verification_queue": [...]}. Never touches training_session.json
    itself -- both callers do their own update_training_session stamping after this returns. Can
    raise ValueError (from _fit_recalibration) -- callers decide how to translate that."""
    new_scale_factor, shape_offsets_result = _fit_recalibration(session.samples)
    _apply_recalibration(new_scale_factor, shape_offsets_result)
    computed_result = {
        "appliedScaleFactor": new_scale_factor,
        "shapeOffsets": shape_offsets_result,
        "computedAt": datetime.now(timezone.utc).isoformat(),
        "sampleCount": len(session.samples),
    }
    return {"computed_result": computed_result, "verification_queue": _select_verification_queue(session.samples)}


@app.post("/training/recalibrate")
def recalibrate_training(authorization: str | None = Header(default=None)):
    """Fits + applies a fresh calibration from the ENTIRE accumulated sample set (see
    _fit_recalibration/_apply_recalibration), then opens a new VERIFICATION ROUND bounded to the
    VERIFICATION_ROUND_SIZE most recent samples (see _select_verification_queue) -- every item in
    that round must be re-scanned under the calibration that was just applied before Finish is
    allowed. Per explicit user requirement, Recalibrate is never "trust the math and move on" --
    it always demands the operator re-prove the items read correctly now, exactly the same
    accurate/not-accurate motion as the initial scans.

    Safe to call repeatedly as the operator keeps scanning more items: every sample's raw_sorted
    was captured pre-correction, so it's independent of whatever calibration happens to be active
    right now -- recalibrating never invalidates previously-recorded samples. Does NOT re-snapshot
    session.initial_* (those stay pinned to how the device was BEFORE this session started, for
    /training/cancel's full-session rollback). Unlike the auto-trigger in /training/verify, this
    endpoint surfaces a real failure to the operator who explicitly clicked it -- it never
    silently swallows an error."""
    _check_auth(authorization)
    session = _require_active_training_session()
    if len(session.samples) < MIN_TRAINING_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {MIN_TRAINING_SAMPLES} samples to calibrate (have {len(session.samples)}).",
        )

    try:
        result = _recalibrate_and_open_round(session)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    now_iso = datetime.now(timezone.utc).isoformat()

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None or current.status != "active":
            raise HTTPException(status_code=409, detail="No active training session.")
        current.computed_result = result["computed_result"]
        current.recalibration_count += 1
        current.round_number += 1
        current.verification_queue = result["verification_queue"]
        current.round_results = []
        current.last_round_accuracy_pct = None
        current.last_activity_at = now_iso
        return current

    session = update_training_session(mutate)
    return {
        "session": asdict(session),
        "appliedScaleFactor": result["computed_result"]["appliedScaleFactor"],
        "appliedShapeOffsets": result["computed_result"]["shapeOffsets"],
    }


@app.post("/training/verify/{sample_index}")
def verify_training_sample(sample_index: int, body: dict, authorization: str | None = Header(default=None)):
    """Re-scans one item from the current verification round (opened by /training/recalibrate)
    and records whether it now reads correctly. Body: { verdict: 'accurate'|'not_accurate' } --
    UNLIKE /training/sample, no true* fields: the true size for this item was already recorded
    on `original_sample` when it was first scanned, and doesn't change on a re-scan -- asking the
    operator to retype it here (the original behavior) was a real reported bug ("I already did
    that"). `verdict` stays a required, explicit human judgment call either way (this system's
    round accuracy is defined as the % of re-scans the OPERATOR marked accurate, never an
    automatic tolerance check -- see training.py's MIN_REQUIRED_ACCURACY_PCT docstring), it just
    no longer implies "type the ground truth in again."

    This ALSO appends a fresh TrainingSample to the session's ever-growing sample set (so a
    "not accurate" re-scan directly feeds the NEXT recalibration, same as an original scan would),
    using `original_sample.true_sorted` as that new sample's ground truth too -- NOT this scan's
    live reading, even when verdict is "accurate". Reusing the live reading as ground truth was
    right for the FIRST scan (the true size wasn't known yet, so confirming "accurate" meant
    "trust this reading"), but during verification the real true size is already on file; feeding
    a noisy fresh reading back in as if it were ground truth would inject drift into the next
    recalibration's fit instead of reinforcing it with the one number actually known to be
    correct.

    Removes `sample_index` from the round's pending verification_queue. Once the queue is empty,
    the round's accuracy (% of re-scans marked accurate) is computed and stored -- Finish checks
    this against MIN_REQUIRED_ACCURACY_PCT (see training.py). That round's outcome is ALSO appended
    to `round_accuracy_history` (a permanent, append-only log) so the operator can see round-over-
    round trend, not just the latest number.

    Per explicit user requirement, a round closing BELOW MIN_REQUIRED_ACCURACY_PCT immediately
    triggers an automatic Recalibrate + opens the next round, with no manual click required -- see
    the auto-recalibrate block below, capped at MAX_CONSECUTIVE_AUTO_RECALIBRATIONS consecutive
    failures (training.py) so this can't loop forever against data that genuinely can't reach the
    target. The response's `autoRecalibrated`/`previousRoundAccuracy` fields tell the caller
    whether that just happened."""
    _check_auth(authorization)
    session = _require_active_training_session()
    if sample_index not in session.verification_queue:
        raise HTTPException(status_code=409, detail="That item is not pending verification in the current round.")
    original_sample = next((s for s in session.samples if s.index == sample_index), None)
    if original_sample is None:
        raise HTTPException(status_code=404, detail="No such sample index in this session.")

    try:
        verdict = str(body["verdict"])
        if verdict not in ("accurate", "not_accurate"):
            raise ValueError("verdict must be 'accurate' or 'not_accurate'")
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid verify payload: {exc}")

    adapter = get_aurora_adapter()
    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        raise HTTPException(status_code=503, detail="Camera feed is stale -- device may be disconnected.")

    depth_frame = adapter.get_depth_frame()
    intrinsics = adapter.get_intrinsics()
    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)
    exclude_mask = _scope_exclude_to_single_detection(depth_frame, intrinsics, exclude_mask, include_mask)

    try:
        result = measure(depth_frame, intrinsics, shape_type=original_sample.shape_type, exclude_mask=exclude_mask, include_mask=include_mask)
    except NotCalibratedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NoObjectDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    measured_sorted = sorted([result.length_in, result.width_in, result.height_in], reverse=True)
    true_sorted = original_sample.true_sorted

    selected_id = _selected_detection_id()
    raw_sorted = sorted([result.raw_length_in, result.raw_width_in, result.raw_height_in], reverse=True)
    delta_in = [round(t - m, 3) for t, m in zip(true_sorted, measured_sorted)]
    now_iso = datetime.now(timezone.utc).isoformat()
    round_result = {
        "sampleIndex": sample_index,
        "shapeType": original_sample.shape_type,
        "verdict": verdict,
        "measuredSorted": measured_sorted,
        "trueSorted": true_sorted,
        "deltaIn": delta_in,
    }

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None or current.status != "active":
            raise HTTPException(status_code=409, detail="No active training session.")
        if sample_index not in current.verification_queue:
            raise HTTPException(status_code=409, detail="That item is not pending verification in the current round.")
        # Feed this re-scan back in as a fresh sample too, using the SAME true_sorted as the
        # original scan (see this endpoint's docstring for why re-deriving it from the live
        # reading would be wrong here) -- a fresh raw_sorted/measured_sorted paired with that
        # already-known-correct true size is exactly as valuable to the NEXT recalibration's fit
        # as an original scan was.
        new_sample = TrainingSample(
            index=len(current.samples),
            shape_type=original_sample.shape_type,
            verdict=verdict,
            true_sorted=true_sorted,
            measured_sorted=measured_sorted,
            raw_sorted=raw_sorted,
            delta_in=delta_in,
            detection_id=selected_id if selected_id is not None else -1,
            captured_at=now_iso,
        )
        current.samples.append(new_sample)
        current.verification_queue = [i for i in current.verification_queue if i != sample_index]
        current.round_results = [r for r in current.round_results if r["sampleIndex"] != sample_index]
        current.round_results.append(round_result)
        if not current.verification_queue:
            accurate_count = sum(1 for r in current.round_results if r["verdict"] == "accurate")
            current.last_round_accuracy_pct = round(100.0 * accurate_count / len(current.round_results), 1) if current.round_results else 0.0
            # Permanent, append-only trend log -- see training.py's round_accuracy_history
            # docstring for why this is separate from last_round_accuracy_pct (overwritten every
            # round) and round_results (reset every recalibrate). Recorded unconditionally on every
            # round close, whether this round passed, failed, or is about to be auto-recalibrated
            # below -- a failed auto-recalibration attempt must never lose this round's entry.
            current.round_accuracy_history.append({
                "round_number": current.round_number,
                "accuracy_pct": current.last_round_accuracy_pct,
                "sample_count": len(current.round_results),
                "computed_at": now_iso,
            })
            if current.last_round_accuracy_pct >= MIN_REQUIRED_ACCURACY_PCT:
                # A PASSING round resets the consecutive-failure counter -- see
                # MAX_CONSECUTIVE_AUTO_RECALIBRATIONS's docstring for why this is the exact
                # condition that re-arms the auto-trigger after it was previously capped out.
                current.consecutive_auto_recalibrate_failures = 0
        current.last_activity_at = now_iso
        return current

    session = update_training_session(mutate)

    # Auto-recalibrate: per explicit user requirement, a round closing below target immediately
    # refits and opens the next round -- the operator should never have to notice a failed round
    # and remember to click Recalibrate themselves. Deliberately a SECOND, separate
    # update_training_session call performed AFTER the one above returns (never nested inside its
    # mutate closure) -- see _apply_recalibration's docstring for why nesting would introduce a new
    # lock-ordering invariant this codebase doesn't otherwise need.
    auto_recalibrated = False
    previous_round_accuracy = session.last_round_accuracy_pct
    if (
        session.status == "active"
        and not session.verification_queue
        and session.last_round_accuracy_pct is not None
        and session.last_round_accuracy_pct < MIN_REQUIRED_ACCURACY_PCT
        and session.consecutive_auto_recalibrate_failures < MAX_CONSECUTIVE_AUTO_RECALIBRATIONS
    ):
        try:
            auto_result = _recalibrate_and_open_round(session)
        except Exception as exc:
            # Broad + swallowed on purpose: a successful verify action must never turn into a 500
            # because the auto-trigger hit an edge case (e.g. a corrupted raw_sorted value). Falls
            # back to today's behavior -- the round stays closed-but-not-recalibrated, and the
            # operator can still click Recalibrate manually.
            print(f"[training] auto-recalibrate after round {session.round_number} failed, leaving for manual Recalibrate: {exc}")
        else:
            now_iso2 = datetime.now(timezone.utc).isoformat()

            def mutate2(current: Optional[TrainingSession]) -> TrainingSession:
                if current is None or current.status != "active":
                    return current  # session ended/cancelled in the meantime -- do nothing
                current.computed_result = auto_result["computed_result"]
                current.recalibration_count += 1
                current.round_number += 1
                current.verification_queue = auto_result["verification_queue"]
                current.round_results = []
                current.last_round_accuracy_pct = None
                current.consecutive_auto_recalibrate_failures += 1
                current.last_activity_at = now_iso2
                return current

            session = update_training_session(mutate2)
            auto_recalibrated = True

    return {
        "session": asdict(session),
        "result": round_result,
        "autoRecalibrated": auto_recalibrated,
        "previousRoundAccuracy": previous_round_accuracy,
    }


@app.post("/training/finish")
def finish_training(authorization: str | None = Header(default=None)):
    """Locks in the session's current calibration. Requires: (1) at least one
    /training/recalibrate to have run, (2) that recalibration's verification round to be fully
    complete (verification_queue empty -- every item re-scanned), and (3) that round's accuracy
    to meet MIN_REQUIRED_ACCURACY_PCT (see training.py) -- per explicit user requirement, this
    system must prove itself 90% accurate on a real re-verification pass before an operator is
    allowed to consider it done, not just trust the fitted math. Idempotent if already completed.
    """
    _check_auth(authorization)
    now_iso = datetime.now(timezone.utc).isoformat()

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None:
            raise HTTPException(status_code=404, detail="No training session in progress.")
        if current.status == "completed":
            return current
        if current.status != "active":
            raise HTTPException(status_code=409, detail=f"Training session is '{current.status}', cannot finish.")
        if current.computed_result is None:
            raise HTTPException(status_code=422, detail="Recalibrate at least once before finishing.")
        if current.verification_queue:
            raise HTTPException(
                status_code=422,
                detail=f"Re-verify all {len(current.verification_queue)} remaining item(s) from this round before finishing.",
            )
        if current.last_round_accuracy_pct is None or current.last_round_accuracy_pct < MIN_REQUIRED_ACCURACY_PCT:
            raise HTTPException(
                status_code=422,
                detail=f"This round's verified accuracy is {current.last_round_accuracy_pct or 0}% -- needs to reach "
                f"{MIN_REQUIRED_ACCURACY_PCT}% before finishing. Recalibrate and re-verify again.",
            )
        current.status = "completed"
        current.completed_at = now_iso
        current.last_activity_at = now_iso
        return current

    session = update_training_session(mutate)
    return {"session": asdict(session)}


def _rollback_training_session(session: TrainingSession) -> None:
    """Restores system_scale_factor + shape profiles to EXACTLY how they were before this
    session started (session.initial_*), regardless of how many times /training/recalibrate ran
    in between -- a full-session undo, not an undo of just the most recent recalibration."""
    set_system_scale_factor(session.initial_scale_factor)
    # Shape types touched by this session that DIDN'T exist beforehand must be deleted outright;
    # types that existed get their exact prior profile restored.
    touched_types = {s.shape_type for s in session.samples}
    for shape_type in touched_types:
        previous = session.initial_shape_profiles.get(shape_type)
        if previous is None:
            reset_shape_profile(shape_type)
        else:
            replace_shape_profile(
                shape_type,
                previous["correction_count"],
                previous["length_offset_in"],
                previous["width_offset_in"],
                previous["height_offset_in"],
            )


@app.post("/training/cancel")
def cancel_training(authorization: str | None = Header(default=None)):
    """Rolls back to EXACTLY how this device was before this session started (see
    _rollback_training_session) -- regardless of how many times /training/recalibrate ran in
    between. If /training/recalibrate never ran, this is a no-op restore (nothing changed) plus
    marking the session cancelled."""
    _check_auth(authorization)
    session = load_training_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No training session in progress.")

    _rollback_training_session(session)

    def mutate(current: Optional[TrainingSession]) -> TrainingSession:
        if current is None:
            raise HTTPException(status_code=404, detail="Training session disappeared mid-cancel.")
        current.status = "cancelled"
        current.last_activity_at = datetime.now(timezone.utc).isoformat()
        return current

    session = update_training_session(mutate)
    return {"session": asdict(session)}


def _serialize_measurement(result: MeasurementResult, product_id: str | None) -> dict:
    payload = {
        "length": result.length_in,
        "width": result.width_in,
        "height": result.height_in,
        "unit": "in",
        "cubicFeet": result.cubic_feet,
        "classification": result.classification,
        "confidence": result.confidence,
    }
    if product_id:
        product = get_product(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail=f"Unknown productId '{product_id}'.")
        measured = sorted([result.length_in, result.width_in, result.height_in], reverse=True)
        expected = sorted(
            [product.expected_length_in, product.expected_width_in, product.expected_height_in],
            reverse=True,
        )
        deltas = [round(m - e, 2) for m, e in zip(measured, expected)]
        within_tolerance = all(abs(d) <= product.tolerance_in for d in deltas)
        payload["expected"] = {
            "productId": product.id,
            "name": product.name,
            "length": product.expected_length_in,
            "width": product.expected_width_in,
            "height": product.expected_height_in,
            "toleranceIn": product.tolerance_in,
        }
        payload["deltaIn"] = {"length": deltas[0], "width": deltas[1], "height": deltas[2]}
        payload["withinTolerance"] = within_tolerance
    return payload


@app.post("/capture")
def capture(
    productId: str | None = Query(default=None),
    shapeType: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    global _last_bbox_overlay
    _check_auth(authorization)
    adapter = get_aurora_adapter()

    age = adapter.get_frame_age_sec()
    if age is None or age > STALE_FRAME_THRESHOLD_SEC:
        raise HTTPException(
            status_code=503,
            detail=f"Camera feed is stale ({age if age is not None else 'never received'}s) -- device may be disconnected.",
        )

    depth_frame = adapter.get_depth_frame()
    rgb_frame = adapter.get_rgb_frame()
    intrinsics = adapter.get_intrinsics()

    exclude_mask, include_mask = _active_override_masks(depth_frame.shape)

    # Scope capture to the real, blob-filtered detection(s) detect_live() already found -- NEVER
    # the raw whole-frame foreground mask. See _scope_exclude_to_single_detection's docstring.
    exclude_mask = _scope_exclude_to_single_detection(depth_frame, intrinsics, exclude_mask, include_mask)

    try:
        result = measure(depth_frame, intrinsics, shape_type=shapeType, exclude_mask=exclude_mask, include_mask=include_mask)
    except NotCalibratedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except NoObjectDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _last_bbox_overlay = {"corners_mm": result.corners_mm, "at": time.monotonic()}
    annotated_rgb = _draw_bbox_overlay(rgb_frame, result.corners_mm, adapter)

    payload = _serialize_measurement(result, productId)
    payload["shapeType"] = shapeType
    payload["imageBase64"] = _rgb_to_jpeg_base64(annotated_rgb)
    payload["capturedAt"] = datetime.now(timezone.utc).isoformat()

    log_entry = add_entry(
        captured_at=payload["capturedAt"],
        measured_length=result.length_in,
        measured_width=result.width_in,
        measured_height=result.height_in,
        measured_cubic_feet=result.cubic_feet,
        classification=result.classification,
        confidence=result.confidence,
        shape_type=shapeType,
        image_base64=payload["imageBase64"],
    )
    payload["logEntryId"] = log_entry.id
    return payload


@app.get("/shape-profiles")
def get_shape_profiles(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    profiles = {p.shape_type: p.__dict__ for p in list_shape_profiles()}
    return {
        "builtinShapeTypes": BUILTIN_SHAPE_TYPES,
        "profiles": profiles,
    }


@app.post("/shape-profiles/correct")
def correct_shape_profile(body: dict, authorization: str | None = Header(default=None)):
    """Records one correction into a shape type's running offset -- the actual training step.
    Body: { shapeType, measuredLength, measuredWidth, measuredHeight, correctedLength,
            correctedWidth, correctedHeight }.
    """
    _check_auth(authorization)
    try:
        shape_type = str(body["shapeType"]).strip()
        if not shape_type:
            raise ValueError("shapeType must not be empty")
        profile = record_correction(
            shape_type=shape_type,
            measured_length_in=float(body["measuredLength"]),
            measured_width_in=float(body["measuredWidth"]),
            measured_height_in=float(body["measuredHeight"]),
            corrected_length_in=float(body["correctedLength"]),
            corrected_width_in=float(body["correctedWidth"]),
            corrected_height_in=float(body["correctedHeight"]),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid correction payload: {exc}")
    return {"profile": profile.__dict__}


@app.delete("/shape-profiles/{shape_type}")
def delete_shape_profile(shape_type: str, authorization: str | None = Header(default=None)):
    """Deletes a shape type's learned correction offset entirely -- e.g. a bad correction (typo,
    wrong units, wrong shape selected) skewed the running mean far enough that it's now making
    every future measurement of that shape WORSE, not better. Starts that shape fresh."""
    _check_auth(authorization)
    reset_shape_profile(shape_type)
    return {"deleted": True}


@app.get("/local-captures")
def get_local_captures(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    """Scan Log data for the standalone Pi /viewer -- most-recent-first, image included (small
    thumbnails would be nicer at scale, but MAX_ENTRIES=100 keeps this file small enough that
    it doesn't matter for a single-device local log)."""
    _check_auth(authorization, token)
    entries = list_entries()
    return {"captures": [e.__dict__ for e in reversed(entries)]}


@app.post("/local-captures/{entry_id}/correct")
def correct_local_capture(
    entry_id: str,
    body: dict,
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    try:
        corrected_length = float(body["length"])
        corrected_width = float(body["width"])
        corrected_height = float(body["height"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid correction payload: {exc}")

    # Corrects only this Pi-local audit record now -- no longer feeds shape_profiles.py's
    # running offset. That feedback loop is exclusively driven by Training Mode (see
    # /training/compute), which fits shape offsets from a whole batch at once instead of one-off
    # corrections that used to conflict with the global system_scale_factor. See training.py's
    # module docstring for the full root-cause writeup.
    entry = correct_entry(entry_id, corrected_length, corrected_width, corrected_height)
    if entry is None:
        raise HTTPException(status_code=404, detail="Capture log entry not found.")
    return {"capture": entry.__dict__}


@app.get("/products")
def get_products(authorization: str | None = Header(default=None), token: str | None = Query(default=None)):
    _check_auth(authorization, token)
    return {"products": [p.__dict__ for p in list_products()]}


@app.post("/products")
def post_product(body: dict, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    try:
        product = create_product(
            name=body["name"],
            expected_length_in=float(body["expectedLengthIn"]),
            expected_width_in=float(body["expectedWidthIn"]),
            expected_height_in=float(body["expectedHeightIn"]),
            tolerance_in=float(body.get("toleranceIn", 0.5)),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid product payload: {exc}")
    return {"product": product.__dict__}


@app.delete("/products/{product_id}")
def delete_product_route(product_id: str, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    deleted = delete_product(product_id)
    return {"deleted": deleted}


@app.get("/viewer", response_class=HTMLResponse)
def viewer(token: str | None = Query(default=None)):
    if not token:
        return HTMLResponse("<h1>Missing ?token=... in the URL</h1>", status_code=400)
    return HTMLResponse(_VIEWER_HTML.replace("__TOKEN__", token))


_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dimensioner Live View</title>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #0b0b0d; color: #eee; display: flex; flex-direction: column; }
    .statusbar { display: flex; align-items: center; gap: 16px; background: #1a1a1e; padding: 10px 16px; font-size: 13px; flex-wrap: wrap; border-bottom: 1px solid #26262c; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .dot.live { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
    .dot.offline { background: #e74c3c; box-shadow: 0 0 6px #e74c3c; }
    .layout { display: flex; flex: 1; min-height: 0; }
    .previews { flex: 1; display: flex; gap: 10px; padding: 12px; min-width: 0; }
    .preview-box { flex: 1; position: relative; background: #000; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; min-width: 0; }
    .preview-box .preview-label { position: absolute; top: 8px; left: 10px; z-index: 5; font-size: 11px; color: #ccc; text-shadow: 0 1px 2px #000; background: rgba(0,0,0,0.4); padding: 2px 8px; border-radius: 10px; }
    .preview-box select.mode-select { position: absolute; top: 6px; right: 6px; z-index: 5; width: auto; font-size: 12px; padding: 4px 8px; }
    .preview-box img { flex: 1; width: 100%; height: 100%; object-fit: contain; display: block; }
    .preview-box canvas#overlayCanvas { position: absolute; inset: 0; z-index: 4; cursor: pointer; }
    .preview-box canvas#overlayCanvas.region-mode-active { cursor: crosshair; }
    .preview-box.depth-collapsed { flex: 0 0 40px; max-width: 40px; cursor: pointer; }
    .preview-box.depth-collapsed img, .preview-box.depth-collapsed select.mode-select { display: none; }
    .preview-box.depth-collapsed .depth-collapsed-label { display: flex; }
    .depth-collapsed-label { display: none; position: absolute; inset: 0; align-items: center; justify-content: center; }
    .depth-collapsed-label span { writing-mode: vertical-rl; transform: rotate(180deg); font-size: 11px; color: #999; letter-spacing: 0.05em; }
    .depth-toggle-btn { position: absolute; top: 6px; left: 6px; z-index: 6; background: rgba(0,0,0,0.5); color: #ddd; border: none; border-radius: 6px; width: 22px; height: 22px; font-size: 12px; line-height: 1; cursor: pointer; padding: 0; }
    .region-toolbar { position: absolute; bottom: 8px; left: 10px; z-index: 6; display: flex; gap: 6px; }
    .region-toolbar button { font-size: 11px; padding: 5px 10px; }
    .region-toolbar button.mode-active { outline: 2px solid #fff; }
    .region-list-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; font-size: 12px; color: #ccc; border-bottom: 1px solid #22222633; }
    .region-list-row .swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
    .accuracy-result { font-size: 13px; margin-top: 10px; }
    .accuracy-result table { width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 6px; }
    .accuracy-result td, .accuracy-result th { padding: 4px 6px; text-align: right; border-bottom: 1px solid #26262c; }
    .accuracy-result th:first-child, .accuracy-result td:first-child { text-align: left; }
    .capture-row { padding: 0 12px 12px; display: flex; gap: 10px; align-items: center; }
    .rightpanel { width: 320px; flex-shrink: 0; background: #141416; border-left: 1px solid #26262c; display: flex; }
    .tabs { width: 96px; flex-shrink: 0; display: flex; flex-direction: column; border-right: 1px solid #26262c; padding: 8px 0; }
    .tabs button { background: none; border: none; color: #999; text-align: left; padding: 12px 12px; font-size: 12px; font-weight: 600; cursor: pointer; border-left: 3px solid transparent; }
    .tabs button.active { color: #fff; border-left-color: #3b82f6; background: #1c1c20; }
    .tabcontent { flex: 1; overflow-y: auto; padding: 14px; min-width: 0; }
    .tabcontent h2 { font-size: 13px; margin: 0 0 12px; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }
    label { font-size: 12px; color: #bbb; display: block; margin-bottom: 4px; }
    input, select { background: #26262c; border: 1px solid #3a3a42; color: #eee; border-radius: 6px; padding: 7px 9px; font-size: 13px; width: 100%; }
    .field-row { display: flex; gap: 8px; align-items: end; margin-bottom: 10px; flex-wrap: wrap; }
    .field { flex: 1; min-width: 100px; }
    button { font-size: 13px; padding: 9px 14px; border-radius: 8px; border: none; cursor: pointer; font-weight: 600; }
    button.primary { background: #3b82f6; color: white; }
    button.primary:hover { background: #2563eb; }
    button.secondary { background: #2a2a30; color: #ddd; }
    button.secondary:hover { background: #35353c; }
    button.big { font-size: 16px; padding: 14px 20px; flex: 1; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result { margin-top: 10px; font-size: 14px; }
    .result .dims { font-size: 22px; font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
    .result .dims input { width: 62px; display: inline-block; font-size: 16px; padding: 4px 6px; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge.box { background: #14532d; color: #86efac; }
    .badge.irregular-item { background: #713f12; color: #fde68a; }
    .badge.pass { background: #14532d; color: #86efac; }
    .badge.fail { background: #7f1d1d; color: #fca5a5; }
    .badge.corrected { background: #4c1d95; color: #ddd6fe; }
    .badge.measured { background: #27272a; color: #a1a1aa; }
    .legend { font-size: 11px; color: #888; margin-top: 6px; }
    .spinner { display: inline-block; width: 13px; height: 13px; border: 2px solid #555; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .divider { border-top: 1px solid #26262c; margin: 14px 0; }
    .log-row { display: flex; gap: 8px; align-items: center; padding: 8px 0; border-bottom: 1px solid #22222633; cursor: pointer; }
    .log-row:hover { background: #1c1c20; }
    .log-row img { width: 44px; height: 44px; object-fit: cover; border-radius: 6px; flex-shrink: 0; }
    .log-row .meta { flex: 1; font-size: 12px; min-width: 0; }
    .log-row .meta .dims-small { color: #ddd; font-weight: 600; }
    .log-row .meta .ts { color: #888; }
    .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 50; }
    .modal { background: #1a1a1e; border-radius: 12px; padding: 20px; width: 420px; max-width: 90vw; max-height: 85vh; overflow-y: auto; }
    .modal img { width: 100%; border-radius: 8px; margin-bottom: 12px; }
    .modal h3 { margin: 0 0 12px; font-size: 15px; }
  </style>
</head>
<body>
  <div class="statusbar" id="statusbar">
    <span><span class="dot offline" id="camDot"></span><span id="camText">Checking camera...</span></span>
    <span id="calText">Calibration: --</span>
  </div>

  <div class="layout">
    <div class="previews">
      <div class="preview-box">
        <span class="preview-label">RGB &middot; cyan = live detections, yellow = last capture</span>
        <img id="rgbImg" src="/stream/rgb?token=__TOKEN__" />
        <canvas id="overlayCanvas"></canvas>
        <div class="region-toolbar">
          <button class="secondary" id="excludeModeBtn" onclick="setRegionMode('exclude')">Exclude</button>
          <button class="secondary" id="includeModeBtn" onclick="setRegionMode('include')">Include</button>
          <button class="secondary" id="clearRegionsBtn" onclick="clearOneShotRegions()">Clear</button>
        </div>
      </div>
      <div class="preview-box depth-collapsed" id="depthBox">
        <span class="preview-label">Depth</span>
        <button class="depth-toggle-btn" id="depthToggleBtn" onclick="toggleDepthCollapsed()">&#8250;</button>
        <select class="mode-select" id="depthModeSelect" onchange="onDepthModeChange()">
          <option value="colorized">Colorized</option>
          <option value="mask">Foreground mask only</option>
          <option value="raw">Raw grayscale</option>
        </select>
        <img id="depthImg" src="/stream/depth?token=__TOKEN__&mode=colorized" />
        <div class="depth-collapsed-label" onclick="toggleDepthCollapsed()"><span>Depth</span></div>
      </div>
    </div>

    <div class="rightpanel">
      <div class="tabs" id="tabButtons"></div>
      <div class="tabcontent" id="tabContent"></div>
    </div>
  </div>

  <div id="modalRoot"></div>

  <script>
    const TOKEN = "__TOKEN__";
    const headers = { Authorization: "Bearer " + TOKEN };
    let activeTab = "measure";

    function reconnectImg(imgEl) {
      imgEl.onerror = () => {
        setTimeout(() => { imgEl.src = imgEl.src.split("&_r=")[0] + "&_r=" + Date.now(); }, 1000);
      };
    }
    reconnectImg(document.getElementById("rgbImg"));
    reconnectImg(document.getElementById("depthImg"));

    function onDepthModeChange() {
      const mode = document.getElementById("depthModeSelect").value;
      document.getElementById("depthImg").src = "/stream/depth?token=" + TOKEN + "&mode=" + mode + "&_r=" + Date.now();
    }

    // Depth panel starts collapsed (see .depth-collapsed on #depthBox above) so the RGB view gets the room.
    function toggleDepthCollapsed() {
      const box = document.getElementById("depthBox");
      const collapsed = box.classList.toggle("depth-collapsed");
      document.getElementById("depthToggleBtn").innerHTML = collapsed ? "&#8250;" : "&#8249;";
    }

    // ---- Click-to-include/exclude region overlay (v7) ----
    let regionMode = null; // null | "exclude" | "include"
    let oneShotRegions = { exclude: [], include: [] };
    let drawingRect = null; // {x0,y0,x1,y1} in canvas-display pixels, while dragging
    let selectedDetection = null; // { selectedId, length, width, height, bbox } | null -- Include = click-to-select this
    const rgbImg = document.getElementById("rgbImg");
    const overlayCanvas = document.getElementById("overlayCanvas");
    const overlayCtx = overlayCanvas.getContext("2d");

    function setRegionMode(mode) {
      regionMode = regionMode === mode ? null : mode;
      document.getElementById("excludeModeBtn").classList.toggle("mode-active", regionMode === "exclude");
      document.getElementById("includeModeBtn").classList.toggle("mode-active", regionMode === "include");
      overlayCanvas.classList.toggle("region-mode-active", !!regionMode);
    }

    function resizeOverlayCanvas() {
      overlayCanvas.width = rgbImg.clientWidth;
      overlayCanvas.height = rgbImg.clientHeight;
      drawRegionOverlay();
    }
    window.addEventListener("resize", resizeOverlayCanvas);
    rgbImg.addEventListener("load", resizeOverlayCanvas, { once: false });

    // rgbImg has CSS `object-fit: contain` (see .preview-box img above), which letterboxes the
    // image inside its container whenever the container's aspect ratio doesn't match the
    // streamed frame's -- near-guaranteed since the container is a flex-sized panel, not a fixed
    // box matching the camera's exact resolution. Treating clientWidth/clientHeight as the
    // image's actual on-screen size (the old code) is off by however wide the letterbox bars
    // are, which is exactly "clicks don't map to what's on screen." This computes the real
    // rendered image rectangle within the container.
    function letterboxRect() {
      const containerW = overlayCanvas.width, containerH = overlayCanvas.height;
      const naturalW = rgbImg.naturalWidth, naturalH = rgbImg.naturalHeight;
      if (!containerW || !containerH || !naturalW || !naturalH) {
        return { offsetX: 0, offsetY: 0, width: containerW, height: containerH };
      }
      const containerRatio = containerW / containerH;
      const naturalRatio = naturalW / naturalH;
      if (naturalRatio > containerRatio) {
        const width = containerW;
        const height = containerW / naturalRatio;
        return { offsetX: 0, offsetY: (containerH - height) / 2, width, height };
      }
      const height = containerH;
      const width = containerH * naturalRatio;
      return { offsetX: (containerW - width) / 2, offsetY: 0, width, height };
    }

    function frameRectToDisplay(frameRect) {
      const box = letterboxRect();
      if (!box.width || !box.height) return frameRect;
      const scaleX = box.width / rgbImg.naturalWidth;
      const scaleY = box.height / rgbImg.naturalHeight;
      return {
        x: box.offsetX + frameRect.x * scaleX,
        y: box.offsetY + frameRect.y * scaleY,
        w: frameRect.w * scaleX,
        h: frameRect.h * scaleY,
      };
    }

    function drawRegionOverlay() {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      const drawSet = (regions, color) => {
        overlayCtx.strokeStyle = color;
        overlayCtx.lineWidth = 2;
        // regions are stored in native frame coordinates (same as what's sent to the backend) --
        // must convert back to display pixels before drawing, or saved rectangles render offset
        // by the letterbox bars just like the click mapping does.
        for (const r of regions) {
          const d = frameRectToDisplay(r);
          overlayCtx.strokeRect(d.x, d.y, d.w, d.h);
        }
      };
      drawSet(oneShotRegions.exclude, "#f87171");
      if (drawingRect) {
        // drawingRect is already in display/canvas pixels (live drag feedback) -- Exclude only,
        // Include has no drawn rectangle since it's a click-to-select, not a drag gesture.
        overlayCtx.strokeStyle = "#f87171";
        overlayCtx.setLineDash([4, 4]);
        overlayCtx.strokeRect(drawingRect.x, drawingRect.y, drawingRect.w, drawingRect.h);
        overlayCtx.setLineDash([]);
      }
      if (selectedDetection && selectedDetection.bbox) {
        // Highlight the currently-selected item's bbox in green so it's visually obvious which
        // item Capture is scoped to, matching the Include button's green accent color.
        const d = frameRectToDisplay(selectedDetection.bbox);
        overlayCtx.strokeStyle = "#4ade80";
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeRect(d.x, d.y, d.w, d.h);
      }
    }

    function canvasToFrameRect(displayRect) {
      // Subtract the letterbox offset first so a rect drawn against the letterbox bar's edge
      // (0,0 of the canvas) maps to (0,0) of the actual image, then scale by the image's real
      // on-screen size -- not the canvas/container size, which includes the letterbox bars.
      const box = letterboxRect();
      if (!box.width || !box.height) return displayRect;
      const scaleX = rgbImg.naturalWidth / box.width;
      const scaleY = rgbImg.naturalHeight / box.height;
      const x = (displayRect.x - box.offsetX) * scaleX;
      const y = (displayRect.y - box.offsetY) * scaleY;
      return {
        x: Math.round(Math.max(0, Math.min(rgbImg.naturalWidth, x))),
        y: Math.round(Math.max(0, Math.min(rgbImg.naturalHeight, y))),
        w: Math.round(displayRect.w * scaleX),
        h: Math.round(displayRect.h * scaleY),
      };
    }

    overlayCanvas.addEventListener("mousedown", (e) => {
      const rect = overlayCanvas.getBoundingClientRect();
      const startX = e.clientX - rect.left;
      const startY = e.clientY - rect.top;

      if (regionMode === "include" || !regionMode) {
        // Include mode (and idle, so a click always works even before pressing a button):
        // clicking a detected item selects it as the one Capture will measure -- see
        // /detections/select. This is deliberately NOT a drag-a-rectangle gesture: forcing raw
        // pixels under an arbitrary rectangle into the foreground mask (the old Include behavior)
        // swept in background pixels around the item too, which is exactly why it silently
        // failed to measure the clicked item correctly. Still tracks movement so an accidental
        // drag doesn't fire a select at the drag's start point.
        const onUpIdle = (upEvent) => {
          window.removeEventListener("mouseup", onUpIdle);
          const curX = upEvent.clientX - rect.left;
          const curY = upEvent.clientY - rect.top;
          const moved = Math.hypot(curX - startX, curY - startY);
          if (moved < 5) {
            const framePoint = canvasToFrameRect({ x: curX, y: curY, w: 0, h: 0 });
            selectDetectionAt(framePoint.x, framePoint.y);
          }
        };
        window.addEventListener("mouseup", onUpIdle);
        return;
      }

      // Exclude mode only, from here on: drag-a-rectangle to mark a region always ignored.
      drawingRect = { x: startX, y: startY, w: 0, h: 0 };

      const onMove = (moveEvent) => {
        const curX = moveEvent.clientX - rect.left;
        const curY = moveEvent.clientY - rect.top;
        drawingRect = { x: Math.min(startX, curX), y: Math.min(startY, curY), w: Math.abs(curX - startX), h: Math.abs(curY - startY) };
        drawRegionOverlay();
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (drawingRect && drawingRect.w > 4 && drawingRect.h > 4) {
          oneShotRegions.exclude.push({ ...drawingRect });
          pushDetectionOverrides();
        }
        drawingRect = null;
        drawRegionOverlay();
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    // ---- Click-to-select scoped capture (v7 follow-up) ----
    async function selectDetectionAt(x, y) {
      try {
        const res = await fetch("/detections/select", {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ x, y }),
        });
        if (!res.ok) return; // 404 = clicked empty background between items, not worth surfacing
        selectedDetection = await res.json();
        // Pre-select Training Mode's shape-type dropdown from the depth-geometry-derived
        // suggestion (see measure.py's _suggest_shape_type) whenever a different item gets
        // selected, instead of always defaulting to "box" -- mirrors the dashboard's
        // TrainingPanel.tsx equivalent. Purely a starting point; the operator can still change it.
        if (selectedDetection && selectedDetection.suggestedShapeType) {
          trainingShapeType = selectedDetection.suggestedShapeType;
        }
        renderSelectionStatus();
        drawRegionOverlay();
        refreshTrainingSection();
      } catch (e) {}
    }

    // ---- Selection status: read-only display now. Click-to-correct ("Recalibrate") was removed
    // here -- it's superseded by Training Mode (see the Calibrate tab), which fits one
    // batch calibration from several varied items instead of ad-hoc single-item corrections
    // that could conflict with each other and with per-shape offsets. See training.py's module
    // docstring on the Pi for the full root-cause writeup. ----
    function renderSelectionStatus() {
      const el = document.getElementById("selectionStatus");
      if (!el) return;
      if (!selectedDetection) { el.innerHTML = ""; return; }
      el.className = "";
      el.innerHTML = '<div class="legend" style="color:#93c5fd">Selected: ' + selectedDetection.length + 'x' + selectedDetection.width + 'x' + selectedDetection.height + 'in</div>';
    }

    async function pushDetectionOverrides() {
      // Include no longer sends includeRegions -- it's a click-to-select instead of a forced
      // rectangle (see the mousedown handler above), so only exclude ever needs pushing here.
      const excludeRegions = oneShotRegions.exclude.map(canvasToFrameRect);
      try {
        await fetch("/detection-overrides", {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ excludeRegions, includeRegions: [] }),
        });
      } catch (e) {}
    }

    async function clearOneShotRegions() {
      // Resets everything: drawn Exclude rectangles AND the current selection (Include), per the
      // "Clear should be a better button" request -- one button returns the live view to a
      // clean slate rather than only clearing one of the two mechanisms.
      oneShotRegions = { exclude: [], include: [] };
      selectedDetection = null;
      renderSelectionStatus();
      drawRegionOverlay();
      await pushDetectionOverrides();
      try {
        await fetch("/detections/deselect", { method: "POST", headers });
      } catch (e) {}
    }

    async function pollStatus() {
      try {
        const res = await fetch("/status?token=" + TOKEN);
        const data = await res.json();
        const dot = document.getElementById("camDot");
        const text = document.getElementById("camText");
        if (data.cameraLive) {
          dot.className = "dot live";
          text.textContent = "Camera live (frame age " + data.frameAgeSec + "s)";
        } else {
          dot.className = "dot offline";
          text.textContent = "Camera OFFLINE" + (data.frameAgeSec != null ? " (last frame " + data.frameAgeSec + "s ago, auto-recovering...)" : " (no frames received yet)");
        }
        const cal = data.calibration;
        const calText = document.getElementById("calText");
        if (cal.isCalibrated) {
          calText.textContent = "Calibrated (" + cal.mode + " mode" + (data.cameraHeightMm ? ", height " + (data.cameraHeightMm/25.4).toFixed(1) + "in" : "") + ")";
        } else {
          calText.textContent = "NOT CALIBRATED";
        }
        window.__lastHeightMm = data.cameraHeightMm;
        const heightInput = document.getElementById("heightIn");
        if (heightInput && data.cameraHeightMm && !heightInput.matches(":focus")) {
          heightInput.value = (data.cameraHeightMm / 25.4).toFixed(2);
        }
      } catch (e) {}
    }
    pollStatus();
    setInterval(pollStatus, 1500);

    // ---- Selection live-presence check ----
    // Confirmed via investigation that once /detections/select sets a selection, NOTHING
    // proactively clears it when the item is physically removed from frame -- not the 300s
    // server TTL (pure time check, no live-presence check), not the 2+-items disambiguation
    // paths (only fire when 2+ detections are simultaneously visible, never for the common
    // "the only/selected item was removed" case). This viewer had NO /detections polling loop at
    // all before (live borders come for free from the MJPEG stream itself), so this is new: poll
    // at the same 1500ms cadence as pollStatus, and if the selected id is no longer in the live
    // list, clear it here AND tell the backend to drop its own selection instead of leaving it to
    // linger for the full TTL. Mirrors the identical fix in the dashboard's page.tsx.
    async function pollSelectionPresence() {
      if (!selectedDetection) return;
      try {
        const res = await fetch("/detections?token=" + TOKEN);
        const data = await res.json();
        const live = (data.detections || []).find((d) => d.id === selectedDetection.selectedId);
        if (!live) {
          selectedDetection = null;
          renderSelectionStatus();
          drawRegionOverlay();
          await fetch("/detections/deselect", { method: "POST", headers });
          return;
        }
        if (live.length !== selectedDetection.length || live.width !== selectedDetection.width || live.height !== selectedDetection.height) {
          selectedDetection = { ...selectedDetection, length: live.length, width: live.width, height: live.height };
          renderSelectionStatus();
        }
      } catch (e) {}
    }
    setInterval(pollSelectionPresence, 1500);

    // ---- Tabs ----
    const TABS = [
      { id: "measure", label: "Measure" },
      { id: "calibration", label: "Calibrate" },
      { id: "products", label: "Products" },
      { id: "log", label: "Scan Log" },
    ];

    function renderTabButtons() {
      const el = document.getElementById("tabButtons");
      el.innerHTML = "";
      for (const t of TABS) {
        const btn = document.createElement("button");
        btn.textContent = t.label;
        btn.className = t.id === activeTab ? "active" : "";
        btn.onclick = () => { activeTab = t.id; renderTabButtons(); renderTabContent(); };
        el.appendChild(btn);
      }
    }

    function renderTabContent() {
      const el = document.getElementById("tabContent");
      if (activeTab === "measure") renderMeasureTab(el);
      else if (activeTab === "calibration") renderCalibrationTab(el);
      else if (activeTab === "products") renderProductsTab(el);
      else if (activeTab === "log") renderLogTab(el);
    }

    // ---- Measure tab: shape type + capture button + inline-editable result ----
    let lastCapture = null; // { logEntryId, length, width, height, ... }

    function renderMeasureTab(el) {
      el.innerHTML = `
        <h2>Capture &amp; Measure</h2>
        <div class="field-row">
          <div class="field">
            <label>Shape type (uses this shape's Training-derived offset, if any)</label>
            <select id="shapeTypeSelect"></select>
          </div>
        </div>
        <div class="field-row">
          <div class="field">
            <label>Product profile (optional, pass/fail check)</label>
            <select id="productSelect"><option value="">-- none --</option></select>
          </div>
        </div>
        <div class="legend" id="selectionStatus" style="margin-bottom:8px"></div>
        <div class="field-row">
          <button class="primary big" onclick="doCapture()" id="captureBtn">Capture &amp; Measure</button>
        </div>
        <div class="result" id="result"></div>
      `;
      loadShapeTypesInto("shapeTypeSelect");
      loadProductsInto("productSelect");
      renderSelectionStatus();
    }

    async function loadShapeTypesInto(selectId) {
      try {
        const res = await fetch("/shape-profiles?token=" + TOKEN);
        const data = await res.json();
        const select = document.getElementById(selectId);
        if (!select) return;
        select.innerHTML = '<option value="">-- none --</option>';
        for (const shapeType of data.builtinShapeTypes) {
          const opt = document.createElement("option");
          opt.value = shapeType; opt.textContent = shapeType;
          select.appendChild(opt);
        }
      } catch (e) {}
    }

    async function resetSelectedShapeProfile() {
      const select = document.getElementById("resetShapeTypeSelect");
      const shapeType = select ? select.value : "";
      if (!shapeType) { alert("Select a shape type first."); return; }
      if (!confirm('Delete the learned correction offset for "' + shapeType + '"? Future scans of this shape start fresh.')) return;
      try {
        await fetch("/shape-profiles/" + encodeURIComponent(shapeType), { method: "DELETE", headers });
        alert('Reset "' + shapeType + '" -- it will start learning fresh from the next correction.');
      } catch (e) { alert("Error resetting shape profile: " + e); }
    }

    async function doCapture() {
      const el = document.getElementById("result");
      const btn = document.getElementById("captureBtn");
      btn.disabled = true;
      el.innerHTML = '<span class="spinner"></span>Capturing...';
      const productId = document.getElementById("productSelect").value;
      const shapeType = document.getElementById("shapeTypeSelect").value;
      try {
        const params = new URLSearchParams();
        if (productId) params.set("productId", productId);
        if (shapeType) params.set("shapeType", shapeType);
        const url = "/capture" + (params.toString() ? "?" + params.toString() : "");
        const res = await fetch(url, { method: "POST", headers });
        const data = await res.json();
        if (!res.ok) {
          el.innerHTML = '<span style="color:#f87171">' + (data.detail || JSON.stringify(data)) + '</span>';
        } else {
          lastCapture = data;
          renderCaptureResult(el, data);
          document.getElementById("rgbImg").src = "/stream/rgb?token=" + TOKEN + "&_r=" + Date.now();
        }
      } catch (e) { el.innerHTML = '<span style="color:#f87171">Error: ' + e + '</span>'; }
      btn.disabled = false;
    }

    // Same 3 fixed colors as _AXIS_LABEL_COLORS in api.py's on-image dimension labels --
    // pairing each input with the color of its matching label on the picture.
    function axisDot(color) { return '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + ';margin-right:3px;vertical-align:middle"></span>'; }

    function renderCaptureResult(el, data) {
      let html = '<div class="dims">';
      html += axisDot("#00dcff") + '<span>' + data.length + '</span> x ';
      html += axisDot("#ff3cdc") + '<span>' + data.width + '</span> x ';
      html += axisDot("#ffd200") + '<span>' + data.height + '</span> in';
      html += '<span class="badge ' + data.classification + '">' + data.classification + '</span>';
      if (data.withinTolerance !== undefined) {
        html += '<span class="badge ' + (data.withinTolerance ? "pass" : "fail") + '">' + (data.withinTolerance ? "PASS" : "FAIL") + '</span>';
      }
      html += '</div>';
      html += '<div class="legend">' + data.cubicFeet + ' cu ft &middot; Confidence: ' + (data.confidence * 100).toFixed(0) + '%';
      if (data.shapeType) html += ' &middot; Shape: ' + data.shapeType;
      if (data.expected) {
        html += ' &middot; Expected: ' + data.expected.length + ' x ' + data.expected.width + ' x ' + data.expected.height + ' in (±' + data.expected.toleranceIn + ') &middot; Delta: ' + data.deltaIn.length + ', ' + data.deltaIn.width + ', ' + data.deltaIn.height;
      }
      html += '</div>';
      html += '<div class="legend" style="margin-top:10px">To correct measurements or retrain the system, use Training Calibration in the Calibrate tab.</div>';
      el.innerHTML = html;
    }

    // ---- Calibration tab ----
    function renderCalibrationTab(el) {
      el.innerHTML = `
        <h2>Calibration</h2>
        <div class="legend" style="margin-bottom:10px">Recommended order: (1) Set height &amp; Calibrate, (2) run a Training Calibration with several varied items.</div>
        <div class="field-row">
          <div class="field">
            <label>Camera mount height (inches, straight down to surface)</label>
            <input id="heightIn" type="number" step="0.25" placeholder="e.g. 36" value="${window.__lastHeightMm ? (window.__lastHeightMm/25.4).toFixed(2) : ''}" />
          </div>
        </div>
        <div class="field-row">
          <button class="secondary" onclick="saveHeight()" style="flex:1">Save Height</button>
        </div>
        <div class="field-row">
          <button class="primary" onclick="calibrate('height')" style="flex:1">Quick Calibrate</button>
        </div>
        <div class="field-row">
          <button class="secondary" onclick="calibrate('live')" style="flex:1">Precise Calibrate (clear surface first)</button>
        </div>
        <div class="legend" id="calibStatus"></div>

        <div class="divider"></div>
        <div id="trainingSection"></div>
        <div class="field-row" style="margin-top:10px">
          <div class="field">
            <label>Reset a shape type's learned offset</label>
            <select id="resetShapeTypeSelect"></select>
          </div>
          <button class="secondary" onclick="resetSelectedShapeProfile()" style="align-self:end" title="Delete this shape type's learned correction offset -- use if a bad Training run skewed it">Reset</button>
        </div>

        <div class="divider"></div>
        <h2>Excluded Zones</h2>
        <div class="legend" style="margin-bottom:8px">Persistent regions always ignored by detection (e.g. a fixed shadow or shelf edge) -- draw with the Exclude tool on the Measure tab's preview, then save here.</div>
        <div class="field-row">
          <button class="secondary" onclick="savePermanentExcludeRegions()" style="flex:1">Save Current Exclude Drawing as Permanent Zone(s)</button>
        </div>
        <div id="permanentZoneList"></div>

        <div class="divider"></div>
        <h2>Auto-clear Static Items</h2>
        <div class="legend" style="margin-bottom:8px">An item left untouched under the camera for longer than the timeout is folded into the background and stops being detected -- picking it up and setting it back down immediately makes it detected again.</div>
        <div class="field-row">
          <label style="display:flex;align-items:center;gap:8px;font-size:13px;color:#ddd;margin-bottom:0">
            <input type="checkbox" id="autoAbsorbEnabled" style="width:auto" onchange="saveAutoAbsorbConfig()" /> Enabled
          </label>
        </div>
        <div class="field-row">
          <div class="field"><label>Timeout (minutes)</label><input id="autoAbsorbMinutes" type="number" step="0.5" min="0.25" /></div>
          <button class="secondary" onclick="saveAutoAbsorbConfig()" style="align-self:end">Save</button>
        </div>
        <div class="legend" id="autoAbsorbStatus"></div>
      `;
      loadShapeTypesInto("resetShapeTypeSelect");
      loadPermanentExcludeRegions();
      loadAutoAbsorbConfig();
      renderTrainingSection(document.getElementById("trainingSection"));
    }

    async function loadAutoAbsorbConfig() {
      try {
        const res = await fetch("/config/auto-absorb?token=" + TOKEN);
        const data = await res.json();
        const enabledEl = document.getElementById("autoAbsorbEnabled");
        const minutesEl = document.getElementById("autoAbsorbMinutes");
        if (enabledEl) enabledEl.checked = data.enabled;
        if (minutesEl) minutesEl.value = (data.timeoutSec / 60).toFixed(1);
      } catch (e) {}
    }

    async function saveAutoAbsorbConfig() {
      const enabled = document.getElementById("autoAbsorbEnabled").checked;
      const minutes = parseFloat(document.getElementById("autoAbsorbMinutes").value);
      const statusEl = document.getElementById("autoAbsorbStatus");
      if (!minutes || minutes <= 0) { if (statusEl) statusEl.textContent = "Enter a valid timeout in minutes."; return; }
      try {
        await fetch("/config/auto-absorb", {
          method: "PUT",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ enabled, timeoutSec: minutes * 60 }),
        });
        if (statusEl) statusEl.textContent = "Saved.";
      } catch (e) { if (statusEl) statusEl.textContent = "Error: " + e; }
    }

    // ---- Training Mode (replaces Accuracy Check + click-to-correct + per-shape Save
    // Correction, which conflicted with each other -- see training.py's module docstring for the
    // full root-cause writeup). Open-ended loop, not fixed phases: scan an item, mark it
    // Accurate or Not Accurate (entering the true size for the latter), which becomes one more
    // sample immediately available to Recalibrate; Recalibrate can run repeatedly against the
    // ever-growing sample set as more items get scanned/rescanned, and the operator decides when
    // to Finish or Cancel & Roll Back. Mirrors the dashboard's TrainingPanel.tsx one-to-one;
    // reuses the same `selectedDetection` module-level var the click-to-select mechanism
    // already sets. ----
    let trainingShapeType = "box";
    let trainingMarkingNotAccurate = false;
    const MIN_REQUIRED_ACCURACY_PCT = 90;
    const MAX_CONSECUTIVE_AUTO_RECALIBRATIONS = 3;
    // One-shot notice from the auto-recalibrate-on-failed-round trigger -- not derived from
    // polled session state since it describes a transition ("round just closed and was auto-
    // recalibrated"), not a persistent field. Cleared on Recalibrate/Cancel, same as the
    // dashboard's TrainingPanel.tsx equivalent.
    let trainingAutoNotice = "";

    async function fetchTrainingSession() {
      try {
        const res = await fetch("/training/session?token=" + TOKEN);
        const data = await res.json();
        return data.session || null;
      } catch (e) { return null; }
    }

    async function fetchLegacyCalibrationCheck() {
      try {
        const res = await fetch("/training/legacy-check?token=" + TOKEN);
        const data = await res.json();
        return !!data.hasLegacyData;
      } catch (e) { return false; }
    }

    async function renderTrainingSection(el) {
      if (!el) return;
      const session = await fetchTrainingSession();
      const hasLegacyData = !session || session.status !== "active" ? await fetchLegacyCalibrationCheck() : false;
      el.innerHTML = trainingSectionHtml(session, hasLegacyData);
      if (session && session.status === "active") {
        renderSelectionStatus(); // keep the click-to-select badge visible while a session is open
      }
    }

    function trainingSectionHtml(session, hasLegacyData) {
      let html = '<h2>Training Calibration</h2>';
      if (hasLegacyData) {
        html += '<div class="legend" style="margin-bottom:8px;color:#fbbf24">This device has legacy calibration data that predates Training Mode. Start a session below to replace it with a verified calibration.</div>';
      }
      if (!session || session.status !== "active") {
        if (session && session.status === "completed" && session.computed_result) {
          const f = session.computed_result.appliedScaleFactor;
          html += '<div class="legend" style="margin-bottom:8px">Last run: ' + (session.completed_at || '').slice(0, 19).replace('T', ' ') + ' &middot; ' + session.samples.length + ' items &middot; scale L&times;' + f.length + ' W&times;' + f.width + ' H&times;' + f.height + '</div>';
        }
        html += '<div class="legend" style="margin-bottom:10px">Scan an item, mark it Accurate or Not Accurate, and repeat. Once you have 5+ scans, Recalibrate any time -- keep scanning and recalibrating until it looks right, then Finish.</div>';
        html += '<div class="field-row"><button class="primary" onclick="startTraining()" style="flex:1">Start Training Calibration</button></div>';
        return html;
      }

      const samples = session.samples || [];
      const canRecalibrate = samples.length >= 5;
      const hasCalibrated = !!session.computed_result;
      const verificationQueue = session.verification_queue || [];
      const roundResults = session.round_results || [];
      const inVerificationRound = verificationQueue.length > 0;
      const currentVerifyIndex = inVerificationRound ? verificationQueue[0] : null;
      const currentVerifySample = currentVerifyIndex != null ? samples.find((s) => s.index === currentVerifyIndex) : null;
      const lastRoundAccuracy = session.last_round_accuracy_pct;
      const roundPassed = lastRoundAccuracy != null && lastRoundAccuracy >= MIN_REQUIRED_ACCURACY_PCT;
      const canFinish = hasCalibrated && !inVerificationRound && roundPassed;

      html += '<div class="legend" style="margin-bottom:8px">' + samples.length + ' scan' + (samples.length === 1 ? '' : 's') + ' recorded'
        + (canRecalibrate ? '' : ' -- need ' + (5 - samples.length) + ' more before you can Recalibrate') + '.'
        + (session.recalibration_count > 0 ? ' Recalibrated ' + session.recalibration_count + 'x so far.' : '') + '</div>';

      // Live accuracy gauge -- while a round is in progress, its RUNNING accuracy so far; before
      // any round has ever closed, the session's overall accurate-verdict rate across all scans
      // instead, so the gauge is never blank on a fresh session. Mirrors TrainingPanel.tsx's
      // identical gaugeAccuracy computation.
      let gaugeAccuracy = null;
      if (inVerificationRound) {
        gaugeAccuracy = roundResults.length > 0 ? Math.round((100 * roundResults.filter((r) => r.verdict === "accurate").length) / roundResults.length) : null;
      } else if (lastRoundAccuracy != null) {
        gaugeAccuracy = lastRoundAccuracy;
      } else if (samples.length > 0) {
        gaugeAccuracy = Math.round((100 * samples.filter((s) => s.verdict === "accurate").length) / samples.length);
      }
      if (gaugeAccuracy != null) {
        const gaugePassed = gaugeAccuracy >= MIN_REQUIRED_ACCURACY_PCT;
        html += '<div style="margin-bottom:8px">';
        html += '<div style="display:flex;justify-content:space-between;font-size:11px;color:#9ca3af;margin-bottom:3px">';
        html += '<span>' + (inVerificationRound ? "This round" : "Accuracy") + '</span>';
        html += '<span style="font-weight:600;' + (gaugePassed ? 'color:#86efac' : '') + '">' + gaugeAccuracy + '% (target ' + MIN_REQUIRED_ACCURACY_PCT + '%)</span>';
        html += '</div>';
        html += '<div style="position:relative;height:10px;border-radius:5px;background:#374151;overflow:hidden">';
        html += '<div style="height:100%;border-radius:5px;width:' + Math.min(100, gaugeAccuracy) + '%;background:' + (gaugePassed ? '#22c55e' : '#f87171') + '"></div>';
        html += '<div style="position:absolute;top:0;bottom:0;width:1px;background:#9ca3af;left:' + MIN_REQUIRED_ACCURACY_PCT + '%"></div>';
        html += '</div></div>';
      }

      const roundAccuracyHistory = session.round_accuracy_history || [];
      if (roundAccuracyHistory.length > 0) {
        html += '<div style="margin-bottom:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center">';
        roundAccuracyHistory.forEach((h, i) => {
          if (i > 0) html += '<span style="color:#6b7280">&rarr;</span>';
          const passed = h.accuracy_pct >= MIN_REQUIRED_ACCURACY_PCT;
          html += '<span class="badge ' + (passed ? 'pass' : 'fail') + '">R' + h.round_number + ': ' + h.accuracy_pct + '%</span>';
        });
        html += '</div>';
      }

      if (trainingAutoNotice) {
        html += '<div class="legend" style="margin-bottom:8px;color:#a5b4fc">' + trainingAutoNotice + '</div>';
      }

      if (inVerificationRound) {
        html += '<div class="legend" style="margin-bottom:6px;color:#a5b4fc">Verification round ' + session.round_number + ' -- ' + roundResults.length + ' of ' + (roundResults.length + verificationQueue.length) + ' re-checked.</div>';
        if (currentVerifySample) {
          html += '<div class="legend" style="margin-bottom:6px">Place item #' + (currentVerifySample.index + 1) + ' back under the camera: ' + currentVerifySample.shape_type + ', true size ' + currentVerifySample.true_sorted.join('x') + 'in -- the exact item scanned in that recalibration.</div>';
        }
        html += '<div class="legend" style="margin-bottom:6px">Select that item in the RGB preview, then mark whether it now reads correctly.</div>';
        html += '<div class="field-row">';
        html += '<button class="secondary" style="flex:1;background:#14532d;color:#86efac" onclick="verifyTrainingSample(true)"' + (selectedDetection ? '' : ' disabled') + '>Accurate</button>';
        html += '<button class="secondary" style="flex:1;background:#7f1d1d;color:#fca5a5" onclick="verifyTrainingSample(false)"' + (selectedDetection ? '' : ' disabled') + '>Not Accurate</button>';
        html += '</div>';
      } else {
        if (lastRoundAccuracy != null) {
          html += '<div class="legend" style="margin-bottom:6px;color:' + (roundPassed ? '#86efac' : '#fca5a5') + '">Round ' + session.round_number + ' verified ' + lastRoundAccuracy + '% accurate (need ' + MIN_REQUIRED_ACCURACY_PCT + '%)' + (roundPassed ? ' -- you can Finish.' : ' -- scan more, Recalibrate, and verify again.') + '</div>';
        }
        html += '<div class="field-row"><div class="field"><label>Shape type</label><select id="trainShapeType" onchange="trainingShapeType = this.value">';
        for (const t of ["box", "bag", "can", "irregular"]) {
          html += '<option value="' + t + '"' + (t === trainingShapeType ? ' selected' : '') + '>' + t + '</option>';
        }
        html += '</select></div></div>';
        html += '<div class="legend" style="margin-bottom:6px">Select an item in the RGB preview, then mark whether its reading is right.</div>';
        if (!trainingMarkingNotAccurate) {
          html += '<div class="field-row">';
          html += '<button class="secondary" style="flex:1;background:#14532d;color:#86efac" onclick="markTrainingSample(true)"' + (selectedDetection ? '' : ' disabled') + '>Accurate</button>';
          html += '<button class="secondary" style="flex:1;background:#7f1d1d;color:#fca5a5" onclick="startMarkNotAccurate()"' + (selectedDetection ? '' : ' disabled') + '>Not Accurate</button>';
          html += '</div>';
        } else {
          html += '<div class="legend" style="margin-bottom:4px">Enter this item\\'s true size:</div>';
          html += '<div class="field-row">';
          html += '<div class="field"><label>True L (in)</label><input id="trainTrueLen" type="number" step="0.01" value="' + (selectedDetection ? selectedDetection.length : '') + '" /></div>';
          html += '<div class="field"><label>True W (in)</label><input id="trainTrueWid" type="number" step="0.01" value="' + (selectedDetection ? selectedDetection.width : '') + '" /></div>';
          html += '<div class="field"><label>True H (in)</label><input id="trainTrueHei" type="number" step="0.01" value="' + (selectedDetection ? selectedDetection.height : '') + '" /></div>';
          html += '</div>';
          html += '<div class="field-row">';
          html += '<button class="secondary" onclick="trainingMarkingNotAccurate = false; refreshTrainingSection();" style="flex:1">Cancel</button>';
          html += '<button class="primary" onclick="markTrainingSample(false)" style="flex:1">Submit</button>';
          html += '</div>';
        }
      }
      html += '<div class="legend" id="trainingStatus"></div>';

      if (roundResults.length) {
        html += '<div style="margin-top:6px;max-height:120px;overflow-y:auto">';
        for (const r of roundResults.slice().reverse()) {
          html += '<div class="region-list-row"><span class="badge ' + (r.verdict === 'accurate' ? 'pass' : 'fail') + '" style="margin-right:6px">' + (r.verdict === 'accurate' ? 'OK' : 'FIX') + '</span>';
          html += '<span>re-check #' + (r.sampleIndex + 1) + ' &middot; ' + r.shapeType + '</span></div>';
        }
        html += '</div>';
      }

      if (samples.length) {
        html += '<div style="margin-top:8px;max-height:180px;overflow-y:auto">';
        for (const s of samples.slice().reverse()) {
          html += '<div class="region-list-row"><span class="badge ' + (s.verdict === 'accurate' ? 'pass' : 'fail') + '" style="margin-right:6px">' + (s.verdict === 'accurate' ? 'OK' : 'FIX') + '</span>';
          html += '<span>#' + (s.index + 1) + ' &middot; ' + s.shape_type + ' &middot; true ' + s.true_sorted.join('x') + 'in</span>';
          if (!inVerificationRound) {
            html += '<button class="secondary" style="font-size:11px;padding:2px 8px;margin-left:auto" onclick="deleteTrainingSample(' + s.index + ')">Remove</button>';
          }
          html += '</div>';
        }
        html += '</div>';
      }

      let finishTitle = '';
      if (hasCalibrated && !canFinish) {
        finishTitle = inVerificationRound
          ? 'Re-verify all ' + verificationQueue.length + ' remaining item(s) from this round first.'
          : "This round's verified accuracy is " + (lastRoundAccuracy || 0) + '% -- needs ' + MIN_REQUIRED_ACCURACY_PCT + '%. Recalibrate and verify again.';
      }
      html += '<div class="field-row" style="margin-top:10px">';
      html += '<button class="secondary" onclick="cancelTraining()" style="flex:1;' + (hasCalibrated ? 'color:#f87171' : '') + '">' + (hasCalibrated ? 'Cancel & Roll Back' : 'Cancel') + '</button>';
      html += '<button class="secondary" onclick="recalibrateTraining()" style="flex:1"' + (canRecalibrate ? '' : ' disabled') + '>Recalibrate</button>';
      html += '<button class="primary" onclick="finishTraining()" style="flex:1" title="' + finishTitle + '"' + (canFinish ? '' : ' disabled') + '>Finish</button>';
      html += '</div>';
      return html;
    }

    async function refreshTrainingSection() {
      renderTrainingSection(document.getElementById("trainingSection"));
    }

    async function startTraining() {
      try {
        const res = await fetch("/training/start", { method: "POST", headers });
        const data = await res.json();
        if (!res.ok) { alert(data.detail || "Error starting training."); return; }
        refreshTrainingSection();
      } catch (e) { alert("Error: " + e); }
    }

    function startMarkNotAccurate() {
      trainingMarkingNotAccurate = true;
      refreshTrainingSection();
    }

    async function markTrainingSample(accurate) {
      const statusEl = document.getElementById("trainingStatus");
      if (!selectedDetection) { if (statusEl) statusEl.textContent = "Select an item in the RGB preview first."; return; }
      const body = { shapeType: trainingShapeType, verdict: accurate ? "accurate" : "not_accurate" };
      if (!accurate) {
        const trueLength = parseFloat(document.getElementById("trainTrueLen").value);
        const trueWidth = parseFloat(document.getElementById("trainTrueWid").value);
        const trueHeight = parseFloat(document.getElementById("trainTrueHei").value);
        if (!trueLength || !trueWidth || !trueHeight) { if (statusEl) statusEl.textContent = "Enter all three true dimensions."; return; }
        body.trueLength = trueLength; body.trueWidth = trueWidth; body.trueHeight = trueHeight;
      }
      if (statusEl) statusEl.textContent = "Recording...";
      try {
        const res = await fetch("/training/sample", {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) { if (statusEl) statusEl.textContent = data.detail || "Error."; return; }
        trainingMarkingNotAccurate = false;
        refreshTrainingSection();
      } catch (e) { if (statusEl) statusEl.textContent = "Error: " + e; }
    }

    async function verifyTrainingSample(accurate) {
      // UNLIKE markTrainingSample, no true-size prompt here even for "Not Accurate" -- the true
      // size was already recorded when this item was first scanned and doesn't change on a
      // re-scan, so the backend reuses it. Asking the operator to retype it was a real reported
      // bug ("I already did that").
      const statusEl = document.getElementById("trainingStatus");
      if (!selectedDetection) { if (statusEl) statusEl.textContent = "Select the item in the RGB preview first."; return; }
      const session = await fetchTrainingSession();
      const queue = (session && session.verification_queue) || [];
      if (!queue.length) { if (statusEl) statusEl.textContent = "No item pending verification."; return; }
      const sampleIndex = queue[0];
      const body = { verdict: accurate ? "accurate" : "not_accurate" };
      if (statusEl) statusEl.textContent = "Recording...";
      try {
        const res = await fetch("/training/verify/" + sampleIndex, {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) { if (statusEl) statusEl.textContent = data.detail || "Error."; return; }
        if (data.autoRecalibrated) {
          const failures = (data.session && data.session.consecutive_auto_recalibrate_failures) || 0;
          trainingAutoNotice = failures >= MAX_CONSECUTIVE_AUTO_RECALIBRATIONS
            ? "Round scored " + data.previousRoundAccuracy + "% -- " + MAX_CONSECUTIVE_AUTO_RECALIBRATIONS + " auto-recalibrations in a row didn't reach " + MIN_REQUIRED_ACCURACY_PCT + "%. Scan more or different items, then Recalibrate manually."
            : "Round scored " + data.previousRoundAccuracy + "% -- auto-recalibrated, next round started.";
        } else {
          trainingAutoNotice = "";
        }
        refreshTrainingSection();
      } catch (e) { if (statusEl) statusEl.textContent = "Error: " + e; }
    }

    async function deleteTrainingSample(index) {
      try {
        await fetch("/training/samples/" + index, { method: "DELETE", headers });
        refreshTrainingSection();
      } catch (e) {}
    }

    async function recalibrateTraining() {
      try {
        const res = await fetch("/training/recalibrate", { method: "POST", headers });
        const data = await res.json();
        if (!res.ok) { alert(data.detail || "Error recalibrating."); return; }
        trainingAutoNotice = "";
        refreshTrainingSection();
      } catch (e) { alert("Error: " + e); }
    }

    async function finishTraining() {
      try {
        const res = await fetch("/training/finish", { method: "POST", headers });
        const data = await res.json();
        if (!res.ok) { alert(data.detail || "Error finishing."); return; }
        refreshTrainingSection();
      } catch (e) {}
    }

    async function cancelTraining() {
      try {
        await fetch("/training/cancel", { method: "POST", headers });
        trainingMarkingNotAccurate = false;
        trainingAutoNotice = "";
        refreshTrainingSection();
      } catch (e) {}
    }

    async function loadPermanentExcludeRegions() {
      const list = document.getElementById("permanentZoneList");
      if (!list) return;
      try {
        const res = await fetch("/permanent-exclude-regions?token=" + TOKEN);
        const data = await res.json();
        renderPermanentZoneList(data.regions);
      } catch (e) {}
    }

    function renderPermanentZoneList(regions) {
      const list = document.getElementById("permanentZoneList");
      if (!list) return;
      list.innerHTML = "";
      if (!regions.length) { list.innerHTML = '<div class="legend">No permanent exclude zones saved.</div>'; return; }
      regions.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "region-list-row";
        row.innerHTML = '<span class="swatch" style="background:#f87171"></span><span>Zone ' + (i + 1) + ': x=' + r.x + ', y=' + r.y + ', w=' + r.w + ', h=' + r.h + '</span>';
        const delBtn = document.createElement("button");
        delBtn.textContent = "Remove"; delBtn.className = "secondary";
        delBtn.style.fontSize = "11px"; delBtn.style.padding = "2px 8px"; delBtn.style.marginLeft = "auto";
        delBtn.onclick = async () => {
          const next = regions.filter((_, idx) => idx !== i);
          await savePermanentExcludeRegionsList(next);
        };
        row.appendChild(delBtn);
        list.appendChild(row);
      });
    }

    async function savePermanentExcludeRegionsList(regions) {
      try {
        const res = await fetch("/permanent-exclude-regions", {
          method: "PUT",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ regions }),
        });
        const data = await res.json();
        renderPermanentZoneList(data.regions);
      } catch (e) {}
    }

    async function savePermanentExcludeRegions() {
      // Promotes the Measure tab's currently-drawn one-shot exclude rectangles (in native frame
      // pixel coords, same shape /detection-overrides already uses) into the persistent list.
      const newRects = oneShotRegions.exclude.map(canvasToFrameRect);
      if (!newRects.length) { alert("Draw one or more Exclude rectangles on the live preview first."); return; }
      try {
        const res = await fetch("/permanent-exclude-regions?token=" + TOKEN);
        const data = await res.json();
        await savePermanentExcludeRegionsList([...data.regions, ...newRects]);
        oneShotRegions.exclude = [];
        drawRegionOverlay();
        await pushDetectionOverrides();
      } catch (e) { alert("Error saving permanent zone: " + e); }
    }

    async function saveHeight() {
      const inches = parseFloat(document.getElementById("heightIn").value);
      if (!inches || inches <= 0) { alert("Enter a valid height in inches."); return; }
      const mm = inches * 25.4;
      const res = await fetch("/config", { method: "PUT", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify({ cameraHeightMm: mm }) });
      document.getElementById("calibStatus").textContent = res.ok ? "Height saved." : "Error saving height.";
    }

    async function calibrate(mode) {
      const el = document.getElementById("calibStatus");
      el.innerHTML = '<span class="spinner"></span>Calibrating (' + mode + ')...';
      try {
        const res = await fetch("/calibrate?mode=" + mode, { method: "POST", headers });
        const data = await res.json();
        el.textContent = res.ok ? "Calibrated (" + data.mode + ") at " + data.capturedAt : "Error: " + (data.detail || JSON.stringify(data));
        pollStatus();
      } catch (e) { el.textContent = "Error: " + e; }
    }

    // ---- Products tab ----
    function renderProductsTab(el) {
      el.innerHTML = `
        <h2>Product Profiles</h2>
        <div class="field-row">
          <div class="field"><label>Name</label><input id="pName" placeholder="e.g. SKU-1234" /></div>
        </div>
        <div class="field-row">
          <div class="field"><label>L (in)</label><input id="pLen" type="number" step="0.1" /></div>
          <div class="field"><label>W (in)</label><input id="pWid" type="number" step="0.1" /></div>
          <div class="field"><label>H (in)</label><input id="pHei" type="number" step="0.1" /></div>
        </div>
        <div class="field-row">
          <div class="field"><label>Tolerance (in)</label><input id="pTol" type="number" step="0.1" value="0.5" /></div>
          <button class="primary" onclick="addProduct()">Add</button>
        </div>
        <div class="divider"></div>
        <div id="productList"></div>
      `;
      loadProductsInto("productSelect", true);
    }

    async function loadProductsInto(selectId, alsoRenderList) {
      try {
        const res = await fetch("/products?token=" + TOKEN);
        const data = await res.json();
        const select = document.getElementById(selectId);
        if (select) {
          select.innerHTML = '<option value="">-- none --</option>';
          for (const p of data.products) {
            const opt = document.createElement("option");
            opt.value = p.id; opt.textContent = p.name;
            select.appendChild(opt);
          }
        }
        const list = document.getElementById("productList");
        if (alsoRenderList && list) {
          list.innerHTML = "";
          for (const p of data.products) {
            const row = document.createElement("div");
            row.className = "legend";
            row.style.marginBottom = "6px";
            row.textContent = p.name + ": " + p.expected_length_in + " x " + p.expected_width_in + " x " + p.expected_height_in + " in (±" + p.tolerance_in + ") ";
            const delBtn = document.createElement("button");
            delBtn.textContent = "Remove"; delBtn.className = "secondary";
            delBtn.style.fontSize = "11px"; delBtn.style.padding = "2px 8px"; delBtn.style.marginLeft = "6px";
            delBtn.onclick = async () => { await fetch("/products/" + p.id, { method: "DELETE", headers }); renderProductsTab(document.getElementById("tabContent")); };
            row.appendChild(delBtn);
            list.appendChild(row);
          }
        }
      } catch (e) {}
    }

    async function addProduct() {
      const name = document.getElementById("pName").value;
      const expectedLengthIn = parseFloat(document.getElementById("pLen").value);
      const expectedWidthIn = parseFloat(document.getElementById("pWid").value);
      const expectedHeightIn = parseFloat(document.getElementById("pHei").value);
      const toleranceIn = parseFloat(document.getElementById("pTol").value) || 0.5;
      if (!name || !expectedLengthIn || !expectedWidthIn || !expectedHeightIn) { alert("Fill in name and all dimensions."); return; }
      const res = await fetch("/products", { method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify({ name, expectedLengthIn, expectedWidthIn, expectedHeightIn, toleranceIn }) });
      if (res.ok) { renderProductsTab(document.getElementById("tabContent")); }
      else { alert("Error adding product."); }
    }

    // ---- Scan Log tab: table with thumbnails, click a row to review + correct with the image ----
    async function renderLogTab(el) {
      el.innerHTML = '<h2>Scan Log</h2><div id="logList" class="legend">Loading...</div>';
      try {
        const res = await fetch("/local-captures?token=" + TOKEN);
        const data = await res.json();
        const list = document.getElementById("logList");
        list.innerHTML = "";
        if (!data.captures.length) { list.textContent = "No captures yet."; return; }
        for (const c of data.captures) {
          const row = document.createElement("div");
          row.className = "log-row";
          row.onclick = () => openLogModal(c);
          const img = document.createElement("img");
          img.src = "data:image/jpeg;base64," + c.image_base64;
          row.appendChild(img);
          const meta = document.createElement("div");
          meta.className = "meta";
          const dims = c.status === "corrected"
            ? c.corrected_length + " x " + c.corrected_width + " x " + c.corrected_height + " in"
            : c.measured_length + " x " + c.measured_width + " x " + c.measured_height + " in";
          meta.innerHTML = '<div class="dims-small">' + dims + '</div><div class="ts">' + new Date(c.captured_at).toLocaleString() + (c.shape_type ? " &middot; " + c.shape_type : "") + '</div>';
          row.appendChild(meta);
          const badge = document.createElement("span");
          badge.className = "badge " + c.status;
          badge.textContent = c.status;
          row.appendChild(badge);
          list.appendChild(row);
        }
      } catch (e) {
        document.getElementById("logList").textContent = "Error loading scan log: " + e;
      }
    }

    function openLogModal(c) {
      const root = document.getElementById("modalRoot");
      root.innerHTML = `
        <div class="modal-overlay" onclick="if(event.target===this) closeLogModal()">
          <div class="modal">
            <h3>Review &amp; Correct</h3>
            <img src="data:image/jpeg;base64,${c.image_base64}" />
            <div class="legend">Captured ${new Date(c.captured_at).toLocaleString()}${c.shape_type ? " &middot; shape: " + c.shape_type : ""}</div>
            <div class="field-row" style="margin-top:10px">
              <div class="field"><label>Length (in)</label><input id="modalLen" type="number" step="0.01" value="${c.status === 'corrected' ? c.corrected_length : c.measured_length}" /></div>
              <div class="field"><label>Width (in)</label><input id="modalWid" type="number" step="0.01" value="${c.status === 'corrected' ? c.corrected_width : c.measured_width}" /></div>
              <div class="field"><label>Height (in)</label><input id="modalHei" type="number" step="0.01" value="${c.status === 'corrected' ? c.corrected_height : c.measured_height}" /></div>
            </div>
            <div class="legend">Camera originally measured: ${c.measured_length} x ${c.measured_width} x ${c.measured_height} in</div>
            <div class="field-row" style="margin-top:12px">
              <button class="secondary" onclick="closeLogModal()" style="flex:1">Close</button>
              <button class="primary" onclick="saveModalCorrection('${c.id}')" style="flex:1">Save Correction</button>
            </div>
            <div class="legend" id="modalStatus"></div>
          </div>
        </div>
      `;
    }

    function closeLogModal() {
      document.getElementById("modalRoot").innerHTML = "";
    }

    async function saveModalCorrection(entryId) {
      const length = parseFloat(document.getElementById("modalLen").value);
      const width = parseFloat(document.getElementById("modalWid").value);
      const height = parseFloat(document.getElementById("modalHei").value);
      const statusEl = document.getElementById("modalStatus");
      statusEl.textContent = "Saving...";
      try {
        await fetch("/local-captures/" + entryId + "/correct", {
          method: "POST",
          headers: { ...headers, "Content-Type": "application/json" },
          body: JSON.stringify({ length, width, height }),
        });
        statusEl.textContent = "Saved.";
        setTimeout(() => { closeLogModal(); renderLogTab(document.getElementById("tabContent")); }, 500);
      } catch (e) { statusEl.textContent = "Error: " + e; }
    }

    renderTabButtons();
    renderTabContent();
  </script>
</body>
</html>
"""
