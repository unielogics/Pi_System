"""
Training Mode: the guided, open-ended calibration loop that replaces the two ad-hoc correction
mechanisms (per-shape "Save Correction" and click-to-select "Recalibrate") this system used to
have. Those two mechanisms conflicted with each other -- see measure.py's _apply_corrections and
api.py's _compound_scale_factor docstrings for the root-cause writeup -- because (a) an additive
shape offset saved while a multiplicative scale factor was active got silently re-multiplied by
whatever scale factor was in effect on every later read, not the one in effect when it was
recorded, and (b) length/width/height are per-item RANK labels (whichever of THAT item's own 3
principal axes is currently biggest/middle/smallest), not fixed camera axes, so sequentially
calibrating against different items could map "length" onto a different physical sensor axis
each time and never converge.

Training Mode fixes both by computing ONE calibration from the WHOLE accumulated sample set at
once (median-based, robust to a rank-crossed outlier) instead of compounding single-item
corrections sequentially, and by deriving per-shape offsets from that SAME fresh scale factor in
the same atomic step (so the ordering bug is structurally impossible).

Per explicit user design (open-ended loop, not fixed phases): a session has ONE active state, not
"collecting" then "verifying" then "completed." An operator scans an item, sees its live
measurement, and marks it 'accurate' (confirms the current reading as ground truth) or
'not_accurate' (enters the true size) -- either way, that becomes one more sample feeding the
next Recalibrate. Recalibrate can be run as many times as the operator wants once >=
MIN_TRAINING_SAMPLES samples exist; each run recomputes from the ENTIRE accumulated sample set
(safe to do repeatedly because `raw_sorted` is captured BEFORE any correction is applied, so it's
calibration-independent -- recalibrating never invalidates a previously-recorded sample). The
operator decides when to Finish (locking in the current calibration) or Cancel (rolling back to
exactly how this device was before this session started), based on how the accurate/not-accurate
marks are trending -- there is no hard system-imposed accuracy gate beyond requiring at least one
completed Recalibrate before Finish is allowed (finishing a session that never actually changed
anything would be misleading).

Same local-JSON-file-with-lock pattern as device_config.py/shape_profiles.py. A single global
session lives on the Pi (one camera = one operator at a time), persisted (not just in-memory)
because this is a multi-step, multi-minute workflow that should survive a page reload -- unlike
the short-TTL in-memory selections/overrides elsewhere in api.py, which only need to survive a
few seconds/minutes of accidental staleness, not a deliberate multi-item session.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

SESSION_PATH = os.path.join(os.path.dirname(__file__), "data", "training_session.json")

# Median-based aggregation (see api.py's /training/recalibrate) only meaningfully rejects outliers
# once N is large enough that 1-2 bad/rank-crossed samples can be outvoted -- at N=3, median just
# discards the min and max and returns the single middle value, with no real averaging benefit.
# At N=5, median discards 2 extremes and needs 2 (not 1) bad samples to break it, giving real
# protection margin against the exact contamination problem this feature exists to solve.
MIN_TRAINING_SAMPLES = 5

# How long a session can sit idle (in 'active') before a NEW /training/start call is allowed to
# auto-abandon it rather than 409. Protects against an operator closing the tab mid-session and
# being unable to ever start a fresh one; long enough that a real, actively-used session (walking
# around placing items) is never mistaken for abandoned.
IDLE_ABANDON_SEC = 30 * 60

# Per explicit user requirement: Finish is blocked until a completed verification round (every
# item re-scanned after the most recent Recalibrate) hits at least this accuracy. "Accuracy" here
# means the percentage of re-scanned items the OPERATOR marked accurate during that round -- a
# human judgment call on each item, not an automatic tolerance check, matching the exact
# accurate/not-accurate marking motion used for the initial scans.
MIN_REQUIRED_ACCURACY_PCT = 90.0

# A verification round always re-queues the operator's MOST RECENT VERIFICATION_ROUND_SIZE
# samples for re-scanning (see api.py's _select_verification_queue), never the whole accumulated
# set -- fitting still uses every sample (more data = better fit), but the physical re-scanning
# workload must not grow unboundedly as samples pile up across rounds (round 1 = 5 items, round 2
# = 10, round 3 = 15+... an escalating busywork trap a user explicitly flagged live). Deliberately
# its own constant, not an alias for MIN_TRAINING_SAMPLES, even though it starts at the same
# value, so it can diverge later without looking like a change to the fitting-statistics
# threshold.
VERIFICATION_ROUND_SIZE = 5

# Per explicit user requirement: once a round closes below MIN_REQUIRED_ACCURACY_PCT, the system
# auto-refits and opens the next round immediately -- no manual Recalibrate click needed to keep
# progressing. But that auto-loop must not run forever against a sample set that genuinely can't
# reach 90% (e.g. all-identical items, or geometry outside this device's calibratable range) --
# capped at this many CONSECUTIVE auto-triggered rounds before the system stops and asks the
# operator to scan more/different items, then manually Recalibrate to try again (which resets the
# counter once that manual attempt's own round passes -- see TrainingSession.
# consecutive_auto_recalibrate_failures).
MAX_CONSECUTIVE_AUTO_RECALIBRATIONS = 3

_SESSION_LOCK = threading.Lock()


@dataclass
class TrainingSample:
    index: int
    shape_type: str
    verdict: str  # 'accurate' | 'not_accurate'
    true_sorted: list  # [length, width, height] -- ground truth for this sample
    measured_sorted: list  # what the system reported (fully corrected) at capture time
    raw_sorted: list  # PRE shape-offset/scale-factor extents (inches) -- calibration-independent
    delta_in: list  # true_sorted - measured_sorted, informational (always [0,0,0] for 'accurate')
    detection_id: int
    captured_at: str
    # True if the operator's manually-selected shape_type (above) disagreed with the geometry-
    # derived suggestion measure() offered for this scan (see measure.py's suggested_shape_type) --
    # None for samples recorded before this field existed, or when no suggestion was available.
    # Tracked purely as a signal: frequent overrides would indicate the geometry-based suggestion
    # is unreliable and a real image classifier might be worth revisiting; nothing so far indicates
    # that's the case.
    shape_type_overridden: Optional[bool] = None


@dataclass
class TrainingSession:
    id: str
    status: str  # 'active' | 'completed' | 'cancelled'
    started_at: str
    last_activity_at: str
    completed_at: Optional[str] = None
    samples: list = field(default_factory=list)  # list[TrainingSample]
    # Snapshot taken at /training/start, for a full-session rollback regardless of how many times
    # /training/recalibrate ran in between -- Cancel always reverts to "before this session",
    # never to some intermediate recalibration.
    initial_scale_factor: Optional[dict] = None
    initial_shape_profiles: dict = field(default_factory=dict)  # {shape_type: profile_dict}
    # The most recently applied result -- {appliedScaleFactor, shapeOffsets, computedAt,
    # sampleCount}. None until the first /training/recalibrate call; Finish requires this to be
    # set (can't finish a session that never actually calibrated anything).
    computed_result: Optional[dict] = None
    recalibration_count: int = 0
    # Per explicit user requirement: every /training/recalibrate opens a VERIFICATION ROUND --
    # every sample used in that recalibration must be re-scanned and marked accurate/not-accurate
    # again, under the calibration that was just applied, before Finish is allowed. This proves
    # the fix actually worked rather than just trusting the math. `round_number` starts at 0 (no
    # round yet); `verification_queue` holds the sample indices still pending re-verification THIS
    # round (empty = round complete or none started); `round_results` holds this round's outcomes
    # so far; `last_round_accuracy_pct` is None until a round fully completes, then holds that
    # round's percentage of re-scans marked accurate -- Finish requires this to be
    # >= MIN_REQUIRED_ACCURACY_PCT AND the queue to be empty (nothing left half-verified).
    round_number: int = 0
    verification_queue: list = field(default_factory=list)  # list[int] sample indices
    round_results: list = field(default_factory=list)  # list[dict]
    last_round_accuracy_pct: Optional[float] = None
    # Append-only log of every round this session has closed (manually or auto-triggered),
    # {round_number, accuracy_pct, sample_count, computed_at} -- unlike last_round_accuracy_pct
    # (overwritten every round) and round_results (reset to [] every recalibrate), this is never
    # overwritten, so the operator can see round-over-round trend ("Round 1: 60% -> Round 2: 80% ->
    # Round 3: 100%"), not just the latest number with no context on whether it's improving.
    round_accuracy_history: list = field(default_factory=list)  # list[dict]
    # How many CONSECUTIVE times the auto-recalibrate-on-failed-round trigger has fired in a row
    # (see MAX_CONSECUTIVE_AUTO_RECALIBRATIONS) -- reset to 0 whenever a round PASSES, incremented
    # each time the auto-trigger opens a new round after a failing one. Once this hits the cap, a
    # failing round closes normally but no longer auto-reopens the next one; the operator must scan
    # more/different items and manually Recalibrate.
    consecutive_auto_recalibrate_failures: int = 0


def _read_unlocked() -> Optional[TrainingSession]:
    """Raw read, assumes the caller already holds _SESSION_LOCK. Never call directly outside
    this module."""
    if not os.path.exists(SESSION_PATH):
        return None
    import json

    with open(SESSION_PATH, "r") as f:
        data = json.load(f)
    if data is None:
        return None
    samples = [TrainingSample(**s) for s in data.get("samples", [])]
    return TrainingSession(
        id=data["id"],
        status=data["status"],
        started_at=data["started_at"],
        last_activity_at=data["last_activity_at"],
        completed_at=data.get("completed_at"),
        samples=samples,
        initial_scale_factor=data.get("initial_scale_factor"),
        initial_shape_profiles=data.get("initial_shape_profiles", {}),
        computed_result=data.get("computed_result"),
        recalibration_count=data.get("recalibration_count", 0),
        round_number=data.get("round_number", 0),
        verification_queue=data.get("verification_queue", []),
        round_results=data.get("round_results", []),
        last_round_accuracy_pct=data.get("last_round_accuracy_pct"),
        round_accuracy_history=data.get("round_accuracy_history", []),
        consecutive_auto_recalibrate_failures=data.get("consecutive_auto_recalibrate_failures", 0),
    )


def _write_unlocked(session: Optional[TrainingSession]) -> None:
    """Raw write, assumes the caller already holds _SESSION_LOCK. Never call directly outside
    this module. Atomic temp-file+os.replace, matching device_config.py/shape_profiles.py."""
    import json

    directory = os.path.dirname(SESSION_PATH)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{SESSION_PATH}.tmp-{os.getpid()}"
    payload = asdict(session) if session is not None else None
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, SESSION_PATH)


def load_training_session() -> Optional[TrainingSession]:
    with _SESSION_LOCK:
        return _read_unlocked()


def update_training_session(mutate: "Callable[[Optional[TrainingSession]], Optional[TrainingSession]]") -> Optional[TrainingSession]:
    """Atomic read-modify-write, same pattern as device_config.py's update_device_config /
    shape_profiles.py's update_shape_profiles. `mutate` receives the current session (or None if
    none exists) and returns the session to persist (or None to clear it) -- unlike the other two
    modules' in-place-mutation callbacks, this one returns a value because a session can be
    replaced wholesale (e.g. /training/start creating a brand-new one) or cleared entirely, not
    just field-patched."""
    with _SESSION_LOCK:
        current = _read_unlocked()
        new_session = mutate(current)
        _write_unlocked(new_session)
        return new_session


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
