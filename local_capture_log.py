"""
Local capture history for the standalone Pi viewer's Scan Log tab.

The Pi has no access to the WMS backend's DimensionCaptureLog (no auth to it, no
zone/warehouse context) -- when the dashboard drives a capture, IT persists the result to that
backend log itself (see UnieBackend's POST /dimension-capture/logs). This file exists so the
standalone Pi-local /viewer page has an equivalent "recent captures, click to correct, see the
picture" experience on its own, without depending on the backend at all -- same local-JSON-file
pattern as products.py/shape_profiles.py. Capped ring buffer so this never grows unbounded.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "local_capture_log.json")
MAX_ENTRIES = 100


@dataclass
class CaptureLogEntry:
    id: str
    captured_at: str
    measured_length: float
    measured_width: float
    measured_height: float
    measured_cubic_feet: float
    classification: str
    confidence: float
    shape_type: Optional[str]
    image_base64: str
    corrected_length: Optional[float] = None
    corrected_width: Optional[float] = None
    corrected_height: Optional[float] = None
    status: str = "measured"  # 'measured' | 'corrected'


def _load_all() -> list[CaptureLogEntry]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        data = json.load(f)
    return [CaptureLogEntry(**item) for item in data]


def _save_all(entries: list[CaptureLogEntry]) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump([asdict(e) for e in entries], f)


def list_entries() -> list[CaptureLogEntry]:
    return _load_all()


def add_entry(
    captured_at: str,
    measured_length: float,
    measured_width: float,
    measured_height: float,
    measured_cubic_feet: float,
    classification: str,
    confidence: float,
    shape_type: Optional[str],
    image_base64: str,
) -> CaptureLogEntry:
    entries = _load_all()
    entry = CaptureLogEntry(
        id=uuid.uuid4().hex[:12],
        captured_at=captured_at,
        measured_length=measured_length,
        measured_width=measured_width,
        measured_height=measured_height,
        measured_cubic_feet=measured_cubic_feet,
        classification=classification,
        confidence=confidence,
        shape_type=shape_type,
        image_base64=image_base64,
    )
    entries.append(entry)
    # Ring buffer: keep only the most recent MAX_ENTRIES.
    entries = entries[-MAX_ENTRIES:]
    _save_all(entries)
    return entry


def correct_entry(
    entry_id: str, corrected_length: float, corrected_width: float, corrected_height: float
) -> Optional[CaptureLogEntry]:
    entries = _load_all()
    for entry in entries:
        if entry.id == entry_id:
            entry.corrected_length = corrected_length
            entry.corrected_width = corrected_width
            entry.corrected_height = corrected_height
            entry.status = "corrected"
            _save_all(entries)
            return entry
    return None
