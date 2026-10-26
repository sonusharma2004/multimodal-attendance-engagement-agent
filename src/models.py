"""Session objects shared by the pipeline and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

PRESENT = "Present"
ABSENT = "Absent"

DISPLAY_COLUMNS = [
    "Student ID",
    "First Seen",
    "Last Seen",
    "Total Duration",
    "Status",
    "Presence",
    "Notes",
]

CSV_COLUMNS = DISPLAY_COLUMNS + ["Presence Ratio", "Visible Seconds"]


@dataclass
class Settings:
    presence_threshold: float = 0.70
    alert_level: float = 45.0
    alert_minutes: float = 5.0
    expected_class_size: int | None = None


@dataclass
class Advisory:
    t_start: float
    t_end: float
    severity: str
    title: str
    message: str
    rationale: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)


@dataclass
class Bundle:
    """Raw signals, before thresholds and advisories are applied."""

    title: str
    source: str
    narrative: str
    duration_s: float
    attendance: pd.DataFrame
    visual: pd.DataFrame | None
    audio: pd.DataFrame | None
    warnings: list[str]
    pose_method: str
    modality: str
    roster_size: int | None
    annotated_frames: list[dict[str, Any]] = field(default_factory=list)
    snapshot: dict[str, Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    title: str
    source: str
    narrative: str
    duration_s: float
    modality: str
    pose_method: str
    attendance: pd.DataFrame
    timeline: pd.DataFrame
    audio: pd.DataFrame | None
    advisories: list[Advisory]
    kpis: dict[str, Any]
    warnings: list[str]
    threshold: float
    alert_level: float
    alert_minutes: float
    roster_size: int | None
    annotated_frames: list[dict[str, Any]]
    snapshot: dict[str, Any] | None
    stats: dict[str, Any]
    generated_at: datetime
    report_markdown: str
    report_json: str
