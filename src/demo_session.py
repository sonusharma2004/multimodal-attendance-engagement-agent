"""A scripted class period for the dashboard.

The sensor timelines are simulated and labeled as such. Sustained-presence
marks, fusion, and advisories are the same functions used for an upload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.audio_dynamics import DISCUSSION, DISTRACTION, LECTURE, SILENCE, score_for_label
from src.models import Bundle

DURATION_S = 42 * 60
BIN_S = 10.0


def make_demo_bundle() -> Bundle:
    attendance = _attendance()
    visual, audio = _signals()
    narrative = (
        "Simulated period, not a recording. Twenty-four students are on the roll. "
        "One arrives a few minutes late and stays. One leaves at minute 26, under the 70% line. "
        "One hallway pass-through is visible for about 50 seconds and is not marked present. "
        "Fusion, the sustained-presence rule, and the advisory engine are the same code that runs on an upload. "
        "The acoustic self-check elsewhere on this page is a separate, real signal test."
    )
    return Bundle(
        title="Period 3 · Linear equations",
        source="Simulated period",
        narrative=narrative,
        duration_s=float(DURATION_S),
        attendance=attendance,
        visual=visual,
        audio=audio,
        warnings=[],
        pose_method="Simulated attentiveness signal (no camera)",
        modality="simulated",
        roster_size=24,
        annotated_frames=[],
        snapshot=None,
        stats={
            "samples": int(DURATION_S / BIN_S),
            "sample_interval_s": BIN_S,
            "detector": "simulated",
            "note": "Scripted 10-second bins. No frames were captured.",
        },
    )


def _attendance() -> pd.DataFrame:
    duration = float(DURATION_S)
    rows = []
    for index in range(1, 23):
        ratio = 0.93 + (index % 5) * 0.014
        rows.append(_row(index, 0.0, duration, ratio * duration, ratio))
    late_visible = duration - 3 * 60
    rows.append(_row(23, 3 * 60, duration, late_visible, late_visible / duration))
    early_visible = 26 * 60
    rows.append(_row(24, 0.0, early_visible, early_visible, early_visible / duration))
    rows.append(_row(25, 11 * 60, 11 * 60 + 50, 50.0, 50.0 / duration))
    return pd.DataFrame(rows)


def _row(number: int, first_s: float, last_s: float, visible_s: float, ratio: float) -> dict:
    student_id = f"P-{number:02d}"
    return {
        "Student ID": student_id,
        "first_s": float(first_s),
        "last_s": float(last_s),
        "visible_s": float(visible_s),
        "Presence Ratio": float(ratio),
    }


def _signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    n_bins = int(DURATION_S / BIN_S)
    visual_rows = []
    audio_rows = []
    for index in range(n_bins):
        t0 = index * BIN_S
        t1 = t0 + BIN_S
        tmin = t0 / 60.0
        if tmin < 2:
            label = SILENCE
            visual = 0.70
            faces = 23
        elif tmin < 15:
            label = LECTURE
            visual = 0.90 - ((tmin - 2) / 13.0) * 0.42
            faces = 23 if tmin < 3 else 24
        elif tmin < 22:
            label = DISTRACTION
            visual = 0.30
            faces = 25 if 11 <= tmin < 11.9 else 24
        elif tmin < 30:
            label = DISCUSSION if index % 5 else LECTURE
            visual = 0.84
            faces = 24 if tmin < 26 else 23
        elif tmin < 36:
            label = LECTURE if index % 4 else DISCUSSION
            visual = 0.74
            faces = 23
        else:
            label = SILENCE if index % 2 == 0 else LECTURE
            visual = 0.77
            faces = 23
        visual = float(np.clip(visual + 0.012 * np.sin(index / 3.0), 0.0, 1.0))
        visual_rows.append({"t": t0 + BIN_S / 2.0, "attention": visual, "faces": faces})
        audio_rows.append(
            {
                "t_start": t0,
                "t_end": t1,
                "label": label,
                "score": score_for_label(label),
                "rms": _rms_hint(label),
                "zcr": 0.0,
                "flatness": 0.8 if label == DISTRACTION else 0.05,
                "centroid_norm": 0.2,
                "sub_cv": 0.5 if label == DISCUSSION else 0.05,
                "flux": 0.0,
            }
        )
    return pd.DataFrame(visual_rows), pd.DataFrame(audio_rows)


def _rms_hint(label: str) -> float:
    return {
        SILENCE: 0.004,
        LECTURE: 0.12,
        DISCUSSION: 0.16,
        DISTRACTION: 0.2,
    }.get(label, 0.0)
