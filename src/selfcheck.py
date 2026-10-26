"""Offline checks the dashboard can run without a classroom recording."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.advisory import generate_advisories
from src.attendance import build_register, format_attendance
from src.audio_dynamics import DISTRACTION, LECTURE, SILENCE, extract_and_classify, score_for_label
from src.fusion import fuse
from src.synthetic_audio import room_mix


def library_versions() -> dict[str, str]:
    versions = {"python": _python()}
    versions["numpy"] = _dist("numpy")
    versions["pandas"] = _dist("pandas")
    versions["matplotlib"] = _dist("matplotlib")
    versions["streamlit"] = _dist("streamlit")
    versions["librosa"] = _dist("librosa")
    versions["mediapipe"] = _dist("mediapipe")
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except Exception as exc:
        versions["opencv"] = f"unavailable ({exc.__class__.__name__})"
    return versions


def run_selfcheck(engine=None) -> list[dict]:
    checks = [
        _check_presence(),
        _check_audio(),
        _check_silence_is_not_zero(),
        _check_advisory(),
        _check_detector(engine),
    ]
    return checks


def _check_presence() -> dict:
    samples = []
    for index in range(100):
        dets = [_det((80, 80, 40, 40))]
        if index == 4:
            dets.append(_det((420, 70, 36, 36)))
        samples.append({"t": float(index), "dets": dets})
    raw = build_register(samples, 100.0, presence_bridge_s=15.0, identity_gap_s=180.0)
    table = format_attendance(raw, 0.70)
    present = int((table["Status"] == "Present").sum())
    absent = table[table["Status"] == "Absent"]
    brief = int(absent["Notes"].str.contains("Brief appearance").sum()) if not absent.empty else 0
    passed = present == 1 and len(absent) == 1 and brief == 1
    return {
        "name": "Sustained presence rejects a one-frame pass-through",
        "passed": passed,
        "detail": f"{present} present, {len(absent)} below threshold, {brief} brief.",
    }


def _check_audio() -> dict:
    waveform, sr, segments = room_mix()
    try:
        classified = extract_and_classify(waveform, sr)
    except Exception as exc:
        return {"name": "Acoustic classifier", "passed": False, "detail": str(exc)}
    mismatches = []
    for start, end, label in segments:
        chunk = classified[(classified["t_start"] >= start + 0.5) & (classified["t_end"] <= end - 0.5)]
        if chunk.empty:
            mismatches.append(f"{label}: no windows")
            continue
        predicted = chunk["label"].mode().iloc[0]
        if predicted != label:
            mismatches.append(f"{label} classified as {predicted}")
    return {
        "name": "Acoustic classifier on a synthetic room mix",
        "passed": not mismatches,
        "detail": "All four scripted blocks matched." if not mismatches else "; ".join(mismatches),
    }


def _check_silence_is_not_zero() -> dict:
    audio = pd.DataFrame(
        {
            "t_start": [0.0, 1.0, 2.0],
            "t_end": [1.0, 2.0, 3.0],
            "label": [SILENCE, SILENCE, SILENCE],
            "score": [score_for_label(SILENCE)] * 3,
        }
    )
    timeline = fuse(None, audio, 3.0, bin_s=1.0)
    mean_engagement = float(timeline["engagement"].mean())
    passed = 40.0 <= mean_engagement <= 60.0
    return {
        "name": "A quiet room is not scored as zero",
        "passed": passed,
        "detail": f"Mean engagement during silence was {mean_engagement:.1f}.",
    }


def _check_advisory() -> dict:
    rows = []
    for index in range(12):
        low = 2 <= index <= 8
        rows.append(
            {
                "t_start": index * 60.0,
                "t_end": (index + 1) * 60.0,
                "t_mid": index * 60.0 + 30.0,
                "engagement": 30.0 if low else 80.0,
                "audio_label": DISTRACTION if low else LECTURE,
                "visual": 0.3 if low else 0.8,
                "audio_score": 0.22 if low else 0.74,
                "faces": 12,
            }
        )
    advisories = generate_advisories(pd.DataFrame(rows), alert_level=45, alert_minutes=5)
    passed = any("quick poll" in item.message for item in advisories)
    titles = ", ".join(item.title for item in advisories) or "none"
    return {
        "name": "Low-engagement stretch recommends a poll, talk, or stretch",
        "passed": passed,
        "detail": titles,
    }


def _check_detector(engine) -> dict:
    if engine is None:
        return {"name": "Face detector", "passed": False, "detail": "Detector was not loaded."}
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        found = engine.detect(blank)
    except Exception as exc:
        return {"name": "Face detector", "passed": False, "detail": str(exc)}
    return {
        "name": "Face detector runs on a blank frame",
        "passed": isinstance(found, list),
        "detail": f"Backend {engine.backend}. Faces on a blank frame: {len(found)}.",
    }


def _det(bbox: tuple[int, int, int, int]) -> dict:
    return {"bbox": bbox, "yaw": 0.0, "pitch": 0.0, "forward": True, "pose_known": True, "confidence": 0.9}


def _python() -> str:
    import sys

    return sys.version.split()[0]


def _dist(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"
