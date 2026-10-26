"""Sustained presence, pose geometry, and the attendance register."""

import numpy as np
import pandas as pd

from src.attendance import build_register, format_attendance, note_for
from src.face_engine import apply_forward_flags, is_forward, pose_from_points
from src.models import CSV_COLUMNS, DISPLAY_COLUMNS


def _det(bbox, yaw=0.0, pitch=0.0):
    return {
        "bbox": bbox,
        "yaw": yaw,
        "pitch": pitch,
        "forward": True,
        "pose_known": True,
        "confidence": 0.9,
    }


def test_full_presence_and_hallway_passthrough():
    samples = []
    for index in range(100):
        dets = [_det((80, 80, 40, 40))]
        if index == 4:
            dets.append(_det((420, 70, 36, 36)))
        samples.append({"t": float(index), "dets": dets})
    raw = build_register(samples, 100.0, presence_bridge_s=15.0, identity_gap_s=180.0)
    table = format_attendance(raw, 0.70)
    present = table[table["Status"] == "Present"]
    absent = table[table["Status"] == "Absent"]
    assert len(present) == 1
    assert len(absent) == 1
    assert float(present["Presence Ratio"].iloc[0]) >= 0.99
    assert float(absent["Presence Ratio"].iloc[0]) < 0.05
    assert "Brief appearance" in absent["Notes"].iloc[0]
    assert set(DISPLAY_COLUMNS).issubset(table.columns)
    assert "Attentiveness" not in table.columns
    assert "Engagement" not in table.columns


def test_short_gap_is_bridged_and_long_gap_is_one_track():
    samples = []
    for index in range(100):
        if index == 50:
            samples.append({"t": float(index), "dets": []})
        else:
            samples.append({"t": float(index), "dets": [_det((80, 80, 40, 40))]})
    bridged = build_register(samples, 100.0, presence_bridge_s=15.0, identity_gap_s=900.0)
    assert len(bridged) == 1
    assert float(bridged["Presence Ratio"].iloc[0]) >= 0.99

    gapped = []
    for index in range(100):
        if 40 <= index <= 60:
            gapped.append({"t": float(index), "dets": []})
        else:
            gapped.append({"t": float(index), "dets": [_det((30, 30, 40, 40))]})
    raw = build_register(gapped, 100.0, presence_bridge_s=15.0, identity_gap_s=900.0)
    assert len(raw) == 1
    ratio = float(raw["Presence Ratio"].iloc[0])
    assert 0.7 < ratio < 0.9


def test_distant_reentry_does_not_inherit_an_id():
    samples = []
    for index in range(100):
        if index <= 20:
            dets = [_det((10, 10, 40, 40))]
        elif index >= 40:
            dets = [_det((520, 10, 40, 40))]
        else:
            dets = []
        samples.append({"t": float(index), "dets": dets})
    raw = build_register(samples, 100.0, presence_bridge_s=15.0, identity_gap_s=900.0)
    assert list(raw["Student ID"]) == ["P-01", "P-02"]
    assert float(raw["Presence Ratio"].iloc[0]) < 0.35
    assert float(raw["Presence Ratio"].iloc[1]) > 0.5


def test_threshold_changes_status_without_changing_ratio():
    raw = pd.DataFrame(
        [
            {
                "Student ID": "P-01",
                "first_s": 0.0,
                "last_s": 100.0,
                "visible_s": 62.0,
                "Presence Ratio": 0.62,
            }
        ]
    )
    low = format_attendance(raw, 0.50)
    high = format_attendance(raw, 0.70)
    assert low["Status"].iloc[0] == "Present"
    assert high["Status"].iloc[0] == "Absent"
    assert "Brief appearance" not in high["Notes"].iloc[0]
    assert list(CSV_COLUMNS) == list(high.columns)


def test_note_for_brief_appearance():
    assert "Brief appearance" in note_for(0.02, 40, 0.70)
    assert "Sustained presence met" in note_for(0.8, 800, 0.70)


def test_frontal_down_and_turned_pose():
    yaw, pitch = pose_from_points((0.0, 0.0), (40.0, 0.0), (20.0, 18.0))
    assert abs(yaw) < 1.5
    assert abs(pitch) < 1.5
    assert is_forward(yaw, pitch)

    yaw, pitch = pose_from_points((0.0, 0.0), (40.0, 0.0), (20.0, 32.0))
    assert pitch > 22
    assert not is_forward(yaw, pitch)

    yaw, pitch = pose_from_points((0.0, 0.0), (40.0, 0.0), (36.0, 18.0))
    assert abs(yaw) > 25
    assert not is_forward(yaw, pitch)


def test_pose_baseline_follows_the_camera():
    samples = [{"t": float(i), "dets": [{"yaw": 1.0, "pitch": 15.0}]} for i in range(10)]
    samples.append({"t": 10.0, "dets": [{"yaw": 0.0, "pitch": 40.0}]})
    baseline = apply_forward_flags(samples)
    assert abs(baseline - 15.0) < 1e-6
    assert samples[0]["dets"][0]["forward"] is True
    assert samples[-1]["dets"][0]["forward"] is False
    assert np.isfinite(baseline)
