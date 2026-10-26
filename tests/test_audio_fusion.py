"""Acoustic labels, fusion weights, and teaching advisories."""

import pandas as pd

from src.advisory import generate_advisories
from src.audio_dynamics import (
    DISCUSSION,
    DISTRACTION,
    LECTURE,
    SILENCE,
    extract_and_classify,
    score_for_label,
)
from src.fusion import AUDIO_WEIGHT, VISUAL_WEIGHT, fuse
from src.synthetic_audio import room_mix


def test_synthetic_room_mix_labels():
    waveform, sample_rate, segments = room_mix()
    classified = extract_and_classify(waveform, sample_rate)
    mismatches = []
    for start, end, label in segments:
        chunk = classified[(classified["t_start"] >= start + 0.5) & (classified["t_end"] <= end - 0.5)]
        assert not chunk.empty, label
        predicted = chunk["label"].mode().iloc[0]
        if predicted != label:
            preview = chunk[["t_start", "label", "rms", "flatness", "zcr", "sub_cv", "flux"]].head(3)
            mismatches.append(f"{label} -> {predicted}\n{preview}")
    assert not mismatches, "\n".join(mismatches)


def test_silence_is_not_scored_as_absence():
    audio = pd.DataFrame(
        {
            "t_start": [0.0, 1.0, 2.0],
            "t_end": [1.0, 2.0, 3.0],
            "label": [SILENCE, SILENCE, SILENCE],
            "score": [score_for_label(SILENCE)] * 3,
        }
    )
    timeline = fuse(None, audio, 3.0, bin_s=1.0)
    assert timeline["engagement"].between(40, 60).all()
    assert set(timeline["w_audio"]) == {1.0}


def test_fusion_weights_and_distraction():
    visual = pd.DataFrame({"t": [5.0], "attention": [1.0], "faces": [12]})
    audio = pd.DataFrame(
        {
            "t_start": [0.0],
            "t_end": [10.0],
            "label": [LECTURE],
            "score": [score_for_label(LECTURE)],
        }
    )
    timeline = fuse(visual, audio, 10.0, bin_s=10.0)
    expected = 100.0 * (VISUAL_WEIGHT * 1.0 + AUDIO_WEIGHT * score_for_label(LECTURE))
    assert abs(float(timeline["engagement_raw"].iloc[0]) - expected) < 0.2
    assert 0 <= float(timeline["engagement"].iloc[0]) <= 100

    slumped = pd.DataFrame({"t": [5.0], "attention": [0.2], "faces": [12]})
    noise = pd.DataFrame(
        {
            "t_start": [0.0],
            "t_end": [10.0],
            "label": [DISTRACTION],
            "score": [score_for_label(DISTRACTION)],
        }
    )
    low = fuse(slumped, noise, 10.0, bin_s=10.0)
    assert float(low["engagement"].iloc[0]) < 40

    visual_only = fuse(visual, None, 10.0, bin_s=10.0)
    assert abs(float(visual_only["engagement_raw"].iloc[0]) - 100.0) < 0.2
    assert float(visual_only["w_visual"].iloc[0]) == 1.0


def test_low_stretch_recommends_a_poll():
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
            }
        )
    advisories = generate_advisories(pd.DataFrame(rows), alert_level=45, alert_minutes=5)
    assert any("quick poll" in item.message for item in advisories)
    assert all("P-0" not in item.message for item in advisories)


def test_long_lecture_slide_is_a_check_for_understanding():
    rows = []
    for index in range(15):
        engagement = 80.0 - (80.0 - 58.0) * (index / 14.0)
        rows.append(
            {
                "t_start": index * 60.0,
                "t_end": (index + 1) * 60.0,
                "t_mid": index * 60.0 + 30.0,
                "engagement": engagement,
                "audio_label": LECTURE,
            }
        )
    advisories = generate_advisories(pd.DataFrame(rows), alert_level=45, alert_minutes=5)
    assert any("check for understanding" in item.message for item in advisories)
    assert not any("quick poll" in item.message for item in advisories)
    assert DISCUSSION == "Student Discussion / Verbal Interaction"
