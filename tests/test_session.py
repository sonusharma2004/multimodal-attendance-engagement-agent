"""Simulated period, report privacy guarantees, and a blank-video smoke test."""

import json

import numpy as np
import pytest

from src.models import Settings
from src.pipeline import load_demo_session, session_from_bundle
from src.demo_session import make_demo_bundle


def test_demo_period_marks_presence_and_asks_for_a_poll():
    session = load_demo_session()
    present = int((session.attendance["Status"] == "Present").sum())
    absent = session.attendance[session.attendance["Status"] == "Absent"]
    assert present == 23
    assert len(absent) == 2
    assert int(absent["Notes"].str.contains("Brief appearance").sum()) == 1
    assert session.kpis["roster_size"] == 24
    engagement = session.timeline["engagement"]
    assert engagement.between(0, 100).all()
    assert float(engagement.min()) < 45
    assert float(engagement.max()) > 75
    messages = [item.message for item in session.advisories]
    assert any("quick poll" in message for message in messages), messages
    assert "Attentiveness" not in session.attendance.columns

    looser = session_from_bundle(make_demo_bundle(), Settings(presence_threshold=0.50))
    assert int((looser.attendance["Status"] == "Present").sum()) == 24
    assert int((looser.attendance["Status"] == "Absent").sum()) == 1


def test_report_is_collective_and_offline():
    session = load_demo_session()
    text = session.report_markdown
    assert "never transcribed" in text
    assert "not a disciplinary tool" in text
    assert "facial recognition" in text
    payload = json.loads(session.report_json)
    assert payload["schema"] == "mea.session.v1"
    assert payload["privacy"]["speech_to_text"] is False
    assert payload["privacy"]["facial_recognition"] is False
    assert payload["privacy"]["individual_engagement_grades"] is False
    assert payload["attendance"]
    assert "image" not in json.dumps(payload["snapshot"])
    joined = " ".join(item["message"] for item in payload["advisories"])
    assert "P-01" not in joined


def test_blank_video_does_not_invent_attendance(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.attendance import analyze_video
    from src.face_engine import FaceEngine

    path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(12):
        writer.write(frame)
    writer.release()
    engine = FaceEngine()
    if engine.backend == "unavailable":
        pytest.skip(f"no face detector: {engine.load_error}")
    result = analyze_video(str(path), engine, max_samples=8)
    assert result["attendance"].empty or len(result["attendance"]) == 0
    assert result["duration_s"] > 0


def test_mediapipe_backend_when_model_is_bundled():
    from src.face_engine import MODEL_PATH, FaceEngine

    if not MODEL_PATH.is_file():
        pytest.skip("blaze face model is not in the repo")
    engine = FaceEngine()
    assert engine.backend == "mediapipe", engine.load_error
    blank = np.zeros((180, 240, 3), dtype=np.uint8)
    found = engine.detect(blank)
    assert isinstance(found, list)
