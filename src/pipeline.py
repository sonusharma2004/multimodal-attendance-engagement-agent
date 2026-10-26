"""Turn raw signals or uploaded media into one session the dashboard can render."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.advisory import generate_advisories
from src.attendance import analyze_image, analyze_video, format_attendance
from src.audio_dynamics import AudioDecodeProxy, extract_and_classify, load_audio_file
from src.demo_session import make_demo_bundle
from src.errors import AudioReadError, PipelineError
from src.formatutil import now_ist
from src.fusion import fuse
from src.models import Bundle, Session, Settings
from src.reporting import compute_kpis, render_json, render_markdown


def load_demo_session(settings: Settings | None = None) -> Session:
    return session_from_bundle(make_demo_bundle(), settings or Settings())


def session_from_bundle(bundle: Bundle, settings: Settings) -> Session:
    warnings = list(bundle.warnings)
    roster = settings.expected_class_size or bundle.roster_size
    attendance = format_attendance(bundle.attendance, settings.presence_threshold)
    timeline = fuse(bundle.visual, bundle.audio, bundle.duration_s)
    has_index = not timeline.empty and bool(timeline["engagement"].notna().any())
    if bundle.duration_s > 0 and not has_index:
        warnings.append("No visual or acoustic signal was available, so there is no engagement index.")
    single_modality = _single_modality_warning(bundle, timeline)
    if single_modality:
        warnings.append(single_modality)
    advisories = generate_advisories(
        timeline,
        alert_level=settings.alert_level,
        alert_minutes=settings.alert_minutes,
    )
    parts = {
        "attendance": attendance,
        "timeline": timeline,
        "duration_s": bundle.duration_s,
        "alert_level": settings.alert_level,
        "roster_size": roster,
        "advisories": advisories,
    }
    kpis = compute_kpis(parts)
    session = Session(
        title=bundle.title,
        source=bundle.source,
        narrative=bundle.narrative,
        duration_s=float(bundle.duration_s or 0.0),
        modality=bundle.modality,
        pose_method=bundle.pose_method,
        attendance=attendance,
        timeline=timeline,
        audio=None if bundle.audio is None else bundle.audio.copy(),
        advisories=advisories,
        kpis=kpis,
        warnings=warnings,
        threshold=float(settings.presence_threshold),
        alert_level=float(settings.alert_level),
        alert_minutes=float(settings.alert_minutes),
        roster_size=int(roster) if roster else None,
        annotated_frames=bundle.annotated_frames,
        snapshot=bundle.snapshot,
        stats=dict(bundle.stats),
        generated_at=now_ist(),
        report_markdown="",
        report_json="",
    )
    session.report_markdown = render_markdown(session)
    session.report_json = render_json(session)
    return session


def ingest_media(
    *,
    video_path: str | None = None,
    audio_path: str | None = None,
    image_bgr=None,
    video_name: str = "video",
    audio_name: str = "audio",
    image_name: str = "still",
    engine=None,
    progress=None,
) -> Bundle:
    if engine is None:
        from src.face_engine import FaceEngine

        engine = FaceEngine()
    if not any([video_path, audio_path, image_bgr is not None]):
        raise PipelineError("Add a classroom video, a soundtrack, or a still frame first.")

    warnings: list[str] = []
    visual = None
    audio = None
    attendance = _empty_raw_attendance()
    annotated: list[dict] = []
    snapshot = None
    stats: dict[str, Any] = {}
    duration = 0.0
    pose_method = engine.pose_method
    title_bits = []

    if video_path and image_bgr is not None:
        warnings.append("A video was uploaded, so the still frame was not used for attendance.")
        image_bgr = None

    if video_path:
        try:
            video = analyze_video(video_path, engine, progress=progress)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Video processing stopped: {exc}") from exc
        attendance = video["attendance"]
        visual = video["visual"]
        annotated = video["annotated_frames"]
        warnings.extend(video["warnings"])
        pose_method = video["pose_method"]
        stats.update(video["stats"])
        duration = max(duration, float(video["duration_s"]))
        title_bits.append(video_name)

    if image_bgr is not None:
        try:
            snapshot = analyze_image(image_bgr, engine)
        except Exception as exc:
            raise PipelineError(f"The still frame could not be read: {exc}") from exc
        pose_method = snapshot["pose_method"]
        stats["detector"] = engine.backend
        stats["snapshot_faces"] = snapshot["faces"]
        title_bits.append(image_name)
        if snapshot["faces"] == 0:
            warnings.append("No faces were found in the still. Try a brighter frame with faces large enough to see.")

    if audio_path:
        try:
            waveform, sample_rate, audio_warnings = load_audio_file(audio_path)
        except AudioDecodeProxy as exc:
            raise AudioReadError(
                "That soundtrack could not be decoded. WAV is the most reliable format. "
                f"Detail: {exc}"
            ) from exc
        except Exception as exc:
            raise AudioReadError(f"That soundtrack could not be decoded. Detail: {exc}") from exc
        warnings.extend(audio_warnings)
        try:
            audio = extract_and_classify(waveform, sample_rate)
        except Exception as exc:
            raise PipelineError(f"Acoustic analysis stopped: {exc}") from exc
        audio_duration = float(len(waveform) / sample_rate)
        stats["audio_windows"] = int(len(audio))
        stats["audio_duration_s"] = audio_duration
        if duration and abs(duration - audio_duration) > 0.1 * max(duration, audio_duration):
            warnings.append(
                "The video and the soundtrack have different lengths. Fusion uses a shared clock from the "
                "start of each file. After the shorter one ends, the index follows the signal that remains."
            )
        duration = max(duration, audio_duration)
        title_bits.append(audio_name)

    if video_path and audio_path:
        modality = "video+audio"
    elif video_path:
        modality = "video"
    elif audio_path and snapshot is not None:
        modality = "still+audio"
    elif audio_path:
        modality = "audio"
    else:
        modality = "still"

    if not title_bits:
        title = "Classroom session"
    else:
        title = "Classroom session · " + " + ".join(title_bits)

    narrative = (
        "Uploaded media was processed in this Streamlit process. Frames and audio were not sent to a vision "
        "or speech API, and nothing was written to a database. Track IDs exist only for this session."
    )
    return Bundle(
        title=title,
        source="Uploaded media",
        narrative=narrative,
        duration_s=float(duration),
        attendance=attendance,
        visual=visual,
        audio=audio,
        warnings=warnings,
        pose_method=pose_method,
        modality=modality,
        roster_size=None,
        annotated_frames=annotated,
        snapshot=snapshot,
        stats=stats,
    )


def _has_visual_attention(bundle: Bundle) -> bool:
    visual = bundle.visual
    if visual is None or visual.empty or "attention" not in visual.columns:
        return False
    return bool(pd.to_numeric(visual["attention"], errors="coerce").notna().any())


def _single_modality_warning(bundle: Bundle, timeline: pd.DataFrame) -> str | None:
    has_visual = _has_visual_attention(bundle)
    has_audio = bundle.audio is not None and not bundle.audio.empty
    if timeline is None or timeline.empty:
        return None
    if has_audio and not has_visual and bundle.modality != "simulated":
        return (
            "The engagement index is acoustic only. A quiet stretch scores as listening, not as absence, "
            "because no head-pose signal was available."
        )
    if has_visual and not has_audio and bundle.modality != "simulated":
        return (
            "The engagement index is visual only. It reflects the share of faces with a forward head pose, "
            "not the sound of the room."
        )
    return None


def _empty_raw_attendance() -> pd.DataFrame:
    return pd.DataFrame(columns=["Student ID", "first_s", "last_s", "visible_s", "Presence Ratio"])


def bgr_from_encoded(data: bytes):
    import cv2

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError("Could not read that image. Use a JPG or PNG.")
    return image
