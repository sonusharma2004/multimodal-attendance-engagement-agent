"""Educator-facing KPIs, Markdown, and JSON. No database."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.advisory import carry_forward
from src.audio_dynamics import AUDIO_LABELS
from src.formatutil import format_clock, format_duration, format_number, format_stamp
from src.models import PRESENT, Advisory, Session


def compute_kpis(session_parts: dict[str, Any]) -> dict[str, Any]:
    attendance: pd.DataFrame = session_parts["attendance"]
    timeline: pd.DataFrame = session_parts["timeline"]
    duration_s = float(session_parts["duration_s"] or 0.0)
    alert_level = float(session_parts["alert_level"])
    roster = session_parts.get("roster_size")

    present = absent = brief = 0
    if attendance is not None and not attendance.empty and "Status" in attendance.columns:
        present = int((attendance["Status"] == PRESENT).sum())
        absent = int((attendance["Status"] != PRESENT).sum())
        if "Notes" in attendance.columns:
            brief = int(attendance["Notes"].astype(str).str.contains("Brief appearance").sum())

    observed = present + absent
    engagement = _series(timeline, "engagement")
    finite = engagement[np.isfinite(engagement)] if engagement.size else engagement
    time_below = 0.0
    if timeline is not None and not timeline.empty and finite.size:
        mask = timeline["engagement"] < alert_level
        time_below = float((timeline.loc[mask, "t_end"] - timeline.loc[mask, "t_start"]).sum())

    low_t = high_t = None
    low_v = high_v = None
    mean_e = median_e = None
    if finite.size:
        mean_e = float(np.mean(finite))
        median_e = float(np.median(finite))
        low_idx = int(np.nanargmin(engagement))
        high_idx = int(np.nanargmax(engagement))
        low_v = float(engagement[low_idx])
        high_v = float(engagement[high_idx])
        low_t = float(timeline["t_mid"].iloc[low_idx])
        high_t = float(timeline["t_mid"].iloc[high_idx])

    audio_mix = _audio_mix(timeline)
    faces = _series(timeline, "faces")
    face_finite = faces[np.isfinite(faces)] if faces.size else faces
    median_faces = float(np.median(face_finite)) if face_finite.size else None

    advisories: list[Advisory] = session_parts.get("advisories") or []
    return {
        "duration_s": duration_s,
        "present": present,
        "below_threshold": absent,
        "brief_appearances": brief,
        "observed_tracks": observed,
        "roster_size": int(roster) if roster else None,
        "coverage": (present / int(roster)) if roster else None,
        "observed_presence_rate": (present / observed) if observed else None,
        "mean_engagement": mean_e,
        "median_engagement": median_e,
        "min_engagement": low_v,
        "max_engagement": high_v,
        "min_engagement_at_s": low_t,
        "max_engagement_at_s": high_t,
        "time_below_alert_s": time_below,
        "pct_time_below_alert": (time_below / duration_s) if duration_s else None,
        "audio_mix": audio_mix,
        "discussion_share": audio_mix.get("Student Discussion / Verbal Interaction"),
        "median_faces": median_faces,
        "advisory_count": len(advisories),
        "nudge_count": sum(1 for item in advisories if item.severity == "nudge"),
    }


def render_markdown(session: Session) -> str:
    k = session.kpis
    lines = [
        f"# Educator session report",
        f"## {session.title}",
        "",
        f"Analyzed {format_stamp(session.generated_at)}.",
        f"Source: {session.source}.",
        "",
        session.narrative.strip(),
        "",
        "This report describes the room as a whole. Track IDs are ephemeral labels for a face that stayed in view. "
        "They are not names, they are not a roll imported from a student information system, and they are not the "
        "output of facial recognition. The soundtrack is never transcribed.",
        "",
        "## Attendance",
        "",
        f"- Session length: {format_duration(k.get('duration_s'))}",
        f"- Sustained-presence threshold: {session.threshold:.0%}",
        f"- Tracks observed: {k.get('observed_tracks', 0)}",
        f"- Marked present: {k.get('present', 0)}",
        f"- Not marked present: {k.get('below_threshold', 0)} "
        f"({k.get('brief_appearances', 0)} brief appearance"
        f"{'' if k.get('brief_appearances') == 1 else 's'})",
    ]
    if k.get("roster_size"):
        lines.append(
            f"- Roll size entered for comparison: {k['roster_size']}. "
            "Coverage is a headcount hint only. This app cannot tell which person was away."
        )
        if k.get("coverage") is not None:
            lines.append(f"- Present tracks compared with that roll: {k['coverage']:.0%}")
    lines.extend(["", "### Register", ""])
    lines.extend(_markdown_table(session.attendance))
    lines.extend(["", "## Room sound", ""])
    if not k.get("audio_mix"):
        lines.append("No soundtrack was analyzed for this session.")
    else:
        lines.append("No words were transcribed and no individual speaker was identified.")
        for label in AUDIO_LABELS:
            share = k["audio_mix"].get(label)
            if share is None:
                continue
            lines.append(f"- {label}: {share:.0%} of the scored soundtrack")
        if k.get("discussion_share") is not None:
            lines.append(
                f"- Verbal interaction time (student discussion): {k['discussion_share']:.0%}"
            )
    lines.extend(["", "## Engagement", ""])
    if k.get("mean_engagement") is None:
        lines.append("There was not enough paired signal to compute an engagement index.")
    else:
        lines.extend(
            [
                f"- Mean index: {k['mean_engagement']:.0f}",
                f"- Median index: {k['median_engagement']:.0f}",
                f"- Lowest: {k['min_engagement']:.0f} at {format_clock(k.get('min_engagement_at_s'))}",
                f"- Highest: {k['max_engagement']:.0f} at {format_clock(k.get('max_engagement_at_s'))}",
                f"- Time below {session.alert_level:.0f}: {format_duration(k.get('time_below_alert_s'))}",
                "",
                "The index blends the share of visible faces with a forward head pose (weight 0.58) and an acoustic "
                "interaction score (weight 0.42). Silence scores 0.48 rather than zero, because a quiet room is often "
                "a listening room. High ambient distraction scores 0.22. When one sensor is missing, the index uses "
                "the sensor that is present and says so. The curve is smoothed so a single noisy second does not fire an advisory.",
            ]
        )
    if k.get("median_faces") is not None:
        lines.append(f"Median faces in frame across scored steps: {k['median_faces']:.0f}.")
    lines.extend(["", "## Advisories the engine would have offered", ""])
    if not session.advisories:
        lines.append("No advisory fired. The engine stays quiet when the room index does not cross a rule.")
    else:
        for item in session.advisories:
            lines.extend(
                [
                    f"### {item.title}",
                    "",
                    f"{format_clock(item.t_start)}–{format_clock(item.t_end)} · {item.severity}",
                    "",
                    item.message,
                    "",
                    f"Why: {item.rationale}",
                    "",
                ]
            )
    lines.extend(
        [
            "## What to carry into the next period",
            "",
            carry_forward(session.advisories),
            "",
            "## Method and limits",
            "",
            f"- Pose method: {session.pose_method}",
            f"- Samples: {session.stats.get('samples', '—')}; "
            f"typical interval: {format_number(session.stats.get('sample_interval_s'), 1)} s",
            "- This is not a disciplinary tool, not a special-education diagnosis, and not a participation grade.",
            "- Looking down at a notebook can remove a face from the detector. Short gaps (15 seconds) still count as in the room. Longer gaps do not.",
            "- If someone leaves and comes back much later, they may receive a new track ID. That is deliberate: the app does not re-identify people.",
            "- Acoustic classes are heuristics over energy and spectrum. They confuse music, video playback, and overlapping talk in predictable ways.",
            "- Uploaded media is processed in memory for this session and is not written to a database by this app.",
            "",
        ]
    )
    if session.warnings:
        lines.extend(["## Pipeline notes", ""])
        for warning in session.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_json(session: Session) -> str:
    payload = {
        "schema": "mea.session.v1",
        "title": session.title,
        "source": session.source,
        "narrative": session.narrative,
        "analyzed_at_ist": format_stamp(session.generated_at),
        "duration_s": session.duration_s,
        "modality": session.modality,
        "pose_method": session.pose_method,
        "privacy": {
            "offline_processing": True,
            "external_vision_api": False,
            "speech_to_text": False,
            "face_database": False,
            "facial_recognition": False,
            "individual_engagement_grades": False,
        },
        "settings": {
            "presence_threshold": session.threshold,
            "alert_level": session.alert_level,
            "alert_minutes": session.alert_minutes,
            "roster_size": session.roster_size,
        },
        "kpis": _jsonable(session.kpis),
        "attendance": _records(session.attendance),
        "advisories": [
            {
                "t_start": item.t_start,
                "t_end": item.t_end,
                "severity": item.severity,
                "title": item.title,
                "message": item.message,
                "rationale": item.rationale,
            }
            for item in session.advisories
        ],
        "timeline": _records(session.timeline),
        "snapshot": _snapshot_counts(session.snapshot),
        "warnings": list(session.warnings),
        "stats": _jsonable(session.stats),
    }
    return json.dumps(payload, indent=2)


def _snapshot_counts(snapshot: dict | None) -> dict | None:
    if not snapshot:
        return None
    return {
        "faces": snapshot.get("faces"),
        "forward": snapshot.get("forward"),
        "turned": snapshot.get("turned"),
        "pose_unknown": snapshot.get("pose_unknown"),
        "pose_method": snapshot.get("pose_method"),
    }


def _audio_mix(timeline: pd.DataFrame | None) -> dict[str, float]:
    if timeline is None or timeline.empty or "audio_label" not in timeline.columns:
        return {}
    scored = timeline[timeline["audio_label"].isin(AUDIO_LABELS)].copy()
    if scored.empty:
        return {}
    weights = (scored["t_end"] - scored["t_start"]).clip(lower=0)
    total = float(weights.sum())
    if total <= 0:
        return {}
    mix = {}
    for label in AUDIO_LABELS:
        mask = scored["audio_label"] == label
        mix[label] = float(weights[mask].sum() / total)
    return mix


def _series(frame: pd.DataFrame | None, column: str) -> np.ndarray:
    if frame is None or frame.empty or column not in frame.columns:
        return np.array([])
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return ["No attendance register. A still frame, an empty detector, or an audio-only session does not invent rows."]
    columns = [
        "Student ID",
        "First Seen",
        "Last Seen",
        "Total Duration",
        "Status",
        "Presence",
        "Notes",
    ]
    columns = [column for column in columns if column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in frame.iterrows():
        cells = [str(row[column]).replace("|", "/") for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    return [header, rule, *body]


def _records(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    records = frame.to_dict(orient="records")
    return [_jsonable(row) for row in records]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
