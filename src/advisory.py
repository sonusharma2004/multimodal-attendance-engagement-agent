"""Heuristic teaching advisories. Collective, specific, and quiet unless needed.

Rules fire on the room index and the room-sound label. They never name a track
and they do not call an external model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.audio_dynamics import DISCUSSION, DISTRACTION, LECTURE
from src.formatutil import format_clock, format_duration
from src.models import Advisory


def generate_advisories(
    timeline: pd.DataFrame | None,
    *,
    alert_level: float = 45.0,
    alert_minutes: float = 5.0,
) -> list[Advisory]:
    if timeline is None or timeline.empty or "engagement" not in timeline.columns:
        return []
    frame = timeline.reset_index(drop=True)
    engagement = pd.to_numeric(frame["engagement"], errors="coerce").to_numpy(dtype=float)
    labels = frame["audio_label"].astype(str).to_numpy() if "audio_label" in frame.columns else np.array([""] * len(frame))
    candidates: list[Advisory] = []

    low_mask = np.isfinite(engagement) & (engagement < float(alert_level))
    for start, end in _true_runs(low_mask):
        span = _span(frame, start, end)
        if span < float(alert_minutes) * 60.0 - 1e-6:
            continue
        distraction_share = _label_share(labels, start, end, DISTRACTION)
        minutes = max(1, int(round(span / 60.0)))
        unit = "minute" if minutes == 1 else "minutes"
        rationale = (
            "The smoothed room index stayed under the alert line. "
            "The cue mixes forward head pose, when the camera can see it, with the acoustic score."
        )
        if distraction_share >= 0.4:
            rationale += " Distraction-like room noise was also high during this stretch."
        candidates.append(
            Advisory(
                t_start=float(frame.loc[start, "t_start"]),
                t_end=float(frame.loc[end, "t_end"]),
                severity="nudge",
                title="Shift the activity",
                message=(
                    f"Collective engagement dropped below {alert_level:.0f}% for {minutes} {unit} "
                    f"— recommend launching a quick poll, interactive discussion, or a 2-minute stretch break."
                ),
                rationale=rationale,
            )
        )

    lecture_mask = (labels == LECTURE) & np.isfinite(engagement) & (engagement >= float(alert_level))
    for start, end in _true_runs(lecture_mask):
        span = _span(frame, start, end)
        if span < 12 * 60 - 1e-6:
            continue
        values = engagement[start : end + 1]
        if _segment_drop(values) < 12.0:
            continue
        minutes = max(1, int(round(span / 60.0)))
        candidates.append(
            Advisory(
                t_start=float(frame.loc[start, "t_start"]),
                t_end=float(frame.loc[end, "t_end"]),
                severity="nudge",
                title="Lecture is running long",
                message=(
                    f"Direct instruction has run for about {minutes} minutes and the room index has been sliding. "
                    "Before adding more explanation, try a check for understanding: a worked step in pairs, "
                    "a mini-whiteboard, or one question to the whole room."
                ),
                rationale="A long lecture stretch lost at least 12 points of collective engagement without yet crossing the low-attention alert.",
            )
        )

    kept = _suppress(candidates)

    # Recovery looks forward from a low-engagement nudge that was actually kept.
    recovery: list[Advisory] = []
    for advisory in kept:
        if advisory.title != "Shift the activity":
            continue
        window_end = advisory.t_end + 4 * 60
        after = frame[(frame["t_start"] >= advisory.t_end - 1e-6) & (frame["t_end"] <= window_end + 1e-6)]
        if after.empty:
            continue
        mean_after = float(pd.to_numeric(after["engagement"], errors="coerce").mean())
        if not np.isfinite(mean_after) or mean_after < float(alert_level) + 18:
            continue
        recovery.append(
            Advisory(
                t_start=float(advisory.t_end),
                t_end=float(min(window_end, frame["t_end"].iloc[-1])),
                severity="celebrate",
                title="Engagement recovered",
                message=(
                    "The room came back after the low stretch. If you changed the activity, keep the next "
                    "instruction short so the recovery sticks, and let one or two students share what the room just did."
                ),
                rationale=f"In the four minutes after the alert, mean engagement was about {mean_after:.0f}.",
            )
        )

    discussion_windows = _sliding_label_windows(
        frame,
        label=DISCUSSION,
        minutes=4,
        min_share=0.60,
        min_engagement=70.0,
    )
    for start_s, end_s, mean_e in discussion_windows:
        recovery.append(
            Advisory(
                t_start=start_s,
                t_end=end_s,
                severity="celebrate",
                title="Discussion is doing real work",
                message=(
                    "Student talk is carrying this stretch and the engagement index is healthy. "
                    "Give the room a minute to share one idea aloud, then name the thread you want to keep."
                ),
                rationale=f"Discussion dominated about four minutes while mean engagement stayed near {mean_e:.0f}.",
            )
        )

    closing = _closing_advisory(frame, alert_level)
    ordered = _suppress(kept + recovery + ([closing] if closing else []))
    return _cap(ordered)


def carry_forward(advisories: list[Advisory]) -> str:
    nudges = [item for item in advisories if item.severity == "nudge"]
    if nudges:
        longest = max(nudges, key=lambda item: item.duration_s)
        return (
            f"Carry this into the next opening: around {format_clock(longest.t_start)} the room needed a change of activity. "
            f"{longest.message}"
        )
    celebrations = [item for item in advisories if item.severity == "celebrate"]
    if celebrations:
        return (
            "The strongest stretch was interactive. Open the next period with a short pair prompt before returning to direct instruction, "
            "so the pattern that worked is still in the room."
        )
    return (
        "Nothing in this session crossed an alert. The next period can open on the planned lesson. "
        "Still build in one short check for understanding — a quiet index is not proof that every student learned it."
    )


def _closing_advisory(frame: pd.DataFrame, alert_level: float) -> Advisory | None:
    end = float(frame["t_end"].iloc[-1])
    if end < 6 * 60 - 1e-6:
        return None
    start = end - 6 * 60
    chunk = frame[frame["t_end"] > start]
    if chunk.empty:
        return None
    engagement = pd.to_numeric(chunk["engagement"], errors="coerce")
    if float(engagement.mean()) < 65 or not np.isfinite(engagement.mean()):
        return None
    below = float(((engagement < alert_level) * (chunk["t_end"] - chunk["t_start"])).sum())
    if below > 0.15 * float((chunk["t_end"] - chunk["t_start"]).sum()):
        return None
    labels = chunk["audio_label"].astype(str)
    distraction = float((chunk.loc[labels == DISTRACTION, "t_end"] - chunk.loc[labels == DISTRACTION, "t_start"]).sum()) if len(chunk) else 0.0
    total = float((chunk["t_end"] - chunk["t_start"]).sum())
    if total <= 0 or distraction / total >= 0.25:
        return None
    return Advisory(
        t_start=float(chunk["t_start"].iloc[0]),
        t_end=float(chunk["t_end"].iloc[-1]),
        severity="info",
        title="Close while the room is with you",
        message=(
            "The closing stretch is steady enough for an exit ticket or a one-minute written check. "
            "Read it as a class signal, not as a ranking of students."
        ),
        rationale="The last six minutes stayed clear of the alert line and were not dominated by ambient distraction.",
    )


def _sliding_label_windows(frame, label: str, minutes: float, min_share: float, min_engagement: float):
    if frame.empty:
        return []
    found = []
    width = minutes * 60.0
    # Step roughly one bin at a time, but collapse to non-overlapping keepers later.
    starts = frame["t_start"].to_numpy(dtype=float)
    for t0 in starts:
        t1 = t0 + width
        chunk = frame[(frame["t_start"] >= t0 - 1e-6) & (frame["t_end"] <= t1 + 1e-6)]
        if chunk.empty:
            continue
        span = float((chunk["t_end"] - chunk["t_start"]).sum())
        if span < width * 0.85:
            continue
        share = _frame_share(chunk, label)
        mean_e = float(pd.to_numeric(chunk["engagement"], errors="coerce").mean())
        if share >= min_share and np.isfinite(mean_e) and mean_e >= min_engagement:
            found.append((float(chunk["t_start"].iloc[0]), float(chunk["t_end"].iloc[-1]), mean_e))
    return _thin_windows(found)


def _thin_windows(windows: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Keep a few non-overlapping discussion windows, preferring earlier ones."""

    kept: list[tuple[float, float, float]] = []
    for window in windows:
        if any(_interval_overlap(window[0], window[1], prev[0], prev[1]) > 0.2 for prev in kept):
            continue
        kept.append(window)
        if len(kept) >= 2:
            break
    return kept


def _frame_share(chunk: pd.DataFrame, label: str) -> float:
    total = float((chunk["t_end"] - chunk["t_start"]).sum())
    if total <= 0:
        return 0.0
    mask = chunk["audio_label"].astype(str) == label
    part = float((chunk.loc[mask, "t_end"] - chunk.loc[mask, "t_start"]).sum())
    return part / total


def _segment_drop(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return 0.0
    k = max(1, int(round(finite.size * 0.2)))
    return float(np.mean(finite[:k]) - np.mean(finite[-k:]))


def _span(frame: pd.DataFrame, start: int, end: int) -> float:
    return float(frame.loc[end, "t_end"] - frame.loc[start, "t_start"])


def _label_share(labels: np.ndarray, start: int, end: int, label: str) -> float:
    chunk = labels[start : end + 1]
    if len(chunk) == 0:
        return 0.0
    return float(np.mean(chunk == label))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    base = min(a1 - a0, b1 - b0)
    if base <= 0:
        return 0.0
    return inter / base


def _suppress(advisories: list[Advisory]) -> list[Advisory]:
    priority = {"nudge": 0, "celebrate": 1, "info": 2}

    def rank(item: Advisory) -> tuple:
        score = priority.get(item.severity, 9)
        if item.title == "Engagement recovered":
            score -= 0.4
        return (score, item.t_start, -item.duration_s)

    ordered = sorted(advisories, key=rank)
    kept: list[Advisory] = []
    for advisory in ordered:
        if any(_interval_overlap(advisory.t_start, advisory.t_end, prev.t_start, prev.t_end) > 0.45 for prev in kept):
            continue
        kept.append(advisory)
    kept.sort(key=lambda item: item.t_start)
    return kept


def _cap(advisories: list[Advisory]) -> list[Advisory]:
    nudges = [item for item in advisories if item.severity == "nudge"]
    others = [item for item in advisories if item.severity != "nudge"]
    trimmed = nudges + others[: max(0, 6 - len(nudges))]
    trimmed.sort(key=lambda item: item.t_start)
    return trimmed


def advisory_rows(advisories: list[Advisory]) -> pd.DataFrame:
    if not advisories:
        return pd.DataFrame(columns=["From", "To", "Severity", "Title", "Message", "Rationale"])
    return pd.DataFrame(
        [
            {
                "From": format_clock(item.t_start),
                "To": format_clock(item.t_end),
                "Severity": item.severity,
                "Title": item.title,
                "Message": item.message,
                "Rationale": item.rationale,
                "Duration": format_duration(item.duration_s),
            }
            for item in advisories
        ]
    )
