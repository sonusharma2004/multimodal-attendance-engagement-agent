"""Sustained-presence attendance from ephemeral face tracks.

A track is Present only when it stays in frame for the configured share of the
session (default 70%). A hallway pass-through stays in the register as Absent
so it cannot be exported as attendance. Track ids follow a place in the room
for a limited time. They are not a facial-recognition identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.formatutil import format_clock, format_duration
from src.models import ABSENT, CSV_COLUMNS, PRESENT

RAW_COLUMNS = ["Student ID", "first_s", "last_s", "visible_s", "Presence Ratio"]


@dataclass
class _Track:
    student_id: str
    last_bbox: tuple[int, int, int, int]
    last_index: int
    obs_indices: list[int] = field(default_factory=list)
    closed: bool = False


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def note_for(ratio: float, _visible_s: float, threshold: float) -> str:
    percent = int(round(threshold * 100))
    if ratio >= threshold:
        return f"Sustained presence met (at least {percent}% of the session)."
    # A brief appearance is a small fraction of the session (a hallway pass),
    # not merely a short clock time inside a short clip.
    if ratio < 0.15:
        return "Below the sustained-presence threshold. Brief appearance — not marked present."
    return "Below the sustained-presence threshold. Not marked present."


def empty_attendance() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_COLUMNS)


def format_attendance(raw: pd.DataFrame | None, threshold: float) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_attendance()
    ratio = raw["Presence Ratio"].astype(float)
    visible = raw["visible_s"].astype(float)
    status = np.where(ratio.to_numpy() >= float(threshold), PRESENT, ABSENT)
    notes = [note_for(float(r), float(v), float(threshold)) for r, v in zip(ratio, visible)]
    presence = [f"{int(round(float(r) * 100))}%" for r in ratio]
    return pd.DataFrame(
        {
            "Student ID": raw["Student ID"].astype(str).to_numpy(),
            "First Seen": [format_clock(v) for v in raw["first_s"]],
            "Last Seen": [format_clock(v) for v in raw["last_s"]],
            "Total Duration": [format_duration(v) for v in visible],
            "Status": status,
            "Presence": presence,
            "Notes": notes,
            "Presence Ratio": ratio.round(4).to_numpy(),
            "Visible Seconds": visible.round(1).to_numpy(),
        }
    )


def build_register(
    samples: list[dict],
    duration_s: float,
    *,
    presence_bridge_s: float = 15.0,
    identity_gap_s: float = 15 * 60,
) -> pd.DataFrame:
    """Build a raw attendance frame and write ``student_id`` back onto detections.

    ``samples`` is a list of ``{"t": seconds, "dets": [detection, ...]}``.
    Detections are mutated in place with a ``student_id``.
    """

    if not samples:
        return pd.DataFrame(columns=RAW_COLUMNS)
    times = np.asarray([float(sample["t"]) for sample in samples], dtype=float)
    order = np.argsort(times)
    if np.any(order != np.arange(len(times))):
        samples = [samples[int(i)] for i in order]
        times = times[order]
    if len(times) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)
    if not math.isfinite(duration_s) or duration_s <= 0:
        duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    if times[-1] > duration_s:
        duration_s = float(times[-1])
    if duration_s <= 0:
        return pd.DataFrame(columns=RAW_COLUMNS)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else duration_s
    max_miss_s = max(float(identity_gap_s), dt)
    tracks: list[_Track] = []
    next_id = 1

    for index, sample in enumerate(samples):
        dets = list(sample.get("dets") or [])
        eligible = [
            track
            for track in tracks
            if not track.closed and (times[index] - times[track.last_index]) <= max_miss_s + 1e-6
        ]
        matches, used_dets = _match(eligible, dets, index, times)
        matched_tracks = set()
        for track, det_index in matches:
            det = dets[det_index]
            det["student_id"] = track.student_id
            track.last_bbox = tuple(det["bbox"])
            track.last_index = index
            track.obs_indices.append(index)
            matched_tracks.add(id(track))
        for det_index, det in enumerate(dets):
            if det_index in used_dets:
                continue
            student_id = f"P-{next_id:02d}" if next_id < 100 else f"P-{next_id:03d}"
            next_id += 1
            det["student_id"] = student_id
            tracks.append(
                _Track(
                    student_id=student_id,
                    last_bbox=tuple(det["bbox"]),
                    last_index=index,
                    obs_indices=[index],
                )
            )
        for track in eligible:
            if id(track) in matched_tracks:
                continue
            if (times[index] - times[track.last_index]) > max_miss_s:
                track.closed = True

    left, right = _sample_cells(times, float(duration_s))
    rows = []
    for track in sorted(tracks, key=lambda item: item.obs_indices[0] if item.obs_indices else 0):
        if not track.obs_indices:
            continue
        visible = _visible_seconds(track.obs_indices, times, left, right, presence_bridge_s)
        ratio = float(visible / duration_s) if duration_s else 0.0
        ratio = float(min(1.0, max(0.0, ratio)))
        first = int(track.obs_indices[0])
        last = int(track.obs_indices[-1])
        rows.append(
            {
                "Student ID": track.student_id,
                "first_s": float(times[first]),
                "last_s": float(times[last]),
                "visible_s": float(visible),
                "Presence Ratio": ratio,
            }
        )
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.DataFrame(rows)


def visual_rows_from_samples(samples: list[dict]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        dets = sample.get("dets") or []
        faces = len(dets)
        known = [det for det in dets if det.get("pose_known") and det.get("forward") is not None]
        forward = sum(1 for det in known if det.get("forward"))
        attention = float(forward / len(known)) if known else float("nan")
        rows.append(
            {
                "t": float(sample["t"]),
                "faces": faces,
                "forward_faces": forward,
                "pose_faces": len(known),
                "attention": attention,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["t", "faces", "forward_faces", "pose_faces", "attention"])
    return pd.DataFrame(rows)


def analyze_video(path: str, engine, *, max_samples: int = 180, progress=None) -> dict:
    """Sample a video, detect faces, and return raw attendance plus a visual series."""

    import cv2

    from src.errors import VideoReadError
    from src.face_engine import apply_forward_flags, draw_tracks, resize_width

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(
            "OpenCV could not open that video. Export an MP4 (H.264) or an MJPEG AVI and try again."
        )
    warnings: list[str] = []
    try:
        times, grab_at, planned_duration = _plan_samples(cap, max_samples=max_samples)
        if len(times) == 0:
            raise VideoReadError("The video did not contain any readable frames.")
        samples: list[dict] = []
        stored: dict[int, bytes] = {}
        candidate_idx = set(int(i) for i in np.linspace(0, len(times) - 1, num=min(8, len(times)), dtype=int))
        total = len(times)
        for index, at_ms in enumerate(grab_at):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(at_ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                samples.append({"t": float(times[index]), "dets": []})
            else:
                resized = resize_width(frame, 1280)
                dets = engine.detect(resized)
                samples.append({"t": float(times[index]), "dets": dets})
                if index in candidate_idx:
                    encoded, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if encoded:
                        stored[index] = buffer.tobytes()
            if progress is not None:
                progress(index, total, "detect")
        duration_s = float(planned_duration) if planned_duration > 0 else float(times[-1])
        baseline = apply_forward_flags(samples)
        attendance = build_register(samples, duration_s)
        visual = visual_rows_from_samples(samples)
        annotated = _annotate_stored(stored, samples, times)
        if progress is not None:
            progress(total - 1, total, "annotate")
    finally:
        cap.release()

    interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    pose_faces = int(visual["pose_faces"].sum()) if not visual.empty else 0
    faces = int(visual["faces"].sum()) if not visual.empty else 0
    if engine.backend != "mediapipe":
        warnings.append(
            "The MediaPipe face detector was not available, so presence uses OpenCV's built-in cascade. "
            "Head pose was not estimated, and the visual half of the engagement index is left blank "
            "rather than guessed."
        )
    if duration_s < 60:
        warnings.append(
            "This clip is shorter than a minute. Sustained presence is relative to the clip only — "
            "it is not a class-period attendance mark."
        )
    if interval > 5:
        warnings.append(
            f"Frames were sampled about every {interval:.0f} seconds so a long recording stays responsive. "
            "Presence is estimated from those samples, not from every frame."
        )
    if faces == 0:
        warnings.append(
            "No faces were found. Check lighting, that students face the camera, and that the picture is large enough."
        )
    elif pose_faces == 0 and engine.backend == "mediapipe":
        warnings.append("Faces were found, but keypoints were not, so head pose was left blank.")

    return {
        "duration_s": duration_s,
        "attendance": attendance,
        "visual": visual,
        "annotated_frames": annotated,
        "warnings": warnings,
        "pose_method": engine.pose_method,
        "stats": {
            "samples": int(len(times)),
            "sample_interval_s": interval,
            "frames_with_faces": int((visual["faces"] > 0).sum()) if not visual.empty else 0,
            "face_detections": faces,
            "pose_detections": pose_faces,
            "pitch_baseline": baseline,
            "detector": engine.backend,
        },
    }


def analyze_image(bgr: np.ndarray, engine) -> dict:
    """Summarize one still. A still cannot support a sustained-presence mark."""

    import cv2

    from src.face_engine import apply_forward_flags, draw_tracks, resize_width

    frame = resize_width(bgr, 1280)
    dets = engine.detect(frame)
    samples = [{"t": 0.0, "dets": dets}]
    apply_forward_flags(samples, min_faces_for_baseline=8)
    known = [det for det in dets if det.get("pose_known")]
    forward = sum(1 for det in known if det.get("forward"))
    turned = sum(1 for det in known if det.get("forward") is False)
    drawn = draw_tracks(frame, dets)
    rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    summary = {
        "faces": len(dets),
        "forward": forward,
        "turned": turned,
        "pose_unknown": len(dets) - len(known),
        "image": rgb,
        "pose_method": engine.pose_method,
    }
    return summary


def _match(tracks: list[_Track], dets: list[dict], index: int, times: np.ndarray):
    pairs = []
    for track_index, track in enumerate(tracks):
        for det_index, det in enumerate(dets):
            bbox = det.get("bbox")
            if not bbox:
                continue
            cost = _association_cost(track.last_bbox, tuple(bbox))
            if cost is None:
                continue
            gap = float(times[index] - times[track.last_index])
            pairs.append((cost + 0.002 * gap, track_index, det_index))
    pairs.sort()
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches = []
    for _cost, track_index, det_index in pairs:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matches.append((tracks[track_index], det_index))
    return matches, used_dets


def _association_cost(previous: tuple, current: tuple) -> float | None:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    dist = math.hypot((px + pw / 2) - (cx + cw / 2), (py + ph / 2) - (cy + ch / 2))
    scale = max(math.hypot(pw, ph), math.hypot(cw, ch), 1.0)
    norm = dist / scale
    overlap = iou(previous, current)
    if norm > 1.35 and overlap < 0.08:
        return None
    return float(norm - overlap)


def _sample_cells(times: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    count = len(times)
    left = np.zeros(count, dtype=float)
    right = np.full(count, duration_s, dtype=float)
    if count == 1:
        return left, right
    mids = (times[:-1] + times[1:]) / 2.0
    right[:-1] = mids
    left[1:] = mids
    left[0] = min(left[0], times[0])
    left[0] = 0.0 if times[0] >= 0 else float(times[0])
    right[-1] = max(float(duration_s), float(times[-1]))
    return left, right


def _visible_seconds(
    obs_indices: list[int],
    times: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bridge_s: float,
) -> float:
    obs = sorted(set(int(i) for i in obs_indices))
    if not obs:
        return 0.0
    covered = set(obs)
    for start, end in zip(obs, obs[1:]):
        gap = float(times[end] - times[start])
        if gap <= bridge_s + 1e-6:
            covered.update(range(start, end + 1))
    return float(sum(float(right[i] - left[i]) for i in covered))


def _plan_samples(cap, max_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return sample times in seconds, seek positions in milliseconds, and duration."""

    fps = float(cap.get(5) or 0.0)
    count = float(cap.get(7) or 0.0)
    duration = count / fps if fps > 1 and count > 1 else 0.0
    if not (0.2 < duration < 60 * 60 * 6):
        duration = _duration_by_counting(cap)
    if duration <= 0:
        return np.array([]), np.array([]), 0.0
    target = int(min(max_samples, max(8, round(duration * 2))))
    target = max(1, min(target, max_samples))
    if duration < 1.0:
        target = 1
    elif duration < 8:
        target = max(1, min(target, int(math.ceil(duration * 2))))
    times = np.linspace(0.0, duration, target, endpoint=False)
    return times, times * 1000.0, float(duration)


def _duration_by_counting(cap) -> float:
    fps = float(cap.get(5) or 0.0) or 30.0
    counted = 0
    while counted < 200_000 and cap.grab():
        counted += 1
    cap.set(1, 0)  # CAP_PROP_POS_FRAMES
    if counted <= 0:
        return 0.0
    return float(counted / fps)


def _annotate_stored(stored: dict[int, bytes], samples: list[dict], times: np.ndarray) -> list[dict]:
    import cv2

    from src.face_engine import draw_tracks

    if not stored:
        return []
    with_faces = [index for index in sorted(stored) if samples[index].get("dets")]
    pool = with_faces or sorted(stored)
    if len(pool) > 4:
        pick = np.linspace(0, len(pool) - 1, 4, dtype=int)
        chosen = []
        for key in pick:
            index = pool[int(key)]
            if index not in chosen:
                chosen.append(index)
    else:
        chosen = pool
    frames = []
    for index in chosen:
        buffer = np.frombuffer(stored[index], dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        drawn = draw_tracks(frame, samples[index].get("dets") or [])
        rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        frames.append({"t": float(times[index]), "image": rgb})
    return frames
