"""Fuse visual attentiveness and room-sound scores into one engagement index.

The index is a property of the room at each time step, not a grade for a student.
Silence scores 0.48 rather than zero, because a quiet room is often listening.
Missing pose is left blank instead of being treated as distraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.audio_dynamics import UNOBSERVED, score_for_label

VISUAL_WEIGHT = 0.58
AUDIO_WEIGHT = 0.42
EMA_ALPHA = 0.35


def choose_bin_s(duration_s: float) -> float:
    if duration_s <= 90:
        return max(1.0, float(duration_s) / 20.0)
    if duration_s <= 600:
        return 5.0
    return 10.0


def fuse(
    visual: pd.DataFrame | None,
    audio: pd.DataFrame | None,
    duration_s: float,
    bin_s: float | None = None,
) -> pd.DataFrame:
    if duration_s is None or duration_s <= 0:
        return _empty_timeline()
    if (visual is None or visual.empty) and (audio is None or audio.empty):
        return _empty_timeline()

    width = float(bin_s) if bin_s and bin_s > 0 else choose_bin_s(float(duration_s))
    n_bins = max(1, int(np.ceil(float(duration_s) / width)))
    n_bins = min(n_bins, 2000)
    visual_idx = _index_visual(visual)
    audio_idx = _index_audio(audio)
    rows = []
    for index in range(n_bins):
        t0 = index * width
        t1 = min(float(duration_s), (index + 1) * width)
        if t1 <= t0:
            continue
        visual_value, faces = _visual_from_index(visual_idx, t0, t1)
        audio_value, label = _audio_from_index(audio_idx, t0, t1)
        has_visual = visual_value is not None and np.isfinite(visual_value)
        has_audio = audio_value is not None and np.isfinite(audio_value)
        if has_visual and has_audio:
            raw = VISUAL_WEIGHT * float(visual_value) + AUDIO_WEIGHT * float(audio_value)
            w_visual, w_audio = VISUAL_WEIGHT, AUDIO_WEIGHT
        elif has_visual:
            raw = float(visual_value)
            w_visual, w_audio = 1.0, 0.0
        elif has_audio:
            raw = float(audio_value)
            w_visual, w_audio = 0.0, 1.0
        else:
            raw = float("nan")
            w_visual, w_audio = 0.0, 0.0
        rows.append(
            {
                "t_start": t0,
                "t_end": t1,
                "t_mid": (t0 + t1) / 2.0,
                "visual": float(visual_value) if has_visual else float("nan"),
                "audio_score": float(audio_value) if has_audio else float("nan"),
                "audio_label": label or UNOBSERVED,
                "faces": faces,
                "w_visual": w_visual,
                "w_audio": w_audio,
                "engagement_raw": raw * 100.0 if np.isfinite(raw) else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_timeline()
    smoothed = _ema_with_gaps(frame["engagement_raw"].to_numpy(dtype=float) / 100.0, EMA_ALPHA)
    engagement = np.clip(smoothed * 100.0, 0.0, 100.0)
    engagement[~np.isfinite(smoothed)] = np.nan
    frame["engagement"] = engagement
    return frame


def _index_visual(visual: pd.DataFrame | None) -> dict | None:
    if visual is None or visual.empty or "t" not in visual.columns:
        return None
    times = pd.to_numeric(visual["t"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(times)
    attention = (
        pd.to_numeric(visual["attention"], errors="coerce").to_numpy(dtype=float)
        if "attention" in visual.columns
        else np.full(len(visual), np.nan)
    )
    faces = (
        pd.to_numeric(visual["faces"], errors="coerce").to_numpy(dtype=float)
        if "faces" in visual.columns
        else np.full(len(visual), np.nan)
    )
    return {"t": times[order], "attention": attention[order], "faces": faces[order]}


def _index_audio(audio: pd.DataFrame | None) -> dict | None:
    if audio is None or audio.empty or "t_start" not in audio.columns:
        return None
    ordered = audio.sort_values("t_start")
    labels = ordered["label"].astype(str).to_numpy()
    if "score" in ordered.columns:
        scores = pd.to_numeric(ordered["score"], errors="coerce").to_numpy(dtype=float)
    else:
        scores = np.full(len(ordered), np.nan)
    for index, label in enumerate(labels):
        if not np.isfinite(scores[index]):
            scores[index] = score_for_label(label)
    return {
        "start": ordered["t_start"].to_numpy(dtype=float),
        "end": ordered["t_end"].to_numpy(dtype=float),
        "score": scores,
        "label": labels,
    }


def _visual_from_index(index: dict | None, t0: float, t1: float) -> tuple[float | None, float]:
    if index is None:
        return None, float("nan")
    left = int(np.searchsorted(index["t"], t0, side="left"))
    right = int(np.searchsorted(index["t"], t1, side="left"))
    if right <= left:
        return None, float("nan")
    faces = index["faces"][left:right]
    face_mean = float(np.nanmean(faces)) if np.isfinite(faces).any() else float("nan")
    values = index["attention"][left:right]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, face_mean
    return float(np.mean(finite)), face_mean


def _audio_from_index(index: dict | None, t0: float, t1: float) -> tuple[float | None, str | None]:
    if index is None:
        return None, None
    left = int(np.searchsorted(index["end"], t0, side="right"))
    right = int(np.searchsorted(index["start"], t1, side="left"))
    if right <= left:
        return None, None
    starts = index["start"][left:right]
    ends = index["end"][left:right]
    overlap = np.minimum(ends, t1) - np.maximum(starts, t0)
    mask = overlap > 0
    if not np.any(mask):
        return None, None
    weights = overlap[mask]
    scores = index["score"][left:right][mask]
    labels = index["label"][left:right][mask]
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None, None
    weights = weights[finite]
    scores = scores[finite]
    labels = labels[finite]
    total = float(weights.sum())
    if total <= 0:
        return None, None
    score = float(np.dot(weights, scores) / total)
    # Majority label by overlap. Class counts stay tiny.
    best_label = None
    best_weight = -1.0
    for label in np.unique(labels):
        weight = float(weights[labels == label].sum())
        if weight > best_weight:
            best_label = str(label)
            best_weight = weight
    return score, best_label


def _ema_with_gaps(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    previous = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            previous = None
            continue
        previous = float(value) if previous is None else alpha * float(value) + (1.0 - alpha) * previous
        out[index] = previous
    return out


def _empty_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "t_start",
            "t_end",
            "t_mid",
            "visual",
            "audio_score",
            "audio_label",
            "faces",
            "w_visual",
            "w_audio",
            "engagement_raw",
            "engagement",
        ]
    )
