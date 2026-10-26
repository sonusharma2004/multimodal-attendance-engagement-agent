"""Room-sound classification without transcription or speaker identity.

Windows are labeled from spectral shape and energy stability:

* Silence — very low energy
* Instructor Lecture — harmonic energy that stays steady
* Student Discussion / Verbal Interaction — harmonic energy that keeps changing
* High Ambient Classroom Distraction — noise-like, inharmonic energy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SILENCE = "Silence"
LECTURE = "Instructor Lecture"
DISCUSSION = "Student Discussion / Verbal Interaction"
DISTRACTION = "High Ambient Classroom Distraction"
UNOBSERVED = "Unobserved"

AUDIO_LABELS = (SILENCE, LECTURE, DISCUSSION, DISTRACTION)

# Tuned against the synthetic room mix in src/synthetic_audio.py.
# Librosa spectral flatness for white noise lands near 0.16 with these STFT
# settings, while a harmonic voice is near 0. Zero-crossing rate separates them.
SILENCE_RMS = 0.015
FLAT_DISTRACTION = 0.12
ZCR_WITH_FLATNESS = 0.08
ZCR_DISTRACTION = 0.22
DISCUSSION_CV = 0.28

# Silence is intentionally not zero: a quiet room is often listening.
AUDIO_SCORE = {
    SILENCE: 0.48,
    LECTURE: 0.74,
    DISCUSSION: 0.92,
    DISTRACTION: 0.22,
    UNOBSERVED: float("nan"),
}


def score_for_label(label: str) -> float:
    return float(AUDIO_SCORE.get(label, float("nan")))


def classify_feature_row(
    rms: float,
    flatness: float,
    zcr: float,
    sub_cv: float,
    flux: float,
    centroid_norm: float,
) -> str:
    """Label one analysis window. ``centroid_norm`` is kept for callers and tests."""

    del centroid_norm  # reserved for later calibration; not required by the rules below
    if not np.isfinite(rms) or rms < SILENCE_RMS:
        return SILENCE
    flat = 0.0 if not np.isfinite(flatness) else float(flatness)
    crossings = 0.0 if not np.isfinite(zcr) else float(zcr)
    if crossings >= ZCR_DISTRACTION or (flat >= FLAT_DISTRACTION and crossings >= ZCR_WITH_FLATNESS):
        return DISTRACTION
    variation = 0.0 if not np.isfinite(sub_cv) else float(sub_cv)
    # Flux is retained on each window for inspection. The lecture/discussion
    # split uses energy stability, which is steadier across recordings.
    del flux
    if variation >= DISCUSSION_CV:
        return DISCUSSION
    return LECTURE


def extract_and_classify(y: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 1.0) -> pd.DataFrame:
    """Classify a mono waveform with librosa spectral features."""

    import librosa

    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(-1))
    if y.size == 0:
        return _empty_audio()
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    n_fft = 1024
    hop = 512
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    power = magnitude**2
    rms_f = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    zcr_f = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    flat_f = librosa.feature.spectral_flatness(S=power)[0]
    cent_f = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    log_mag = np.log1p(magnitude)
    delta = np.diff(log_mag, axis=1, prepend=log_mag[:, :1])
    flux_f = np.mean(np.maximum(delta, 0.0), axis=0)

    frame_count = min(len(rms_f), len(zcr_f), len(flat_f), len(cent_f), len(flux_f))
    rms_f = rms_f[:frame_count]
    zcr_f = zcr_f[:frame_count]
    flat_f = flat_f[:frame_count]
    cent_f = cent_f[:frame_count]
    flux_f = flux_f[:frame_count]
    times = librosa.frames_to_time(np.arange(frame_count), sr=sr, hop_length=hop, n_fft=n_fft)

    duration = float(len(y) / sr)
    starts = np.arange(0.0, max(duration - 1e-6, 1e-6), hop_s)
    rows: list[dict] = []
    for t0 in starts:
        t1 = min(duration, float(t0) + win_s)
        mask = (times >= t0) & (times < t1)
        if not np.any(mask):
            continue
        rms = float(np.mean(rms_f[mask]))
        zcr = float(np.mean(zcr_f[mask]))
        flatness = float(np.nanmean(flat_f[mask]))
        centroid = float(np.mean(cent_f[mask]))
        flux = float(np.mean(flux_f[mask]))
        sub = rms_f[mask]
        sub_cv = float(np.std(sub) / (np.mean(sub) + 1e-8))
        centroid_norm = centroid / (sr / 2.0)
        label = classify_feature_row(rms, flatness, zcr, sub_cv, flux, centroid_norm)
        rows.append(
            {
                "t_start": float(t0),
                "t_end": float(t1),
                "rms": rms,
                "zcr": zcr,
                "flatness": flatness,
                "centroid_norm": float(centroid_norm),
                "sub_cv": sub_cv,
                "flux": flux,
                "label": label,
                "score": score_for_label(label),
            }
        )
    if not rows:
        return _empty_audio()
    return pd.DataFrame(rows)


def load_audio_file(path: str, target_sr: int = 22050, max_seconds: float = 90 * 60) -> tuple[np.ndarray, int, list[str]]:
    """Load a soundtrack as mono float audio. Returns waveform, sample rate, warnings."""

    import librosa

    warnings: list[str] = []
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as exc:  # decoder failures vary by backend
        raise AudioDecodeProxy(str(exc)) from exc
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size == 0:
        raise AudioDecodeProxy("The audio file did not contain any samples.")
    limit = int(max_seconds * sr)
    if y.size > limit:
        y = y[:limit]
        warnings.append(
            "The soundtrack was longer than 90 minutes, so only the first 90 minutes were analyzed."
        )
    return y, int(sr), warnings


class AudioDecodeProxy(Exception):
    """Internal decode error. ``pipeline`` rewrites this as ``AudioReadError``."""


def _empty_audio() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "t_start",
            "t_end",
            "rms",
            "zcr",
            "flatness",
            "centroid_norm",
            "sub_cv",
            "flux",
            "label",
            "score",
        ]
    )
