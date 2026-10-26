"""A short synthetic room mix used to test and demonstrate the acoustic classifier.

The signal is generated, not recorded. It is not a substitute for a classroom
soundtrack. It only checks that silence, steady harmonics, changing harmonics,
and noise land in different bins.
"""

from __future__ import annotations

import numpy as np

from src.audio_dynamics import DISCUSSION, DISTRACTION, LECTURE, SILENCE


def room_mix(sr: int = 22050, seconds: int = 8) -> tuple[np.ndarray, int, list[tuple[float, float, str]]]:
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    rng = np.random.default_rng(5)

    silence = rng.normal(0.0, 0.0008, n).astype(np.float32)

    lecture = (
        0.22 * np.sin(2 * np.pi * 170 * t)
        + 0.11 * np.sin(2 * np.pi * 340 * t)
        + 0.05 * np.sin(2 * np.pi * 510 * t)
    )
    lecture *= 0.94 + 0.06 * np.sin(2 * np.pi * 2.2 * t)
    lecture = lecture.astype(np.float32)

    discussion = np.zeros(n, dtype=np.float32)
    chunk = max(1, sr // 4)
    for index, start in enumerate(range(0, n, chunk)):
        stop = min(n, start + chunk)
        local = np.linspace(0.0, (stop - start) / sr, stop - start, endpoint=False)
        f0 = 150.0 if index % 2 == 0 else 240.0
        amp = 0.10 if index % 2 == 0 else 0.45
        piece = amp * (np.sin(2 * np.pi * f0 * local) + 0.35 * np.sin(2 * np.pi * 2 * f0 * local))
        fade = min(int(0.02 * sr), max(1, (stop - start) // 4))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        discussion[start:stop] = piece.astype(np.float32)

    distraction = rng.normal(0.0, 0.22, n).astype(np.float32)
    y = np.concatenate([silence, lecture, discussion, distraction]).astype(np.float32)
    span = float(seconds)
    segments = [
        (0.0, span, SILENCE),
        (span, span * 2, LECTURE),
        (span * 2, span * 3, DISCUSSION),
        (span * 3, span * 4, DISTRACTION),
    ]
    return y, sr, segments
