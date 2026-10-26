"""Matplotlib figures for the educator dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.audio_dynamics import AUDIO_LABELS, DISCUSSION, DISTRACTION, LECTURE, SILENCE, UNOBSERVED
from src.theme import (
    ABSENT,
    DISTRACTION as DISTRACTION_COLOR,
    DISCUSSION as DISCUSSION_COLOR,
    INK,
    LECTURE as LECTURE_COLOR,
    MUTED,
    PAPER,
    PRESENT,
    SILENCE as SILENCE_COLOR,
    TEAL,
    UNOBSERVED as UNOBSERVED_COLOR,
    CORAL,
)

LABEL_COLORS = {
    SILENCE: SILENCE_COLOR,
    LECTURE: LECTURE_COLOR,
    DISCUSSION: DISCUSSION_COLOR,
    DISTRACTION: DISTRACTION_COLOR,
    UNOBSERVED: UNOBSERVED_COLOR,
}


def fig_engagement(timeline: pd.DataFrame, alert_level: float, clock_s: float | None = None):
    import matplotlib.pyplot as plt

    frame = _until(timeline, clock_s)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 0.55, 1.7], "hspace": 0.12},
        layout="constrained",
    )
    _style_fig(fig)
    if frame is None or frame.empty or "engagement" not in frame.columns or not frame["engagement"].notna().any():
        axes[0].text(0.5, 0.5, "Not enough signal to draw an engagement curve.", ha="center", va="center", color=MUTED)
        for axis in axes:
            axis.axis("off")
        return fig

    x, xlabel = _x(frame["t_mid"].to_numpy(dtype=float), float(frame["t_end"].iloc[-1]))
    engagement = frame["engagement"].to_numpy(dtype=float)
    raw = frame["engagement_raw"].to_numpy(dtype=float) if "engagement_raw" in frame.columns else None
    ax = axes[0]
    if raw is not None:
        ax.plot(x, raw, color=TEAL, alpha=0.28, lw=1.2, label="Unsmoothed")
    ax.plot(x, engagement, color=TEAL, lw=2.2, label="Engagement index")
    ax.axhline(alert_level, color=CORAL, ls="--", lw=1.1, label=f"Alert {alert_level:.0f}")
    ax.fill_between(x, engagement, alert_level, where=np.isfinite(engagement) & (engagement < alert_level), color=CORAL, alpha=0.16, interpolate=True)
    ax.set_ylim(-2, 104)
    ax.set_ylabel("Index")
    ax.set_title("Collective engagement")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _style_ax(ax)

    _paint_labels(axes[1], frame, x, legend=False)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Sound", color=MUTED)
    _style_ax(axes[1], grid=False)

    visual = pd.to_numeric(frame["visual"], errors="coerce").to_numpy(dtype=float) * 100.0 if "visual" in frame else None
    axv = axes[2]
    if visual is not None and np.isfinite(visual).any():
        axv.plot(x, visual, color="#2C6BED", lw=1.8, label="Forward head pose")
        axv.set_ylim(-2, 104)
        axv.legend(frameon=False, loc="upper right")
    else:
        axv.text(0.01, 0.5, "No head-pose signal in this view.", transform=axv.transAxes, color=MUTED, va="center")
        axv.set_ylim(0, 1)
    axv.set_ylabel("Pose %")
    axv.set_xlabel(xlabel)
    _style_ax(axv)
    return fig


def fig_audio_strip(frame: pd.DataFrame, clock_s: float | None = None):
    import matplotlib.pyplot as plt

    data = frame
    time_col = "t_mid" if "t_mid" in frame.columns else None
    if time_col is None and {"t_start", "t_end"}.issubset(frame.columns):
        data = frame.copy()
        data["t_mid"] = (data["t_start"] + data["t_end"]) / 2.0
    data = _until(data, clock_s) if "t_start" in data.columns else data
    fig, ax = plt.subplots(figsize=(11.2, 2.4), layout="constrained")
    _style_fig(fig)
    if data.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No soundtrack windows.", ha="center", va="center", color=MUTED)
        return fig
    x, xlabel = _x(data["t_mid"].to_numpy(dtype=float), float(data["t_end"].iloc[-1]))
    _paint_labels(ax, data, x)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title("Room-sound labels")
    _style_ax(ax, grid=False)
    return fig


def fig_presence(attendance: pd.DataFrame, threshold: float):
    import matplotlib.pyplot as plt

    fig_height = max(3.2, min(0.32 * max(len(attendance), 1) + 1.2, 12))
    fig, ax = plt.subplots(figsize=(11.2, fig_height), layout="constrained")
    _style_fig(fig)
    if attendance is None or attendance.empty or "Presence Ratio" not in attendance.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "No tracks to chart.", ha="center", va="center", color=MUTED)
        return fig
    view = attendance.head(40).iloc[::-1]
    colors = [PRESENT if status == "Present" else ABSENT for status in view["Status"]]
    ax.barh(view["Student ID"].astype(str), view["Presence Ratio"].astype(float) * 100.0, color=colors, height=0.72)
    ax.axvline(threshold * 100.0, color=INK, ls="--", lw=1.0, label=f"{threshold:.0%} sustained-presence line")
    ax.set_xlabel("Share of the session in frame")
    ax.set_xlim(0, 100)
    ax.set_title("Sustained presence")
    ax.legend(frameon=False, loc="lower right")
    _style_ax(ax, grid=False)
    ax.grid(axis="x", color="#EFE8DE", lw=0.8)
    return fig


def fig_mix(mix: dict[str, float]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.2, 3.1), layout="constrained")
    _style_fig(fig)
    labels = [label for label in AUDIO_LABELS if label in mix]
    if not labels:
        ax.axis("off")
        ax.text(0.5, 0.5, "No room-sound mix for this session.", ha="center", va="center", color=MUTED)
        return fig
    values = [mix[label] * 100.0 for label in labels]
    colors = [LABEL_COLORS[label] for label in labels]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of scored time")
    ax.set_title("How the room sounded")
    _style_ax(ax, grid=False)
    ax.grid(axis="x", color="#EFE8DE", lw=0.8)
    return fig


def _paint_labels(axis, frame: pd.DataFrame, x: np.ndarray, legend: bool = True) -> None:
    import matplotlib.patches as mpatches

    if "audio_label" not in frame.columns or len(frame) == 0:
        return
    labels = frame["audio_label"].astype(str).to_numpy()
    colors = [LABEL_COLORS.get(label, UNOBSERVED_COLOR) for label in labels]
    axis.bar(x, np.ones(len(x)), width=_bar_width(x), color=colors, align="center")
    axis.set_ylim(0, 1)
    handles = [
        mpatches.Patch(color=LABEL_COLORS[label], label=label)
        for label in list(AUDIO_LABELS) + [UNOBSERVED]
        if label in set(labels)
    ]
    if legend and handles:
        axis.legend(
            handles=handles,
            frameon=False,
            ncol=min(2, len(handles)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.55),
        )


def _bar_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.8
    return float(np.median(np.diff(x))) * 0.98


def _until(frame: pd.DataFrame, clock_s: float | None) -> pd.DataFrame:
    if frame is None or frame.empty or clock_s is None or "t_start" not in frame.columns:
        return frame
    return frame[frame["t_start"] < float(clock_s) + 1e-9].copy()


def _x(seconds: np.ndarray, duration_s: float) -> tuple[np.ndarray, str]:
    if duration_s >= 180:
        return seconds / 60.0, "Minutes into the session"
    return seconds, "Seconds into the session"


def _style_fig(fig) -> None:
    fig.patch.set_facecolor(PAPER)


def _style_ax(axis, grid: bool = True) -> None:
    axis.set_facecolor(PAPER)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color("#E4DDD2")
    axis.spines["bottom"].set_color("#E4DDD2")
    if grid:
        axis.grid(axis="y", color="#EFE8DE", lw=0.8)
        axis.set_axisbelow(True)
