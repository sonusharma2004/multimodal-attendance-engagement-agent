"""Streamlit layout for the educator dashboard."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from src import __version__
from src.advisory import advisory_rows
from src.formatutil import format_clock, format_duration, format_number
from src.models import DISPLAY_COLUMNS, Session, Settings
from src.selfcheck import library_versions, run_selfcheck
from src.theme import CORAL, CREAM, INK, LINE, PAPER, TEAL, TEAL_DARK

AUDIO_HELP = {
    "Silence": "Very low energy. Scored as listening, not as an empty room.",
    "Instructor Lecture": "Steady harmonic energy, read as one sustained voice.",
    "Student Discussion / Verbal Interaction": "Harmonic energy that keeps changing, read as talk between people.",
    "High Ambient Classroom Distraction": "Noise-like energy: scraping, clatter, playback, or a roar.",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background:
              radial-gradient(1100px 420px at 0% -10%, #E4F2EF 0%, {CREAM} 46%, {CREAM} 100%);
            color: {INK};
          }}
          .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            max-width: 1180px;
          }}
          section[data-testid="stSidebar"] {{
            background: #F3EFE8;
            border-right: 1px solid {LINE};
          }}
          section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
          }}
          h1, h2, h3, .mea-hero h1 {{
            font-family: Georgia, "Iowan Old Style", Palatino, "Palatino Linotype", serif;
            letter-spacing: -0.02em;
            color: {TEAL_DARK};
          }}
          .mea-hero {{
            margin-bottom: 0.4rem;
          }}
          .mea-kicker {{
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            color: {TEAL};
            font-weight: 650;
            margin-bottom: 0.35rem;
          }}
          .mea-hero h1 {{
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0 0 0.45rem 0;
            font-weight: 650;
          }}
          .mea-dek {{
            font-size: 1.05rem;
            line-height: 1.45;
            color: #243140;
            max-width: 46rem;
            margin: 0;
          }}
          .mea-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.85rem 0 0.2rem 0;
          }}
          .mea-pills span, .mea-source {{
            border: 1px solid {LINE};
            background: {PAPER};
            color: {TEAL_DARK};
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
          }}
          .mea-source {{
            display: inline-block;
            margin-bottom: 0.6rem;
            background: #E7F3F1;
          }}
          .mea-card {{
            background: {PAPER};
            border: 1px solid {LINE};
            border-radius: 18px;
            padding: 0.95rem 1rem 0.4rem 1rem;
            min-height: 9.5rem;
          }}
          .mea-card h3 {{
            margin: 0 0 0.35rem 0;
            font-size: 1.15rem;
          }}
          .mea-card p {{
            color: #334155;
            font-size: 0.95rem;
          }}
          .adv {{
            background: {PAPER};
            border: 1px solid {LINE};
            border-left: 6px solid {TEAL};
            border-radius: 16px;
            padding: 0.85rem 1rem 0.2rem 1rem;
            margin-bottom: 0.75rem;
          }}
          .adv.nudge {{ border-left-color: {CORAL}; }}
          .adv.celebrate {{ border-left-color: {TEAL}; }}
          .adv.info {{ border-left-color: #2C6BED; }}
          .adv .kicker {{
            color: #5C6776;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
          }}
          .adv h3 {{ margin: 0 0 0.35rem 0; font-size: 1.25rem; }}
          .adv .why {{ color: #5C6776; font-size: 0.92rem; }}
          div[data-testid="stMetric"] {{
            background: {PAPER};
            border: 1px solid {LINE};
            border-radius: 16px;
            padding: 0.7rem 0.85rem 0.4rem 0.85rem;
          }}
          div[data-testid="stMetric"] label {{
            color: #5C6776;
          }}
          @media (max-width: 800px) {{
            .mea-hero h1 {{ font-size: 1.7rem; }}
            section.main div[data-testid="stHorizontalBlock"] {{
              flex-direction: column;
            }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="mea-hero">
          <div class="mea-kicker">Privacy-first classroom analytics</div>
          <h1>Multimodal Attendance &amp; Engagement Agent</h1>
          <p class="mea-dek">
            A classroom instrument for sustained presence, room sound, and a collective engagement index.
            It advises the teacher about the room. It does not grade a student, store a face, or transcribe a word.
          </p>
          <div class="mea-pills">
            <span>Processed in this app process</span>
            <span>No face database</span>
            <span>No speech-to-text</span>
            <span>Room-level advisories</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(status) -> tuple[Settings, float | None]:
    st.sidebar.markdown("### Start a session")
    st.sidebar.caption("Real classrooms should run this app on a computer you control. Nothing here calls a cloud vision or speech API.")
    if st.sidebar.button("Explore a simulated period", type="primary", key="demo-side"):
        _load_demo()
    video = st.sidebar.file_uploader(
        "Classroom video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        help="Sampled on this machine. Used for presence and head pose.",
    )
    audio = st.sidebar.file_uploader(
        "Classroom soundtrack",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Energy and spectrum only. WAV is the most reliable. Words are not transcribed.",
    )
    photo = st.sidebar.camera_input("Still from this computer's camera", help="A snapshot cannot mark attendance.")
    still = st.sidebar.file_uploader("Or upload a still", type=["jpg", "jpeg", "png", "webp"])
    if st.sidebar.button("Process uploaded media", key="process-side"):
        _process_uploads(video, audio, photo, still, status)

    st.sidebar.markdown("### Thresholds")
    presence = st.sidebar.slider(
        "Sustained presence",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        format="%d%%",
        help="A track is marked present only if it stays in frame for at least this share of the session. The default stops a hallway pass-through from becoming an attendance mark.",
    )
    alert = st.sidebar.slider(
        "Engagement alert",
        min_value=20,
        max_value=80,
        value=45,
        step=5,
        help="Advisories look for a collective index below this line.",
    )
    minutes = st.sidebar.slider(
        "Alert persistence (minutes)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="The low-engagement advisory waits until the dip has lasted this long.",
    )
    expected = st.sidebar.number_input(
        "Expected class size",
        min_value=0,
        max_value=200,
        value=0,
        step=1,
        help="Optional denominator for a coverage hint. Unseen students are not invented as rows. Leave 0 to use the simulated roll, or none for an upload.",
    )
    settings = Settings(
        presence_threshold=presence / 100.0,
        alert_level=float(alert),
        alert_minutes=float(minutes),
        expected_class_size=int(expected) if expected else None,
    )

    bundle = st.session_state.get("bundle")
    clock = None
    if bundle is not None and bundle.duration_s > 0:
        st.sidebar.markdown("### Replay")
        duration = float(bundle.duration_s)
        step = 1.0 if duration <= 180 else 15.0 if duration <= 20 * 60 else 30.0
        clock = st.sidebar.slider(
            "Session clock",
            min_value=0.0,
            max_value=duration,
            value=duration,
            step=step,
            key=f"clock-{bundle.source}-{bundle.title}-{int(duration)}",
            help="Engagement and advisories reveal themselves up to this moment.",
        )
        st.sidebar.caption(f"Through {format_clock(clock)} of {format_clock(duration)}.")

    with st.sidebar.expander("Privacy principles"):
        st.markdown(
            """
            - Video is not sent to an external vision API.
            - Audio is not transcribed and speakers are not identified.
            - Track IDs are ephemeral. Returning after a long gap can create a new ID on purpose.
            - The engagement index describes the room. The register does not store a per-student attention score.
            - A hosted demo still runs on the machine that serves this app. For a real class, run it locally.
            """
        )
    versions = library_versions()
    st.sidebar.caption(
        f"v{__version__} · Python {versions.get('python', '')} · Streamlit {versions.get('streamlit', '')}"
    )
    return settings, clock


def render_empty() -> None:
    st.markdown("### Two ways to open the dashboard")
    st.markdown(
        "Load a simulated algebra period to see attendance, sound, engagement, and advisories. "
        "Or process a recording. A hallway glimpse is not marked present, and a quiet minute is not treated as a failure."
    )
    left, mid, right = st.columns(3)
    with left:
        st.markdown(
            """
            <div class="mea-card">
              <h3>Simulated period</h3>
              <p>A 42-minute scripted class. The scoring code is real. The camera is not.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            """
            <div class="mea-card">
              <h3>Video</h3>
              <p>MediaPipe tracks faces across the clip. Presence needs most of the session, not one frame.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="mea-card">
              <h3>Soundtrack</h3>
              <p>Librosa sorts energy into silence, lecture, discussion, or distraction. No transcript.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    action_left, action_right = st.columns(2)
    with action_left:
        if st.button("Load the simulated period", type="primary", key="demo-main"):
            _load_demo()
            st.rerun()
    with action_right:
        if st.button("Run the offline self-check", key="check-empty"):
            _run_checks()
    if st.session_state.get("checks"):
        render_checks(st.session_state["checks"])


def render_error() -> None:
    error = st.session_state.get("error")
    if not error:
        return
    message, trace = error
    st.error(message)
    with st.expander("Technical detail"):
        st.code(trace)


def render_session(session: Session, clock_s: float | None) -> None:
    st.markdown(f'<span class="mea-source">{_escape(session.source)} · {_escape(session.modality)}</span>', unsafe_allow_html=True)
    st.markdown(session.narrative)
    for warning in session.warnings:
        st.warning(warning)
    presence, sound, engage, advice, report = st.tabs(
        [
            "Presence attendance",
            "Acoustic dynamics",
            "Engagement fusion",
            "Teaching advisories",
            "Session report",
        ]
    )
    with presence:
        render_presence(session)
    with sound:
        render_acoustic(session, clock_s)
    with engage:
        render_engagement(session, clock_s)
    with advice:
        render_advisories(session, clock_s)
    with report:
        render_report(session)


def render_presence(session: Session) -> None:
    st.markdown("#### Who stayed in the frame")
    st.caption(
        "Present means the track covered the sustained-presence threshold for this session. "
        "Absent means the track was seen and did not. Unseen students are not invented as rows."
    )
    kpis = session.kpis
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session length", format_duration(session.duration_s))
    c2.metric("Marked present", f"{kpis.get('present', 0)}")
    c3.metric("Not marked present", f"{kpis.get('below_threshold', 0)}")
    c4.metric("Brief appearances", f"{kpis.get('brief_appearances', 0)}")
    if session.roster_size:
        coverage = kpis.get("coverage")
        st.caption(
            f"Roll size used for comparison: {session.roster_size}. "
            + (f"Present tracks cover {coverage:.0%} of that number. " if coverage is not None else "")
            + "That comparison is not an identified absence list."
        )
    if session.attendance is not None and not session.attendance.empty:
        show_fig(fig_presence(session.attendance, session.threshold))
        show_df(_styled_attendance(session.attendance[DISPLAY_COLUMNS]))
        st.download_button(
            "Download attendance register (CSV)",
            data=session.attendance.to_csv(index=False).encode("utf-8"),
            file_name="attendance_register.csv",
            mime="text/csv",
            key="csv-attendance",
        )
    elif session.snapshot is None:
        st.info(
            "No faces were tracked long enough to build a register. If this was a real recording, "
            "check light, camera angle, and that faces are large enough to see. An empty detector is not an absence list."
        )
    if session.snapshot is not None:
        _render_snapshot(session)
    if session.annotated_frames:
        st.markdown("#### Sampled frames")
        st.caption("Boxes are ephemeral track IDs, not names. Color is not an attentiveness grade.")
        columns = st.columns(2)
        for index, frame in enumerate(session.annotated_frames):
            with columns[index % 2]:
                show_image(frame["image"], f"Sample at {format_clock(frame['t'])}")
    with st.expander("How sustained presence is decided"):
        st.markdown(
            f"""
            A track is **{session.threshold:.0%}** present only when the time it was actually covered reaches that share of the session.
            Gaps of 15 seconds or less still count as in the room, so a turned head or a person walking in front of someone else does not erase them.
            The tracker will keep the same ID across gaps up to 15 minutes if the person returns to a similar place in the frame.
            It will not match a face to a name. Pose method for this session: **{session.pose_method}**.
            """
        )


def render_acoustic(session: Session, clock_s: float | None) -> None:
    st.markdown("#### How the room sounded")
    st.caption("Four labels, from spectrum and energy stability. This is not a transcript and not a speaker list.")
    mix = session.kpis.get("audio_mix") or {}
    short_labels = {
        "Silence": "Silence",
        "Instructor Lecture": "Lecture",
        "Student Discussion / Verbal Interaction": "Discussion",
        "High Ambient Classroom Distraction": "Distraction",
    }
    if mix:
        cols = st.columns(4)
        for column, label in zip(cols, mix):
            column.metric(short_labels.get(label, label), f"{mix[label]:.0%}", help=AUDIO_HELP.get(label, label))
        show_fig(fig_mix(mix))
        if session.timeline is not None and not session.timeline.empty:
            show_fig(fig_audio_strip(session.timeline, clock_s))
            st.caption("Silence is grey, lecture is teal, discussion is blue, distraction is coral.")
        if session.audio is not None and not session.audio.empty:
            with st.expander("Window table"):
                show_df(session.audio.head(300))
            st.download_button(
                "Download sound windows (CSV)",
                data=session.audio.to_csv(index=False).encode("utf-8"),
                file_name="room_sound_windows.csv",
                mime="text/csv",
                key="csv-audio",
            )
    else:
        st.info("This session has no soundtrack. Upload audio, or run the synthetic mix below to see the real classifier.")
    st.markdown("#### Classifier proof")
    st.caption("A 32-second generated signal: silence, a steady harmonic voice, alternating voices, then noise. Librosa classifies it on this machine.")
    if st.button("Classify the synthetic room mix", key="mix-button"):
        _run_mix()
    mix_frame = st.session_state.get("mix")
    if mix_frame is not None:
        show_fig(fig_audio_strip(mix_frame, None))
        summary = (
            mix_frame.groupby("label", sort=False)["t_end"]
            .count()
            .rename("windows")
            .reset_index()
        )
        show_df(summary)
        with st.expander("Windows"):
            show_df(mix_frame[["t_start", "t_end", "label", "rms", "flatness", "sub_cv", "flux"]])


def render_engagement(session: Session, clock_s: float | None) -> None:
    st.markdown("#### One index for the room")
    st.caption(
        "0.58 × share of faces looking forward, plus 0.42 × acoustic interaction. "
        "Silence contributes 0.48, not zero. The line is smoothed."
    )
    timeline = session.timeline
    if timeline is None or timeline.empty or not timeline["engagement"].notna().any():
        st.info("There is not enough signal yet to draw an engagement curve.")
        return
    view = timeline if clock_s is None else timeline[timeline["t_start"] < clock_s + 1e-9]
    if view.empty or not view["engagement"].notna().any():
        st.info("Move the session clock forward to reveal the curve.")
        return
    kpis_now = _slice_kpis(view, session.alert_level)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean index", format_number(kpis_now["mean"], 0))
    c2.metric("Lowest", format_number(kpis_now["low"], 0))
    c3.metric("Highest", format_number(kpis_now["high"], 0))
    c4.metric(f"Time below {session.alert_level:.0f}", format_duration(kpis_now["below_s"]))
    show_fig(fig_engagement(timeline, session.alert_level, clock_s))
    st.caption(f"Showing through {format_clock(clock_s if clock_s is not None else session.duration_s)}.")
    weights = view[["w_visual", "w_audio"]].mean(numeric_only=True) if {"w_visual", "w_audio"}.issubset(view.columns) else None
    if weights is not None:
        st.caption(
            f"Average weight in this view: visual {weights['w_visual']:.2f}, acoustic {weights['w_audio']:.2f}. "
            "A weight of 1 means the other sensor was missing."
        )
    st.download_button(
        "Download engagement timeline (CSV)",
        data=timeline.to_csv(index=False).encode("utf-8"),
        file_name="engagement_timeline.csv",
        mime="text/csv",
        key="csv-timeline",
    )
    with st.expander("How to read the index"):
        st.markdown(
            """
            Forward head pose is the share of detected faces whose nose sits near the middle of the eyes,
            after the session's own camera angle is subtracted. A high-mounted camera does not automatically
            mark the whole class as looking down.

            Acoustic scores: silence 0.48, instructor lecture 0.74, student discussion 0.92,
            high ambient distraction 0.22. Discussion raises the index. Clatter pulls it down.
            A silent, forward-facing room stays in the middle, which is what listening often looks like.

            The advisory engine reads the smoothed index, so one shouted second does not page the teacher.
            """
        )


def render_advisories(session: Session, clock_s: float | None) -> None:
    st.markdown("#### What the teacher might do next")
    st.caption("Heuristic prompts about the room. They name an activity, not a student. Drag the session clock to replay them.")
    visible = [
        item
        for item in session.advisories
        if clock_s is None or item.t_start <= clock_s + 1e-6
    ]
    if not visible:
        st.info("No advisory at this point in the session. The engine stays quiet while the room is inside its normal range.")
    for item in visible:
        st.markdown(
            f"""
            <div class="adv {item.severity}">
              <div class="kicker">{format_clock(item.t_start)}–{format_clock(item.t_end)} · {item.severity}</div>
              <h3>{_escape(item.title)}</h3>
              <p>{_escape(item.message)}</p>
              <p class="why">{_escape(item.rationale)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Rules the engine actually uses"):
        st.markdown(
            f"""
            - If the smoothed index stays under **{session.alert_level:.0f}** for **{session.alert_minutes:.0f} minutes**, suggest a poll, a discussion, or a two-minute stretch.
            - If lecture runs 12 minutes and the index slides by 12 points without crossing that line, suggest a check for understanding.
            - If the index recovers by about 18 points after a dip, say so.
            - If student talk dominates a healthy stretch, suggest a short share-out.
            - If the last six minutes are steady, suggest an exit ticket read as a class signal.
            - Overlapping prompts collapse. A nudge wins over a celebration of the same minutes.
            """
        )
    if visible:
        table = advisory_rows(visible)
        st.download_button(
            "Download advisories (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="teaching_advisories.csv",
            mime="text/csv",
            key="csv-advice",
        )


def render_report(session: Session) -> None:
    st.markdown("#### Session summary")
    k = session.kpis
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Marked present", f"{k.get('present', 0)}")
    c2.metric("Mean engagement", format_number(k.get("mean_engagement"), 0))
    c3.metric("Verbal interaction", _pct(k.get("discussion_share")))
    c4.metric("Advisories", f"{k.get('advisory_count', 0)}")
    from src.advisory import carry_forward

    st.markdown("##### Carry into the next opening")
    st.markdown(carry_forward(session.advisories))
    if k.get("audio_mix"):
        show_fig(fig_mix(k["audio_mix"]))
    st.markdown(session.report_markdown)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download report (Markdown)",
            data=session.report_markdown.encode("utf-8"),
            file_name="session_report.md",
            mime="text/markdown",
            key="dl-md",
        )
    with d2:
        st.download_button(
            "Download summary (JSON)",
            data=session.report_json.encode("utf-8"),
            file_name="session_summary.json",
            mime="application/json",
            key="dl-json",
        )
    with d3:
        if session.attendance is not None and not session.attendance.empty:
            st.download_button(
                "Download register (CSV)",
                data=session.attendance.to_csv(index=False).encode("utf-8"),
                file_name="attendance_register.csv",
                mime="text/csv",
                key="dl-csv-report",
            )
    st.markdown("#### Offline self-check")
    if st.button("Run self-check", key="check-report"):
        _run_checks()
    if st.session_state.get("checks"):
        render_checks(st.session_state["checks"])


def render_checks(checks: list[dict]) -> None:
    passed = sum(1 for item in checks if item["passed"])
    st.markdown(f"**{passed} of {len(checks)} checks passed.**")
    for item in checks:
        mark = "Pass" if item["passed"] else "Fail"
        st.markdown(f"**{mark} · {item['name']}.** {item['detail']}")


def show_fig(fig) -> None:
    st.pyplot(fig, clear_figure=True)


def show_image(image, caption: str) -> None:
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def show_df(frame) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _styled_attendance(frame: pd.DataFrame):
    def paint(row: pd.Series) -> list[str]:
        if row.get("Status") == "Present":
            color = "background-color: #E5F4EC; color: #145C32"
        elif row.get("Status") == "Absent":
            color = "background-color: #F8EEE8; color: #8A4B22"
        else:
            color = ""
        return [color] * len(row)

    return frame.style.apply(paint, axis=1)


def _render_snapshot(session: Session) -> None:
    snap = session.snapshot or {}
    st.markdown("#### Still frame")
    st.caption("A single picture cannot support sustained presence. Counts below are a detector check, not an attendance mark and not a ranking.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Faces detected", f"{snap.get('faces', 0)}")
    c2.metric("Forward pose", f"{snap.get('forward', 0)}")
    c3.metric("Turned from the camera", f"{snap.get('turned', 0)}")
    if snap.get("pose_unknown"):
        st.caption(f"{snap['pose_unknown']} face(s) had no pose estimate.")
    image = snap.get("image")
    if image is not None:
        show_image(image, "Ephemeral boxes only. Color is not an attentiveness grade.")


def _slice_kpis(view: pd.DataFrame, alert_level: float) -> dict:
    import numpy as np

    values = pd.to_numeric(view["engagement"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    below = view[pd.to_numeric(view["engagement"], errors="coerce") < alert_level]
    below_s = float((below["t_end"] - below["t_start"]).sum()) if not below.empty else 0.0
    return {
        "mean": float(np.mean(finite)) if finite.size else None,
        "low": float(np.min(finite)) if finite.size else None,
        "high": float(np.max(finite)) if finite.size else None,
        "below_s": below_s,
    }


def _pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "—"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _load_demo() -> None:
    from src.demo_session import make_demo_bundle

    st.session_state["bundle"] = make_demo_bundle()
    st.session_state["error"] = None
    st.toast("Simulated period loaded. No camera was used.")


def _run_checks() -> None:
    with st.spinner("Running presence, acoustics, fusion, and the face detector locally…"):
        st.session_state["checks"] = run_selfcheck(get_engine())


def _run_mix() -> None:
    from src.audio_dynamics import extract_and_classify
    from src.synthetic_audio import room_mix

    with st.spinner("Classifying a generated 32-second mix with librosa…"):
        waveform, sample_rate, _segments = room_mix()
        st.session_state["mix"] = extract_and_classify(waveform, sample_rate)


def _process_uploads(video, audio, photo, still, status) -> None:
    from src.pipeline import bgr_from_encoded, ingest_media

    if not any([video, audio, photo, still]):
        st.sidebar.warning("Add a video, a soundtrack, or a still first.")
        return
    paths: list[str] = []
    try:
        video_path = _spill(video, ".mp4", paths) if video is not None else None
        audio_path = _spill(audio, _suffix(audio.name if audio else "", ".wav"), paths) if audio is not None else None
        image = None
        image_name = "still"
        chosen = photo if photo is not None else still
        if chosen is not None:
            image = bgr_from_encoded(chosen.getvalue())
            image_name = getattr(chosen, "name", "camera-still")
        bar = status.progress(0.0, text="Starting local processing…")

        def progress(index: int, total: int, stage: str) -> None:
            frac = (index + 1) / max(total, 1)
            if stage == "detect":
                frac *= 0.92
            bar.progress(min(frac, 1.0), text=f"Sampling on this machine · frame {index + 1} of {total}")

        bundle = ingest_media(
            video_path=video_path,
            audio_path=audio_path,
            image_bgr=image,
            video_name=getattr(video, "name", "video"),
            audio_name=getattr(audio, "name", "audio"),
            image_name=image_name,
            engine=get_engine(),
            progress=progress if video_path else None,
        )
        bar.progress(1.0, text="Finished. Media was not written to a database.")
        st.session_state["bundle"] = bundle
        st.session_state["error"] = None
        st.toast("Session processed in this app.")
    except Exception as exc:
        import traceback

        st.session_state["error"] = (str(exc), traceback.format_exc())
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


def _spill(upload, suffix: str, paths: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.write(upload.getvalue())
    handle.close()
    paths.append(handle.name)
    return handle.name


def _suffix(name: str, default: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext and len(ext) <= 5:
        return ext
    return default


@st.cache_resource(show_spinner=False)
def get_engine():
    from src.face_engine import FaceEngine

    return FaceEngine()


# Imported at the bottom of chart helpers so a missing matplotlib install fails inside the tab, not at import.
def fig_presence(attendance, threshold):
    from src.visualization import fig_presence as build

    return build(attendance, threshold)


def fig_mix(mix):
    from src.visualization import fig_mix as build

    return build(mix)


def fig_audio_strip(frame, clock_s):
    from src.visualization import fig_audio_strip as build

    return build(frame, clock_s)


def fig_engagement(timeline, alert_level, clock_s):
    from src.visualization import fig_engagement as build

    return build(timeline, alert_level, clock_s)
