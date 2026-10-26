"""Multimodal Attendance & Engagement Agent.

Streamlit entry point. Processing stays in this Python process: no vision API,
no speech-to-text, and no database.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import streamlit as st

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.pipeline import session_from_bundle
from src.ui import inject_css, render_empty, render_error, render_hero, render_session, render_sidebar


def main() -> None:
    inject_css()
    render_hero()
    status = st.empty()
    settings, clock = render_sidebar(status)
    render_error()
    bundle = st.session_state.get("bundle")
    if bundle is None:
        render_empty()
        return
    try:
        session = session_from_bundle(bundle, settings)
    except Exception as exc:
        st.error(f"The session could not be scored: {exc}")
        return
    render_session(session, clock)


main()
