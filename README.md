# Multimodal Attendance & Engagement Agent

Privacy-first classroom intelligence for **Skill Wallet Capstone Project 5**. The app estimates sustained presence from faces, classifies room sound without transcribing it, fuses both into a collective engagement index, and offers teaching prompts about the room rather than about a named student.

Platform ID: `6ac775fb32e2e6be34c17468`

It runs entirely inside the Python process that serves Streamlit. There is no vision API, no speech-to-text, no speaker identification, and no database.

## What you can do

1. **Presence attendance.** Sample a webcam still, a photo, or a classroom video. MediaPipe BlazeFace assigns ephemeral track IDs. A track is marked Present only after it covers the sustained-presence threshold (default 70% of the session), so a hallway pass-through stays off the present list. Download the register as CSV: Student ID, First Seen, Last Seen, Total Duration, Status.
2. **Acoustic dynamics.** Classify a soundtrack into Silence, Instructor Lecture, Student Discussion / Verbal Interaction, and High Ambient Classroom Distraction. Words are never transcribed.
3. **Engagement fusion.** Combine forward head pose and acoustic energy into an index from 0 to 100, and plot the timeline.
4. **Teaching advisories.** When the room index stays under the alert line (default 45 for 5 minutes), the engine suggests a poll, a discussion, or a short stretch. Prompts do not name tracks.
5. **Session report.** KPIs, the register, the sound mix, and the advisories, as Markdown or JSON.

A simulated 42-minute algebra period exercises the same scoring code without a camera. A separate 32-second generated mix exercises the real librosa classifier.

## Run locally

Python 3.9 or newer. This repo is developed on 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints. For a classroom, use a computer you control. A still from the browser camera and any uploaded file are processed in that local process.

Run the tests:

```bash
pip install pytest
pytest
```

## Privacy, stated plainly

- Track IDs follow a place in the frame for a while. They are not names and not matches against a gallery. Someone who leaves for a long time and returns may get a new ID. That is intentional.
- The attendance table does not include a per-student attention score.
- Head pose is a geometric cue from eyes and nose, then shifted so a high camera does not mark the whole class as looking down. The OpenCV Haar cascade, used only if MediaPipe cannot load, does not invent a pose.
- Silence scores as 0.48, not zero. A quiet room is often listening.
- Gaps of about 15 seconds still count as in the room. Longer gaps do not. Looking down at a notebook can hide a face from the detector; that limit is called out in the report.
- Uploads are written to a temporary file, analyzed, and deleted. Session state holds results in memory until you close the app.

If you deploy to Streamlit Community Cloud, uploads are processed on Streamlit's machine for that session. This app still does not call a separate vision or speech service and still does not write a database. For a real class, run it locally.

## How the index is built

Engagement = 100 × (0.58 × share of faces with a forward head pose + 0.42 × acoustic score), then lightly smoothed.

| Room sound | Score |
| --- | --- |
| Silence | 0.48 |
| Instructor Lecture | 0.74 |
| Student Discussion / Verbal Interaction | 0.92 |
| High Ambient Classroom Distraction | 0.22 |

If one sensor is missing, the index uses the other and the report says so.

## Project layout

```
app.py                  Streamlit entry point
src/attendance.py       Tracking and sustained presence
src/face_engine.py      MediaPipe / OpenCV detection and pose
src/audio_dynamics.py   Librosa window features and labels
src/fusion.py           Time-step engagement index
src/advisory.py         Teaching prompts
src/reporting.py        Markdown and JSON reports
src/demo_session.py     Scripted period
src/ui.py               Dashboard
models/                 Bundled BlazeFace full-range model
tests/                  Presence, audio, fusion, report
```

Long videos are sampled (about two frames a second, capped) so a class period stays responsive. Presence ratios use those samples.

## Deploy on Streamlit Community Cloud

1. Publish this folder as a public GitHub repository. The main file is `app.py` on branch `main`.
2. Open [share.streamlit.io](https://share.streamlit.io), choose **New app**, and pick that repository.
3. Set the branch to `main` and the main file path to `app.py`.
4. Deploy. `packages.txt` installs the system libraries OpenCV and audio decoding need. `runtime.txt` asks for Python 3.12. No secrets are required.

The bundled face detector is Google's MediaPipe BlazeFace full-range model (`models/blaze_face_full_range.tflite`), Apache License 2.0. See `models/NOTICE`.

## Limits

This is a pedagogical instrument, not a disciplinary system, not an identification system, and not a diagnosis. Acoustic labels confuse music, video playback, and overlapping voices. Pose is a proxy for where a face is pointed, not for learning.
