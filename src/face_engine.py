"""On-device face detection and a geometric head-pose cue.

MediaPipe's full-range BlazeFace detector is the primary backend because a
classroom camera sees many faces at once. Head pose is estimated from eye and
nose keypoints, then calibrated to the session median so a high-mounted camera
does not mark the whole room as looking down.

The OpenCV Haar cascade is a presence-only fallback. It does not invent a pose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "blaze_face_full_range.tflite"
CASCADE_PATH = Path(__file__).resolve().parents[1] / "models" / "haarcascade_frontalface_default.xml"

YAW_LIMIT = 25.0
PITCH_LIMIT = 16.0


def pose_from_points(
    eye_a: tuple[float, float] | None,
    eye_b: tuple[float, float] | None,
    nose: tuple[float, float] | None,
    mouth: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Return yaw and pitch in degrees from image-space keypoints.

    The estimate is scale-invariant. Yaw sign depends on which eye is passed
    first; callers should use ``abs(yaw)``. Pitch is positive when the nose sits
    lower than a neutral frontal face (a downward gaze in a typical webcam view).
    ``mouth`` is accepted so callers can pass it through, and reserved for a
    finer pitch estimate.
    """

    del mouth
    if eye_a is None or eye_b is None or nose is None:
        return None
    lx, ly = eye_a
    rx, ry = eye_b
    nx, ny = nose
    dx, dy = rx - lx, ry - ly
    inter = float(np.hypot(dx, dy))
    if inter < 1e-6:
        return None
    ux, uy = dx / inter, dy / inter
    px, py = -uy, ux
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    vx, vy = nx - mx, ny - my
    if (vx * px + vy * py) < 0:
        px, py = -px, -py
    along = vx * ux + vy * uy
    perp = vx * px + vy * py
    yaw = (along / inter) * 70.0
    pitch = ((perp / inter) - 0.45) * 80.0
    return float(yaw), float(pitch)


def is_forward(yaw: float, pitch: float, baseline_pitch: float = 0.0) -> bool:
    return abs(yaw) <= YAW_LIMIT and abs(pitch - baseline_pitch) <= PITCH_LIMIT


def apply_forward_flags(samples: list[dict], min_faces_for_baseline: int = 8) -> float:
    """Set ``forward`` and ``pose_known`` on each detection. Returns the pitch baseline."""

    pitches: list[float] = []
    for sample in samples:
        for det in sample.get("dets", []):
            yaw = det.get("yaw")
            pitch = det.get("pitch")
            if yaw is None or pitch is None:
                continue
            if abs(float(yaw)) <= 20.0:
                pitches.append(float(pitch))
    if len(pitches) < min_faces_for_baseline:
        baseline = 0.0
    else:
        baseline = float(np.median(np.asarray(pitches, dtype=float)))
    for sample in samples:
        for det in sample.get("dets", []):
            yaw = det.get("yaw")
            pitch = det.get("pitch")
            if yaw is None or pitch is None:
                det["pose_known"] = False
                det["forward"] = None
            else:
                det["pose_known"] = True
                det["forward"] = is_forward(float(yaw), float(pitch), baseline)
    return baseline


def resize_width(bgr: np.ndarray, width: int = 1280) -> np.ndarray:
    import cv2

    height, current = bgr.shape[:2]
    if current <= width:
        return bgr
    scale = width / float(current)
    return cv2.resize(bgr, (width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


class FaceEngine:
    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.backend = "unavailable"
        self.load_error: str | None = None
        self._detector = None
        self._cascade = None
        self._try_mediapipe()
        if self._detector is None:
            self._init_haar()

    @property
    def pose_method(self) -> str:
        if self.backend == "mediapipe":
            return "MediaPipe BlazeFace full-range keypoints, calibrated to this recording"
        if self.backend == "opencv-haar":
            return "OpenCV Haar cascade, presence only — head pose is not estimated"
        return "No face detector available"

    def _try_mediapipe(self) -> None:
        if not self.model_path.is_file():
            self.load_error = f"Face model not found at {self.model_path}"
            return
        try:
            import os

            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import mediapipe as mp

            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            self._mp = mp
            self.backend = "mediapipe"
            self.load_error = None
            import atexit

            atexit.register(self.close)
        except Exception as exc:  # import errors, missing delegates, bad model
            self._detector = None
            self.backend = "unavailable"
            self.load_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        """Drop the detector before interpreter shutdown.

        MediaPipe's own destructor schedules work on a thread pool that is
        already gone by the time ``__del__`` runs, which prints a shutdown
        error. Clearing the handle makes that destructor a no-op. The process
        is exiting, so the native graph is released with it.
        """

        detector = getattr(self, "_detector", None)
        self._detector = None
        if detector is None:
            return
        try:
            detector._handle = None
        except Exception:
            return

    def _init_haar(self) -> None:
        try:
            import cv2

            candidates = [CASCADE_PATH]
            bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if bundled:
                candidates.append(Path(bundled) / "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier()
            loaded = False
            for path in candidates:
                if path.is_file() and cascade.load(str(path)):
                    loaded = True
                    break
            if not loaded or cascade.empty():
                self.load_error = (self.load_error or "") + " Haar cascade failed to load."
                return
            self._cascade = cascade
            self.backend = "opencv-haar"
        except Exception as exc:
            self._cascade = None
            self.load_error = f"{self.load_error or ''} {type(exc).__name__}: {exc}".strip()

    def detect(self, bgr: np.ndarray) -> list[dict]:
        if bgr is None or bgr.size == 0:
            return []
        if self._detector is not None:
            try:
                return self._detect_mediapipe(bgr)
            except Exception as exc:
                self.load_error = f"MediaPipe detect failed ({type(exc).__name__}: {exc}). Switched to OpenCV."
                self._detector = None
                self.backend = "unavailable"
                if self._cascade is None:
                    self._init_haar()
        if self._cascade is not None:
            return self._detect_haar(bgr)
        return []

    def _detect_mediapipe(self, bgr: np.ndarray) -> list[dict]:
        import cv2

        rgb = _align_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        height, width = bgr.shape[:2]
        detections = []
        for found in getattr(result, "detections", None) or []:
            parsed = _parse_mediapipe_detection(found, width, height)
            if parsed is not None:
                detections.append(parsed)
        return _nms(detections)

    def _detect_haar(self, bgr: np.ndarray) -> list[dict]:
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detections = []
        for x, y, w, h in rects:
            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": None,
                    "yaw": None,
                    "pitch": None,
                    "forward": None,
                    "pose_known": False,
                }
            )
        return _nms(detections, iou_thresh=0.4)


def draw_tracks(bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Draw ephemeral track ids. Color is not an attentiveness mark."""

    import cv2

    out = bgr.copy()
    color = (100, 107, 14)  # BGR teal
    for det in dets:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = str(det.get("student_id") or "")
        if label:
            cv2.putText(
                out,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _align_rgb(rgb: np.ndarray, alignment: int = 64) -> np.ndarray:
    """Copy into an aligned buffer so LiteRT does not reallocate every frame."""

    rgb = np.ascontiguousarray(rgb)
    if rgb.ctypes.data % alignment == 0:
        return rgb
    flat = np.ascontiguousarray(rgb).reshape(-1)
    buffer = np.empty(flat.size + alignment, dtype=np.uint8)
    offset = (-int(buffer.ctypes.data)) % alignment
    view = buffer[offset : offset + flat.size]
    view[:] = flat
    return view.reshape(rgb.shape)


def _parse_mediapipe_detection(detection, width: int, height: int) -> dict | None:
    box = getattr(detection, "bounding_box", None)
    if box is None:
        return None
    x = float(getattr(box, "origin_x", 0))
    y = float(getattr(box, "origin_y", 0))
    w = float(getattr(box, "width", 0))
    h = float(getattr(box, "height", 0))
    if 0 < w <= 2 and 0 < h <= 2:
        x, y, w, h = x * width, y * height, w * width, h * height
    x_i, y_i, w_i, h_i = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if w_i < 18 or h_i < 18:
        return None
    x_i = max(0, min(x_i, width - 1))
    y_i = max(0, min(y_i, height - 1))
    score = None
    categories = getattr(detection, "categories", None) or []
    if categories:
        score = float(getattr(categories[0], "score", 0.0))
        if score < 0.5:
            return None
    eye_a, eye_b, nose, mouth = _named_keypoints(detection)
    pose = pose_from_points(eye_a, eye_b, nose, mouth)
    yaw = pitch = None
    if pose is not None:
        yaw, pitch = pose
    return {
        "bbox": (x_i, y_i, max(1, w_i), max(1, h_i)),
        "confidence": score,
        "yaw": yaw,
        "pitch": pitch,
        "forward": None,
        "pose_known": pose is not None,
    }


def _named_keypoints(detection) -> tuple:
    keypoints = getattr(detection, "keypoints", None) or []
    labeled: dict[str, tuple[float, float]] = {}
    ordered: list[tuple[float, float]] = []
    for kp in keypoints:
        xy = (float(kp.x), float(kp.y))
        ordered.append(xy)
        label = str(getattr(kp, "label", "") or "").strip().lower()
        if label:
            labeled[label] = xy

    def find(*needles: str, exclude: tuple[str, ...] = ()) -> tuple[float, float] | None:
        for label, xy in labeled.items():
            if any(needle in label for needle in needles) and not any(item in label for item in exclude):
                return xy
        return None

    nose = find("nose")
    mouth = find("mouth")
    eyes = [
        xy
        for label, xy in labeled.items()
        if "eye" in label and "ear" not in label and "trag" not in label
    ]
    if (len(eyes) < 2 or nose is None) and len(ordered) >= 3 and not labeled:
        eyes = ordered[:2]
        nose = ordered[2]
        mouth = ordered[3] if len(ordered) > 3 else mouth
    if len(eyes) < 2 or nose is None:
        return None, None, None, mouth
    return eyes[0], eyes[1], nose, mouth


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _nms(detections: list[dict], iou_thresh: float = 0.45, limit: int = 40) -> list[dict]:
    ranked = sorted(detections, key=lambda det: -1.0 if det.get("confidence") is None else -float(det["confidence"]))
    kept: list[dict] = []
    for det in ranked:
        if all(_iou(det["bbox"], other["bbox"]) < iou_thresh for other in kept):
            kept.append(det)
        if len(kept) >= limit:
            break
    return kept
