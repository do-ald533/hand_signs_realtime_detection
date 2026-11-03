"""MediaPipe landmark extraction: video frames to raw ``(frames, 75, 3)`` arrays.

Conventions (shared by training and the live demo, so they must not drift apart):

* Coordinates are in **pixels of the frame that was processed**: ``(x * width, y * height,
  z * width)``. MediaPipe's normalized x/y are anisotropic on non-square frames, which would turn
  rotation augmentation into a shear; pixels are isotropic. z shares x's scale in MediaPipe.
* Anything not detected is **NaN**, never 0 (see :func:`signlang.preprocess.normalize`).
* Left/right are the **signer's** anatomical left/right on an **unmirrored** camera image. A selfie
  preview must be flipped only for display; the frame handed to the extractor must not be.
* Hands are labelled by which pose wrist they are nearest to, not by MediaPipe's handedness
  classifier, whose mirroring assumption differs between APIs and is easy to get backwards.
"""

import urllib.request
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from signlang.constants import (
    LEFT_HAND,
    LEFT_SHOULDER,
    N_COORDS,
    N_POINTS,
    N_POSE,
    POSE,
    RIGHT_HAND,
    RIGHT_SHOULDER,
)

PoseModel = Literal["lite", "full", "heavy"]

_MODEL_ROOT: Final = "https://storage.googleapis.com/mediapipe-models"
POSE_MODEL_URLS: Final[dict[str, str]] = {
    size: f"{_MODEL_ROOT}/pose_landmarker/pose_landmarker_{size}/float16/latest/"
    f"pose_landmarker_{size}.task"
    for size in ("lite", "full", "heavy")
}
HAND_MODEL_URL: Final = (
    f"{_MODEL_ROOT}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_DIR: Final = Path("models/mediapipe")

LEFT_WRIST: Final = 15
RIGHT_WRIST: Final = 16


class _Landmark(Protocol):
    x: float
    y: float
    z: float
    visibility: float | None


def ensure_model(url: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Path to the MediaPipe ``.task`` file behind ``url``, downloading it on first use."""
    dest = model_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.replace(dest)
    return dest


@cache
def _model_bytes(url: str, model_dir: Path) -> bytes:
    return ensure_model(url, model_dir).read_bytes()


def assign_hands(
    hand_wrists: NDArray[np.floating], pose: NDArray[np.floating]
) -> tuple[int | None, int | None]:
    """Decide which detected hand is the signer's left and which is the right.

    Args:
        hand_wrists: ``(k, 2+)`` pixel position of each detected hand's wrist, ``k`` at most 2.
        pose: ``(33, 3)`` pose landmarks in pixels, NaN where not detected.

    Returns:
        ``(left_index, right_index)`` into ``hand_wrists``; ``None`` for a side with no hand.
        Hands are matched to the nearest pose wrists. If a pose wrist is missing, the image x
        position decides instead (the signer's left hand appears on the image's right).
    """
    count = len(hand_wrists)
    if count == 0:
        return None, None
    wrists_xy = hand_wrists[:count, :2]
    pose_wrists = np.stack([pose[LEFT_WRIST, :2], pose[RIGHT_WRIST, :2]])  # (2, 2): left, right

    if np.isfinite(pose_wrists).all():
        cost = np.linalg.norm(wrists_xy[:, None] - pose_wrists[None], axis=-1)  # (k, 2)
        if count == 1:
            return (0, None) if cost[0, 0] <= cost[0, 1] else (None, 0)
        straight = cost[0, 0] + cost[1, 1]  # hand 0 is left, hand 1 is right
        crossed = cost[0, 1] + cost[1, 0]
        return (0, 1) if straight <= crossed else (1, 0)

    if count >= 2:
        left = int(np.argmax(wrists_xy[:, 0]))
        return left, 1 - left
    shoulders_x = pose[[LEFT_SHOULDER, RIGHT_SHOULDER], 0]
    shoulders_x = shoulders_x[np.isfinite(shoulders_x)]
    if len(shoulders_x) == 0:
        return None, None
    return (0, None) if wrists_xy[0, 0] > shoulders_x.mean() else (None, 0)


def landmarks_from_results(
    pose_landmarks: Sequence[Sequence[_Landmark]],
    hand_landmarks: Sequence[Sequence[_Landmark]],
    width: int,
    height: int,
    min_visibility: float = 0.5,
) -> NDArray[np.float32]:
    """Assemble one ``(75, 3)`` frame from MediaPipe pose and hand results.

    Pose points with ``visibility < min_visibility`` (typically legs and hips in upper-body
    footage, where MediaPipe only extrapolates) are dropped to NaN.
    """
    scale = np.array([width, height, width], dtype=np.float32)
    frame = np.full((N_POINTS, N_COORDS), np.nan, dtype=np.float32)

    if pose_landmarks:
        points = pose_landmarks[0]
        pose = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32) * scale
        visibility = np.array([1.0 if p.visibility is None else p.visibility for p in points])
        pose[visibility < min_visibility] = np.nan
        frame[POSE] = pose

    if hand_landmarks:
        hands = [
            np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32) * scale for h in hand_landmarks
        ]
        left, right = assign_hands(np.stack([h[0] for h in hands]), frame[POSE])
        if left is not None:
            frame[LEFT_HAND] = hands[left]
        if right is not None:
            frame[RIGHT_HAND] = hands[right]
    return frame


class LandmarkExtractor:
    """Runs MediaPipe pose and hand landmarkers in VIDEO mode (they track across frames).

    Use one instance per video or per live session: VIDEO mode requires strictly increasing
    timestamps, and tracking state from a previous clip should not leak into the next one.
    """

    def __init__(
        self,
        pose_model: PoseModel = "full",
        *,
        min_visibility: float = 0.5,
        model_dir: Path = DEFAULT_MODEL_DIR,
        confidence: float = 0.5,
    ) -> None:
        self.min_visibility = min_visibility
        self._pose_url = POSE_MODEL_URLS[pose_model]
        self._model_dir = model_dir
        self._confidence = confidence
        self._pose: Any = None
        self._hands: Any = None

    def __enter__(self) -> Self:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        running = vision.RunningMode.VIDEO

        def base(url: str) -> Any:
            return BaseOptions(model_asset_buffer=_model_bytes(url, self._model_dir))

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(self._pose_url),
                running_mode=running,
                num_poses=1,
                min_pose_detection_confidence=self._confidence,
                min_pose_presence_confidence=self._confidence,
                min_tracking_confidence=self._confidence,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(HAND_MODEL_URL),
                running_mode=running,
                num_hands=2,
                min_hand_detection_confidence=self._confidence,
                min_hand_presence_confidence=self._confidence,
                min_tracking_confidence=self._confidence,
            )
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for task in (self._pose, self._hands):
            if task is not None:
                task.close()
        self._pose = self._hands = None

    def process(self, frame_bgr: NDArray[Any], timestamp_ms: int) -> NDArray[np.float32]:
        """Landmarks of one unmirrored BGR frame as a ``(75, 3)`` array (NaN where missing)."""
        import cv2

        if self._pose is None:
            raise RuntimeError("use LandmarkExtractor as a context manager")
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        pose = self._pose.detect_for_video(image, timestamp_ms)
        hands = self._hands.detect_for_video(image, timestamp_ms)
        return landmarks_from_results(
            pose.pose_landmarks, hands.hand_landmarks, width, height, self.min_visibility
        )


def _clamp_box(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    return (x1, y1, x2, y2) if x2 - x1 >= 32 and y2 - y1 >= 32 else None


def extract_video(
    path: Path,
    *,
    frame_start: int = 1,
    frame_end: int = -1,
    bbox: tuple[int, int, int, int] | None = None,
    pose_model: PoseModel = "full",
) -> NDArray[np.float32]:
    """Raw landmarks for frames ``frame_start..frame_end`` (1-based, inclusive) of a video file.

    ``bbox`` crops each frame to the signer first (WLASL ships one per clip), which helps the hand
    model on wide, low-resolution footage. Returns shape ``(0, 75, 3)`` if no frame could be read.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return np.empty((0, N_POINTS, N_COORDS), dtype=np.float32)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 <= fps <= 240.0 else 25.0

    frames: list[NDArray[np.float32]] = []
    with LandmarkExtractor(pose_model) as extractor:
        index = 0  # 1-based index of the frame about to be read
        while True:
            index += 1
            if frame_end != -1 and index > frame_end:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if index < frame_start:
                continue
            if bbox is not None and (box := _clamp_box(bbox, frame.shape[1], frame.shape[0])):
                x1, y1, x2, y2 = box
                frame = frame[y1:y2, x1:x2]
            timestamp_ms = round(len(frames) * 1000.0 / fps)
            frames.append(extractor.process(frame, timestamp_ms))
    capture.release()
    if not frames:
        return np.empty((0, N_POINTS, N_COORDS), dtype=np.float32)
    return np.stack(frames)


def detection_rates(raw: NDArray[np.floating]) -> dict[str, float]:
    """Share of frames in which the pose, the left hand and the right hand were detected."""
    if len(raw) == 0:
        return {"pose": 0.0, "left": 0.0, "right": 0.0}
    present = np.isfinite(raw)
    return {
        "pose": float(present[:, :N_POSE].any(axis=(1, 2)).mean()),
        "left": float(present[:, LEFT_HAND].any(axis=(1, 2)).mean()),
        "right": float(present[:, RIGHT_HAND].any(axis=(1, 2)).mean()),
    }
