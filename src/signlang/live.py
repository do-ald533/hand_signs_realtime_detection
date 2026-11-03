"""Live recognition: sliding window of landmarks, a confidence/stability gate, and the demo loop."""

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from signlang.constants import (
    LEFT_HAND,
    LEFT_SHOULDER,
    N_COORDS,
    N_POINTS,
    RIGHT_HAND,
    RIGHT_SHOULDER,
    SEQ_LEN,
)
from signlang.inference import LoadedModel, load_checkpoint
from signlang.landmarks import LandmarkExtractor, PoseModel
from signlang.preprocess import preprocess


class SlidingWindow:
    """The most recent ``length`` raw landmark frames, turned into model input on demand.

    Training clips are whole signs resampled to ``SEQ_LEN`` frames, so the window is resampled the
    same way. ``length`` therefore sets the time scale the model sees: it should roughly match the
    duration of a typical sign at the camera's frame rate, not necessarily ``SEQ_LEN``.
    """

    def __init__(self, length: int = 64, seq_len: int = SEQ_LEN) -> None:
        if length < seq_len // 2:
            raise ValueError(f"window of {length} frames is too short for {seq_len}-frame input")
        self.length = length
        self.seq_len = seq_len
        self._frames: deque[NDArray[np.float32]] = deque(maxlen=length)

    def push(self, frame: NDArray[np.float32]) -> None:
        if frame.shape != (N_POINTS, N_COORDS):
            raise ValueError(f"expected a ({N_POINTS}, {N_COORDS}) frame, got {frame.shape}")
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def full(self) -> bool:
        return len(self._frames) == self.length

    def hand_activity(self) -> float:
        """Share of frames in the window where at least one hand was detected."""
        if not self._frames:
            return 0.0
        window = np.stack(self._frames)
        hands = np.isfinite(window[:, LEFT_HAND]).any(axis=(1, 2)) | np.isfinite(
            window[:, RIGHT_HAND]
        ).any(axis=(1, 2))
        return float(hands.mean())

    def model_input(self) -> NDArray[np.float32] | None:
        """``(seq_len, 225)`` model input; ``None`` if the window is not full or lacks shoulders."""
        if not self.full:
            return None
        try:
            return preprocess(np.stack(self._frames), self.seq_len)
        except ValueError:  # shoulders never visible in this window
            return None


class PredictionGate:
    """Only let a word through when the model is confident *and* keeps saying the same thing.

    A word is shown after ``stable`` consecutive predictions agree on it with probability at least
    ``threshold``. It then stays up for ``hold`` more updates so the display does not flicker
    between predictions. Passing ``None`` (no usable window, e.g. no hands) breaks the streak.
    """

    def __init__(self, threshold: float = 0.6, stable: int = 3, hold: int = 6) -> None:
        if not 0.0 < threshold <= 1.0 or stable < 1 or hold < 0:
            raise ValueError("need 0 < threshold <= 1, stable >= 1 and hold >= 0")
        self.threshold, self.stable, self.hold = threshold, stable, hold
        self.reset()

    def reset(self) -> None:
        self._candidate: int | None = None
        self._streak = 0
        self._shown: int | None = None
        self._hold_left = 0

    def update(self, proba: NDArray[np.floating] | None) -> int | None:
        """Feed one prediction; returns the class to display, or ``None`` for nothing."""
        if proba is not None and proba.max() >= self.threshold:
            top = int(proba.argmax())
            self._streak = self._streak + 1 if top == self._candidate else 1
            self._candidate = top
        else:
            self._candidate, self._streak = None, 0

        if self._candidate is not None and self._streak >= self.stable:
            self._shown, self._hold_left = self._candidate, self.hold
        elif self._hold_left > 0:
            self._hold_left -= 1
        else:
            self._shown = None
        return self._shown


class Predictor:
    """A loaded checkpoint that maps a ``(seq_len, 225)`` window to class probabilities."""

    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded
        self.class_names = loaded.class_names

    @classmethod
    def from_checkpoint(cls, path: Path) -> "Predictor":
        return cls(load_checkpoint(path))

    @torch.inference_mode()
    def __call__(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.loaded.model.predict_proba(torch.from_numpy(window)[None])[0].numpy()


@dataclass(frozen=True, slots=True)
class LiveStats:
    frames: int
    seconds: float
    fps: float  # mean over the run, excluding the first second of warm-up
    ms_per_frame: float  # mean landmark extraction + inference time per frame
    words: tuple[str, ...]  # distinct words that were shown, in order


def run_live(
    source: int | Path,
    checkpoint: Path,
    *,
    window: int = 64,
    stride: int = 4,
    threshold: float = 0.6,
    stable: int = 3,
    hold: int = 6,
    min_activity: float = 0.25,
    pose_model: PoseModel = "full",
    headless: bool = False,
    mirror_view: bool = True,
    draw_landmarks: bool = True,
    max_frames: int | None = None,
) -> LiveStats:
    """Run the recognizer on a webcam index or a video file.

    Landmarks are always extracted from the *unmirrored* frame, because that is what the model was
    trained on; ``mirror_view`` only flips the picture that is shown. Press ``q`` or Esc to quit.
    """
    import cv2

    predictor = Predictor.from_checkpoint(checkpoint)
    names = predictor.class_names
    is_camera = isinstance(source, int)
    capture = cv2.VideoCapture(source if is_camera else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    if is_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    nominal_fps = capture.get(cv2.CAP_PROP_FPS)
    nominal_fps = nominal_fps if 1.0 <= nominal_fps <= 240.0 else 30.0

    buffer = SlidingWindow(window, predictor.loaded.seq_len)
    gate = PredictionGate(threshold, stable, hold)
    words: list[str] = []
    candidate_text = ""
    shown: int | None = None
    fps_ema = 0.0
    work_ms = 0.0
    counted = 0
    frames = 0
    started = time.perf_counter()
    last_ts = -1

    with LandmarkExtractor(pose_model) as extractor:
        while max_frames is None or frames < max_frames:
            tick = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            ts = int((tick - started) * 1000) if is_camera else int(frames * 1000 / nominal_fps)
            last_ts = ts = max(ts, last_ts + 1)

            work_start = time.perf_counter()
            landmarks = extractor.process(frame, ts)
            buffer.push(landmarks)
            frames += 1
            if frames % stride == 0 and buffer.full:
                window_input = (
                    buffer.model_input() if buffer.hand_activity() >= min_activity else None
                )
                proba = predictor(window_input) if window_input is not None else None
                previous = shown
                shown = gate.update(proba)
                candidate_text = (
                    f"{names[int(proba.argmax())]} {proba.max():.0%}" if proba is not None else ""
                )
                if shown is not None and shown != previous:
                    words.append(names[shown])
                    if headless:
                        print(f"frame {frames:5d}  {names[shown]}  ({candidate_text})")
            elapsed_ms = (time.perf_counter() - work_start) * 1000
            if time.perf_counter() - started > 1.0:  # skip warm-up (model load, first inferences)
                work_ms += elapsed_ms
                counted += 1

            period = time.perf_counter() - tick
            fps_ema = 1.0 / period if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * (1.0 / period)

            if not headless:
                view = cv2.flip(frame, 1) if mirror_view else frame.copy()
                if draw_landmarks:
                    _draw_landmarks(view, landmarks, flipped=mirror_view)
                _draw_overlay(
                    view, names[shown] if shown is not None else "", candidate_text, fps_ema
                )
                cv2.imshow("signlang", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    capture.release()
    if not headless:
        cv2.destroyAllWindows()

    seconds = time.perf_counter() - started
    steady = max(frames - int(nominal_fps), 1)
    return LiveStats(
        frames=frames,
        seconds=seconds,
        fps=steady / max(seconds - 1.0, 1e-6) if frames > nominal_fps else frames / seconds,
        ms_per_frame=work_ms / max(counted, 1),
        words=tuple(words),
    )


def _draw_landmarks(view: NDArray[Any], landmarks: NDArray[np.float32], *, flipped: bool) -> None:
    """Dots for the shoulders and both hands, so it is obvious what the model is being fed."""
    import cv2

    width = view.shape[1]
    groups = (
        ([LEFT_SHOULDER, RIGHT_SHOULDER], (255, 200, 0)),
        (range(LEFT_HAND.start, LEFT_HAND.stop), (0, 165, 255)),
        (range(RIGHT_HAND.start, RIGHT_HAND.stop), (60, 220, 90)),
    )
    for indices, color in groups:
        for i in indices:
            x, y = landmarks[i, :2]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(view, (int(width - 1 - x) if flipped else int(x), int(y)), 3, color, -1)


def _draw_overlay(view: NDArray[Any], word: str, candidate: str, fps: float) -> None:
    import cv2

    width = view.shape[1]
    cv2.rectangle(view, (0, 0), (width, 64), (0, 0, 0), -1)
    cv2.putText(view, word or "...", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (60, 220, 90) if word else (140, 140, 140), 3, cv2.LINE_AA)  # fmt: skip
    cv2.putText(view, f"{fps:4.1f} FPS", (width - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
    if candidate:
        cv2.putText(view, f"top guess: {candidate}", (width - 260, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)  # fmt: skip
