"""Turn raw landmark sequences into fixed-size, position- and scale-invariant model input."""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from signlang.constants import (
    LEFT_SHOULDER,
    N_COORDS,
    N_FEATURES,
    N_POINTS,
    RIGHT_SHOULDER,
    SEQ_LEN,
)

_MIN_SHOULDER_WIDTH = 1e-6
_Scalar = TypeVar("_Scalar", bound=np.generic)


def _check_shape(seq: NDArray[np.floating]) -> None:
    if seq.ndim != 3 or seq.shape[1:] != (N_POINTS, N_COORDS):
        raise ValueError(f"expected shape (frames, {N_POINTS}, {N_COORDS}), got {seq.shape}")
    if len(seq) == 0:
        raise ValueError("sequence has no frames")


def normalize(seq: NDArray[np.floating]) -> NDArray[np.float32]:
    """Center each frame on the shoulder midpoint and scale by the mean shoulder width.

    The result does not depend on where the signer stands or how far they are from the camera.

    Missing landmarks (NaN) come out as exact zeros. They are masked *after* normalizing, so they
    are not shifted to ``-center / width`` the way a zero fed into the formula would be. Frames
    whose shoulders were not detected reuse the sequence's mean shoulder midpoint.

    Args:
        seq: Raw landmarks with shape ``(frames, 75, 3)``.

    Returns:
        Normalized landmarks with the same shape, as ``float32``.

    Raises:
        ValueError: If the shape is wrong or no frame has both shoulders detected.
    """
    seq = np.asarray(seq, dtype=np.float32)
    _check_shape(seq)

    left, right = seq[:, LEFT_SHOULDER], seq[:, RIGHT_SHOULDER]
    widths = np.linalg.norm(left - right, axis=-1)
    visible = np.isfinite(widths)
    if not visible.any():
        raise ValueError("no frame has both shoulders detected")

    width = float(widths[visible].mean())
    if width < _MIN_SHOULDER_WIDTH:
        raise ValueError(f"shoulder width is degenerate ({width:.2e})")

    center = (left + right) / 2
    center[~visible] = center[visible].mean(axis=0)

    normalized = (seq - center[:, None]) / width
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32, copy=False)


def resample(seq: NDArray[_Scalar], n: int = SEQ_LEN) -> NDArray[_Scalar]:
    """Pick ``n`` evenly spaced frames, repeating frames if the sequence is shorter than ``n``."""
    if len(seq) == 0:
        raise ValueError("sequence has no frames")
    idx = np.linspace(0, len(seq) - 1, n).round().astype(np.intp)
    return seq[idx]


def preprocess(raw: NDArray[np.floating], n: int = SEQ_LEN) -> NDArray[np.float32]:
    """Raw ``(frames, 75, 3)`` landmarks to a ``(n, 225)`` model input."""
    return resample(normalize(raw), n).reshape(n, N_FEATURES)
