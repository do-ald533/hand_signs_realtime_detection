"""Landmark layout shared by extraction, preprocessing, augmentation and the model.

One frame is 75 points with (x, y, z) each::

    [0, 33)   pose        MediaPipe Pose landmarks
    [33, 54)  left hand   MediaPipe Hand landmarks
    [54, 75)  right hand  MediaPipe Hand landmarks

Landmarks that were not detected are stored as NaN in raw sequences and become zeros after
:func:`signlang.preprocess.normalize`.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

N_POSE: Final = 33
N_HAND: Final = 21
N_POINTS: Final = N_POSE + 2 * N_HAND
N_COORDS: Final = 3
N_FEATURES: Final = N_POINTS * N_COORDS

POSE: Final = slice(0, N_POSE)
LEFT_HAND: Final = slice(N_POSE, N_POSE + N_HAND)
RIGHT_HAND: Final = slice(N_POSE + N_HAND, N_POINTS)

LEFT_SHOULDER: Final = 11
RIGHT_SHOULDER: Final = 12

SEQ_LEN: Final = 32
N_CLASSES: Final = 100

# MediaPipe Pose landmarks that are left/right counterparts of each other (the nose, index 0,
# is its own mirror image).
POSE_MIRROR_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 4),  # eye (inner)
    (2, 5),  # eye
    (3, 6),  # eye (outer)
    (7, 8),  # ear
    (9, 10),  # mouth
    (11, 12),  # shoulder
    (13, 14),  # elbow
    (15, 16),  # wrist
    (17, 18),  # pinky
    (19, 20),  # index finger
    (21, 22),  # thumb
    (23, 24),  # hip
    (25, 26),  # knee
    (27, 28),  # ankle
    (29, 30),  # heel
    (31, 32),  # foot index
)


def _mirror_permutation() -> NDArray[np.intp]:
    """Point order that turns a frame into its left/right mirror image."""
    perm = np.arange(N_POINTS)
    for left, right in POSE_MIRROR_PAIRS:
        perm[left], perm[right] = right, left
    perm[LEFT_HAND], perm[RIGHT_HAND] = (
        np.arange(N_POSE + N_HAND, N_POINTS),
        np.arange(N_POSE, N_POSE + N_HAND),
    )
    return perm


MIRROR_PERMUTATION: Final = _mirror_permutation()
