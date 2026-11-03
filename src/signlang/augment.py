"""Landmark-space augmentation. Operates on *normalized* ``(frames, 75, 3)`` sequences.

Normalization puts the origin at the shoulder midpoint, so rotating and scaling about the origin
is a rotation/zoom of the signer, and all three transforms map missing (zero) landmarks to zero.
"""

import numpy as np
from numpy.typing import NDArray

from signlang.constants import LEFT_HAND, MIRROR_PERMUTATION, RIGHT_HAND

Sequence = NDArray[np.float32]


def rotate_xy(seq: Sequence, angle_rad: float) -> Sequence:
    """Rotate in the image plane (x, y) about the origin; z is untouched."""
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    out = seq.copy()
    out[..., 0] = cos * seq[..., 0] - sin * seq[..., 1]
    out[..., 1] = sin * seq[..., 0] + cos * seq[..., 1]
    return out


def scale(seq: Sequence, factor: float) -> Sequence:
    """Uniformly scale all coordinates about the origin."""
    return (seq * factor).astype(np.float32, copy=False)


def mirror(seq: Sequence) -> Sequence:
    """Left/right flip.

    Negating x is not enough: the left and right hands must also trade places, and so must every
    left/right pose pair (shoulders, elbows, ...). Otherwise the model sees a signer whose left
    hand is drawn on the right side of the body, which no real signer produces.
    """
    out = seq[:, MIRROR_PERMUTATION].copy()
    out[..., 0] *= -1
    return out


def time_crop(seq: Sequence, rng: np.random.Generator, max_fraction: float) -> Sequence:
    """Trim up to ``max_fraction`` of the clip from each end, then stretch back to full length.

    Simulates signers who start late, finish early or sign at a different speed.
    """
    last = len(seq) - 1
    start = rng.uniform(0, max_fraction) * last
    stop = (1 - rng.uniform(0, max_fraction)) * last
    return seq[np.linspace(start, stop, len(seq)).round().astype(np.intp)]


def drop_hands(seq: Sequence, rng: np.random.Generator, prob: float) -> Sequence:
    """Zero each hand independently in each frame with probability ``prob``.

    Live hand detection misses frames that the training videos did not, so the model should cope.
    """
    out = seq.copy()
    for hand in (LEFT_HAND, RIGHT_HAND):
        out[rng.random(len(seq)) < prob, hand] = 0.0
    return out


def add_noise(seq: Sequence, rng: np.random.Generator, std: float) -> Sequence:
    """Gaussian jitter on detected landmarks; missing (zero) landmarks stay exactly zero."""
    noise = rng.normal(0.0, std, seq.shape).astype(np.float32)
    return seq + noise * (seq != 0)


def augment(
    seq: Sequence,
    rng: np.random.Generator,
    *,
    max_rotation_deg: float = 10.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    mirror_prob: float = 0.5,
    crop_fraction: float = 0.0,
    hand_dropout: float = 0.0,
    noise_std: float = 0.0,
) -> Sequence:
    """Random small rotation and scale, a mirror with probability ``mirror_prob``, and optionally
    time cropping, per-frame hand dropout and landmark noise (all off by default)."""
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    out = scale(rotate_xy(seq, float(angle)), float(rng.uniform(*scale_range)))
    if rng.random() < mirror_prob:
        out = mirror(out)
    if crop_fraction > 0:
        out = time_crop(out, rng, crop_fraction)
    if hand_dropout > 0:
        out = drop_hands(out, rng, hand_dropout)
    if noise_std > 0:
        out = add_noise(out, rng, noise_std)
    return out
