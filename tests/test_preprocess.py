import numpy as np
import pytest

from signlang.constants import (
    LEFT_HAND,
    LEFT_SHOULDER,
    N_FEATURES,
    N_POINTS,
    RIGHT_SHOULDER,
    SEQ_LEN,
)
from signlang.preprocess import normalize, preprocess, resample


def test_normalize_centers_on_shoulders_and_scales_to_unit_width(raw_sequence):
    out = normalize(raw_sequence)
    midpoint = (out[:, LEFT_SHOULDER] + out[:, RIGHT_SHOULDER]) / 2
    width = np.linalg.norm(out[:, LEFT_SHOULDER] - out[:, RIGHT_SHOULDER], axis=-1)
    np.testing.assert_allclose(midpoint, 0, atol=1e-5)
    assert width.mean() == pytest.approx(1.0, abs=1e-5)
    assert out.dtype == np.float32


def test_normalize_ignores_position_and_distance_from_camera(raw_sequence):
    moved = raw_sequence * 3.7 + np.array([0.2, -0.1, 0.05], dtype=np.float32)
    np.testing.assert_allclose(normalize(moved), normalize(raw_sequence), atol=1e-4)


def test_missing_hand_is_exactly_zero_after_normalize(raw_sequence):
    raw_sequence[:, LEFT_HAND] = np.nan
    out = normalize(raw_sequence)
    assert np.isfinite(out).all()
    assert (out[:, LEFT_HAND] == 0).all()
    assert (out[:, : LEFT_HAND.start] != 0).any()


def test_frames_without_shoulders_reuse_the_sequence_center(raw_sequence):
    raw_sequence[5, [LEFT_SHOULDER, RIGHT_SHOULDER]] = np.nan
    out = normalize(raw_sequence)
    assert np.isfinite(out).all()
    # The other landmarks of that frame are still normalized, not zeroed.
    assert (out[5, 0] != 0).any()


def test_normalize_rejects_sequences_without_visible_shoulders(raw_sequence):
    raw_sequence[:, [LEFT_SHOULDER, RIGHT_SHOULDER]] = np.nan
    with pytest.raises(ValueError, match="shoulders"):
        normalize(raw_sequence)


def test_normalize_rejects_degenerate_shoulder_width(raw_sequence):
    raw_sequence[:, LEFT_SHOULDER] = raw_sequence[:, RIGHT_SHOULDER]
    with pytest.raises(ValueError, match="degenerate"):
        normalize(raw_sequence)


@pytest.mark.parametrize("shape", [(10, 33, 3), (10, N_POINTS, 2), (N_POINTS, 3)])
def test_normalize_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match="expected shape"):
        normalize(np.zeros(shape))


def test_resample_hits_first_and_last_frame():
    seq = np.arange(100)[:, None]
    out = resample(seq, 32)
    assert len(out) == 32
    assert out[0, 0] == 0
    assert out[-1, 0] == 99
    assert (np.diff(out[:, 0]) > 0).all()


def test_resample_repeats_frames_of_short_sequences():
    seq = np.arange(5)[:, None]
    out = resample(seq, 32)
    assert len(out) == 32
    assert set(out[:, 0]) == {0, 1, 2, 3, 4}


def test_normalize_rejects_empty_sequence():
    with pytest.raises(ValueError, match="no frames"):
        normalize(np.empty((0, N_POINTS, 3)))


def test_resample_rejects_empty_sequence():
    with pytest.raises(ValueError, match="no frames"):
        resample(np.empty((0, N_POINTS, 3)))


def test_preprocess_returns_flat_fixed_size_input(raw_sequence):
    out = preprocess(raw_sequence)
    assert out.shape == (SEQ_LEN, N_FEATURES)
    assert out.dtype == np.float32
