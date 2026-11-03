import numpy as np
import pytest

from signlang.augment import add_noise, augment, drop_hands, mirror, rotate_xy, scale, time_crop
from signlang.constants import LEFT_HAND, LEFT_SHOULDER, RIGHT_HAND, RIGHT_SHOULDER
from signlang.preprocess import normalize


@pytest.fixture
def normalized(raw_sequence):
    raw_sequence[:, RIGHT_HAND] = np.nan  # one hand missing, as in many one-handed signs
    return normalize(raw_sequence)


def test_mirror_twice_is_identity(normalized):
    np.testing.assert_array_equal(mirror(mirror(normalized)), normalized)


def test_mirror_moves_left_hand_to_right_hand_with_x_flipped(normalized):
    out = mirror(normalized)
    np.testing.assert_array_equal(out[:, RIGHT_HAND, 0], -normalized[:, LEFT_HAND, 0])
    np.testing.assert_array_equal(out[:, RIGHT_HAND, 1:], normalized[:, LEFT_HAND, 1:])
    assert (out[:, LEFT_HAND] == 0).all()  # the missing hand is now on the other side


def test_mirror_swaps_shoulders(normalized):
    out = mirror(normalized)
    np.testing.assert_array_equal(out[:, LEFT_SHOULDER, 0], -normalized[:, RIGHT_SHOULDER, 0])


def test_mirror_does_not_modify_its_input(normalized):
    before = normalized.copy()
    mirror(normalized)
    np.testing.assert_array_equal(normalized, before)


def test_rotation_preserves_distances_and_zeros(normalized):
    out = rotate_xy(normalized, np.deg2rad(9))
    np.testing.assert_allclose(
        np.linalg.norm(out[..., :2], axis=-1),
        np.linalg.norm(normalized[..., :2], axis=-1),
        atol=1e-5,
    )
    np.testing.assert_array_equal(out[..., 2], normalized[..., 2])
    assert (out[:, RIGHT_HAND] == 0).all()


def test_zero_rotation_and_unit_scale_are_identity(normalized):
    np.testing.assert_allclose(rotate_xy(normalized, 0.0), normalized, atol=1e-7)
    np.testing.assert_array_equal(scale(normalized, 1.0), normalized)


def test_time_crop_keeps_length_and_only_uses_existing_frames(normalized):
    out = time_crop(normalized, np.random.default_rng(0), max_fraction=0.25)
    assert out.shape == normalized.shape
    frames = {tuple(f.ravel()[:6]) for f in normalized}
    assert all(tuple(f.ravel()[:6]) in frames for f in out)


def test_time_crop_of_zero_is_identity(normalized):
    np.testing.assert_array_equal(time_crop(normalized, np.random.default_rng(0), 0.0), normalized)


def test_drop_hands_only_zeroes_hand_blocks(normalized):
    out = drop_hands(normalized, np.random.default_rng(0), prob=1.0)
    assert (out[:, LEFT_HAND] == 0).all() and (out[:, RIGHT_HAND] == 0).all()
    np.testing.assert_array_equal(out[:, : LEFT_HAND.start], normalized[:, : LEFT_HAND.start])
    np.testing.assert_array_equal(drop_hands(normalized, np.random.default_rng(0), 0.0), normalized)


def test_noise_leaves_missing_landmarks_exactly_zero(normalized):
    out = add_noise(normalized, np.random.default_rng(0), std=0.05)
    assert (out[:, RIGHT_HAND] == 0).all()  # fixture marks this hand as missing
    assert not np.array_equal(out[:, LEFT_HAND], normalized[:, LEFT_HAND])
    assert out.dtype == np.float32


def test_augment_with_every_option_is_reproducible(normalized):
    options = {"crop_fraction": 0.2, "hand_dropout": 0.2, "noise_std": 0.02}
    a = augment(normalized, np.random.default_rng(9), **options)
    b = augment(normalized, np.random.default_rng(9), **options)
    np.testing.assert_array_equal(a, b)
    assert a.shape == normalized.shape and a.dtype == np.float32


def test_augment_is_reproducible_and_keeps_shape_and_dtype(normalized):
    a = augment(normalized, np.random.default_rng(1))
    b = augment(normalized, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert a.shape == normalized.shape
    assert a.dtype == np.float32


def test_augment_mirror_probability_is_respected(normalized):
    always = augment(
        normalized,
        np.random.default_rng(0),
        mirror_prob=1.0,
        max_rotation_deg=0,
        scale_range=(1, 1),
    )
    never = augment(
        normalized,
        np.random.default_rng(0),
        mirror_prob=0.0,
        max_rotation_deg=0,
        scale_range=(1, 1),
    )
    np.testing.assert_allclose(always, mirror(normalized), atol=1e-6)
    np.testing.assert_allclose(never, normalized, atol=1e-6)
