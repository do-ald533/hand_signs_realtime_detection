import numpy as np

from signlang.constants import (
    LEFT_HAND,
    LEFT_SHOULDER,
    MIRROR_PERMUTATION,
    N_HAND,
    N_POINTS,
    N_POSE,
    RIGHT_HAND,
    RIGHT_SHOULDER,
)


def test_layout_adds_up():
    assert N_POINTS == N_POSE + 2 * N_HAND == 75
    assert LEFT_HAND.stop == RIGHT_HAND.start
    assert RIGHT_HAND.stop == N_POINTS


def test_mirror_permutation_is_an_involution():
    perm = MIRROR_PERMUTATION
    assert sorted(perm) == list(range(N_POINTS))
    np.testing.assert_array_equal(perm[perm], np.arange(N_POINTS))


def test_mirror_permutation_swaps_shoulders_and_hands():
    assert MIRROR_PERMUTATION[LEFT_SHOULDER] == RIGHT_SHOULDER
    assert MIRROR_PERMUTATION[0] == 0  # nose maps to itself
    np.testing.assert_array_equal(MIRROR_PERMUTATION[LEFT_HAND], np.arange(*RIGHT_HAND.indices(75)))
    np.testing.assert_array_equal(MIRROR_PERMUTATION[RIGHT_HAND], np.arange(*LEFT_HAND.indices(75)))
