from types import SimpleNamespace

import numpy as np
import pytest

from signlang.constants import LEFT_HAND, N_POINTS, POSE, RIGHT_HAND
from signlang.landmarks import (
    LEFT_WRIST,
    RIGHT_WRIST,
    _clamp_box,
    assign_hands,
    detection_rates,
    landmarks_from_results,
)


def _pose_array(left_wrist=(300.0, 200.0), right_wrist=(100.0, 200.0)) -> np.ndarray:
    pose = np.full((33, 3), np.nan, dtype=np.float32)
    pose[11] = (260, 100, 0)  # left shoulder: on the image's right for an unmirrored view
    pose[12] = (140, 100, 0)
    pose[LEFT_WRIST, :2] = left_wrist
    pose[RIGHT_WRIST, :2] = right_wrist
    return pose


def test_assign_hands_matches_hands_to_the_nearest_pose_wrist():
    hands = np.array([[105.0, 205.0], [295.0, 195.0]])  # first hand is near the right wrist
    assert assign_hands(hands, _pose_array()) == (1, 0)


def test_assign_hands_with_a_single_hand():
    pose = _pose_array()
    assert assign_hands(np.array([[298.0, 205.0]]), pose) == (0, None)
    assert assign_hands(np.array([[102.0, 205.0]]), pose) == (None, 0)


def test_assign_hands_picks_the_cheaper_global_matching_when_hands_cross():
    # Both hands closer to the left wrist than to the right one; the pairing still uses both sides.
    pose = _pose_array(left_wrist=(200.0, 200.0), right_wrist=(100.0, 200.0))
    hands = np.array([[190.0, 200.0], [150.0, 200.0]])
    assert assign_hands(hands, pose) == (0, 1)


def test_assign_hands_falls_back_to_image_position_without_pose_wrists():
    pose = _pose_array()
    pose[[LEFT_WRIST, RIGHT_WRIST]] = np.nan
    assert assign_hands(np.array([[120.0, 0.0], [310.0, 0.0]]), pose) == (1, 0)  # larger x = left
    assert assign_hands(np.array([[310.0, 0.0]]), pose) == (0, None)  # right of the midline
    assert assign_hands(np.array([[90.0, 0.0]]), pose) == (None, 0)


def test_assign_hands_gives_up_when_nothing_anchors_a_lone_hand():
    assert assign_hands(np.array([[1.0, 1.0]]), np.full((33, 3), np.nan)) == (None, None)
    assert assign_hands(np.empty((0, 2)), _pose_array()) == (None, None)


def _lm(x, y, z=0.0, visibility=None):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _fake_pose(visible_up_to=24):
    return [_lm(0.5, 0.5, 0.1, 1.0 if i < visible_up_to else 0.1) for i in range(33)]


def test_landmarks_from_results_scales_to_pixels_and_masks_invisible_points():
    pose = _fake_pose()
    frame = landmarks_from_results([pose], [], width=200, height=100)
    assert frame.shape == (N_POINTS, 3)
    np.testing.assert_allclose(frame[0], [100.0, 50.0, 20.0])  # x*w, y*h, z*w
    assert np.isnan(frame[30]).all()  # low-visibility landmark dropped
    assert np.isnan(frame[LEFT_HAND]).all() and np.isnan(frame[RIGHT_HAND]).all()


def test_landmarks_from_results_labels_hands_by_pose_wrist_position():
    pose = _fake_pose()
    pose[LEFT_WRIST] = _lm(0.8, 0.5, 0.0, 1.0)
    pose[RIGHT_WRIST] = _lm(0.2, 0.5, 0.0, 1.0)
    near_left = [_lm(0.79 + 0.001 * i, 0.5) for i in range(21)]
    near_right = [_lm(0.21 + 0.001 * i, 0.5) for i in range(21)]
    # MediaPipe reports the hands in arbitrary order; the output must not depend on it.
    for hands in ([near_right, near_left], [near_left, near_right]):
        frame = landmarks_from_results([pose], hands, width=100, height=100)
        assert frame[LEFT_HAND][0, 0] == pytest.approx(79.0)
        assert frame[RIGHT_HAND][0, 0] == pytest.approx(21.0)


def test_landmarks_from_results_with_nothing_detected_is_all_nan():
    assert np.isnan(landmarks_from_results([], [], 640, 480)).all()


def test_detection_rates():
    raw = np.full((4, N_POINTS, 3), np.nan, dtype=np.float32)
    raw[:, POSE] = 1.0
    raw[:2, LEFT_HAND] = 1.0
    assert detection_rates(raw) == {"pose": 1.0, "left": 0.5, "right": 0.0}
    assert detection_rates(raw[:0]) == {"pose": 0.0, "left": 0.0, "right": 0.0}


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((10, 10, 100, 100), (10, 10, 100, 100)),
        ((-20, -5, 5000, 90), (0, 0, 640, 90)),  # clamped to the frame
        ((50, 50, 60, 60), None),  # too small to be a signer
        ((0, 0, 0, 0), None),  # WLASL's "no box" placeholder
    ],
)
def test_clamp_box(box, expected):
    assert _clamp_box(box, 640, 480) == expected


@pytest.mark.slow
def test_extractor_runs_on_a_blank_frame():
    pytest.importorskip("mediapipe")
    from signlang.landmarks import LandmarkExtractor

    with LandmarkExtractor("lite") as extractor:
        first = extractor.process(np.zeros((240, 320, 3), dtype=np.uint8), 0)
        second = extractor.process(np.zeros((240, 320, 3), dtype=np.uint8), 40)
    assert first.shape == second.shape == (N_POINTS, 3)
    assert np.isnan(first).all()  # nobody in the picture
