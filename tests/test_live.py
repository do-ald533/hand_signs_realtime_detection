import numpy as np
import pytest

from signlang.constants import (
    LEFT_HAND,
    LEFT_SHOULDER,
    N_COORDS,
    N_FEATURES,
    N_POINTS,
    RIGHT_SHOULDER,
    SEQ_LEN,
)
from signlang.inference import save_checkpoint
from signlang.live import PredictionGate, Predictor, SlidingWindow
from signlang.models import SignGRU


def _frame(rng, hand=False, shoulders=True) -> np.ndarray:
    frame = np.full((N_POINTS, N_COORDS), np.nan, dtype=np.float32)
    if shoulders:
        frame[LEFT_SHOULDER] = (360, 240, 0)
        frame[RIGHT_SHOULDER] = (280, 240, 0)
    if hand:
        frame[LEFT_HAND] = rng.normal(360, 20, (21, 3))
    return frame


class TestSlidingWindow:
    def test_gives_no_input_until_full(self, rng):
        window = SlidingWindow(length=48)
        for _ in range(47):
            window.push(_frame(rng))
        assert not window.full and window.model_input() is None
        window.push(_frame(rng))
        assert window.full

    def test_model_input_is_resampled_to_training_shape(self, rng):
        window = SlidingWindow(length=48)
        for _ in range(60):  # more than the window holds: only the latest 48 count
            window.push(_frame(rng, hand=True))
        out = window.model_input()
        assert out is not None and out.shape == (SEQ_LEN, N_FEATURES)
        assert np.isfinite(out).all()

    def test_window_without_visible_shoulders_gives_no_input(self, rng):
        window = SlidingWindow(length=32)
        for _ in range(32):
            window.push(_frame(rng, shoulders=False))
        assert window.model_input() is None

    def test_hand_activity_is_the_share_of_frames_with_a_hand(self, rng):
        window = SlidingWindow(length=32)
        for i in range(32):
            window.push(_frame(rng, hand=i < 8))
        assert window.hand_activity() == pytest.approx(0.25)
        assert SlidingWindow(length=32).hand_activity() == 0.0

    def test_rejects_wrongly_shaped_frames_and_tiny_windows(self):
        with pytest.raises(ValueError, match="expected a"):
            SlidingWindow().push(np.zeros((33, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="too short"):
            SlidingWindow(length=8)

    def test_clear_empties_the_window(self, rng):
        window = SlidingWindow(length=32)
        for _ in range(32):
            window.push(_frame(rng))
        window.clear()
        assert not window.full


def _p(top: int, conf: float, n: int = 4) -> np.ndarray:
    proba = np.full(n, (1 - conf) / (n - 1))
    proba[top] = conf
    return proba


class TestPredictionGate:
    def test_needs_consecutive_confident_agreement(self):
        gate = PredictionGate(threshold=0.6, stable=3, hold=0)
        assert [gate.update(_p(2, 0.9)) for _ in range(4)] == [None, None, 2, 2]

    def test_low_confidence_breaks_the_streak(self):
        gate = PredictionGate(threshold=0.6, stable=3, hold=0)
        outputs = [
            gate.update(p) for p in (_p(1, 0.9), _p(1, 0.9), _p(1, 0.4), _p(1, 0.9), _p(1, 0.9))
        ]
        assert outputs == [None] * 5

    def test_changing_the_answer_restarts_the_count(self):
        gate = PredictionGate(threshold=0.5, stable=2, hold=0)
        assert gate.update(_p(0, 0.9)) is None
        assert gate.update(_p(1, 0.9)) is None  # different word: streak restarts
        assert gate.update(_p(1, 0.9)) == 1

    def test_shown_word_is_held_then_cleared(self):
        gate = PredictionGate(threshold=0.5, stable=1, hold=2)
        assert gate.update(_p(3, 0.9)) == 3
        assert [gate.update(None) for _ in range(4)] == [3, 3, None, None]

    def test_no_usable_window_breaks_the_streak(self):
        gate = PredictionGate(threshold=0.5, stable=2, hold=0)
        gate.update(_p(0, 0.9))
        gate.update(None)
        assert gate.update(_p(0, 0.9)) is None

    def test_a_new_stable_word_replaces_the_held_one(self):
        gate = PredictionGate(threshold=0.5, stable=1, hold=5)
        gate.update(_p(0, 0.9))
        assert gate.update(_p(1, 0.9)) == 1

    @pytest.mark.parametrize(
        "kwargs", [{"threshold": 0.0}, {"threshold": 1.5}, {"stable": 0}, {"hold": -1}]
    )
    def test_rejects_invalid_settings(self, kwargs):
        with pytest.raises(ValueError, match="threshold"):
            PredictionGate(**kwargs)


def test_predictor_returns_a_probability_distribution(tmp_path, rng):
    kwargs = {"n_classes": 5, "hidden_size": 8}
    path = tmp_path / "m.pt"
    save_checkpoint(SignGRU(**kwargs), path, class_names=list("abcde"), model_kwargs=kwargs)
    predictor = Predictor.from_checkpoint(path)
    proba = predictor(rng.normal(size=(SEQ_LEN, N_FEATURES)).astype(np.float32))
    assert proba.shape == (5,)
    assert proba.sum() == pytest.approx(1.0, abs=1e-5)
    assert predictor.class_names == list("abcde")
