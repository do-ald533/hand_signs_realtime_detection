import numpy as np
import pytest

from signlang.metrics import confusion_matrix, most_confused, top_k_accuracy

SCORES = np.array(
    [
        [0.7, 0.2, 0.1],  # true 0 -> top-1 hit
        [0.5, 0.4, 0.1],  # true 1 -> top-1 miss, top-2 hit
        [0.6, 0.3, 0.1],  # true 2 -> top-1 miss, top-2 miss
        [0.1, 0.2, 0.7],  # true 2 -> top-1 hit
    ]
)
LABELS = np.array([0, 1, 2, 2])


def test_top_k_accuracy():
    assert top_k_accuracy(SCORES, LABELS, k=1) == pytest.approx(0.5)
    assert top_k_accuracy(SCORES, LABELS, k=2) == pytest.approx(0.75)
    assert top_k_accuracy(SCORES, LABELS, k=3) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, 4])
def test_top_k_accuracy_rejects_bad_k(k):
    with pytest.raises(ValueError, match="k must be"):
        top_k_accuracy(SCORES, LABELS, k=k)


def test_top_k_accuracy_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="expected scores"):
        top_k_accuracy(SCORES, LABELS[:2])


def test_confusion_matrix_keeps_classes_absent_from_predictions():
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), n_classes=3)
    assert cm.shape == (3, 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 0, 0]])


def test_most_confused_lists_biggest_off_diagonal_first():
    cm = np.array([[9, 3, 0], [1, 8, 0], [0, 0, 10]])
    assert most_confused(cm, ["book", "paper", "dog"], top=5) == [
        ("book", "paper", 3),
        ("paper", "book", 1),
    ]


def test_most_confused_is_empty_for_a_perfect_classifier():
    assert most_confused(np.eye(3, dtype=int) * 5, ["a", "b", "c"]) == []
