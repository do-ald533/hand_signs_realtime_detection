"""Evaluation metrics: top-k accuracy, confusion matrix, most-confused sign pairs."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix


def top_k_accuracy(scores: NDArray[np.floating], labels: NDArray[np.integer], k: int = 1) -> float:
    """Fraction of samples whose true label is among the ``k`` highest-scoring classes.

    Args:
        scores: ``(samples, classes)`` logits or probabilities.
        labels: ``(samples,)`` integer class ids.
        k: Number of top predictions that count as a hit.
    """
    if scores.ndim != 2 or labels.shape != (len(scores),):
        raise ValueError(
            f"expected scores (N, C) and labels (N,), got {scores.shape}, {labels.shape}"
        )
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be in [1, {scores.shape[1]}], got {k}")
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return float((top_k == labels[:, None]).any(axis=1).mean())


def bootstrap_ci(
    correct: NDArray[np.bool_], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-sample hits.

    The WLASL100 test split has a few hundred clips, so a single accuracy figure is only known to a
    few percentage points; this makes that visible.
    """
    if len(correct) == 0:
        raise ValueError("no samples")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_bootstrap_diff(
    hits_a: NDArray[np.bool_],
    hits_b: NDArray[np.bool_],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """``mean(a) - mean(b)`` with a percentile CI, resampling the same clips for both models.

    Resampling clips jointly accounts for the models being tested on identical data, which gives a
    much tighter interval than comparing two separate confidence intervals by eye.
    """
    if hits_a.shape != hits_b.shape or len(hits_a) == 0:
        raise ValueError("need two non-empty hit arrays of the same length")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hits_a), size=(n_boot, len(hits_a)))
    diffs = hits_a[draws].mean(axis=1) - hits_b[draws].mean(axis=1)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(hits_a.mean() - hits_b.mean()), (float(low), float(high))


def top_k_hits(
    scores: NDArray[np.floating], labels: NDArray[np.integer], k: int = 1
) -> NDArray[np.bool_]:
    """Per-sample version of :func:`top_k_accuracy`."""
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return (top_k == labels[:, None]).any(axis=1)


def confusion_matrix(
    labels: NDArray[np.integer], preds: NDArray[np.integer], n_classes: int
) -> NDArray[np.int64]:
    """``(n_classes, n_classes)`` counts; rows are true classes, columns are predictions."""
    return _sk_confusion_matrix(labels, preds, labels=np.arange(n_classes)).astype(np.int64)


def most_confused(
    cm: NDArray[np.integer], names: Sequence[str], top: int = 10
) -> list[tuple[str, str, int]]:
    """Largest off-diagonal entries as ``(true sign, predicted sign, count)``, biggest first."""
    off_diagonal = cm.astype(np.int64).copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(-off_diagonal, axis=None, kind="stable")[:top]
    rows, cols = np.unravel_index(order, off_diagonal.shape)
    return [
        (names[int(r)], names[int(c)], int(off_diagonal[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if off_diagonal[r, c] > 0
    ]
