"""Lower-bound baseline: average landmarks over time, then logistic regression.

Averaging discards the order of the frames, so the baseline cannot see motion. The gap between it
and the GRU is therefore a direct measure of how much the temporal model is worth.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def temporal_mean(seqs: NDArray[np.floating]) -> NDArray[np.floating]:
    """``(samples, frames, features)`` to ``(samples, features)``."""
    return seqs.mean(axis=1)


def build_baseline(c: float = 1.0, max_iter: int = 2000, seed: int = 0) -> Pipeline:
    """Un-fitted pipeline that takes ``(samples, frames, features)`` arrays directly."""
    return make_pipeline(
        FunctionTransformer(temporal_mean),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=max_iter, random_state=seed),
    )
