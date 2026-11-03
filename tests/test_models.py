import warnings

import numpy as np
import pytest
import torch

from signlang.constants import N_FEATURES, SEQ_LEN
from signlang.models import SignGRU, build_baseline, temporal_mean


def test_gru_output_shape():
    model = SignGRU(n_classes=100)
    logits = model(torch.randn(4, SEQ_LEN, N_FEATURES))
    assert logits.shape == (4, 100)


def test_gru_backward_reaches_every_parameter():
    model = SignGRU(n_classes=5, hidden_size=16)
    model(torch.randn(2, SEQ_LEN, N_FEATURES)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_gru_predict_proba_is_a_distribution():
    model = SignGRU(n_classes=7, hidden_size=16)
    proba = model.predict_proba(torch.randn(3, SEQ_LEN, N_FEATURES))
    torch.testing.assert_close(proba.sum(dim=-1), torch.ones(3))
    assert (proba >= 0).all()


def test_pooling_modes_read_different_summaries_of_the_sequence():
    last = SignGRU(n_classes=4, hidden_size=8, pooling="last").eval()
    mean = SignGRU(n_classes=4, hidden_size=8, pooling="mean").eval()
    mean.load_state_dict(last.state_dict())
    x = torch.randn(2, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        hidden, _ = last.gru(x)
        torch.testing.assert_close(last(x), last.head(hidden[:, -1]))
        torch.testing.assert_close(mean(x), mean.head(hidden.mean(dim=1)))
        assert not torch.allclose(last(x), mean(x))


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        SignGRU(pooling="max")  # pyright: ignore[reportArgumentType]


def test_single_layer_gru_does_not_warn_about_dropout():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SignGRU(n_classes=3, hidden_size=8, num_layers=1)


def test_temporal_mean_collapses_the_frame_axis():
    seqs = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    np.testing.assert_allclose(temporal_mean(seqs), seqs.mean(axis=1))


def test_baseline_ignores_frame_order():
    rng = np.random.default_rng(0)
    seqs = rng.normal(size=(60, SEQ_LEN, 6))
    labels = np.repeat(np.arange(3), 20)
    seqs += labels[:, None, None] * 2.0
    model = build_baseline().fit(seqs, labels)
    shuffled = seqs[:, rng.permutation(SEQ_LEN)]
    np.testing.assert_array_equal(model.predict(shuffled), model.predict(seqs))


def test_baseline_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 25)
    seqs = rng.normal(size=(100, SEQ_LEN, 8)) + labels[:, None, None] * 3.0
    model = build_baseline().fit(seqs, labels)
    assert (model.predict(seqs) == labels).mean() > 0.95
    assert model.predict_proba(seqs).shape == (100, 4)


@pytest.mark.parametrize("n_classes", [2, 100])
def test_gru_supports_different_vocabulary_sizes(n_classes):
    assert SignGRU(n_classes=n_classes, hidden_size=8)(torch.zeros(1, 4, N_FEATURES)).shape == (
        1,
        n_classes,
    )
