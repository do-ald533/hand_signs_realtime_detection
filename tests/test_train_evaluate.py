import json

import numpy as np
import pytest

from signlang.evaluate import (
    evaluate_all,
    format_table,
    plot_confusion_matrix,
    score,
    signer_overlap,
)
from signlang.inference import load_checkpoint, predict_logits
from signlang.metrics import bootstrap_ci, paired_bootstrap_diff
from signlang.train import (
    TrainConfig,
    fit_baseline,
    fit_gru,
    load_baseline,
    save_baseline,
)

NAMES = ["alpha", "beta", "gamma"]
# Independent of TrainConfig's tuned defaults, which are far too aggressive for 36 toy sequences.
FAST = {
    "epochs": 12,
    "hidden_size": 16,
    "batch_size": 16,
    "patience": 50,
    "dropout": 0.1,
    "crop_fraction": 0.0,
    "hand_dropout": 0.0,
    "noise_std": 0.0,
}


def test_gru_learns_a_separable_problem_and_checkpoint_round_trips(splits, tmp_path):
    result = fit_gru(splits, NAMES, TrainConfig(**FAST), tmp_path / "gru.pt")
    assert result.val_top1 > 0.9
    assert 1 <= result.best_epoch <= FAST["epochs"]
    assert len(result.history) == FAST["epochs"]

    loaded = load_checkpoint(result.checkpoint)
    assert loaded.class_names == NAMES and loaded.seq_len == 32
    assert loaded.meta["config"]["hidden_size"] == 16
    logits = predict_logits(loaded.model, splits["val"].flat)
    assert (logits.argmax(axis=1) == splits["val"].y).mean() == pytest.approx(result.val_top1)


def test_training_is_reproducible_for_a_fixed_seed(splits, tmp_path):
    a = fit_gru(splits, NAMES, TrainConfig(**FAST, seed=5), tmp_path / "a.pt")
    b = fit_gru(splits, NAMES, TrainConfig(**FAST, seed=5), tmp_path / "b.pt")
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]


def test_early_stopping_cuts_training_short(splits, tmp_path):
    config = TrainConfig(epochs=200, hidden_size=16, patience=3, lr=1e-2, augment=False)
    result = fit_gru(splits, NAMES, config, tmp_path / "gru.pt")
    assert len(result.history) < 200


def test_baseline_fits_saves_and_loads(splits, tmp_path):
    model = fit_baseline(splits)
    save_baseline(model, tmp_path / "m" / "baseline.joblib")
    loaded = load_baseline(tmp_path / "m" / "baseline.joblib")
    test = splits["test"]
    assert (loaded.predict(test.flat) == test.y).mean() > 0.9


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.random(50) < 0.7
    large = rng.random(5000) < 0.7
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert lo_s <= small.mean() <= hi_s
    assert hi_l - lo_l < hi_s - lo_s
    with pytest.raises(ValueError, match="no samples"):
        bootstrap_ci(np.array([], dtype=bool))


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(0)
    base = rng.random(400) < 0.4
    better = base | (rng.random(400) < 0.3)  # strictly better on the same clips
    diff, (low, high) = paired_bootstrap_diff(better, base)
    assert diff > 0.1 and low > 0
    assert diff == pytest.approx(better.mean() - base.mean())
    same, (low, high) = paired_bootstrap_diff(base, base.copy())
    assert same == 0 and low == high == 0
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff(base, base[:10])


def test_score_reports_top1_top5_and_intervals():
    logits = np.eye(6)[[0, 1, 2, 3]] * 5
    result = score(logits, np.array([0, 1, 5, 3]))
    assert result.top1 == pytest.approx(0.75) and result.n == 4
    assert result.top5 >= result.top1
    assert result.top1_ci[0] <= result.top1 <= result.top1_ci[1]


def test_signer_overlap(splits):
    assert signer_overlap(splits["train"], splits["val"]) == 1.0  # same four signers
    assert (
        0.0 < signer_overlap(splits["train"], splits["test"]) < 1.0
    )  # signers 2,3 shared; 4,5 new


def test_confusion_matrix_plot_is_written(tmp_path):
    cm = np.array([[5, 1], [0, 6]])
    path = tmp_path / "figs" / "cm.png"
    plot_confusion_matrix(cm, ["a", "b"], path)
    assert path.stat().st_size > 1000


def test_evaluate_all_writes_reports_for_baseline_and_every_seed(splits, tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    save_baseline(fit_baseline(splits), models / "baseline.joblib")
    for seed in (0, 1):
        fit_gru(splits, NAMES, TrainConfig(**FAST, seed=seed), models / f"gru_seed{seed}.pt")
    (models / "gru.pt").write_bytes((models / "gru_seed0.pt").read_bytes())

    report = evaluate_all(splits, models, reports)

    assert report["dataset"] == {
        "num_classes": 3, "train": 36, "val": 12, "test": 12,
        "test_signer_overlap_with_train": pytest.approx(0.5),
    }  # fmt: skip
    assert len(report["gru"]["seeds"]) == 2
    assert set(report["gru_minus_baseline"]) == {"top1", "top5"}
    assert report["baseline"]["top1"] > 0.9
    assert json.loads((reports / "results.json").read_text())["gru"]["primary"]["n"] == 12
    assert "Baseline" in (reports / "results.md").read_text()
    assert (reports / "figures" / "confusion_matrix.png").exists()
    assert "mean ± std over 2 seeds" in format_table(report)
