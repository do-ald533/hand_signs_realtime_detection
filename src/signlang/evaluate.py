"""Test-split evaluation: top-1/top-5 with confidence intervals, confusion matrix, errors."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to disk
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from signlang.data.dataset import SplitData
from signlang.inference import load_checkpoint, predict_logits
from signlang.metrics import (
    bootstrap_ci,
    confusion_matrix,
    most_confused,
    paired_bootstrap_diff,
    top_k_accuracy,
    top_k_hits,
)
from signlang.train import load_baseline


@dataclass(frozen=True, slots=True)
class Scores:
    top1: float
    top5: float
    top1_ci: tuple[float, float]
    top5_ci: tuple[float, float]
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "top1": self.top1, "top5": self.top5, "n": self.n,
            "top1_ci95": list(self.top1_ci), "top5_ci95": list(self.top5_ci),
        }  # fmt: skip


def score(scores: NDArray[np.floating], labels: NDArray[np.integer]) -> Scores:
    top5 = min(5, scores.shape[1])
    return Scores(
        top1=top_k_accuracy(scores, labels, 1),
        top5=top_k_accuracy(scores, labels, top5),
        top1_ci=bootstrap_ci(top_k_hits(scores, labels, 1)),
        top5_ci=bootstrap_ci(top_k_hits(scores, labels, top5)),
        n=len(labels),
    )


def plot_confusion_matrix(cm: NDArray[np.integer], names: Sequence[str], path: Path) -> None:
    """Row-normalized heatmap: row = true sign, column = what the model said."""
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    size = max(8, len(names) * 0.16)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(names)), names, fontsize=5)
    ax.set_xlabel("predicted sign")
    ax.set_ylabel("true sign")
    ax.set_title("WLASL100 test split: share of each sign's clips by prediction")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="share of true class")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def signer_overlap(train: SplitData, other: SplitData) -> float:
    """Share of ``other``'s clips whose signer also appears in ``train``.

    WLASL's official splits are not signer-independent; a high value means test accuracy partly
    measures recognition of familiar signers rather than of signs.
    """
    seen = set(train.signer_ids.tolist())
    return float(np.mean([s in seen for s in other.signer_ids.tolist()])) if len(other) else 0.0


def format_table(report: dict[str, Any]) -> str:
    def cell(s: dict[str, Any]) -> str:
        return f"{s['top1']:.1%} [{s['top1_ci95'][0]:.1%}, {s['top1_ci95'][1]:.1%}]"

    def cell5(s: dict[str, Any]) -> str:
        return f"{s['top5']:.1%} [{s['top5_ci95'][0]:.1%}, {s['top5_ci95'][1]:.1%}]"

    def row(label: str, s: dict[str, Any]) -> str:
        return f"| {label} | {cell(s)} | {cell5(s)} |"

    gru = report["gru"]
    lines = [
        "| Model | Top-1 (95% CI) | Top-5 (95% CI) |",
        "| --- | --- | --- |",
        row("Baseline: mean landmarks + logistic regression", report["baseline"]),
        row("Two-layer GRU (best-validation seed)", gru["primary"]),
    ]
    if gru.get("seeds"):
        top1 = f"{gru['mean_top1']:.1%} ± {gru['std_top1']:.1%}"
        top5 = f"{gru['mean_top5']:.1%} ± {gru['std_top5']:.1%}"
        lines.append(
            f"| Two-layer GRU, mean ± std over {len(gru['seeds'])} seeds | {top1} | {top5} |"
        )
    return "\n".join(lines)


def evaluate_all(
    splits: dict[str, SplitData],
    models_dir: Path,
    reports_dir: Path,
    *,
    top_confused: int = 15,
) -> dict[str, Any]:
    """Score the baseline and every trained GRU on the test split and write the reports."""
    test, train = splits["test"], splits["train"]
    primary = load_checkpoint(models_dir / "gru.pt")
    names = primary.class_names

    baseline = load_baseline(models_dir / "baseline.joblib")
    baseline_proba = np.zeros((len(test), len(names)))
    baseline_proba[:, baseline.classes_] = baseline.predict_proba(
        test.flat
    )  # classes unseen in training stay 0
    baseline_scores = score(baseline_proba, test.y)

    gru_logits = predict_logits(primary.model, test.flat)
    primary_scores = score(gru_logits, test.y)

    seeds = []
    for path in sorted(models_dir.glob("gru_seed*.pt")):
        loaded = load_checkpoint(path)
        s = score(predict_logits(loaded.model, test.flat), test.y)
        seeds.append(
            {"checkpoint": path.name, "val_top1": loaded.meta.get("val_top1"), **s.as_dict()}
        )

    difference = {}
    for key, k in (("top1", 1), ("top5", min(5, len(names)))):
        diff, interval = paired_bootstrap_diff(
            top_k_hits(gru_logits, test.y, k), top_k_hits(baseline_proba, test.y, k)
        )
        difference[key] = {"diff": diff, "ci95": list(interval)}

    cm = confusion_matrix(test.y, gru_logits.argmax(axis=1), len(names))
    gru_report: dict[str, Any] = {"primary": primary_scores.as_dict(), "seeds": seeds}
    if seeds:
        ddof = 1 if len(seeds) > 1 else 0
        for key in ("top1", "top5"):
            values = np.array([s[key] for s in seeds])
            gru_report[f"mean_{key}"] = float(values.mean())
            gru_report[f"std_{key}"] = float(values.std(ddof=ddof))

    report: dict[str, Any] = {
        "dataset": {
            "num_classes": len(names),
            **{name: len(data) for name, data in splits.items()},
            "test_signer_overlap_with_train": signer_overlap(train, test),
        },
        "baseline": baseline_scores.as_dict(),
        "gru": gru_report,
        "gru_minus_baseline": difference,
        "most_confused": [
            {"true": t, "predicted": p, "count": c}
            for t, p, c in most_confused(cm, names, top_confused)
        ],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (reports_dir / "results.md").write_text(format_table(report) + "\n")
    plot_confusion_matrix(cm, names, reports_dir / "figures" / "confusion_matrix.png")
    return report
