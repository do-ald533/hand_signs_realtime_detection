"""Training for the logistic-regression baseline and the GRU.

Model selection uses the validation split only. The test split is touched by
:mod:`signlang.evaluate` and nowhere else.
"""

import copy
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from signlang.constants import N_FEATURES
from signlang.data.dataset import LandmarkDataset, SplitData
from signlang.inference import predict_logits, save_checkpoint
from signlang.metrics import top_k_accuracy
from signlang.models import SignGRU, build_baseline

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Defaults are the configuration chosen on the validation split (see docs/tuning.md).
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 5e-2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 50  # epochs without a better validation top-1 before stopping
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    pooling: Literal["last", "mean"] = "mean"
    augment: bool = True
    crop_fraction: float = 0.4  # time-crop augmentation, see signlang.augment
    hand_dropout: float = 0.35
    noise_std: float = 0.08
    seed: int = 0

    def augment_options(self) -> dict[str, float]:
        return {
            "crop_fraction": self.crop_fraction,
            "hand_dropout": self.hand_dropout,
            "noise_std": self.noise_std,
        }


@dataclass(slots=True)
class TrainResult:
    best_epoch: int
    val_top1: float
    val_top5: float
    checkpoint: Path
    history: list[dict[str, float]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code that still uses the global RNG
    torch.manual_seed(seed)


def fit_gru(
    splits: dict[str, SplitData],
    class_names: list[str],
    config: TrainConfig,
    checkpoint: Path,
) -> TrainResult:
    """Train a :class:`SignGRU`; keeps and saves the epoch with the best validation top-1."""
    set_seed(config.seed)
    train, val = splits["train"], splits["val"]
    model_kwargs = {
        "n_classes": len(class_names),
        "n_feats": N_FEATURES,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "pooling": config.pooling,
    }
    model = SignGRU(**model_kwargs)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loader = DataLoader(
        LandmarkDataset(
            train,
            augment_data=config.augment,
            seed=config.seed,
            augment_options=config.augment_options(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_key, best_epoch, stale = (-1.0, 0.0), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += loss.item() * len(labels)
        scheduler.step()

        logits = predict_logits(model, val.flat)
        val_loss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(val.y)))
        val_top1 = top_k_accuracy(logits, val.y, 1)
        val_top5 = top_k_accuracy(logits, val.y, min(5, len(class_names)))
        history.append(
            {"epoch": epoch, "train_loss": running / len(train), "val_loss": val_loss,
             "val_top1": val_top1, "val_top5": val_top5}
        )  # fmt: skip
        if epoch == 1 or epoch % 10 == 0:
            log.info("epoch %3d  train %.3f  val %.3f  top1 %.3f  top5 %.3f",
                     epoch, history[-1]["train_loss"], val_loss, val_top1, val_top5)  # fmt: skip

        key = (val_top1, -val_loss)  # accuracy first, lower loss breaks ties
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    best = history[best_epoch - 1]
    save_checkpoint(
        model,
        checkpoint,
        class_names=class_names,
        model_kwargs=model_kwargs,
        meta={"config": asdict(config), "best_epoch": best_epoch, "val_top1": best["val_top1"]},
    )
    return TrainResult(best_epoch, best["val_top1"], best["val_top5"], checkpoint, history)


def fit_baseline(splits: dict[str, SplitData], seed: int = 0, c: float = 100.0) -> Pipeline:
    """Temporal-mean logistic regression on the training split.

    ``c`` was picked on the validation split: top-1 rises from C=0.0005 to a plateau around
    C=5..100 and falls again at C=1000.
    """
    train = splits["train"]
    return build_baseline(c=c, seed=seed).fit(train.flat, train.y)


def save_baseline(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline(path: Path) -> Pipeline:
    return joblib.load(path)
