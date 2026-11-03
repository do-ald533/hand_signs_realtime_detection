"""Saving, loading and running trained sign classifiers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from signlang.constants import SEQ_LEN
from signlang.models import SignGRU


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: SignGRU
    class_names: list[str]
    seq_len: int
    meta: dict[str, Any]


def save_checkpoint(
    model: SignGRU,
    path: Path,
    *,
    class_names: list[str],
    model_kwargs: dict[str, Any],
    seq_len: int = SEQ_LEN,
    meta: dict[str, Any] | None = None,
) -> None:
    """Weights plus everything needed to rebuild the model without the training code's config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "class_names": class_names,
            "seq_len": seq_len,
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: Path) -> LoadedModel:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = SignGRU(**blob["model_kwargs"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return LoadedModel(model, list(blob["class_names"]), int(blob["seq_len"]), dict(blob["meta"]))


@torch.inference_mode()
def predict_logits(
    model: SignGRU, x: NDArray[np.float32], batch_size: int = 256
) -> NDArray[np.float32]:
    """Logits for ``(samples, frames, 225)`` or ``(samples, frames, 75, 3)`` input."""
    model.eval()
    flat = x.reshape(len(x), x.shape[1], -1)
    outputs = [
        model(torch.from_numpy(flat[start : start + batch_size])).numpy()
        for start in range(0, len(flat), batch_size)
    ]
    return (
        np.concatenate(outputs) if outputs else np.empty((0, model.head.out_features), np.float32)
    )
