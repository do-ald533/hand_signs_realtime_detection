"""Two-layer GRU over a landmark sequence."""

from typing import Literal

import torch
from torch import Tensor, nn

from signlang.constants import N_CLASSES, N_FEATURES


class SignGRU(nn.Module):
    """Classify a ``(batch, frames, features)`` landmark sequence into one of ``n_classes`` signs.

    The logits come from the hidden state of the last frame (``pooling="last"``) or from the mean
    hidden state over all frames (``pooling="mean"``, steadier on small datasets). Every sequence in
    a batch must have the same length (see :func:`signlang.preprocess.resample`).
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_feats: int = N_FEATURES,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.gru = nn.GRU(
            n_feats,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch warns on dropout with 1 layer
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1] if self.pooling == "last" else out.mean(dim=1))

    @torch.inference_mode()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Softmax class probabilities; switches to eval mode and does not restore training mode."""
        self.eval()
        return self(x).softmax(dim=-1)
