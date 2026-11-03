"""Turn cached landmark files into per-split training arrays and a PyTorch dataset."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from signlang.augment import augment
from signlang.constants import N_COORDS, N_POINTS, SEQ_LEN
from signlang.data.download import ManifestRow
from signlang.data.wlasl import SPLITS, Instance
from signlang.preprocess import normalize, resample

MIN_FRAMES = 4


@dataclass(frozen=True, slots=True)
class SplitData:
    """Normalized, fixed-length sequences of one split."""

    x: NDArray[np.float32]  # (samples, frames, 75, 3)
    y: NDArray[np.int64]  # (samples,)
    video_ids: tuple[str, ...]
    signer_ids: NDArray[np.int64]  # (samples,)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def flat(self) -> NDArray[np.float32]:
        """``(samples, frames, 225)``: the layout the models take."""
        return self.x.reshape(len(self), self.x.shape[1], -1)


def build_splits(
    rows: Sequence[ManifestRow],
    instances: Sequence[Instance],
    landmarks_dir: Path,
    seq_len: int = SEQ_LEN,
) -> tuple[dict[str, SplitData], Counter[str]]:
    """Load, normalize and resample every cached clip, grouped by the official split.

    Returns the splits and a counter of why clips were left out (no landmarks, too short, no
    shoulders detected), so the final dataset size is always accounted for.
    """
    by_id = {i.video_id: i for i in instances}
    columns: dict[str, list] = {s: [] for s in SPLITS}
    dropped: Counter[str] = Counter()

    for row in rows:
        if row.status != "ok":
            dropped[f"not downloaded ({row.kind})"] += 1
            continue
        path = landmarks_dir / f"{row.video_id}.npy"
        if not path.exists():
            dropped["no landmark file"] += 1
            continue
        raw = np.load(path)
        if len(raw) < MIN_FRAMES:
            dropped[f"fewer than {MIN_FRAMES} frames"] += 1
            continue
        try:
            seq = resample(normalize(raw), seq_len)
        except ValueError:
            dropped["shoulders never detected"] += 1
            continue
        instance = by_id[row.video_id]
        columns[instance.split].append((seq, instance.label, row.video_id, instance.signer_id))

    splits = {}
    for split, items in columns.items():
        seqs, labels, ids, signers = zip(*items, strict=True) if items else ((), (), (), ())
        splits[split] = SplitData(
            x=np.stack(seqs) if items else np.empty((0, seq_len, N_POINTS, N_COORDS), np.float32),
            y=np.asarray(labels, dtype=np.int64),
            video_ids=tuple(ids),
            signer_ids=np.asarray(signers, dtype=np.int64),
        )
    return splits, dropped


def save_splits(splits: dict[str, SplitData], class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray] = {"class_names": np.asarray(class_names)}
    for name, data in splits.items():
        arrays[f"x_{name}"] = data.x
        arrays[f"y_{name}"] = data.y
        arrays[f"ids_{name}"] = np.asarray(data.video_ids)
        arrays[f"signers_{name}"] = data.signer_ids
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]


def load_splits(path: Path) -> tuple[dict[str, SplitData], list[str]]:
    with np.load(path) as archive:
        splits = {
            name: SplitData(
                x=archive[f"x_{name}"],
                y=archive[f"y_{name}"],
                video_ids=tuple(str(v) for v in archive[f"ids_{name}"]),
                signer_ids=archive[f"signers_{name}"],
            )
            for name in SPLITS
        }
        return splits, [str(n) for n in archive["class_names"]]


class LandmarkDataset(Dataset[tuple[torch.Tensor, int]]):
    """``(frames, 225)`` tensors with optional on-the-fly augmentation (new draw every epoch)."""

    def __init__(
        self,
        data: SplitData,
        *,
        augment_data: bool = False,
        seed: int = 0,
        augment_options: dict[str, Any] | None = None,
    ) -> None:
        self._data = data
        self._augment = augment_data
        self._options = augment_options or {}
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        seq = self._data.x[index]
        if self._augment:
            seq = augment(seq, self._rng, **self._options)
        return torch.from_numpy(seq.reshape(len(seq), -1)), int(self._data.y[index])
