import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from signlang.constants import LEFT_SHOULDER, N_COORDS, N_POINTS, RIGHT_SHOULDER, SEQ_LEN
from signlang.data.dataset import SplitData


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def raw_sequence(rng: np.random.Generator) -> NDArray[np.float32]:
    """40 frames of random landmarks with shoulders placed like a real, upright signer."""
    seq = rng.normal(0.5, 0.1, size=(40, N_POINTS, N_COORDS)).astype(np.float32)
    seq[:, LEFT_SHOULDER] = np.array([0.62, 0.50, 0.0]) + rng.normal(0, 0.005, (40, N_COORDS))
    seq[:, RIGHT_SHOULDER] = np.array([0.38, 0.50, 0.0]) + rng.normal(0, 0.005, (40, N_COORDS))
    return seq


def make_split(
    rng: np.random.Generator, n_classes: int = 3, per_class: int = 10, *, signer_offset: int = 0
) -> SplitData:
    """Separable synthetic split: each class sits at a different offset in feature space."""
    labels = np.repeat(np.arange(n_classes), per_class).astype(np.int64)
    x = rng.normal(0, 0.3, size=(len(labels), SEQ_LEN, N_POINTS, N_COORDS)).astype(np.float32)
    x += (labels[:, None, None, None] * 1.5).astype(np.float32)
    ids = tuple(f"v{signer_offset}_{i}" for i in range(len(labels)))
    signers = (np.arange(len(labels)) % 4 + signer_offset).astype(np.int64)
    return SplitData(x=x, y=labels, video_ids=ids, signer_ids=signers)


@pytest.fixture
def splits(rng: np.random.Generator) -> dict[str, SplitData]:
    return {
        "train": make_split(rng, per_class=12),
        "val": make_split(rng, per_class=4),
        "test": make_split(rng, per_class=4, signer_offset=2),
    }


@pytest.fixture
def wlasl_json(tmp_path: Path) -> Path:
    """Tiny WLASL-shaped metadata file: two glosses, one clip from each kind of source."""

    def clip(video_id: str, split: str, url: str, signer: int) -> dict:
        return {
            "video_id": video_id, "split": split, "url": url, "fps": 25, "frame_start": 1,
            "frame_end": -1, "bbox": [0, 0, 64, 64], "signer_id": signer,
        }  # fmt: skip

    meta = [
        {
            "gloss": "book",
            "instances": [
                clip("1", "train", "https://www.youtube.com/watch?v=abcdefghijk", 1),
                clip("2", "val", "http://www.aslpro.com/clips/book.swf", 2),
                clip("3", "test", "http://example.org/book.mp4", 3),
            ],
        },
        {"gloss": "drink", "instances": [clip("4", "train", "http://example.org/drink.mp4", 1)]},
    ]
    path = tmp_path / "WLASL_v0.3.json"
    path.write_text(json.dumps(meta))
    return path
