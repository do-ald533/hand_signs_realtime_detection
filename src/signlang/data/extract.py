"""Run MediaPipe over every downloaded video once and cache the landmarks as ``.npy`` files.

Each ``<video_id>.npy`` holds the raw ``(frames, 75, 3)`` float32 array of one clip (NaN where a
landmark was not detected). Training then reads these small arrays instead of decoding video every
epoch. An ``index.csv`` records frame counts and detection rates per clip.
"""

import csv
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from tqdm import tqdm

from signlang.data.download import ManifestRow
from signlang.data.wlasl import Instance


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    video_id: str
    status: str  # "ok", "empty" or an error description
    frames: int
    pose_rate: float
    left_rate: float
    right_rate: float
    seconds: float


def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)  # one process per core already; avoid oversubscription


def _extract_one(
    video_id: str,
    video_path: str,
    frame_start: int,
    frame_end: int,
    bbox: tuple[int, int, int, int],
    out_dir: str,
    pose_model: str,
) -> ExtractionResult:
    from signlang.landmarks import detection_rates, extract_video

    started = time.perf_counter()
    target = Path(out_dir) / f"{video_id}.npy"
    try:
        raw = extract_video(
            Path(video_path),
            frame_start=frame_start,
            frame_end=frame_end,
            bbox=bbox,
            pose_model=pose_model,  # pyright: ignore[reportArgumentType]
        )
    except Exception as error:  # one corrupt clip must not abort a 2000-video run
        message = f"{type(error).__name__}: {error}".splitlines()[0][:120]
        return ExtractionResult(video_id, message, 0, 0, 0, 0, time.perf_counter() - started)
    if len(raw) == 0:
        return ExtractionResult(video_id, "empty", 0, 0, 0, 0, time.perf_counter() - started)
    np.save(target, raw)
    rates = detection_rates(raw)
    return ExtractionResult(
        video_id, "ok", len(raw), rates["pose"], rates["left"], rates["right"],
        time.perf_counter() - started,
    )  # fmt: skip


def extract_all(
    rows: Sequence[ManifestRow],
    instances: Iterable[Instance],
    data_root: Path,
    out_dir: Path,
    *,
    workers: int = 6,
    pose_model: str = "full",
    overwrite: bool = False,
) -> list[ExtractionResult]:
    """Extract landmarks for every downloaded clip; clips with a ``.npy`` already are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {i.video_id: i for i in instances}
    jobs = []
    for row in rows:
        if row.status != "ok" or ((out_dir / f"{row.video_id}.npy").exists() and not overwrite):
            continue
        i = by_id[row.video_id]
        jobs.append(
            (row.video_id, str(data_root / row.path), i.frame_start, i.frame_end, i.bbox,
             str(out_dir), pose_model)
        )  # fmt: skip

    results: list[ExtractionResult] = []
    # "spawn" because MediaPipe/TensorFlow-Lite state does not survive fork() safely.
    with ProcessPoolExecutor(
        workers, mp_context=get_context("spawn"), initializer=_init_worker
    ) as pool:
        futures = [pool.submit(_extract_one, *job) for job in jobs]
        done = as_completed(futures)
        results.extend(
            f.result() for f in tqdm(done, total=len(futures), desc="landmarks", unit="clip")
        )
    return results


def write_index(results: Iterable[ExtractionResult], path: Path) -> None:
    """Merge ``results`` into the extraction index (new results replace older rows of a clip)."""
    merged: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="") as handle:
            merged = {row["video_id"]: row for row in csv.DictReader(handle)}
    fields = ["video_id", "status", "frames", "pose_rate", "left_rate", "right_rate", "seconds"]
    for r in results:
        merged[r.video_id] = {
            "video_id": r.video_id,
            "status": r.status,
            "frames": str(r.frames),
            "pose_rate": f"{r.pose_rate:.3f}",
            "left_rate": f"{r.left_rate:.3f}",
            "right_rate": f"{r.right_rate:.3f}",
            "seconds": f"{r.seconds:.2f}",
        }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged.values())
