"""WLASL metadata: which videos make up the WLASL-N subset and which official split each is in.

``WLASL_v0.3.json`` lists glosses by decreasing number of videos, so the first ``N`` entries are the
WLASL-N subset (100 for WLASL100). Every instance carries its official ``train``/``val``/``test``
assignment. :func:`check_against_official` cross-checks both facts against the authors' own
``nslt_N.json``.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlparse

Split = Literal["train", "val", "test"]
SPLITS: Final[tuple[Split, ...]] = ("train", "val", "test")
UrlKind = Literal["youtube", "swf", "direct"]


@dataclass(frozen=True, slots=True)
class Instance:
    """One labelled video clip."""

    video_id: str
    gloss: str
    label: int
    split: Split
    url: str
    fps: int
    frame_start: int  # 1-based, inclusive
    frame_end: int  # 1-based, inclusive; -1 means "to the end of the video"
    bbox: tuple[int, int, int, int]  # signer box (x1, y1, x2, y2) in pixels
    signer_id: int

    @property
    def kind(self) -> UrlKind:
        return url_kind(self.url)


def url_kind(url: str) -> UrlKind:
    """How to fetch a video: YouTube needs yt-dlp, aslpro serves Flash, the rest is plain HTTP."""
    host = urlparse(url).netloc.lower()
    if "youtube" in host or "youtu.be" in host:
        return "youtube"
    if "aslpro" in host:
        return "swf"
    return "direct"


def load_wlasl(meta_path: Path | str, num_classes: int = 100) -> list[Instance]:
    """Instances of the first ``num_classes`` glosses of ``WLASL_v0.3.json``."""
    entries = json.loads(Path(meta_path).read_text())
    if not 1 <= num_classes <= len(entries):
        raise ValueError(f"num_classes must be in [1, {len(entries)}], got {num_classes}")

    instances: list[Instance] = []
    for label, entry in enumerate(entries[:num_classes]):
        for raw in entry["instances"]:
            bbox = raw.get("bbox") or [0, 0, 0, 0]
            instances.append(
                Instance(
                    video_id=str(raw["video_id"]),
                    gloss=entry["gloss"],
                    label=label,
                    split=raw["split"],
                    url=raw["url"],
                    fps=int(raw.get("fps", 25)),
                    frame_start=int(raw.get("frame_start", 1)),
                    frame_end=int(raw.get("frame_end", -1)),
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    signer_id=int(raw.get("signer_id", -1)),
                )
            )
    return instances


def class_names(instances: Sequence[Instance]) -> list[str]:
    """Gloss for every label id, ordered by label."""
    names: dict[int, str] = {i.label: i.gloss for i in instances}
    return [names[label] for label in sorted(names)]


def check_against_official(instances: Sequence[Instance], nslt_path: Path | str) -> list[str]:
    """Disagreements between ``instances`` and the authors' ``nslt_N.json`` (empty means identical).

    Compares the set of video ids, each video's split and each video's class index.
    """
    official: dict[str, dict] = json.loads(Path(nslt_path).read_text())
    ours = {i.video_id: i for i in instances}
    problems = [f"video {v} only in official file" for v in sorted(official.keys() - ours.keys())]
    problems += [f"video {v} not in official file" for v in sorted(ours.keys() - official.keys())]
    for video_id in sorted(ours.keys() & official.keys()):
        theirs, mine = official[video_id], ours[video_id]
        if theirs["subset"] != mine.split:
            problems.append(f"video {video_id}: split {mine.split} != official {theirs['subset']}")
        if theirs["action"][0] != mine.label:
            problems.append(
                f"video {video_id}: label {mine.label} != official {theirs['action'][0]}"
            )
    return problems
