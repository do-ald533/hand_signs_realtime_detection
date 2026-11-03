"""Download WLASL videos and record, per video, whether it could be fetched.

Many WLASL links are dead, so the outcome of every attempt is written to a manifest CSV. Its ``ok``
rows are the dataset that actually exists; report those counts, not the nominal 2038.

Three kinds of source (see :func:`signlang.data.wlasl.url_kind`):

* ``direct``: plain HTTP(S) files, fetched with ``urllib``.
* ``youtube``: fetched with yt-dlp, single-file formats only (no ffmpeg needed), at most 480p.
* ``swf``: aslpro.com serves Flash, which OpenCV cannot decode; recorded as ``skipped``.

Several instances can point at the same URL; it is downloaded once and shared.
"""

import csv
import logging
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final, Literal

from tqdm import tqdm

from signlang.data.wlasl import Instance, Split

log = logging.getLogger(__name__)

Status = Literal["ok", "failed", "skipped"]
# Same browser-style User-Agent the WLASL authors' own downloader sends; several hosts reject bots.
_USER_AGENT: Final = (
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
)
_MAX_BYTES: Final = 100 * 1024 * 1024
# Video-only streams: signs need no audio, and a single stream needs no ffmpeg to merge.
# AV1 is avoided because OpenCV builds often cannot decode it.
_YT_FORMAT: Final = "bv*[height<=480][vcodec!*=av01]/bv*[vcodec!*=av01]/bv*[height<=480]/bv*/b"


@dataclass(frozen=True, slots=True)
class ManifestRow:
    video_id: str
    gloss: str
    label: int
    split: Split
    url: str
    kind: str
    status: Status
    reason: str
    path: str  # video file, relative to the manifest's directory; empty unless status == "ok"


_META_BASE: Final = "https://raw.githubusercontent.com/dxli94/WLASL/master"
_META_FILES: Final = {
    "WLASL_v0.3.json": "start_kit/WLASL_v0.3.json",
    "nslt_100.json": "code/I3D/preprocess/nslt_100.json",
    "wlasl_class_list.txt": "code/I3D/preprocess/wlasl_class_list.txt",
}


def fetch_metadata(dest_dir: Path) -> list[Path]:
    """Download the WLASL index files (about 12 MB) from the authors' repository."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, remote in _META_FILES.items():
        dest = dest_dir / name
        if not dest.exists():
            request = urllib.request.Request(
                f"{_META_BASE}/{remote}", headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                dest.write_bytes(response.read())
        paths.append(dest)
    return paths


def _polite_pause() -> None:
    time.sleep(random.uniform(0.5, 1.5))  # be kind to small hosts, as the WLASL authors ask


def is_decodable(path: Path) -> bool:
    """True if OpenCV can open ``path`` and decode its first frame."""
    import cv2  # imported lazily so the pure-Python parts of this module need no GUI libraries

    capture = cv2.VideoCapture(str(path))
    try:
        ok, _ = capture.read() if capture.isOpened() else (False, None)
    finally:
        capture.release()
    return bool(ok)


def _fetch_direct(url: str, dest: Path, timeout: float, max_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    part = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
            size = 0
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"larger than {max_bytes >> 20} MB")
                out.write(chunk)
        head = part.read_bytes()[:16].lstrip().lower()
        if head.startswith((b"<!doctype", b"<html", b"<?xml")):
            raise ValueError("got a web page, not a video")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def _fetch_youtube(url: str, dest_stem: Path, timeout: float, max_bytes: int) -> Path:
    from yt_dlp import YoutubeDL

    options = {
        "format": _YT_FORMAT,
        "outtmpl": f"{dest_stem}.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 2,
        "socket_timeout": timeout,
        "max_filesize": max_bytes,
    }
    # YouTube extraction needs a JavaScript runtime; yt-dlp only enables deno by default.
    for runtime in ("deno", "node", "bun"):
        if shutil.which(runtime):
            options["js_runtimes"] = {runtime: {}}
            break
    with YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
        ydl.download([url])
    found = sorted(dest_stem.parent.glob(f"{dest_stem.name}.*"))
    found = [p for p in found if p.suffix not in {".part", ".ytdl"}]
    if not found:
        raise FileNotFoundError("yt-dlp finished without producing a file")
    return found[0]


def _download_one(
    instance: Instance, out_dir: Path, retries: int, timeout: float, max_bytes: int
) -> tuple[Status, str, Path | None]:
    """Fetch the video behind ``instance.url``; returns (status, reason, path)."""
    existing = [p for p in out_dir.glob(f"{instance.video_id}.*") if p.suffix != ".part"]
    if existing and is_decodable(existing[0]):
        return "ok", "already downloaded", existing[0]

    if instance.kind == "swf":
        return "skipped", "Flash (.swf) video; no decoder available", None

    last_error = "unknown error"
    for attempt in range(1, retries + 2):
        try:
            if instance.kind == "youtube":
                path = _fetch_youtube(instance.url, out_dir / instance.video_id, timeout, max_bytes)
            else:
                path = out_dir / f"{instance.video_id}.mp4"
                _fetch_direct(instance.url, path, timeout, max_bytes)
            if not is_decodable(path):
                path.unlink(missing_ok=True)
                raise ValueError("file is not a decodable video")
            _polite_pause()
            return "ok", "", path
        except Exception as error:  # network, HTTP, yt-dlp and decode errors all mean "no video"
            last_error = f"{type(error).__name__}: {error}".splitlines()[0][:200]
            log.debug("attempt %d for %s failed: %s", attempt, instance.video_id, last_error)
            time.sleep(attempt)
    return "failed", last_error, None


def download_all(
    instances: Sequence[Instance],
    out_dir: Path,
    *,
    direct_workers: int = 8,
    youtube_workers: int = 3,
    retries: int = 1,
    timeout: float = 30.0,
    max_bytes: int = _MAX_BYTES,
    on_result: Callable[[ManifestRow], None] | None = None,
) -> list[ManifestRow]:
    """Download every instance (once per distinct URL) and return one manifest row for each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_url: dict[str, list[Instance]] = defaultdict(list)
    for instance in instances:
        by_url[instance.url].append(instance)

    outcomes: dict[str, tuple[Status, str, Path | None]] = {}
    for kind, workers in (("direct", direct_workers), ("youtube", youtube_workers), ("swf", 1)):
        leaders = [group[0] for group in by_url.values() if group[0].kind == kind]
        with (
            ThreadPoolExecutor(max_workers=workers) as pool,
            tqdm(total=len(leaders), desc=f"download {kind}", unit="video") as bar,
        ):
            futures = {
                pool.submit(_download_one, leader, out_dir, retries, timeout, max_bytes): leader
                for leader in leaders
            }
            for future in as_completed(futures):
                outcomes[futures[future].url] = future.result()
                bar.update()

    rows = []
    for instance in instances:
        status, reason, path = outcomes[instance.url]
        row = ManifestRow(
            video_id=instance.video_id,
            gloss=instance.gloss,
            label=instance.label,
            split=instance.split,
            url=instance.url,
            kind=instance.kind,
            status=status,
            reason=reason,
            path=path.relative_to(out_dir.parent).as_posix() if path else "",
        )
        rows.append(row)
        if on_result:
            on_result(row)
    return rows


def write_manifest(rows: Iterable[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ManifestRow)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as handle:
        return [
            ManifestRow(**{**row, "label": int(row["label"])})  # pyright: ignore[reportArgumentType]
            for row in csv.DictReader(handle)
        ]


def summarize(rows: Iterable[ManifestRow]) -> str:
    """Human-readable availability table: how many videos exist per split and why others do not."""
    rows = list(rows)
    lines = [f"{'split':<6}{'ok':>6}{'total':>7}{'kept':>8}"]
    for split in ("train", "val", "test"):
        in_split = [r for r in rows if r.split == split]
        ok = sum(r.status == "ok" for r in in_split)
        share = f"{ok / len(in_split):.1%}" if in_split else "n/a"
        lines.append(f"{split:<6}{ok:>6}{len(in_split):>7}{share:>8}")
    ok_total = sum(r.status == "ok" for r in rows)
    lines.append(f"{'all':<6}{ok_total:>6}{len(rows):>7}{ok_total / max(len(rows), 1):>8.1%}")

    reasons = Counter(f"{r.kind}/{r.status}: {r.reason or '-'}" for r in rows if r.status != "ok")
    if reasons:
        lines.append("")
        lines.append("not downloaded:")
        lines += [f"  {count:>5}  {reason}" for reason, count in reasons.most_common(8)]
    return "\n".join(lines)


def ok_rows(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    return [r for r in rows if r.status == "ok"]
