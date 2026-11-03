import numpy as np
import pytest

from signlang.data import download
from signlang.data.download import (
    ManifestRow,
    download_all,
    ok_rows,
    read_manifest,
    summarize,
    write_manifest,
)
from signlang.data.wlasl import load_wlasl


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(download, "_polite_pause", lambda: None)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)


def _write_video(path):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()


def _row(video_id="1", split="train", status="ok", kind="direct", reason="") -> ManifestRow:
    return ManifestRow(
        video_id, "book", 0, split, "http://x/y.mp4", kind, status, reason, "raw/1.mp4"
    )


def test_manifest_round_trip(tmp_path):
    rows = [_row("1"), _row("2", "val", "failed", reason="HTTPError: 403")]
    write_manifest(rows, tmp_path / "sub" / "manifest.csv")
    assert read_manifest(tmp_path / "sub" / "manifest.csv") == rows


def test_summarize_counts_per_split_and_explains_losses():
    rows = [
        _row("1"),
        _row("2", status="failed", reason="HTTPError: 403"),
        _row("3", "test", "skipped", "swf", "Flash"),
    ]
    text = summarize(rows)
    table = {line.split()[0]: line.split()[1:] for line in text.splitlines()[:5]}
    assert table["train"] == ["1", "2", "50.0%"]
    assert table["test"] == ["0", "1", "0.0%"]
    assert table["val"] == ["0", "0", "n/a"]
    assert table["all"] == ["1", "3", "33.3%"]
    assert "direct/failed: HTTPError: 403" in text
    assert "swf/skipped: Flash" in text
    assert [r.video_id for r in ok_rows(rows)] == ["1"]


def test_download_all_handles_each_source_kind(wlasl_json, tmp_path):
    instances = load_wlasl(wlasl_json, 2)
    good = tmp_path / "good.mp4"
    _write_video(good)
    # Point the two "direct" clips at local files: one real video, one missing file.
    instances = [
        i
        if i.kind != "direct"
        else _with_url(i, good.as_uri() if i.video_id == "3" else "file:///nope.mp4")
        for i in instances
    ]
    # ...and the YouTube clip is turned into a direct failure so the test needs no network.
    instances[0] = _with_url(instances[0], "file:///also-missing.mp4")

    out = tmp_path / "data" / "raw"
    rows = {r.video_id: r for r in download_all(instances, out, retries=0)}

    assert rows["3"].status == "ok" and rows["3"].path == "raw/3.mp4"
    assert (out / "3.mp4").exists()
    assert rows["1"].status == "failed" and rows["4"].status == "failed"
    assert rows["2"].status == "skipped" and "Flash" in rows["2"].reason
    assert not list(out.glob("*.part"))  # no half-written files left behind


def test_download_all_fetches_a_shared_url_once_and_reuses_existing_files(wlasl_json, tmp_path):
    good = tmp_path / "good.mp4"
    _write_video(good)
    base = load_wlasl(wlasl_json, 2)
    twins = [_with_url(base[2], good.as_uri()), _with_url(base[3], good.as_uri())]  # ids 3 and 4

    out = tmp_path / "data" / "raw"
    first = download_all(twins, out, retries=0)
    assert [r.status for r in first] == ["ok", "ok"]
    assert first[0].path == first[1].path  # one file serves both clips

    again = download_all(twins, out, retries=0)
    assert [r.reason for r in again] == ["already downloaded"] * 2


def test_html_error_pages_are_not_accepted_as_videos(wlasl_json, tmp_path):
    page = tmp_path / "page.mp4"
    page.write_text("<!DOCTYPE html><html>Sign in</html>")
    (instance,) = [_with_url(load_wlasl(wlasl_json, 2)[2], page.as_uri())]
    (row,) = download_all([instance], tmp_path / "raw", retries=0)
    assert row.status == "failed" and "web page" in row.reason


def _with_url(instance, url):
    from dataclasses import replace

    return replace(instance, url=url)
