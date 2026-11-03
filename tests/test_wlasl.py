import json
from pathlib import Path

import pytest

from signlang.data.wlasl import check_against_official, class_names, load_wlasl, url_kind


def test_load_wlasl_assigns_labels_in_gloss_order(wlasl_json):
    instances = load_wlasl(wlasl_json, 2)
    assert [(i.video_id, i.gloss, i.label, i.split) for i in instances] == [
        ("1", "book", 0, "train"),
        ("2", "book", 0, "val"),
        ("3", "book", 0, "test"),
        ("4", "drink", 1, "train"),
    ]
    assert instances[0].bbox == (0, 0, 64, 64)
    assert class_names(instances) == ["book", "drink"]


def test_load_wlasl_keeps_only_the_first_n_glosses(wlasl_json):
    instances = load_wlasl(wlasl_json, num_classes=1)
    assert {i.gloss for i in instances} == {"book"}


@pytest.mark.parametrize("n", [0, 3])
def test_load_wlasl_rejects_bad_subset_size(wlasl_json, n):
    with pytest.raises(ValueError, match="num_classes"):
        load_wlasl(wlasl_json, num_classes=n)


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://www.youtube.com/watch?v=abc", "youtube"),
        ("https://youtu.be/abc", "youtube"),
        ("http://www.aslpro.com/clip.swf", "swf"),
        ("https://media.spreadthesign.com/video/mp4/13/1.mp4", "direct"),
    ],
)
def test_url_kind(url, kind):
    assert url_kind(url) == kind


def _official(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "nslt.json"
    path.write_text(json.dumps(content))
    return path


def test_check_against_official_accepts_identical_metadata(wlasl_json, tmp_path):
    official = {
        "1": {"subset": "train", "action": [0, 1, 9]},
        "2": {"subset": "val", "action": [0, 1, 9]},
        "3": {"subset": "test", "action": [0, 1, 9]},
        "4": {"subset": "train", "action": [1, 1, 9]},
    }
    assert check_against_official(load_wlasl(wlasl_json, 2), _official(tmp_path, official)) == []


def test_check_against_official_reports_every_kind_of_difference(wlasl_json, tmp_path):
    official = {
        "1": {"subset": "test", "action": [0, 1, 9]},  # wrong split
        "2": {"subset": "val", "action": [5, 1, 9]},  # wrong label
        "3": {"subset": "test", "action": [0, 1, 9]},
        "99": {"subset": "train", "action": [0, 1, 9]},  # not ours
    }  # video 4 is missing from the official file
    problems = check_against_official(load_wlasl(wlasl_json, 2), _official(tmp_path, official))
    assert len(problems) == 4
    text = "\n".join(problems)
    assert "video 1: split train != official test" in text
    assert "video 2: label 0 != official 5" in text
    assert "video 99 only in official file" in text
    assert "video 4 not in official file" in text


def test_load_wlasl_reads_json_from_disk_as_given(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            [
                {
                    "gloss": "x",
                    "instances": [{"video_id": 7, "split": "test", "url": "http://a/b.mp4"}],
                }
            ]
        )
    )
    (instance,) = load_wlasl(path, 1)
    assert instance.video_id == "7"  # ids are strings even if the file stores numbers
    assert instance.frame_end == -1 and instance.frame_start == 1
