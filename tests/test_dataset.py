import numpy as np
import pytest
import torch

from signlang.constants import LEFT_SHOULDER, N_FEATURES, N_POINTS, RIGHT_SHOULDER, SEQ_LEN
from signlang.data.dataset import LandmarkDataset, build_splits, load_splits, save_splits
from signlang.data.download import ManifestRow
from signlang.data.wlasl import load_wlasl


def _row(instance, status="ok") -> ManifestRow:
    return ManifestRow(
        instance.video_id, instance.gloss, instance.label, instance.split, instance.url,
        instance.kind, status, "", f"raw/{instance.video_id}.mp4",
    )  # fmt: skip


def test_build_splits_groups_by_official_split_and_accounts_for_every_drop(
    wlasl_json, raw_sequence, tmp_path
):
    instances = load_wlasl(wlasl_json, 2)  # ids 1 train, 2 val, 3 test, 4 train
    rows = [_row(i) for i in instances]
    rows[1] = _row(instances[1], status="skipped")  # id 2 was never downloaded
    landmarks = tmp_path / "landmarks"
    landmarks.mkdir()
    np.save(landmarks / "1.npy", raw_sequence)
    np.save(landmarks / "3.npy", raw_sequence[:2])  # too short
    # id 4: no landmark file at all

    splits, dropped = build_splits(rows, instances, landmarks)

    assert [len(splits[s]) for s in ("train", "val", "test")] == [1, 0, 0]
    assert splits["train"].x.shape == (1, SEQ_LEN, N_POINTS, 3)
    assert splits["train"].video_ids == ("1",)
    assert dropped == {"not downloaded (swf)": 1, "fewer than 4 frames": 1, "no landmark file": 1}


def test_build_splits_drops_clips_whose_shoulders_were_never_seen(
    wlasl_json, raw_sequence, tmp_path
):
    instances = load_wlasl(wlasl_json, 2)[:1]
    raw_sequence[:, [LEFT_SHOULDER, RIGHT_SHOULDER]] = np.nan
    np.save(tmp_path / "1.npy", raw_sequence)
    splits, dropped = build_splits([_row(instances[0])], instances, tmp_path)
    assert len(splits["train"]) == 0
    assert dropped == {"shoulders never detected": 1}


def test_splits_survive_a_save_load_round_trip(splits, tmp_path):
    path = tmp_path / "p" / "wlasl.npz"
    save_splits(splits, ["a", "b", "c"], path)
    loaded, names = load_splits(path)
    assert names == ["a", "b", "c"]
    for name in splits:
        np.testing.assert_array_equal(loaded[name].x, splits[name].x)
        np.testing.assert_array_equal(loaded[name].y, splits[name].y)
        np.testing.assert_array_equal(loaded[name].signer_ids, splits[name].signer_ids)
        assert loaded[name].video_ids == splits[name].video_ids


def test_flat_view_has_model_layout(splits):
    assert splits["train"].flat.shape == (len(splits["train"]), SEQ_LEN, N_FEATURES)


def test_dataset_yields_model_ready_tensors(splits):
    features, label = LandmarkDataset(splits["train"])[0]
    assert features.shape == (SEQ_LEN, N_FEATURES) and features.dtype == torch.float32
    assert label == int(splits["train"].y[0])


def test_augmentation_changes_samples_but_is_reproducible_per_seed(splits):
    plain = LandmarkDataset(splits["train"])[0][0]
    a = LandmarkDataset(splits["train"], augment_data=True, seed=3)[0][0]
    b = LandmarkDataset(splits["train"], augment_data=True, seed=3)[0][0]
    c = LandmarkDataset(splits["train"], augment_data=True, seed=4)[0][0]
    assert not torch.equal(plain, a)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_augmentation_does_not_modify_the_stored_data(splits):
    before = splits["train"].x.copy()
    dataset = LandmarkDataset(splits["train"], augment_data=True)
    for i in range(len(dataset)):
        dataset[i]
    np.testing.assert_array_equal(splits["train"].x, before)


def test_empty_split_is_representable(wlasl_json, tmp_path):
    splits, dropped = build_splits([], load_wlasl(wlasl_json, 2), tmp_path)
    assert all(len(s) == 0 for s in splits.values()) and not dropped
    assert splits["test"].x.shape == (0, SEQ_LEN, N_POINTS, 3)
    with pytest.raises(IndexError):
        LandmarkDataset(splits["test"])[0]
