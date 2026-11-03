import pytest

from signlang import __version__
from signlang.cli import build_parser, main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_every_pipeline_stage_is_a_command():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # pyright: ignore[reportOptionalMemberAccess]
    assert list(commands) == [
        "fetch-meta",
        "info",
        "download",
        "extract",
        "prepare",
        "train",
        "evaluate",
        "demo",
    ]


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_demo_defaults_match_the_documented_settings():
    args = build_parser().parse_args(["demo"])
    assert (args.source, args.window, args.stride, args.threshold, args.stable) == (
        "0",
        64,
        4,
        0.6,
        3,
    )


def test_info_on_tiny_metadata(wlasl_json, tmp_path, capsys):
    data_dir = tmp_path / "data"
    (data_dir / "meta").mkdir(parents=True)
    (data_dir / "meta" / "WLASL_v0.3.json").write_bytes(wlasl_json.read_bytes())
    assert main(["info", "--data-dir", str(data_dir), "--num-classes", "2"]) == 0
    out = capsys.readouterr().out
    assert "4 videos, 2 signs" in out
    assert "'train': 2" in out and "'swf': 1" in out


def test_train_without_prepared_data_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["train", "--data-dir", str(tmp_path), "--model", "baseline"])
