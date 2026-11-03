"""Command-line entry point: ``signlang <command>``.

Pipeline order: fetch-meta -> download -> extract -> prepare -> train -> evaluate -> demo.
Heavy dependencies (MediaPipe, PyTorch, OpenCV) are imported inside the commands that need them so
that ``signlang --help`` and ``signlang info`` start instantly.
"""

import argparse
import logging
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from signlang import __version__


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    root: Path = args.data_dir
    return {
        "meta": root / "meta" / "WLASL_v0.3.json",
        "nslt": root / "meta" / f"nslt_{args.num_classes}.json",
        "raw": root / "raw",
        "manifest": root / "manifest.csv",
    }


def _cmd_fetch_meta(args: argparse.Namespace) -> int:
    from signlang.data.download import fetch_metadata

    for path in fetch_metadata(args.data_dir / "meta"):
        print(f"ok  {path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from signlang.data.download import read_manifest, summarize
    from signlang.data.wlasl import check_against_official, load_wlasl

    paths = _paths(args)
    instances = load_wlasl(paths["meta"], args.num_classes)
    splits = Counter(i.split for i in instances)
    kinds = Counter(i.kind for i in instances)
    print(
        f"WLASL{args.num_classes}: {len(instances)} videos, "
        f"{len({i.label for i in instances})} signs"
    )
    print("official splits:", dict(splits))
    print("sources:", dict(kinds))
    print("signers:", len({i.signer_id for i in instances}))
    if paths["nslt"].exists():
        problems = check_against_official(instances, paths["nslt"])
        print(
            "matches official nslt file:",
            "yes" if not problems else f"NO ({len(problems)} differences)",
        )
    if paths["manifest"].exists():
        print("\n" + summarize(read_manifest(paths["manifest"])))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from signlang.data.download import download_all, summarize, write_manifest
    from signlang.data.wlasl import load_wlasl

    paths = _paths(args)
    instances = load_wlasl(paths["meta"], args.num_classes)
    if args.limit:
        instances = instances[:: max(len(instances) // args.limit, 1)][: args.limit]
    rows = download_all(
        instances,
        paths["raw"],
        direct_workers=args.workers,
        youtube_workers=args.youtube_workers,
    )
    write_manifest(rows, paths["manifest"])
    print(summarize(rows))
    print(f"\nmanifest: {paths['manifest']}")
    return 0


def _load_rows_and_instances(args: argparse.Namespace):
    from signlang.data.download import read_manifest
    from signlang.data.wlasl import load_wlasl

    paths = _paths(args)
    if not paths["manifest"].exists():
        raise SystemExit(f"{paths['manifest']} not found; run `signlang download` first")
    return read_manifest(paths["manifest"]), load_wlasl(paths["meta"], args.num_classes)


def _cmd_extract(args: argparse.Namespace) -> int:
    from signlang.data.extract import extract_all, write_index

    rows, instances = _load_rows_and_instances(args)
    out_dir = args.data_dir / "landmarks"
    results = extract_all(
        rows, instances, args.data_dir, out_dir,
        workers=args.workers, pose_model=args.pose_model, overwrite=args.overwrite,
    )  # fmt: skip
    write_index(results, out_dir / "index.csv")
    bad = Counter(r.status for r in results if r.status != "ok")
    print(
        f"extracted {sum(r.status == 'ok' for r in results)} clips; problems: {dict(bad) or 'none'}"
    )
    return 0


def _processed_path(args: argparse.Namespace) -> Path:
    return args.data_dir / "processed" / f"wlasl{args.num_classes}_t{args.seq_len}.npz"


def _cmd_prepare(args: argparse.Namespace) -> int:
    from signlang.data.dataset import build_splits, save_splits
    from signlang.data.wlasl import class_names

    rows, instances = _load_rows_and_instances(args)
    splits, dropped = build_splits(rows, instances, args.data_dir / "landmarks", args.seq_len)
    save_splits(splits, class_names(instances), _processed_path(args))
    total = {name: len(data) for name, data in splits.items()}
    nominal = Counter(i.split for i in instances)
    print(f"{'split':<6}{'used':>6}{'official':>10}")
    for name in ("train", "val", "test"):
        print(f"{name:<6}{total[name]:>6}{nominal[name]:>10}")
    if dropped:
        print("left out:", dict(dropped))
    print(f"saved {_processed_path(args)}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from signlang.data.dataset import load_splits
    from signlang.train import TrainConfig, fit_baseline, fit_gru, save_baseline

    splits, names = load_splits(_processed_path(args))
    models = args.models_dir
    if args.model == "baseline":
        save_baseline(fit_baseline(splits), models / "baseline.joblib")
        print(f"saved {models / 'baseline.joblib'}")
        return 0

    best: tuple[float, Path] | None = None
    for offset in range(args.seeds):
        seed = args.seed + offset
        config = TrainConfig(
            epochs=args.epochs, lr=args.lr, hidden_size=args.hidden, dropout=args.dropout,
            weight_decay=args.weight_decay, pooling=args.pooling, augment=not args.no_augment,
            crop_fraction=args.crop, hand_dropout=args.hand_dropout, noise_std=args.noise,
            seed=seed,
        )  # fmt: skip
        path = models / f"gru_seed{seed}.pt"
        result = fit_gru(splits, names, config, path)
        print(
            f"seed {seed}: best epoch {result.best_epoch}, "
            f"val top-1 {result.val_top1:.3f}, top-5 {result.val_top5:.3f}"
        )
        if best is None or result.val_top1 > best[0]:
            best = (result.val_top1, path)
    assert best is not None
    shutil.copyfile(best[1], models / "gru.pt")  # the demo and the confusion matrix use this one
    print(f"best by validation: {best[1].name} -> {models / 'gru.pt'}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from signlang.data.dataset import load_splits
    from signlang.evaluate import evaluate_all, format_table

    splits, _ = load_splits(_processed_path(args))
    report = evaluate_all(splits, args.models_dir, args.reports_dir)
    print(format_table(report))
    print(
        "\ntest signer overlap with train: "
        f"{report['dataset']['test_signer_overlap_with_train']:.0%}"
    )
    print("most confused (true -> predicted):")
    for item in report["most_confused"][:8]:
        print(f"  {item['true']:>12} -> {item['predicted']:<12} x{item['count']}")
    print(f"\nwrote {args.reports_dir}/results.json, results.md, figures/confusion_matrix.png")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from signlang.live import run_live

    source: int | Path = int(args.source) if args.source.isdigit() else Path(args.source)
    stats = run_live(
        source, args.models_dir / "gru.pt",
        window=args.window, stride=args.stride, threshold=args.threshold, stable=args.stable,
        hold=args.hold, min_activity=args.min_activity, pose_model=args.pose_model,
        headless=args.headless, mirror_view=not args.no_mirror_view,
        draw_landmarks=not args.no_landmarks, max_frames=args.max_frames,
    )  # fmt: skip
    print(
        f"{stats.frames} frames in {stats.seconds:.1f}s = {stats.fps:.1f} FPS "
        f"({stats.ms_per_frame:.1f} ms/frame landmarks+model); "
        f"words shown: {list(stats.words) or 'none'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signlang", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=Path("data"))
    common.add_argument("--num-classes", type=int, default=100, help="WLASL subset size")

    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(
        name: str, func: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[common], help=help_, description=help_)
        sub.set_defaults(func=func)
        return sub

    add("fetch-meta", _cmd_fetch_meta, "download the WLASL index files (about 12 MB)")
    add("info", _cmd_info, "describe the dataset and what has been downloaded")

    download = add("download", _cmd_download, "download videos and write data/manifest.csv")
    download.add_argument("--workers", type=int, default=8, help="parallel direct downloads")
    download.add_argument(
        "--youtube-workers", type=int, default=3, help="parallel yt-dlp downloads"
    )
    download.add_argument(
        "--limit", type=int, default=0, help="only a spread-out sample of N videos"
    )

    extract = add("extract", _cmd_extract, "run MediaPipe on every downloaded video, cache .npy")
    extract.add_argument("--workers", type=int, default=6, help="parallel processes")
    extract.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full")
    extract.add_argument(
        "--overwrite", action="store_true", help="redo clips that already have .npy"
    )

    prepare = add(
        "prepare", _cmd_prepare, "normalize + resample cached landmarks into one dataset file"
    )
    prepare.add_argument("--seq-len", type=int, default=32)

    for name, func, help_ in (
        ("train", _cmd_train, "train the baseline or the GRU (validation split only)"),
        ("evaluate", _cmd_evaluate, "score on the TEST split and write reports/"),
    ):
        sub = add(name, func, help_)
        sub.add_argument("--seq-len", type=int, default=32)
        sub.add_argument("--models-dir", type=Path, default=Path("models"))
        if name == "train":
            sub.add_argument("--model", choices=["baseline", "gru"], default="gru")
            sub.add_argument(
                "--seeds", type=int, default=5, help="train this many GRUs, seeds consecutive"
            )
            sub.add_argument("--seed", type=int, default=0, help="first seed")
            sub.add_argument("--epochs", type=int, default=200)
            sub.add_argument("--lr", type=float, default=2e-3)
            sub.add_argument("--hidden", type=int, default=256)
            sub.add_argument("--dropout", type=float, default=0.5)
            sub.add_argument("--pooling", choices=["last", "mean"], default="mean")
            sub.add_argument("--weight-decay", type=float, default=5e-2)
            sub.add_argument("--no-augment", action="store_true", help="disable all augmentation")
            sub.add_argument("--crop", type=float, default=0.4, help="max time-crop fraction")
            sub.add_argument("--hand-dropout", type=float, default=0.35, help="per-frame hand drop")
            sub.add_argument("--noise", type=float, default=0.08, help="landmark jitter std")
        else:
            sub.add_argument("--reports-dir", type=Path, default=Path("reports"))

    demo = add("demo", _cmd_demo, "live recognition from a webcam index (0) or a video file")
    demo.add_argument("--source", default="0", help="camera index or path to a video file")
    demo.add_argument("--models-dir", type=Path, default=Path("models"))
    demo.add_argument("--window", type=int, default=64, help="frames of history per prediction")
    demo.add_argument("--stride", type=int, default=4, help="predict every N frames")
    demo.add_argument("--threshold", type=float, default=0.6, help="minimum top-1 probability")
    demo.add_argument(
        "--stable", type=int, default=3, help="agreeing predictions needed to show a word"
    )
    demo.add_argument("--hold", type=int, default=6, help="predictions a shown word stays up")
    demo.add_argument(
        "--min-activity", type=float, default=0.25, help="min share of window frames with a hand"
    )
    demo.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full")
    demo.add_argument("--headless", action="store_true", help="no window; print shown words")
    demo.add_argument(
        "--no-mirror-view", action="store_true", help="do not flip the displayed image"
    )
    demo.add_argument("--no-landmarks", action="store_true", help="do not draw landmark dots")
    demo.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)
