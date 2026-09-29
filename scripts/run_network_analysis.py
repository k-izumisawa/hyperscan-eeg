"""保存済みconnectivity CSVからhyperbrainネットワーク解析を実行する。

引数を省略すると、設定ファイルにある全条件・全セグメント・全帯域を巡回する。
入力connectivity CSVがない組合せはスキップする。

例:
    python scripts/run_network_analysis.py
    python scripts/run_network_analysis.py --condition speaking --segment gattai_ato --band alpha
    python scripts/run_network_analysis.py --condition speaking --segment gattai_mae --band theta --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from configs.gattai_hyperscan_study import DESIGN, FREQ_BANDS, PATHS  # noqa: E402
from hyperscan_eeg.network import analyze_hyperbrain_network  # noqa: E402
from hyperscan_eeg.network_reporting import segment_sort_key, subject_groups_from_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ペア別connectivity CSVからhyperbrainネットワークを構築・評価します。"
    )
    parser.add_argument(
        "--condition",
        nargs="+",
        default=list(DESIGN.conditions),
        choices=DESIGN.conditions,
        help="対象条件。省略時は設定ファイルの全条件",
    )
    parser.add_argument(
        "--segment",
        nargs="+",
        default=None,
        help="結果ディレクトリ内の区間名。省略時は各条件の設定にある全区間",
    )
    parser.add_argument(
        "--band",
        nargs="+",
        default=[band.name for band in FREQ_BANDS],
        help="周波数帯域名。省略時は設定ファイルの全帯域",
    )
    parser.add_argument("--method", default="plv", help="connectivity指標名（既定: plv）")
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="保持する強度のパーセンタイル閾値（既定: 90＝上位10%%）",
    )
    parser.add_argument(
        "--signed-threshold",
        action="store_true",
        help="絶対値ではなく符号付き値そのものから閾値を算出する",
    )
    parser.add_argument("--show", action="store_true", help="保存に加えて図を画面表示する")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="入力CSVがない組合せをスキップせずエラーにする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_groups = subject_groups_from_pairs(
        tuple(DESIGN.subjects), tuple(DESIGN.pairs)
    )

    completed = 0
    skipped = 0
    for condition in args.condition:
        if args.segment is None:
            condition_dir = PATHS.results_dir / condition
            segments = (
                sorted(
                    (path.name for path in condition_dir.iterdir() if path.is_dir()),
                    key=segment_sort_key,
                )
                if condition_dir.exists()
                else []
            )
            if not segments:
                plan = DESIGN.condition_segments.get(condition)
                segments = list(plan.labels) if plan is not None else ["whole_task"]
        else:
            segments = args.segment

        for subjects in subject_groups:
            subject_label = "_".join(f"sub{subject:02d}" for subject in subjects)
            group_pairs = tuple(
                (subject_a, subject_b)
                for index, subject_a in enumerate(subjects)
                for subject_b in subjects[index + 1 :]
            )
            for segment in segments:
                for band in args.band:
                    pair_files = {
                        pair: PATHS.results_dir
                        / condition
                        / segment
                        / f"pair_{pair[0]:02d}_{pair[1]:02d}_{args.method}_{band}.csv"
                        for pair in group_pairs
                    }
                    missing = [path for path in pair_files.values() if not path.exists()]
                    if missing:
                        message = (
                            f"入力CSV不足: {subject_label} / {condition} / {segment} / "
                            f"{band} ({len(missing)}件)"
                        )
                        if args.strict:
                            raise FileNotFoundError(f"{message}: {missing}")
                        print(f"[SKIP] {message}")
                        skipped += 1
                        continue

                    output_stem = (
                        f"hyperbrain_{subject_label}_{args.method}_{band}_p{args.percentile:g}"
                    )
                    output_dir = Path("data/network_results") / condition / segment
                    title = (
                        f"Hyperbrain Network: {subject_label} / {args.method.upper()} / "
                        f"{band} / {condition} / {segment} (p{args.percentile:g})"
                    )

                    metrics = analyze_hyperbrain_network(
                        subjects=subjects,
                        pair_files=pair_files,
                        output_dir=output_dir,
                        output_stem=output_stem,
                        title=title,
                        percentile=args.percentile,
                        by_absolute_value=not args.signed_threshold,
                        show=args.show,
                    )
                    completed += 1
                    print(
                        f"[DONE] {subject_label} / {condition} / {segment} / {band}: "
                        f"nodes={metrics.n_nodes}, edges={metrics.n_edges}, "
                        f"threshold={metrics.threshold:.6g}, outputs={output_dir}"
                    )

    print(f"[SUMMARY] completed={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
