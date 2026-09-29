"""既存PLV CSVから発表向けnetwork reportを一括生成する。

引数を省略すると、設定済みの全被験者group・全条件・結果として存在する
全セグメント・全帯域を巡回する。

例:
    python scripts/run_network_report.py
    python scripts/run_network_report.py --condition speaking \
        --segments gattai_mae gattai_ato
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from configs.gattai_hyperscan_study import DESIGN, FREQ_BANDS, PATHS  # noqa: E402
from hyperscan_eeg.network_reporting import (  # noqa: E402
    analyze_network_for_report,
    plot_network_metrics,
    plot_participant_network_comparison,
    segment_sort_key,
    subject_groups_from_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="保存済みPLV CSVから状態・帯域別network figureを作成します。"
    )
    parser.add_argument(
        "--condition",
        nargs="+",
        default=list(DESIGN.conditions),
        choices=DESIGN.conditions,
        help="対象条件。省略時は全条件",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=None,
        help="比較する区間名。省略時はresults内の全セグメント",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=[band.name for band in FREQ_BANDS],
        help="対象帯域（既定: 設定ファイルの全帯域）",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="channel-level図で保持するedgeのpercentile（既定: 98）",
    )
    parser.add_argument(
        "--metrics-percentile",
        type=float,
        default=90.0,
        help="network metrics算出用のpercentile（既定: 90）",
    )
    parser.add_argument("--method", default="plv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/network_reports"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="入力CSVがない組合せをスキップせずエラーにする",
    )
    return parser.parse_args()


def _pair_files(
    pairs: tuple[tuple[int, int], ...],
    condition: str,
    segment: str,
    band: str,
    method: str,
) -> dict[tuple[int, int], Path]:
    return {
        pair: PATHS.results_dir
        / condition
        / segment
        / f"pair_{pair[0]:02d}_{pair[1]:02d}_{method}_{band}.csv"
        for pair in pairs
    }


def main() -> None:
    args = parse_args()
    subject_groups = subject_groups_from_pairs(
        tuple(DESIGN.subjects), tuple(DESIGN.pairs)
    )
    completed = 0
    skipped = 0

    for subjects in subject_groups:
        subject_label = "_".join(f"sub{subject:02d}" for subject in subjects)
        group_pairs = tuple(
            (subject_a, subject_b)
            for index, subject_a in enumerate(subjects)
            for subject_b in subjects[index + 1 :]
        )
        for condition in args.condition:
            if args.segments is None:
                condition_dir = PATHS.results_dir / condition
                segments = (
                    sorted(
                        (path.name for path in condition_dir.iterdir() if path.is_dir()),
                        key=segment_sort_key,
                    )
                    if condition_dir.exists()
                    else []
                )
            else:
                segments = args.segments

            output_dir = args.output_dir / subject_label / condition
            metrics_rows: list[dict[str, int | float | bool | str]] = []
            participant_graphs_by_band: dict[str, dict] = {
                band: {} for band in args.bands
            }

            for band in args.bands:
                for segment in segments:
                    pair_files = _pair_files(
                        group_pairs, condition, segment, band, args.method
                    )
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

                    stem = (
                        f"{subject_label}_{segment}_{args.method}_{band}_p{args.percentile:g}"
                    )
                    metrics, participant_graph, _ = analyze_network_for_report(
                        subjects=subjects,
                        pair_files=pair_files,
                        output_dir=output_dir / segment,
                        output_stem=stem,
                        title=(
                            f"Channel-level network: {subject_label} / "
                            f"{args.method.upper()} / {band} / {condition} / {segment} "
                            f"(p{args.percentile:g})"
                        ),
                        percentile=args.percentile,
                        metrics_percentile=args.metrics_percentile,
                    )
                    metrics_rows.append(
                        {
                            "subjects": subject_label,
                            "condition": condition,
                            "segment": segment,
                            "band": band,
                            **metrics.as_dict(),
                        }
                    )
                    participant_graphs_by_band[band][segment] = participant_graph
                    completed += 1

                if participant_graphs_by_band[band]:
                    figure = plot_participant_network_comparison(
                        participant_graphs_by_band[band],
                        subjects,
                        title=(
                            f"Participant-level mean {args.method.upper()} / {band} / "
                            f"{condition} / {subject_label}"
                        ),
                        save_path=(
                            output_dir
                            / f"participant_network_{subject_label}_{args.method}_{band}.png"
                        ),
                    )
                    plt.close(figure)

            if metrics_rows:
                metrics_frame = pd.DataFrame(metrics_rows)
                metrics_path = output_dir / "network_metrics_by_segment_band.csv"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_frame.to_csv(metrics_path, index=False)
                figure = plot_network_metrics(
                    metrics_frame,
                    output_dir / "network_metrics_by_segment_band.png",
                )
                plt.close(figure)
                print(f"[DONE] network report: {output_dir}")
                print(f"[DONE] metrics: {metrics_path}")

    print(f"[SUMMARY] completed={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
