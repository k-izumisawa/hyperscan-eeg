"""前処理済みFIFの波形を被験者・条件ごとに対話表示する。

ウィンドウを閉じると次のファイルへ進む。既定では実験設定に含まれる
全被験者・全条件を表示する。表示のみを行い、データは変更しない。

使用例:
    python scripts/review_preprocessed.py
    python scripts/review_preprocessed.py --subjects 4 5 6 --conditions silent
    python scripts/review_preprocessed.py --duration 30 --start 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from configs.gattai_hyperscan_study import DESIGN, PATHS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="前処理済みEEG波形を被験者・条件ごとに順番に確認します。"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(DESIGN.subjects),
        help="表示する被験者番号（既定: 設定ファイルの全被験者）",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DESIGN.conditions),
        help="表示する条件（既定: 設定ファイルの全条件）",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=None,
        help="表示するsession番号。省略時はrawデータから自動検出",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="最初に表示する開始時刻（秒、既定: 0）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="画面に表示する時間幅（秒、既定: 20）",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=32,
        help="同時に表示するチャンネル数（既定: 32）",
    )
    parser.add_argument(
        "--scalings",
        default="auto",
        help="波形表示スケール（既定: auto、例: 100e-6）",
    )
    return parser.parse_args()


def _parse_scaling(value: str) -> str | dict[str, float]:
    if value == "auto":
        return "auto"
    try:
        return {"eeg": float(value)}
    except ValueError as exc:
        raise ValueError(
            "--scalingsには 'auto' またはボルト単位の数値を指定してください。"
        ) from exc


def main() -> None:
    args = parse_args()
    scalings = _parse_scaling(args.scalings)
    missing: list[Path] = []

    for condition in args.conditions:
        for subject in args.subjects:
            if condition in DESIGN.state_event_plans:
                sessions: tuple[int | None, ...] = tuple(args.sessions or PATHS.raw_sessions(subject, condition))
            else:
                sessions = (None,)
            for session in sessions:
                path = PATHS.preprocessed_file(subject, condition, session)
                if not path.exists():
                    print(f"[SKIP] 前処理済みファイルがありません: {path}")
                    missing.append(path)
                    continue

                session_label = f"session{session:02d}" if session is not None else "legacy"
                print(f"[OPEN] subject{subject} / {condition} / {session_label}: {path}")
                raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
                max_start = max(0.0, raw.times[-1] - min(args.duration, raw.times[-1]))
                start = min(max(args.start, 0.0), max_start)
                raw.plot(
                    start=start,
                    duration=min(args.duration, raw.times[-1]),
                    n_channels=min(args.n_channels, len(raw.ch_names)),
                    scalings=scalings,
                    title=f"subject{subject} / {condition} / {session_label} / preprocessed",
                    block=True,
                )

    if missing:
        print(f"[DONE] 表示完了（未作成のためスキップ: {len(missing)}件）")
    else:
        print("[DONE] すべての前処理済み波形を確認しました。")


if __name__ == "__main__":
    main()
