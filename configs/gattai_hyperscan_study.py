"""この実験（2者間ハイパースキャニング・「合体」ゲーム課題）固有の設定。

`hyperscan_eeg` パッケージ本体（`src/hyperscan_eeg/`）は特定の実験デザインに
依存しない汎用ライブラリであり、被験者数・条件名・マーカーコード・
ペア対応・ゲーム状態などの具体的な値は一切持たない。本ファイルが、
それら実験固有の値を一箇所にまとめて注入する役割を持つ。

別の実験に転用する場合は、このファイルを複製・編集するだけでよく
（例: `configs/my_other_study.py`）、`src/hyperscan_eeg/` 側の変更は不要。
"""

from __future__ import annotations

from pathlib import Path

from hyperscan_eeg.config import (
    ExperimentDesign,
    MontageConfig,
    PathConfig,
    StateEventPlan,
)
from hyperscan_eeg.presets import EMOTIV_FLEX_SALINE_32CH_LABELS, STANDARD_EEG_FREQUENCY_BANDS

PATHS = PathConfig(
    raw_dir=Path("data/raw"),
    marker_dir=Path("markerdata"),
    digitizer_dir=Path("data/digitizer"),
    preprocessed_dir=Path("data/preprocessed"),
    results_dir=Path("data/results"),
    figures_dir=Path("data/figures"),
)

# 現行仕様には終了マーカーがないため、状態マーカー1から解析を開始する。
MARKER_CFG = None

MONTAGE_CFG = MontageConfig(channel_labels=EMOTIV_FLEX_SALINE_32CH_LABELS)

FREQ_BANDS = STANDARD_EEG_FREQUENCY_BANDS

# 現行実験の2チーム（sub07~09、sub10~12）の各triad内pairを列挙する。
DESIGN = ExperimentDesign(
    subjects=(7, 8, 9, 10, 11, 12),
    conditions=("speaking",),
    pairs=((7, 8), (7, 9), (8, 9), (10, 11), (10, 12), (11, 12)),
    condition_segments={},
    state_event_plans={
        "speaking": StateEventPlan(
            state_labels={
                1: "both_unfused",
                2: "own_team_fused",
                3: "both_teams_fused",
                4: "opponent_team_fused",
            },
            event_labels={
                5: "own_special",
                6: "opponent_special",
            },
            game_start_marker=1,
            # 0～3秒はカウントダウン、3～8秒は成否決定後の5秒間。
            event_window_seconds=8.0,
            # 最初の必殺技マーカー（5または6）の8秒後で状態解析を終了する。
            end_at_first_event_window=True,
        )
    },
)
