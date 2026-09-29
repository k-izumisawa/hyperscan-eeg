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
    MarkerConfig,
    MontageConfig,
    PathConfig,
    SegmentPlan,
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

# InletPortマーカー: このタスク制御ソフト固有の割り当て（1=タスク開始, 3=タスク終了）。
MARKER_CFG = MarkerConfig(start_marker=1, end_marker=3)

MONTAGE_CFG = MontageConfig(channel_labels=EMOTIV_FLEX_SALINE_32CH_LABELS)

FREQ_BANDS = STANDARD_EEG_FREQUENCY_BANDS

# Pilot 1 (sub01~03) と Pilot 2 (sub04~06) の各triad内で実施したpairを列挙する。
DESIGN = ExperimentDesign(
    subjects=(1, 2, 3, 4, 5, 6),
    conditions=("silent", "speaking"),
    pairs=((1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)),
    condition_segments={
        # 「合体」マーカー(3番)を境界として、silent条件は1回(2区間)、
        # speaking条件は2回(3区間)出現する、という実験固有の例外仕様。
        "silent": SegmentPlan(labels=("gattai_mae", "gattai_ato"
        ""), boundary_markers=(2,)),
        "speaking": SegmentPlan(
            labels=("gattai_mae", "gattai_ato"), boundary_markers=(2,)
        ),
    },
)
