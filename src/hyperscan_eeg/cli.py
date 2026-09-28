"""コマンドライン実行用エントリポイント。

対話的にパラメータを調整したい場合は `scripts/run_preprocessing.py` /
`scripts/run_analysis.py` を直接編集して実行する方が容易。
本モジュールは自動化・CI実行向けの薄いCLIラッパーを提供する。

`hyperscan_eeg` 自体は実験固有の値を持たないため、CLIは実行対象の実験
設定モジュール（`configs/` 配下、既定は `configs.gattai_hyperscan_study`）を
動的にインポートして使う。別の実験で使う場合は `--config-module` で
自分の設定モジュールを指定する。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType

from .logging_utils import setup_logging
from .pipeline import run_pair_connectivity, run_subject_preprocessing

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config_module(dotted_path: str) -> ModuleType:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    return importlib.import_module(dotted_path)


def preprocess_main() -> None:
    parser = argparse.ArgumentParser(description="EEGハイパースキャニングデータの前処理")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--non-interactive", action="store_true", help="ICA成分の目視確認を省略する")
    parser.add_argument(
        "--use-iclabel",
        action="store_true",
        help="MNE-ICALabelでICA成分を自動分類し、ノイズ成分を自動で除外対象にする",
    )
    parser.add_argument(
        "--save-montage-figures",
        action="store_true",
        help="個人別montageの3D・topomap確認図を保存する",
    )
    parser.add_argument("--config-module", type=str, default="configs.gattai_hyperscan_study")
    args = parser.parse_args()

    setup_logging()
    cfg = _load_config_module(args.config_module)
    from .config import ICAConfig

    ica_cfg = ICAConfig(interactive=not args.non_interactive, use_iclabel=args.use_iclabel)
    run_subject_preprocessing(
        subject=args.subject,
        condition=args.condition,
        paths=cfg.PATHS,
        marker_cfg=cfg.MARKER_CFG,
        montage_cfg=cfg.MONTAGE_CFG,
        segment_plan=cfg.DESIGN.condition_segments.get(args.condition),
        ica_cfg=ica_cfg,
        save_montage_figures=args.save_montage_figures,
    )


def analyze_main() -> None:
    parser = argparse.ArgumentParser(description="脳間同期指標（PLV/PSI等）の算出")
    parser.add_argument("--sub-a", type=int, required=True)
    parser.add_argument("--sub-b", type=int, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--method", type=str, default="plv")
    parser.add_argument("--segment", action="store_true", help="ゲーム状態ごとに区間分割して算出する")
    parser.add_argument("--config-module", type=str, default="configs.gattai_hyperscan_study")
    args = parser.parse_args()

    setup_logging()
    cfg = _load_config_module(args.config_module)
    segment_plan = cfg.DESIGN.condition_segments[args.condition] if args.segment else None

    run_pair_connectivity(
        pair=(args.sub_a, args.sub_b),
        condition=args.condition,
        paths=cfg.PATHS,
        bands=cfg.FREQ_BANDS,
        segment_plan=segment_plan,
        method=args.method,
    )
