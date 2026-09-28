"""前処理の実行スクリプト。

実験固有パラメータは `configs/gattai_hyperscan_study.py` を編集する。
被験者・条件・ICAの対話確認有無など、実行そのものに関わる制御のみ
このファイル内で調整する。
    python scripts/run_preprocessing.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from configs.gattai_hyperscan_study import DESIGN, MARKER_CFG, MONTAGE_CFG, PATHS  # noqa: E402
from hyperscan_eeg.config import FilterConfig, ICAConfig  # noqa: E402
from hyperscan_eeg.logging_utils import setup_logging  # noqa: E402
from hyperscan_eeg.pipeline import run_subject_preprocessing  # noqa: E402

logger = logging.getLogger(__name__)

# ============================== 実行制御パラメータ ==============================
FILTER_CFG = FilterConfig(l_freq=1.0, h_freq=60.0, notch_freqs=(50.0, 100.0))
ICA_CFG = ICAConfig(interactive=True)  # バッチ実行時は False にして目視確認を省略
SAVE_MONTAGE_FIGURES = False  # Trueにした場合のみ3D・topomapの確認図を保存
# ICLabelでノイズ成分を自動判定したい場合は use_iclabel=True を追加する
# （interactive=True と併用すると、自動判定後にGUIで目視確認・修正できる）
# 例: ICAConfig(interactive=True, use_iclabel=True)
# ==================================================================================


def main() -> None:
    setup_logging(log_file=Path("logs/preprocessing.log"))

    for condition in DESIGN.conditions:
        for subject in DESIGN.subjects:
            try:
                run_subject_preprocessing(
                    subject=subject,
                    condition=condition,
                    paths=PATHS,
                    marker_cfg=MARKER_CFG,
                    montage_cfg=MONTAGE_CFG,
                    segment_plan=DESIGN.condition_segments.get(condition),
                    filter_cfg=FILTER_CFG,
                    ica_cfg=ICA_CFG,
                    save_montage_figures=SAVE_MONTAGE_FIGURES,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.error("subject%d / %s の前処理に失敗しました: %s", subject, condition, exc)
                continue

    logger.info("すべての前処理が完了しました。")


if __name__ == "__main__":
    main()
