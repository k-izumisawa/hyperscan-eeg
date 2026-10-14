"""脳間同期指標（PLV / PSI 等）算出の実行スクリプト。

実験固有パラメータ（被験者・ペア・条件・ゲーム状態の区間分割）は
`configs/gattai_hyperscan_study.py` を編集する。同期指標の種類や
区間分割の有無など、実行そのものに関わる制御のみこのファイル内で調整する。
    python scripts/run_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from configs.gattai_hyperscan_study import DESIGN, FREQ_BANDS, PATHS  # noqa: E402
from hyperscan_eeg.logging_utils import setup_logging  # noqa: E402
from hyperscan_eeg.pipeline import run_pair_connectivity  # noqa: E402

logger = logging.getLogger(__name__)

# ============================== 実行制御パラメータ ==============================
METHOD = "plv"  # mne_connectivity.spectral_connectivity_epochs の method ('plv','psi','coh','wpli' 等)
SEGMENT_BY_GAME_STATE = True  # True: gattai_mae/ato1/(ato2) ごとに分割, False: タスク全体を一括算出
SAVE_FIGURES = True
# ==================================================================================


def main() -> None:
    setup_logging(log_file=Path("logs/analysis.log"))

    for condition in DESIGN.conditions:
        segment_plan = (
            DESIGN.condition_segments.get(condition) if SEGMENT_BY_GAME_STATE else None
        )
        state_event_plan = (
            DESIGN.state_event_plans.get(condition) if SEGMENT_BY_GAME_STATE else None
        )

        for pair in DESIGN.pairs:
            if state_event_plan is not None:
                sessions_a = set(PATHS.raw_sessions(pair[0], condition))
                sessions_b = set(PATHS.raw_sessions(pair[1], condition))
                sessions: tuple[int | None, ...] = tuple(sorted(sessions_a & sessions_b))
                if not sessions:
                    logger.warning(
                        "pair%s / %s: 共通sessionがないためスキップします", pair, condition
                    )
                    continue
            else:
                sessions = (None,)

            for session in sessions:
                try:
                    run_pair_connectivity(
                        pair=pair,
                        condition=condition,
                        paths=PATHS,
                        bands=FREQ_BANDS,
                        segment_plan=segment_plan,
                        method=METHOD,
                        save_figures=SAVE_FIGURES,
                        session=session,
                        state_event_plan=state_event_plan,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    logger.error(
                        "pair%s / %s / session%s の同期指標算出に失敗しました: %s",
                        pair,
                        condition,
                        session,
                        exc,
                    )
                    continue

    logger.info("すべてのペア・条件の同期指標算出が完了しました。")


if __name__ == "__main__":
    main()
