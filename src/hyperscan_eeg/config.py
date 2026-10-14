"""実験パラメータのスキーマ定義（データクラス）。

このモジュールは「型」だけを定義し、特定の実験（被験者数、条件名、
マーカーコード、ペア対応、ゲーム状態等）に紐づく具体的な値は一切持たない。
具体的な値は各実験ごとにリポジトリ直下の `configs/` に用意し、
`scripts/` からそれを読み込んで注入する（詳細はREADME参照）。

EMOTIV Flex Saline 32ch 等のハードウェアに紐づく再利用可能なプリセット値は
`presets.py` を参照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Sequence

import numpy as np


def all_subject_pairs(subjects: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """被験者リストから総当たり組み合わせを生成するヘルパー。

    通常の実験デザインでは `ExperimentDesign.pairs` に実施したペアを
    直接列挙するため、本関数は使わない（総当たりを仮定しないための
    設計上、これがデフォルト値になることはない）。

    実データでは組んでいない被験者どうしの組み合わせ（サロゲート/偽ペア）
    を作りたい、permutationテストのような帰無分布構築の場面で、候補となる
    全組み合わせの列挙に転用できる可能性があるため残してある。実際に
    permutationテストを実装する際は、ペア内の左右対応をシャッフルする等
    本関数だけでは要件を満たさない可能性が高く、別途改修が必要。
    """
    return tuple(
        (a, b) for i, a in enumerate(subjects) for b in subjects[i + 1 :]
    )


@dataclass(frozen=True)
class PathConfig:
    """入出力ディレクトリ構成。ディレクトリ名の慣習のみを定義する。
    ディレクトリ構造は raw/condition/subxx.bdf, digitizer/subxx.txt, preprocessed/subxx_yy_preprocessed_raw.fif となる。
    """

    raw_dir: Path = Path("data/raw")
    marker_dir: Path = Path("markerdata")
    digitizer_dir: Path = Path("data/digitizer")
    preprocessed_dir: Path = Path("data/preprocessed")
    results_dir: Path = Path("data/results")
    figures_dir: Path = Path("data/figures")

    def raw_file(
        self, subject: int, condition: str, session: int | None = None
    ) -> Path:
        if session is None:
            return self.raw_dir / condition / f"sub{subject}.bdf"
        directory = self.raw_dir / condition
        candidates = (
            directory / f"sub{subject:02d}_session{session:02d}.bdf",
            directory / f"sub{subject}_session{session}.bdf",
            directory / f"sub{subject:02d}_session{session}.bdf",
            directory / f"sub{subject}_session{session:02d}.bdf",
        )
        return next((path for path in candidates if path.exists()), candidates[0])

    def marker_file(
        self, subject: int, condition: str, session: int | None = None
    ) -> Path:
        if session is None:
            return self.marker_dir / f"sub{subject:02d}_{condition}.csv"
        candidates = (
            self.marker_dir
            / f"sub{subject:02d}_session{session:02d}_{condition}.csv",
            self.marker_dir / f"sub{subject}_session{session}_{condition}.csv",
            self.marker_dir / f"sub{subject:02d}_session{session}_{condition}.csv",
            self.marker_dir / f"sub{subject}_session{session:02d}_{condition}.csv",
        )
        return next((path for path in candidates if path.exists()), candidates[0])

    def digitizer_file(self, subject: int) -> Path:
        return self.digitizer_dir / f"sub{subject}.TXT"

    def preprocessed_file(
        self, subject: int, condition: str, session: int | None = None
    ) -> Path:
        if session is None:
            return self.preprocessed_dir / f"sub{subject}_{condition}_preprocessed_raw.fif"
        return (
            self.preprocessed_dir
            / f"sub{subject:02d}_session{session:02d}_{condition}_preprocessed_raw.fif"
        )

    def raw_sessions(self, subject: int, condition: str) -> tuple[int, ...]:
        """Return available ``subXX_sessionXX.bdf`` session numbers."""
        directory = self.raw_dir / condition
        pattern = re.compile(
            rf"^sub0*{subject}_session0*(\d+)\.bdf$", re.IGNORECASE
        )
        sessions = {
            int(match.group(1))
            for path in directory.glob("*.bdf")
            if (match := pattern.match(path.name)) is not None
        }
        return tuple(sorted(sessions))


@dataclass(frozen=True)
class MarkerConfig:
    """タスク区間の開始・終了を表すイベントマーカーコード。

    計測開始・終了時に対応するInletPortマーカー(LSL想定)を各実験の `configs/` で明示的に指定する。
    """

    start_marker: int
    end_marker: int


@dataclass(frozen=True)
class SegmentPlan:
    """1条件分の区間分割定義（境界マーカーとラベルの対応）。

    `boundary_markers` は時系列順に検出する境界マーカーコードの並びで、
    要素数は `labels` の要素数 - 1 でなければならない。
    例：labels: [task, rest]
        boundary_markers: [2,3]
    """

    labels: tuple[str, ...]
    boundary_markers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.boundary_markers) != len(self.labels) - 1:
            raise ValueError(
                f"boundary_markers の数({len(self.boundary_markers)})は "
                f"labels の数-1({len(self.labels) - 1})と一致する必要があります: "
                f"labels={self.labels}, boundary_markers={self.boundary_markers}"
            )
        if len(set(self.labels)) != len(self.labels):
            # segment_by_markers は labels をそのまま辞書キー・出力ファイル名に使うため、
            # 重複を許すと後勝ちで区間が上書きされ、片方の区間が黙って失われる。
            raise ValueError(f"labels に重複があります: {self.labels}")


@dataclass(frozen=True)
class StateEventPlan:
    """Marker-driven game states and overlapping event-analysis windows."""

    state_labels: dict[int, str]
    event_labels: dict[int, str]
    game_start_marker: int = 1
    event_window_seconds: float = 8.0
    end_at_first_event_window: bool = False

    def __post_init__(self) -> None:
        if self.game_start_marker not in self.state_labels:
            raise ValueError("game_start_marker must be included in state_labels")
        if set(self.state_labels).intersection(self.event_labels):
            raise ValueError("state marker IDs and event marker IDs must not overlap")
        labels = [*self.state_labels.values(), *self.event_labels.values()]
        if len(labels) != len(set(labels)):
            raise ValueError("state/event labels must be unique")
        if self.event_window_seconds <= 0:
            raise ValueError("event_window_seconds must be positive")

    @property
    def marker_ids(self) -> set[int]:
        return set(self.state_labels) | set(self.event_labels)


@dataclass(frozen=True)
class ExperimentDesign:
    """被験者・ペア対応・条件（yy）・ゲーム状態（zz）の対応関係。

    総当たりを仮定せず、実施したペアを `pairs` に明示的に列挙する
    （総当たりの場合は `all_subject_pairs()` を利用して生成してよい）。
    フィールドにデフォルト値を持たせず、実験ごとに `configs/` で
    明示的に構築することを前提とする。
    """

    subjects: tuple[int, ...]
    conditions: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    condition_segments: dict[str, SegmentPlan]
    state_event_plans: dict[str, StateEventPlan] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterConfig:
    l_freq: float = 1.0
    h_freq: float = 60.0
    notch_freqs: tuple[float, ...] = (50.0, 100.0)
    fir_design: str = "firwin"


@dataclass(frozen=True)
class ICAConfig:
    method: str = "infomax"
    random_state: int = 97
    fit_params: dict = field(default_factory=lambda: {"extended": True})
    ica_highpass: float = 1.0
    interactive: bool = True
    # True の場合、ICAフィット前に平均参照を適用したうえで（ICLabelの学習前提に
    # 合わせるため）MNE-ICALabelで各成分を brain/eye blink/muscle artifact 等に
    # 自動分類する。ただし ica.exclude は自動では設定しない（確信度が低くても
    # 該当ラベルなら除外、という挙動を避けるため）。分類ラベルと確信度は
    # ログに出力されるほか、review_ica_interactively のGUI上で各成分名に
    # 付記され、目視判断の補助材料として使う。実際の除外はGUI上の手動選択
    # （interactive=True）で行うこと。
    use_iclabel: bool = True


@dataclass(frozen=True)
class MontageConfig:
    channel_labels: tuple[str, ...]
    n_landmarks: int = 5
    scale_to_meters: float = 100.0  # デジタイザ生データの単位（cm想定）に応じて調整
    fallback_standard_montage: str = "standard_1020"


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    fmin: float
    fmax: float

    def freqs(self, step: float = 1.0) -> np.ndarray:
        """帯域内の解析周波数ビン配列を生成する。"""
        return np.arange(self.fmin, self.fmax, step)
