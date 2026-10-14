"""前処理・同期指標算出の一連の流れをまとめた高水準オーケストレーション関数。

各関数は `config.py` のデータクラスをそのまま引数として受け取り、
被験者・ペア・条件・区間ラベルの組み合わせだけを外側のスクリプトが
制御する構成とする。これにより実行スクリプト側でのパラメータ変更・
実行制御が容易になる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd

from . import connectivity, io, phase, preprocessing, visualization
from .config import (
    FilterConfig,
    FrequencyBand,
    ICAConfig,
    MarkerConfig,
    MontageConfig,
    PathConfig,
    SegmentPlan,
    StateEventPlan,
)
from .presets import EMOTIV_AUX_CHANNEL_PREFIXES

logger = logging.getLogger(__name__)


def _register_digitizer_montage(
    raw: mne.io.BaseRaw,
    subject: int,
    condition: str,
    paths: PathConfig,
    montage_cfg: MontageConfig,
    save_montage_figures: bool,
    session: int | None = None,
) -> None:
    """Register an individual montage or fall back to a standard montage."""
    digitizer_file = paths.digitizer_file(subject)
    if not digitizer_file.exists():
        logger.warning(
            "Digitizer file not found; using MNE standard montage '%s': %s",
            montage_cfg.fallback_standard_montage,
            digitizer_file,
        )
        montage = mne.channels.make_standard_montage(
            montage_cfg.fallback_standard_montage
        )
    else:
        montage = io.build_dig_montage(subject, paths.digitizer_dir, montage_cfg)
    raw.set_montage(montage, match_case=False, on_missing="warn")
    if save_montage_figures:
        session_suffix = f"_session{session:02d}" if session is not None else ""
        visualization.plot_montage(
            montage,
            title=f"sub{subject}{session_suffix}_{condition}",
            save_path=(
                paths.figures_dir
                / condition
                / f"sub{subject}{session_suffix}_montage.png"
            ),
            show=False,
        )


def _apply_analysis_reference(
    raw: mne.io.BaseRaw,
    apply_average_reference: bool,
    ica_cfg: ICAConfig,
) -> None:
    """Apply the analysis reference independently of ICA classification.

    ICLabel assumes average-referenced data. Because average referencing is
    not reversible after ICA has been applied, that combination is rejected
    explicitly instead of silently changing the requested reference.
    """
    if ica_cfg.use_iclabel and not apply_average_reference:
        raise ValueError(
            "use_iclabel=True requires apply_average_reference=True"
        )
    if apply_average_reference:
        raw.set_eeg_reference("average", projection=False, verbose=False)


def run_subject_preprocessing(
    subject: int,
    condition: str,
    paths: PathConfig,
    marker_cfg: MarkerConfig | None,
    montage_cfg: MontageConfig,
    segment_plan: SegmentPlan | None = None,
    filter_cfg: FilterConfig = FilterConfig(),
    ica_cfg: ICAConfig = ICAConfig(),
    aux_prefixes: tuple[str, ...] = EMOTIV_AUX_CHANNEL_PREFIXES,
    save_montage_figures: bool = False,
    apply_average_reference: bool = True,
    session: int | None = None,
    state_event_plan: StateEventPlan | None = None,
) -> Path:
    """1被験者・1条件分の生EEGデータを前処理し、fifとして保存する。

    切り出し -> モンタージュ適用 -> EEGチャンネル抽出 -> フィルタ -> ICA除去、
    の順で処理する。ゲーム状態（zz）区間へのさらなる分割は解析時
    （`run_pair_connectivity`）に行うため、ここではタスク全体を対象に
    ICAを1回だけフィットする（短い区間ごとに分割してからICAを行うと
    独立成分推定が不安定になりやすいため）。

    Returns:
        保存された前処理済みfifファイルのパス。
    """
    session_label = f"session{session:02d}" if session is not None else "legacy"
    logger.info(
        "=== 前処理開始: subject%d / %s / %s ===",
        subject,
        condition,
        session_label,
    )

    if segment_plan is not None and state_event_plan is not None:
        raise ValueError("segment_plan and state_event_plan cannot be used together")
    if marker_cfg is None and state_event_plan is None:
        raise ValueError("marker_cfg or state_event_plan is required")

    raw = io.load_raw_eeg(paths.raw_file(subject, condition, session))
    if state_event_plan is not None:
        allowed_markers = state_event_plan.marker_ids
    else:
        assert marker_cfg is not None
        allowed_markers = {marker_cfg.start_marker, marker_cfg.end_marker}
        if segment_plan is not None:
            allowed_markers.update(segment_plan.boundary_markers)
    events = io.get_marker_events(
        raw,
        csv_file=paths.marker_file(subject, condition, session),
        allowed_markers=allowed_markers,
    )
    if state_event_plan is not None:
        raw_task = preprocessing.crop_from_game_start(
            raw, events, state_event_plan.game_start_marker
        )
    else:
        assert marker_cfg is not None
        raw_task = preprocessing.crop_to_task_window(raw, events, marker_cfg)

    raw_eeg = preprocessing.select_eeg_channels(raw_task, aux_prefixes)

    _register_digitizer_montage(
        raw_eeg,
        subject,
        condition,
        paths,
        montage_cfg,
        save_montage_figures,
        session,
    )
    _apply_analysis_reference(raw_eeg, apply_average_reference, ica_cfg)

    # 解析データへの平均参照は apply_average_reference で独立に制御する。
    # ICLabelは1-100Hzバンドパス・平均参照済みデータでのICA fitを前提とする
    # （詳細は `preprocessing._ica_fit_raw` を参照）。use_iclabel時は、fit・
    # 自動判定・（対話確認時の）GUI確認・適用までを一貫してプロジェクトの
    # 解析用バンドパス適用前の raw_eeg（ブロードバンド）に対して行い、
    # ICA適用後のクリーンなデータに解析用バンドパス（既定60Hzローパス+
    # ノッチ）をかける。use_iclabel=False（手動レビューのみ）の場合は従来
    # 通り、先にバンドパスをかけてからICAをfit・適用する。
    bandpass_before_ica = not ica_cfg.use_iclabel
    ica_input = (
        preprocessing.apply_bandpass_notch(raw_eeg, filter_cfg) if bandpass_before_ica else raw_eeg
    )

    ica = preprocessing.fit_ica(ica_input, ica_cfg)
    if ica_cfg.use_iclabel:
        preprocessing.label_ica_iclabel(ica, ica_input, ica_cfg)
    if ica_cfg.interactive:
        preprocessing.review_ica_interactively(ica, ica_input)
    ica_cleaned = preprocessing.apply_ica(ica_input, ica)
    raw_clean = (
        ica_cleaned if bandpass_before_ica else preprocessing.apply_bandpass_notch(ica_cleaned, filter_cfg)
    )
    session_suffix = f"_session{session:02d}" if session is not None else ""
    overlay_figure = visualization.plot_ica_overlay(
        ica,
        ica_input,
        save_path=(
            paths.figures_dir
            / condition
            / f"sub{subject}{session_suffix}_ica_overlay.png"
        ),
        show=ica_cfg.interactive,
    )
    if not ica_cfg.interactive:
        plt.close(overlay_figure)

    save_path = paths.preprocessed_file(subject, condition, session)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_clean.save(save_path, overwrite=True)
    logger.info("前処理済みデータを保存しました: %s", save_path)
    return save_path


def _load_segments(
    path: Path,
    plan: SegmentPlan | None,
    state_event_plan: StateEventPlan | None = None,
) -> dict[str, mne.io.BaseRaw]:
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    if plan is not None and state_event_plan is not None:
        raise ValueError("plan and state_event_plan cannot be used together")
    if state_event_plan is not None:
        events = io.get_marker_events(raw, allowed_markers=state_event_plan.marker_ids)
        return preprocessing.segment_by_state_events(raw, events, state_event_plan)
    if plan is None:
        return {"whole_task": raw}
    events = io.get_marker_events(raw)
    return preprocessing.segment_by_markers(raw, events, plan)


def run_pair_connectivity(
    pair: tuple[int, int],
    condition: str,
    paths: PathConfig,
    bands: tuple[FrequencyBand, ...],
    segment_plan: SegmentPlan | None = None,
    method: str = "plv",
    save_csv: bool = True,
    save_figures: bool = False,
    session: int | None = None,
    state_event_plan: StateEventPlan | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """1ペア・1条件分の脳間同期指標を、区間ごと・周波数帯域ごとに算出する。

    Args:
        segment_plan: ゲーム状態ごとの区間分割定義（例:
            `design.condition_segments[condition]`）。None の場合は
            タスク全体を単一区間 'whole_task' として扱う。
        method: `mne_connectivity.spectral_connectivity_epochs` の method。

    Returns:
        {区間名: {帯域名: (p1側電極 x p2側電極) DataFrame}}。
    """
    sub_a, sub_b = pair
    pair_label = f"pair_{sub_a:02d}_{sub_b:02d}"
    session_label = f"session{session:02d}" if session is not None else "legacy"
    logger.info(
        "=== 同期指標算出開始: %s / %s / %s ===",
        pair_label,
        condition,
        session_label,
    )

    file_a = paths.preprocessed_file(sub_a, condition, session)
    file_b = paths.preprocessed_file(sub_b, condition, session)
    if not file_a.exists() or not file_b.exists():
        raise FileNotFoundError(f"前処理済みファイルが見つかりません: {file_a} または {file_b}")

    segments_a = _load_segments(file_a, segment_plan, state_event_plan)
    segments_b = _load_segments(file_b, segment_plan, state_event_plan)
    if segments_a.keys() != segments_b.keys():
        raise ValueError(
            f"2者間で検出された区間ラベルが一致しません: {sorted(segments_a)} vs {sorted(segments_b)}"
        )

    results: dict[str, dict[str, pd.DataFrame]] = {}
    for label in segments_a:
        logger.info("-> 区間: %s", label)
        combined_raw, sfreq = phase.combine_dyad_raw(segments_a[label], segments_b[label])
        band_matrices = connectivity.compute_all_bands(combined_raw, sfreq, bands, method=method)
        results[label] = band_matrices

        for band_name, matrix in band_matrices.items():
            if save_csv:
                result_dir = paths.results_dir / condition
                if session is not None:
                    result_dir = result_dir / f"session{session:02d}"
                csv_path = result_dir / label / f"{pair_label}_{method}_{band_name}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                matrix.to_csv(csv_path)
                logger.info("CSVを保存しました: %s", csv_path)

            if save_figures:
                figure_dir = paths.figures_dir / condition
                if session is not None:
                    figure_dir = figure_dir / f"session{session:02d}"
                fig_path = figure_dir / label / f"{pair_label}_{method}_{band_name}.png"
                figure = visualization.plot_band_heatmap(
                    matrix, f"{pair_label} / {condition} / {label} / {band_name}", fig_path
                )
                plt.close(figure)

    return results
