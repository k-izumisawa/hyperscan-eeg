"""区間切り出し・チャンネル選択・フィルタリング・ICAによるノイズ除去。

すべての関数は特定の実験デザイン（条件名やゲーム状態名）に依存せず、
マーカーコードとラベルのリストを引数として受け取る汎用関数として実装する。
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mne
import numpy as np

from .config import FilterConfig, ICAConfig, MarkerConfig, SegmentPlan

logger = logging.getLogger(__name__)


def crop_to_task_window(
    raw: mne.io.BaseRaw, events: np.ndarray, marker_cfg: MarkerConfig
) -> mne.io.BaseRaw:
    """開始マーカーから終了マーカーまでの区間を切り出す。

    Raises:
        ValueError: 開始・終了マーカーのいずれかが見つからない場合。
    """
    starts = events[events[:, 2] == marker_cfg.start_marker][:, 0]
    ends = events[events[:, 2] == marker_cfg.end_marker][:, 0]
    if len(starts) == 0 or len(ends) == 0:
        raise ValueError(
            f"開始マーカー({marker_cfg.start_marker})または"
            f"終了マーカー({marker_cfg.end_marker})が見つかりません。"
        )

    sfreq = raw.info["sfreq"]
    tmin = (starts[0] - raw.first_samp) / sfreq
    tmax = (ends[-1] - raw.first_samp) / sfreq
    raw_task = raw.copy().crop(tmin=tmin, tmax=tmax)
    logger.info("タスク区間を切り出しました: %.2f - %.2f 秒", tmin, tmax)
    return raw_task


def segment_by_markers(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    plan: SegmentPlan,
) -> dict[str, mne.io.BaseRaw]:
    """`SegmentPlan` に従い、Rawを名前付き区間に分割する。
    状態ごとに計測ファイルを分けている場合は実行不要

    `plan.boundary_markers` を時系列順に1つずつ辿り、直前の境界以降で
    最初に出現するマーカーを次の境界として採用する。同一マーカーの反復
    （例: 'speaking' 条件で同じ「合体」マーカーが2回出現 -> 3区間）にも、
    境界ごとに異なるマーカーコードを使う構成にも対応する。状態数や
    境界マーカーの構成が変わっても `SegmentPlan` を差し替えるだけでよく、
    本関数の変更は不要。

    Args:
        raw: 対象のRaw（通常はタスク区間切り出し済みのもの）。
        events: `get_marker_events` 等で得たイベント配列。
        plan: 境界マーカーとラベルの対応定義。

    Raises:
        ValueError: `plan.boundary_markers` に対応するマーカーが
            イベント中に見つからない場合。
    """
    sfreq = raw.info["sfreq"]
    events_sorted = events[np.argsort(events[:, 0])]

    boundary_times: list[float] = []
    cursor = raw.first_samp
    for i, marker in enumerate(plan.boundary_markers):
        matches = events_sorted[
            (events_sorted[:, 2] == marker) & (events_sorted[:, 0] >= cursor)
        ]
        if len(matches) == 0:
            raise ValueError(
                f"境界{i + 1}番目のマーカー({marker})がこれ以降のイベント中に見つかりません。"
            )
        sample = matches[0, 0]
        boundary_times.append((sample - raw.first_samp) / sfreq)
        cursor = sample + 1  # 同一マーカーの反復にも対応するため、次はこのサンプルより後を探索

    boundaries = [raw.times[0], *boundary_times, raw.times[-1]]
    segments = {}
    for label, t0, t1 in zip(plan.labels, boundaries[:-1], boundaries[1:]):
        segments[label] = raw.copy().crop(tmin=t0, tmax=t1)
        logger.info("区間 '%s': %.2f - %.2f 秒", label, t0, t1)
    return segments


def select_eeg_channels(
    raw: mne.io.BaseRaw, aux_prefixes: tuple[str, ...]
) -> mne.io.BaseRaw:
    """補助（非EEG）チャンネルを除外し、EEGチャンネルのみを残す。"""
    eeg_chs = [ch for ch in raw.ch_names if not ch.startswith(aux_prefixes)]
    if not eeg_chs:
        raise ValueError("EEGチャンネルが1つも検出されませんでした。")
    return raw.copy().pick(eeg_chs)


def apply_bandpass_notch(raw: mne.io.BaseRaw, filter_cfg: FilterConfig) -> mne.io.BaseRaw:
    """バンドパスフィルタとノッチフィルタ（電源ノイズ除去）を適用する。"""
    raw_filtered = raw.copy().filter(
        l_freq=filter_cfg.l_freq,
        h_freq=filter_cfg.h_freq,
        fir_design=filter_cfg.fir_design,
        verbose=False,
    )
    raw_filtered.notch_filter(freqs=list(filter_cfg.notch_freqs), verbose=False)
    return raw_filtered


def _ica_fit_raw(raw: mne.io.BaseRaw, ica_cfg: ICAConfig) -> mne.io.BaseRaw:
    """ICAのフィットおよび`label_ica_iclabel`での分類に使う前処理済みコピーを作る。

    `ica_cfg.use_iclabel` が False の場合は、`ica_highpass` Hzのハイパスの
    みを適用する。

    True の場合は、ICLabelの前提に合わせて明示的に `h_freq=100.0` のローパスと平均参照を
    追加で適用する。そのため、この関数には**プロジェクトの解析用バンドパス
    （例: 60Hzローパス）をまだ適用していない `raw`** を渡すこと。既に60Hz等
    でローパス済みのデータに `h_freq=100.0` を指定しても、60Hz超の情報は
    filter()の呼び出し前から失われているため、`info['lowpass']=100`という
    メタデータ上のつじつまは合うが、ICLabelが実際に活用できる高周波成分
    （筋電アーチファクト等の判別に有効）は増えない。
    """
    h_freq = 100.0 if ica_cfg.use_iclabel else None
    raw_for_ica = raw.copy().filter(
        l_freq=ica_cfg.ica_highpass, h_freq=h_freq, fir_design="firwin", verbose=False
    )
    if ica_cfg.use_iclabel:
        raw_for_ica.set_eeg_reference("average", verbose=False)
    return raw_for_ica


def fit_ica(raw: mne.io.BaseRaw, ica_cfg: ICAConfig) -> mne.preprocessing.ICA:
    """フィルタ済みコピー上でICAをフィットする（適用はしない）。

    フィルタ内容は `_ica_fit_raw` を参照。`ica_cfg.use_iclabel=True` で
    `label_ica_iclabel` も併用する場合は、`fit_ica` と `label_ica_iclabel` に
    **同じ `raw`（プロジェクトの解析用バンドパス適用前のもの）** を渡すこと。
    """
    raw_for_ica = _ica_fit_raw(raw, ica_cfg)
    ica = mne.preprocessing.ICA(
        n_components=None,
        random_state=ica_cfg.random_state,
        method=ica_cfg.method,
        fit_params=ica_cfg.fit_params,
    )
    ica.fit(raw_for_ica)
    return ica


def label_ica_iclabel(
    ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw, ica_cfg: ICAConfig
) -> mne.preprocessing.ICA:
    """MNE-ICALabel（ICLabel）でICA成分を分類し、目視確認用の補助情報として付与する。

    `raw` には `fit_ica(raw, ica_cfg)` に渡したものと同じ`raw`を渡すこと
    （`_ica_fit_raw` で同じ1-100Hzバンドパス・平均参照済みコピーを内部で
    再構築し、ICLabelの前提データに合わせる）。

    分類ラベルは "brain" / "muscle artifact" / "eye blink" / "heart beat" /
    "line noise" / "channel noise" / "other" の7種類。確信度がわずかでも
    非brainラベルに割り当てられた成分を機械的に除外すると過剰除去に
    つながりかねないため、**`ica.exclude` は自動では設定しない**。代わりに
    各成分の分類結果を `review_ica_interactively` のGUI上の成分名に
    `ICA000 [eye:92%]` のように付記し、目視判断の補助材料として使う
    （実際の除外はGUI上の手動選択で行う）。

    Returns:
        `_ica_names`（GUI表示名）が更新された同じICAオブジェクト。
    """
    from mne_icalabel import label_components

    if not ica_cfg.interactive:
        logger.warning(
            "use_iclabel=True ですが interactive=False のため、ICLabelの分類結果は"
            "目視確認なしでは ica.exclude に反映されません（成分は除去されません）。"
        )

    raw_for_label = _ica_fit_raw(raw, ica_cfg)
    result = label_components(raw_for_label, ica, method="iclabel")
    labels = result["labels"]
    probs = result["y_pred_proba"]

    short_labels = {
        "brain": "brain",
        "muscle artifact": "muscle",
        "eye blink": "eye",
        "heart beat": "heart",
        "line noise": "line",
        "channel noise": "chan",
        "other": "other",
    }
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        logger.info("IC%03d: %s (確信度 %.2f)", idx, label, prob)
        short = short_labels.get(label, label)
        # plot_sources() は ica._ica_names をGUI上の成分名として表示するため、
        # ここに分類ラベルを書き込んで目視確認時の補助情報とする
        # （MNEの非公開属性への依存だが、表示用途のみで解析結果には影響しない）。
        ica._ica_names[idx] = f"{ica._ica_names[idx]} [{short}:{prob:.0%}]"

    return ica


def review_ica_interactively(
    ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
) -> mne.preprocessing.ICA:
    """成分topographyと時系列の目視確認用ウィンドウを表示する。

    ウィンドウを閉じると選択された成分が `ica.exclude` に反映される。
    非対話環境（バッチ実行等）では呼び出さないこと。
    """
    logger.info("ICA成分のtopography一覧を表示します。波形確認中も開いたままになります。")
    component_figures = ica.plot_components(
        inst=raw,
        title="ICA component topographies",
        show=False,
    )
    if not isinstance(component_figures, (list, tuple)):
        component_figures = [component_figures]
    for figure in component_figures:
        figure.show()
        figure.canvas.flush_events()

    logger.info(
        "topographyを開いたまま波形を表示します。波形画面上で除外したい成分を選択し、"
        "波形画面を閉じてください。"
    )
    try:
        ica.plot_sources(raw, show=True, block=True)
    finally:
        # 波形レビュー終了後は、次の被験者の図と混在しないようtopographyを閉じる。
        for figure in component_figures:
            plt.close(figure)
    logger.info("除外対象として選択された成分: %s", ica.exclude)
    return ica


def apply_ica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> mne.io.BaseRaw:
    """`ica.exclude` に設定された成分を除去したRawを返す。"""
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean
