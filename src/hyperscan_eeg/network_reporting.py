"""既存connectivity CSVから発表向けnetwork figureを作成する。

96ノードのchannel-level hyperbrainだけでは結合が過密になりやすいため、
各被験者ペアの全channel pair平均をedge weightとする3ノードの
participant-level networkも併せて作成する。グラフの構築・描画にはNetworkXを
使用し、状態間でnode位置とedge幅の尺度を共通化して比較可能にする。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .network import (
    NetworkMetrics,
    build_hyperbrain_matrix,
    compute_network_metrics,
    graph_from_adjacency,
    load_connectivity_csv,
    plot_hyperbrain_network,
    threshold_hyperbrain_matrix,
)


def subject_groups_from_pairs(
    subjects: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """設定済みpairの連結成分から独立した被験者groupを復元する。

    各group内はhyperbrain構築に必要な総当たりpairが揃っていることを検証する。
    これにより、複数pilotのtriadを1つのExperimentDesignに列挙できる。
    """
    pair_graph = nx.Graph()
    pair_graph.add_nodes_from(subjects)
    pair_graph.add_edges_from(pairs)
    groups = tuple(
        sorted((tuple(sorted(component)) for component in nx.connected_components(pair_graph)))
    )
    supplied = {tuple(sorted(pair)) for pair in pairs}
    for group in groups:
        if len(group) < 2:
            raise ValueError(f"pairを持たない被験者groupがあります: {group}")
        expected = {
            (subject_a, subject_b)
            for index, subject_a in enumerate(group)
            for subject_b in group[index + 1 :]
        }
        missing = expected.difference(supplied)
        if missing:
            raise ValueError(f"被験者group {group} のpairが不足しています: {sorted(missing)}")
    return groups


def segment_sort_key(label: str) -> tuple[int, str]:
    """ゲーム状態を合体前から合体後への時間順に並べるためのキー。"""
    if label == "gattai_mae":
        return (0, label)
    if label == "gattai_ato":
        return (1, label)
    match = re.fullmatch(r"gattai_ato(\d+)", label)
    if match:
        return (int(match.group(1)), label)
    return (100, label)


def load_pair_matrices(
    pair_files: Mapping[tuple[int, int], Path],
) -> dict[tuple[int, int], pd.DataFrame]:
    """全被験者ペアのconnectivity CSVを検証しながら読み込む。"""
    return {pair: load_connectivity_csv(path) for pair, path in pair_files.items()}


def build_participant_graph(
    subjects: tuple[int, ...],
    pair_matrices: Mapping[tuple[int, int], pd.DataFrame],
) -> nx.Graph:
    """channel pair平均PLVをedge weightとするparticipant-level graphを作る。"""
    graph = nx.Graph()
    graph.add_nodes_from(subjects)
    for (subject_a, subject_b), matrix in pair_matrices.items():
        values = matrix.to_numpy(dtype=float)
        graph.add_edge(
            subject_a,
            subject_b,
            weight=float(values.mean()),
            median=float(np.median(values)),
            n_channel_pairs=int(values.size),
        )
    return graph


def plot_participant_network_comparison(
    state_graphs: Mapping[str, nx.Graph],
    subjects: tuple[int, ...],
    title: str,
    save_path: Path,
) -> plt.Figure:
    """複数状態の3-node networkを同一尺度・同一配置で並べる。"""
    if not state_graphs:
        raise ValueError("描画するparticipant-level networkがありません。")

    all_weights = [
        float(data["weight"])
        for graph in state_graphs.values()
        for _, _, data in graph.edges(data=True)
    ]
    if not all_weights:
        raise ValueError("participant-level networkにedgeがありません。")
    weight_min = min(all_weights)
    weight_max = max(all_weights)
    denominator = weight_max - weight_min

    # 状態間で位置を固定し、上に1名・下に2名の三角形として表示する。
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(subjects), endpoint=False)
    positions = {
        subject: np.array([np.cos(angle), np.sin(angle)])
        for subject, angle in zip(subjects, angles)
    }

    figure, axes = plt.subplots(
        1, len(state_graphs), figsize=(6 * len(state_graphs), 5), squeeze=False
    )
    palette = plt.get_cmap("Set2")
    node_colors = [palette(i % palette.N) for i in range(len(subjects))]

    for axis, (state, graph) in zip(axes[0], state_graphs.items()):
        edges = list(graph.edges(data=True))
        weights = np.array([float(data["weight"]) for _, _, data in edges])
        if denominator <= 0:
            widths = np.full(len(edges), 4.0)
        else:
            widths = 1.5 + 7.5 * (weights - weight_min) / denominator

        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_size=2600,
            node_color=node_colors,
            edgecolors="black",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=axis,
            labels={subject: f"Subject {subject}" for subject in subjects},
            font_size=11,
            font_weight="bold",
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            width=widths,
            edge_color="#4c78a8",
            alpha=0.75,
        )
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            ax=axis,
            edge_labels={(a, b): f"{data['weight']:.3f}" for a, b, data in edges},
            font_size=10,
            label_pos=0.5,
        )
        axis.set_title(state, fontsize=14)
        axis.set_axis_off()

    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure


def analyze_network_for_report(
    subjects: tuple[int, ...],
    pair_files: Mapping[tuple[int, int], Path],
    output_dir: Path,
    output_stem: str,
    title: str,
    percentile: float = 98.0,
    metrics_percentile: float = 90.0,
) -> tuple[NetworkMetrics, nx.Graph, pd.DataFrame]:
    """1状態・1帯域のchannel/participant networkを構築して図を保存する。

    発表図はedgeの交差を抑えるため ``percentile`` で強く絞る一方、network
    metricsは過度に疎なgraphでclustering等がすべて0になるのを避けるため、
    独立した ``metrics_percentile`` のgraphから算出する。
    """
    pair_matrices = load_pair_matrices(pair_files)
    adjacency = build_hyperbrain_matrix(subjects, pair_matrices)
    thresholded, threshold = threshold_hyperbrain_matrix(
        adjacency, percentile=percentile, by_absolute_value=True
    )
    channel_graph = graph_from_adjacency(thresholded)
    metrics_thresholded, metrics_threshold = threshold_hyperbrain_matrix(
        adjacency, percentile=metrics_percentile, by_absolute_value=True
    )
    metrics_graph = graph_from_adjacency(metrics_thresholded)
    metrics = compute_network_metrics(
        metrics_graph, metrics_threshold, metrics_percentile, True
    )
    participant_graph = build_participant_graph(subjects, pair_matrices)

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_figure = plot_hyperbrain_network(
        channel_graph,
        subjects,
        list(next(iter(pair_matrices.values())).index),
        threshold,
        title,
        save_path=output_dir / f"{output_stem}_channel_network.png",
        show=False,
    )
    plt.close(channel_figure)

    participant_edges = pd.DataFrame(
        [
            {
                "subject_a": subject_a,
                "subject_b": subject_b,
                **data,
            }
            for subject_a, subject_b, data in participant_graph.edges(data=True)
        ]
    )
    participant_edges.to_csv(
        output_dir / f"{output_stem}_participant_edges.csv", index=False
    )
    return metrics, participant_graph, thresholded


def plot_network_metrics(
    metrics: pd.DataFrame,
    save_path: Path,
) -> plt.Figure:
    """状態・帯域別の主要な二値network metricsを比較表示する。"""
    required = {
        "segment",
        "band",
        "global_efficiency",
        "local_efficiency",
        "average_clustering",
        "characteristic_path_length",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"network metricsに必要な列がありません: {sorted(missing)}")

    metric_columns = (
        "global_efficiency",
        "local_efficiency",
        "average_clustering",
        "characteristic_path_length",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(metrics["band"].nunique())
    bands = list(dict.fromkeys(metrics["band"]))
    segments = list(dict.fromkeys(metrics["segment"]))
    width = 0.8 / max(len(segments), 1)

    for axis, metric_name in zip(axes.flat, metric_columns):
        for segment_index, segment in enumerate(segments):
            subset = metrics[metrics["segment"] == segment].set_index("band")
            values = [float(subset.loc[band, metric_name]) for band in bands]
            offset = (segment_index - (len(segments) - 1) / 2) * width
            axis.bar(x + offset, values, width=width, label=segment)
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xticks(x)
        axis.set_xticklabels(bands)
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend()
    percentiles = metrics.get("percentile")
    percentile_label = ""
    if percentiles is not None and len(set(percentiles)) == 1:
        percentile_label = f" (p{float(percentiles.iloc[0]):g})"
    figure.suptitle(f"Channel-level PLV network metrics{percentile_label}")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure
