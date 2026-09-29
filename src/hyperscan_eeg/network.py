"""被験者間connectivity行列からhyperbrainネットワークを構築・評価する。

各CSVは、ファイル名のペア ``pair_A_B`` に対して行が被験者A、列が
被験者BのEEGチャンネルを表す。本モジュールは全ペアの行列を厳格に検証し、
被験者内エッジを持たない対称なhyperbrain隣接行列を構築する。

グラフ指標は、閾値処理後の二値トポロジーに対して算出する。connectivity値は
可視化と監査用隣接行列には保持されるが、距離へ暗黙変換して最短経路計算には
使用しない（強度と距離は意味が逆であり、変換規則に研究上の判断が必要なため）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NetworkMetrics:
    """閾値後の二値hyperbrainネットワーク指標。"""

    n_nodes: int
    n_edges: int
    density: float
    n_isolates: int
    n_components: int
    largest_component_nodes: int
    global_efficiency: float
    local_efficiency: float
    average_clustering: float
    characteristic_path_length: float
    threshold: float
    percentile: float
    threshold_by_absolute_value: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "n_isolates": self.n_isolates,
            "n_components": self.n_components,
            "largest_component_nodes": self.largest_component_nodes,
            "global_efficiency": self.global_efficiency,
            "local_efficiency": self.local_efficiency,
            "average_clustering": self.average_clustering,
            "characteristic_path_length": self.characteristic_path_length,
            "threshold": self.threshold,
            "percentile": self.percentile,
            "threshold_by_absolute_value": self.threshold_by_absolute_value,
        }


def load_connectivity_csv(path: Path) -> pd.DataFrame:
    """チャンネル名付きconnectivity CSVを読み込み、数値・形状を検証する。"""
    if not path.exists():
        raise FileNotFoundError(f"connectivity CSVが見つかりません: {path}")

    matrix = pd.read_csv(path, index_col=0)
    if matrix.empty:
        raise ValueError(f"connectivity CSVが空です: {path}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"connectivity行列が正方行列ではありません: {path} {matrix.shape}")
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError(f"チャンネル名に重複があります: {path}")
    if list(matrix.index) != list(matrix.columns):
        raise ValueError(
            f"行と列のチャンネル名・順序が一致しません: {path}\n"
            f"rows={list(matrix.index)}\ncolumns={list(matrix.columns)}"
        )
    try:
        matrix = matrix.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"connectivity行列に数値以外が含まれます: {path}") from exc
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(f"connectivity行列にNaNまたはInfがあります: {path}")
    return matrix


def build_hyperbrain_matrix(
    subjects: tuple[int, ...],
    pair_matrices: Mapping[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """全被験者ペア行列から対称なhyperbrain隣接行列を構築する。"""
    if len(subjects) < 2 or len(set(subjects)) != len(subjects):
        raise ValueError(f"subjectsは重複のない2名以上である必要があります: {subjects}")

    expected_pairs = {
        (subject_a, subject_b)
        for i, subject_a in enumerate(subjects)
        for subject_b in subjects[i + 1 :]
    }
    supplied_pairs = set(pair_matrices)
    if supplied_pairs != expected_pairs:
        raise ValueError(
            "hyperbrain構築に必要なペア行列が揃っていません。"
            f"不足={sorted(expected_pairs - supplied_pairs)}, "
            f"余分={sorted(supplied_pairs - expected_pairs)}"
        )

    first = pair_matrices[next(iter(expected_pairs))]
    channel_names = list(first.index)
    for pair, matrix in pair_matrices.items():
        if matrix.shape != first.shape:
            raise ValueError(f"ペア{pair}の行列形状が一致しません: {matrix.shape} != {first.shape}")
        if list(matrix.index) != channel_names or list(matrix.columns) != channel_names:
            raise ValueError(f"ペア{pair}のチャンネル名・順序が他ペアと一致しません。")

    labels = [f"sub{subject:02d}:{channel}" for subject in subjects for channel in channel_names]
    n_channels = len(channel_names)
    adjacency = pd.DataFrame(0.0, index=labels, columns=labels)
    subject_offsets = {subject: i * n_channels for i, subject in enumerate(subjects)}

    for (subject_a, subject_b), matrix in pair_matrices.items():
        start_a = subject_offsets[subject_a]
        start_b = subject_offsets[subject_b]
        values = matrix.to_numpy(copy=True)
        adjacency.iloc[start_a : start_a + n_channels, start_b : start_b + n_channels] = values
        adjacency.iloc[start_b : start_b + n_channels, start_a : start_a + n_channels] = values.T

    if not np.allclose(adjacency.to_numpy(), adjacency.to_numpy().T, equal_nan=False):
        raise RuntimeError("構築したhyperbrain隣接行列が対称ではありません。")
    return adjacency


def threshold_hyperbrain_matrix(
    adjacency: pd.DataFrame,
    percentile: float = 90.0,
    by_absolute_value: bool = True,
) -> tuple[pd.DataFrame, float]:
    """被験者間エッジの上位強度だけを保持する。"""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentileは0から100の範囲で指定してください: {percentile}")
    values = adjacency.to_numpy()
    upper = values[np.triu_indices_from(values, k=1)]
    candidates = upper[upper != 0]
    if candidates.size == 0:
        raise ValueError("閾値算出に使用できる非ゼロの被験者間エッジがありません。")

    strengths = np.abs(candidates) if by_absolute_value else candidates
    threshold = float(np.percentile(strengths, percentile))
    comparison = np.abs(values) if by_absolute_value else values
    thresholded = values.copy()
    thresholded[comparison < threshold] = 0.0
    np.fill_diagonal(thresholded, 0.0)
    return pd.DataFrame(thresholded, index=adjacency.index, columns=adjacency.columns), threshold


def graph_from_adjacency(adjacency: pd.DataFrame) -> nx.Graph:
    """重み付き対称隣接行列を無向NetworkXグラフへ変換する。"""
    graph = nx.from_numpy_array(adjacency.to_numpy(), create_using=nx.Graph)
    nx.set_node_attributes(graph, dict(enumerate(adjacency.index)), "label")
    return graph


def compute_network_metrics(
    graph: nx.Graph,
    threshold: float,
    percentile: float,
    by_absolute_value: bool,
) -> NetworkMetrics:
    """非連結グラフを明示的に扱い、二値トポロジー指標を算出する。"""
    if graph.number_of_nodes() == 0:
        raise ValueError("ノードがないためネットワーク指標を算出できません。")

    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    largest_graph = graph.subgraph(largest)
    path_length = (
        0.0
        if largest_graph.number_of_nodes() == 1
        else float(nx.average_shortest_path_length(largest_graph))
    )
    return NetworkMetrics(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        density=float(nx.density(graph)),
        n_isolates=nx.number_of_isolates(graph),
        n_components=len(components),
        largest_component_nodes=largest_graph.number_of_nodes(),
        global_efficiency=float(nx.global_efficiency(graph)),
        local_efficiency=float(nx.local_efficiency(graph)),
        average_clustering=float(nx.average_clustering(graph)),
        characteristic_path_length=path_length,
        threshold=threshold,
        percentile=percentile,
        threshold_by_absolute_value=by_absolute_value,
    )


def _standard_channel_positions(channel_names: list[str]) -> dict[str, np.ndarray]:
    montage_positions = mne.channels.make_standard_montage("standard_1020").get_positions()[
        "ch_pos"
    ]
    canonical = {name.casefold(): name for name in montage_positions}
    positions: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for channel in channel_names:
        montage_name = canonical.get(channel.casefold())
        if montage_name is None:
            missing.append(channel)
        else:
            positions[channel] = np.asarray(montage_positions[montage_name][:2], dtype=float)
    if missing:
        raise ValueError(f"standard_1020に存在しないチャンネルがあります: {missing}")
    return positions


def plot_hyperbrain_network(
    graph: nx.Graph,
    subjects: tuple[int, ...],
    channel_names: list[str],
    threshold: float,
    title: str,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """頭皮座標を円周上に配置してhyperbrainネットワークを描画する。"""
    expected_nodes = len(subjects) * len(channel_names)
    if graph.number_of_nodes() != expected_nodes:
        raise ValueError(
            f"グラフのノード数が被験者数×チャンネル数と一致しません: "
            f"{graph.number_of_nodes()} != {expected_nodes}"
        )

    scalp_positions = _standard_channel_positions(channel_names)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(subjects), endpoint=False)
    offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.25
    position: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}
    node_colors: list[str] = []
    palette = plt.get_cmap("Set2")
    for subject_index, (subject, offset) in enumerate(zip(subjects, offsets)):
        for channel_index, channel in enumerate(channel_names):
            node = subject_index * len(channel_names) + channel_index
            position[node] = scalp_positions[channel] + offset
            labels[node] = channel
            node_colors.append(palette(subject_index % palette.N))

    edges = list(graph.edges(data=True))
    magnitudes = np.array([abs(float(data["weight"])) for _, _, data in edges])
    if magnitudes.size:
        maximum = float(magnitudes.max())
        denominator = maximum - threshold
        relative = np.ones_like(magnitudes) if denominator <= 0 else (magnitudes - threshold) / denominator
        widths = 0.5 + np.clip(relative, 0.0, 1.0) * 5.5
        edge_colors = ["#666666" if data["weight"] >= 0 else "#377eb8" for _, _, data in edges]
    else:
        widths = []
        edge_colors = []

    figure, axis = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_nodes(
        graph, position, ax=axis, node_size=100, node_color=node_colors, edgecolors="black"
    )
    nx.draw_networkx_labels(graph, position, ax=axis, labels=labels, font_size=7)
    nx.draw_networkx_edges(
        graph,
        position,
        ax=axis,
        edgelist=[(u, v) for u, v, _ in edges],
        width=widths,
        edge_color=edge_colors,
        alpha=0.6,
    )
    for subject, offset in zip(subjects, offsets):
        axis.text(offset[0], offset[1] + 0.13, f"Subject {subject}", ha="center", fontweight="bold")
    axis.set_title(title, fontsize=16)
    axis.set_axis_off()
    figure.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def analyze_hyperbrain_network(
    subjects: tuple[int, ...],
    pair_files: Mapping[tuple[int, int], Path],
    output_dir: Path,
    output_stem: str,
    title: str,
    percentile: float = 90.0,
    by_absolute_value: bool = True,
    show: bool = False,
) -> NetworkMetrics:
    """CSV読込から行列・図・指標保存までを一括実行する。"""
    pair_matrices = {pair: load_connectivity_csv(path) for pair, path in pair_files.items()}
    adjacency = build_hyperbrain_matrix(subjects, pair_matrices)
    thresholded, threshold = threshold_hyperbrain_matrix(
        adjacency, percentile=percentile, by_absolute_value=by_absolute_value
    )
    graph = graph_from_adjacency(thresholded)
    metrics = compute_network_metrics(graph, threshold, percentile, by_absolute_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    adjacency.to_csv(output_dir / f"{output_stem}_adjacency_full.csv")
    thresholded.to_csv(output_dir / f"{output_stem}_adjacency_thresholded.csv")
    pd.DataFrame([metrics.as_dict()]).to_csv(
        output_dir / f"{output_stem}_network_metrics.csv", index=False
    )
    figure = plot_hyperbrain_network(
        graph,
        subjects,
        list(next(iter(pair_matrices.values())).index),
        threshold,
        title,
        save_path=output_dir / f"{output_stem}_network.png",
        show=show,
    )
    plt.close(figure)
    return metrics
