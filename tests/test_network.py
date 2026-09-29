from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyperscan_eeg.network import (
    build_hyperbrain_matrix,
    compute_network_metrics,
    graph_from_adjacency,
    threshold_hyperbrain_matrix,
)
from hyperscan_eeg.network_reporting import (
    build_participant_graph,
    segment_sort_key,
    subject_groups_from_pairs,
)


def _matrix(values: list[list[float]], channels: tuple[str, ...] = ("Cz", "Fz")) -> pd.DataFrame:
    return pd.DataFrame(values, index=channels, columns=channels)


def _complete_pairs() -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (1, 2): _matrix([[1.0, 2.0], [3.0, 4.0]]),
        (1, 3): _matrix([[5.0, 6.0], [7.0, 8.0]]),
        (2, 3): _matrix([[9.0, 10.0], [11.0, 12.0]]),
    }


def test_build_hyperbrain_places_pair_and_transpose() -> None:
    adjacency = build_hyperbrain_matrix((1, 2, 3), _complete_pairs())

    assert adjacency.shape == (6, 6)
    np.testing.assert_allclose(adjacency, adjacency.T)
    assert adjacency.loc["sub01:Cz", "sub02:Fz"] == 2.0
    assert adjacency.loc["sub02:Fz", "sub01:Cz"] == 2.0
    assert adjacency.loc["sub01:Cz", "sub01:Fz"] == 0.0


def test_build_hyperbrain_rejects_missing_pair() -> None:
    pairs = _complete_pairs()
    del pairs[(2, 3)]

    with pytest.raises(ValueError, match="不足"):
        build_hyperbrain_matrix((1, 2, 3), pairs)


def test_build_hyperbrain_rejects_channel_order_mismatch() -> None:
    pairs = _complete_pairs()
    pairs[(2, 3)] = _matrix([[9.0, 10.0], [11.0, 12.0]], ("Fz", "Cz"))

    with pytest.raises(ValueError, match="チャンネル名・順序"):
        build_hyperbrain_matrix((1, 2, 3), pairs)


def test_absolute_threshold_retains_large_negative_edge() -> None:
    adjacency = pd.DataFrame(
        [[0.0, -10.0, 1.0], [-10.0, 0.0, 2.0], [1.0, 2.0, 0.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )

    thresholded, threshold = threshold_hyperbrain_matrix(
        adjacency, percentile=90.0, by_absolute_value=True
    )

    assert threshold == pytest.approx(8.4)
    assert thresholded.loc["A", "B"] == -10.0
    assert thresholded.loc["A", "C"] == 0.0


def test_metrics_include_isolates_and_largest_component_path() -> None:
    adjacency = pd.DataFrame(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    graph = graph_from_adjacency(adjacency)

    metrics = compute_network_metrics(graph, 1.0, 90.0, True)

    assert metrics.n_nodes == 3
    assert metrics.n_edges == 1
    assert metrics.n_isolates == 1
    assert metrics.n_components == 2
    assert metrics.largest_component_nodes == 2
    assert metrics.characteristic_path_length == 1.0


def test_participant_graph_uses_mean_channel_pair_connectivity() -> None:
    graph = build_participant_graph((1, 2, 3), _complete_pairs())

    assert set(graph.nodes) == {1, 2, 3}
    assert set(graph.edges) == {(1, 2), (1, 3), (2, 3)}
    assert graph[1][2]["weight"] == pytest.approx(2.5)
    assert graph[1][2]["median"] == pytest.approx(2.5)
    assert graph[1][2]["n_channel_pairs"] == 4


def test_subject_groups_are_recovered_from_configured_pairs() -> None:
    groups = subject_groups_from_pairs(
        (1, 2, 3, 4, 5, 6),
        ((1, 2), (1, 3), (2, 3), (4, 5), (4, 5), (4, 6), (5, 6)),
    )

    assert groups == ((1, 2, 3), (4, 5, 6))


def test_subject_group_rejects_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="pairが不足"):
        subject_groups_from_pairs((1, 2, 3), ((1, 2), (2, 3)))


def test_game_segments_sort_in_temporal_order() -> None:
    segments = ["gattai_ato2", "gattai_ato", "gattai_mae", "gattai_ato1"]

    assert sorted(segments, key=segment_sort_key) == [
        "gattai_mae",
        "gattai_ato",
        "gattai_ato1",
        "gattai_ato2",
    ]
