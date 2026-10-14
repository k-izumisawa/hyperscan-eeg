from __future__ import annotations

from unittest.mock import Mock

import mne
import numpy as np

from hyperscan_eeg import preprocessing
from hyperscan_eeg.config import StateEventPlan


def test_ica_topographies_stay_visible_while_sources_are_reviewed(monkeypatch) -> None:
    figures = [Mock(), Mock()]
    ica = Mock()
    ica.exclude = [1, 3]
    ica.plot_components.return_value = figures

    def check_figures_before_source_window(*args, **kwargs) -> None:
        assert all(figure.show.called for figure in figures)
        assert kwargs == {"show": True, "block": True}

    ica.plot_sources.side_effect = check_figures_before_source_window
    close = Mock()
    monkeypatch.setattr(preprocessing.plt, "close", close)

    result = preprocessing.review_ica_interactively(ica, Mock())

    assert result is ica
    ica.plot_components.assert_called_once()
    assert all(figure.canvas.flush_events.called for figure in figures)
    assert close.call_count == len(figures)


def test_state_segments_and_special_event_windows_can_overlap() -> None:
    info = mne.create_info(["Cz"], sfreq=10.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 200)), info, verbose=False)
    events = np.array(
        [
            [10, 0, 1],
            [50, 0, 2],
            [60, 0, 5],
            [100, 0, 3],
        ]
    )
    plan = StateEventPlan(
        state_labels={1: "both_unfused", 2: "own_team_fused", 3: "both_teams_fused"},
        event_labels={5: "own_special"},
        event_window_seconds=8.0,
    )

    cropped = preprocessing.crop_from_game_start(raw, events, start_marker=1)
    segments = preprocessing.segment_by_state_events(cropped, events, plan)

    assert list(segments) == [
        "state_both_unfused_01",
        "state_own_team_fused_01",
        "state_both_teams_fused_01",
        "event_own_special_01",
    ]
    assert segments["state_both_unfused_01"].times[-1] == 3.9
    assert segments["state_own_team_fused_01"].times[-1] == 4.9
    assert segments["state_both_teams_fused_01"].times[-1] == 9.9
    assert segments["event_own_special_01"].times[-1] == 8.0


def test_cropped_state_data_restores_game_start_at_first_sample() -> None:
    info = mne.create_info(["Cz"], sfreq=10.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
    events_without_boundary_start = np.array([[40, 0, 2], [60, 0, 5]])
    plan = StateEventPlan(
        state_labels={1: "both_unfused", 2: "own_team_fused"},
        event_labels={5: "own_special"},
        event_window_seconds=3.0,
    )

    segments = preprocessing.segment_by_state_events(
        raw, events_without_boundary_start, plan
    )

    assert list(segments) == [
        "state_both_unfused_01",
        "state_own_team_fused_01",
        "event_own_special_01",
    ]
    assert segments["state_both_unfused_01"].times[-1] == 3.9


def test_state_segments_end_after_first_special_event_window() -> None:
    info = mne.create_info(["Cz"], sfreq=10.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 250)), info, verbose=False)
    events = np.array(
        [
            [10, 0, 1],
            [50, 0, 2],
            [100, 0, 5],
        ]
    )
    plan = StateEventPlan(
        state_labels={1: "both_unfused", 2: "own_team_fused"},
        event_labels={5: "own_special", 6: "opponent_special"},
        event_window_seconds=8.0,
        end_at_first_event_window=True,
    )

    cropped = preprocessing.crop_from_game_start(raw, events, start_marker=1)
    segments = preprocessing.segment_by_state_events(cropped, events, plan)

    assert segments["state_both_unfused_01"].times[-1] == 3.9
    # State 2 begins at sample 50 and ends at sample 100 + 8 seconds.
    assert segments["state_own_team_fused_01"].times[-1] == 13.0
    assert segments["event_own_special_01"].times[-1] == 8.0
