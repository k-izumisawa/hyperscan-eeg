from __future__ import annotations

from unittest.mock import Mock

from hyperscan_eeg import preprocessing


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

