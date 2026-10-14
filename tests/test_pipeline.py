from __future__ import annotations

from unittest.mock import Mock

import pytest

from hyperscan_eeg import pipeline
from hyperscan_eeg.config import ICAConfig, MontageConfig, PathConfig


def test_missing_digitizer_file_uses_standard_montage(
    tmp_path, caplog, monkeypatch
) -> None:
    raw = Mock()
    paths = PathConfig(digitizer_dir=tmp_path / "digitizer")
    montage_cfg = MontageConfig(channel_labels=("Cz",))
    standard_montage = Mock()
    make_standard_montage = Mock(return_value=standard_montage)
    monkeypatch.setattr(
        pipeline.mne.channels,
        "make_standard_montage",
        make_standard_montage,
    )

    pipeline._register_digitizer_montage(
        raw,
        subject=99,
        condition="speaking",
        paths=paths,
        montage_cfg=montage_cfg,
        save_montage_figures=False,
    )

    make_standard_montage.assert_called_once_with("standard_1020")
    raw.set_montage.assert_called_once_with(
        standard_montage, match_case=False, on_missing="warn"
    )
    assert "using MNE standard montage 'standard_1020'" in caplog.text


def test_session_paths_and_discovery(tmp_path) -> None:
    paths = PathConfig(
        raw_dir=tmp_path / "raw",
        marker_dir=tmp_path / "markers",
        preprocessed_dir=tmp_path / "preprocessed",
    )
    speaking_dir = paths.raw_dir / "speaking"
    speaking_dir.mkdir(parents=True)
    (speaking_dir / "sub07_session02.bdf").touch()
    (speaking_dir / "sub07_session01.bdf").touch()

    assert paths.raw_sessions(7, "speaking") == (1, 2)
    assert paths.raw_file(7, "speaking", 2).name == "sub07_session02.bdf"
    assert paths.marker_file(7, "speaking", 2).name == "sub07_session02_speaking.csv"
    assert (
        paths.preprocessed_file(7, "speaking", 2).name
        == "sub07_session02_speaking_preprocessed_raw.fif"
    )


def test_average_reference_is_independent_of_iclabel() -> None:
    raw = Mock()

    pipeline._apply_analysis_reference(
        raw,
        apply_average_reference=True,
        ica_cfg=ICAConfig(use_iclabel=False),
    )

    raw.set_eeg_reference.assert_called_once_with(
        "average", projection=False, verbose=False
    )


def test_native_reference_without_iclabel_is_allowed() -> None:
    raw = Mock()

    pipeline._apply_analysis_reference(
        raw,
        apply_average_reference=False,
        ica_cfg=ICAConfig(use_iclabel=False),
    )

    raw.set_eeg_reference.assert_not_called()


def test_iclabel_requires_average_reference() -> None:
    with pytest.raises(ValueError, match="requires apply_average_reference=True"):
        pipeline._apply_analysis_reference(
            Mock(),
            apply_average_reference=False,
            ica_cfg=ICAConfig(use_iclabel=True),
        )
