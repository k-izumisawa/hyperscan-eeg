from __future__ import annotations

from unittest.mock import Mock

from hyperscan_eeg import pipeline
from hyperscan_eeg.config import MontageConfig, PathConfig


def test_missing_digitizer_file_skips_montage_registration(tmp_path, caplog) -> None:
    raw = Mock()
    paths = PathConfig(digitizer_dir=tmp_path / "digitizer")
    montage_cfg = MontageConfig(channel_labels=("Cz",))

    pipeline._register_digitizer_montage(
        raw,
        subject=99,
        condition="speaking",
        paths=paths,
        montage_cfg=montage_cfg,
        save_montage_figures=True,
    )

    raw.set_montage.assert_not_called()
    assert "skipping montage registration" in caplog.text
