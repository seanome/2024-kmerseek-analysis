"""A partial parameter search must never become a negative result."""

import importlib.util
from pathlib import Path

import pytest


def test_incomplete_search_does_not_write_verdict(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "nb299", root / "notebooks/299_omrgc_sketch_pilot.py"
    )
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    monkeypatch.setattr(pilot, "work_dir", lambda: tmp_path)
    monkeypatch.setattr(pilot, "DATA", tmp_path)
    # One setting finished; the remaining settings must not be treated as negatives.
    done = tmp_path / "search" / pilot.setting_id(*pilot.SETTINGS[0]) / "done"
    done.parent.mkdir(parents=True)
    done.touch()
    with pytest.raises(SystemExit, match="incomplete search; no verdict written"):
        pilot.cmd_score(None)
    assert not (tmp_path / "verdict.json").exists()
