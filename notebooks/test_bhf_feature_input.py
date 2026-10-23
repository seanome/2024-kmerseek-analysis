"""Reject relocated region data from a different BHF experiment."""

from pathlib import Path

import pytest

import bhf_feature_utils as features


def setup_inputs(tmp_path: Path, monkeypatch):
    run = tmp_path / "null_bhf_dipeptide_regions"
    reference = tmp_path / "null_bhf_dipeptide"
    moved = tmp_path / "287" / run.name
    reference.mkdir()
    (moved / "regions").mkdir(parents=True)
    for name in ["queries.fa", "arms.json"]:
        (reference / name).write_text("same source")
        (moved / name).write_text("same source")
    monkeypatch.setattr(features, "REGIONS_RUN", run)
    return run, moved


def test_relocated_inputs_require_identical_queries_and_settings(tmp_path, monkeypatch):
    run, moved = setup_inputs(tmp_path, monkeypatch)
    assert features.regions_input(run) == moved
    (moved / "queries.fa").write_text("different experiment")
    with pytest.raises(ValueError, match="queries.fa"):
        features.regions_input(run)


def test_explicit_missing_run_does_not_pick_another_experiment(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.regions_input(tmp_path / "other_experiment")
