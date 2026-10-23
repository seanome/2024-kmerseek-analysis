"""Check alphabet axes against the names the analysis registry actually uses."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
from figure_utils import alphabet_axis, alphabet_size
from ortholog_analysis_utils import ALPHABET_AA_GROUPS


@pytest.mark.parametrize("alphabet", list(ALPHABET_AA_GROUPS))
def test_registry_names_get_their_actual_class_count(alphabet):
    assert alphabet_size(alphabet) == len(ALPHABET_AA_GROUPS[alphabet])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_empty_filtered_alphabet_axis_has_no_ticks(axis):
    fig, ax = plt.subplots()
    try:
        assert alphabet_axis(ax, [], axis=axis) == {}
        ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
        assert len(ticks) == 0
    finally:
        plt.close(fig)


def test_merged_reexports_keep_main_hp_keywords(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import figure_utils as fu
    import hp_conservation_utils as hp
    import mhc_region_utils as mhc

    assert hp.finish_figure is fu.finish_figure
    assert mhc.finish_figure is fu.finish_figure
    fig, ax = plt.subplots()
    layout = Mock()
    monkeypatch.setattr(fig, "tight_layout", layout)
    try:
        ax.plot([0, 1], [0, 1], label="observed")
        ax.legend()
        hp.finish_figure(
            fig,
            tmp_path / "legacy.png",
            tools="cached check",
            hypothesis="Keyword compatibility holds.",
            conclusion="Image written.",
            title="Compatibility",
            subtitle="Main branch subtitle",
            font_scale=0.5,
            footer_box={"facecolor": "white"},
            tight_layout=False,
        )
        layout.assert_not_called()
        labels = {t.get_text(): t for t in fig.texts}
        assert labels["Main branch subtitle"].get_fontsize() == 4.5
        assert labels["Compatibility"].get_fontsize() == 6.25
        assert (tmp_path / "legacy.png").stat().st_size > 0
    finally:
        plt.close(fig)


def test_main_probability_helpers_survive_merge():
    import numpy as np
    import hp_conservation_utils as hp

    # Enumerating all eight length-three binary strings gives three with adjacent 1s.
    assert hp.aligned_stretches("AA-AA", "AAAAA").tolist() == [2, 2]
    got = hp.pr_longest_run_at_least(
        np.array([0.5, 0.5]), [np.array([3]), np.array([2, 2])], [2]
    )
    assert got[0].tolist() == pytest.approx([3 / 8, 7 / 16])
