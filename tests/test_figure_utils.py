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
