"""poscounts size factors and the size_factor_mean direction rule (design/sample_scaling/README.md)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from pec_analysis_utils import poscounts_size_factors, scaled_per_million  # noqa: E402


def test_a_sample_with_twice_the_reads_gets_twice_the_factor():
    f = poscounts_size_factors([[10, 20], [5, 10], [100, 200]])
    assert f == pytest.approx([2**-0.5, 2**0.5])


def test_zeros_are_left_out_of_the_median_but_counted_in_the_reference():
    # gene 2 has a read in sample 2 only: its reference is log(8) / 2, by hand
    c = np.array([[4.0, 4.0], [0.0, 8.0], [4.0, 4.0]])
    raw = np.array([1.0, np.exp(np.median([0.0, np.log(8) / 2, 0.0]))])
    assert poscounts_size_factors(c) == pytest.approx(raw / np.sqrt(raw.prod()))


def test_one_gene_with_many_reads_moves_the_total_but_not_the_factor():
    c = np.array([[50, 50]] * 9 + [[50, 5000]])
    assert poscounts_size_factors(c) == pytest.approx([1, 1])
    assert (c / c.sum(axis=0) * 1e6)[0, 1] < 0.2 * (c / c.sum(axis=0) * 1e6)[0, 0]  # CPM calls gene 1 down
    s = scaled_per_million(c)
    assert s[0, 0] == pytest.approx(s[0, 1]) and s.sum(axis=0).mean() == pytest.approx(1e6)


def test_a_sample_with_no_reads_is_refused():
    with pytest.raises(ValueError):
        poscounts_size_factors([[1, 0], [2, 0]])


def test_size_factor_mean_gives_no_direction_where_the_total_alone_changed(tmp_path):
    genes = [f"g{i}" for i in range(10)]
    rows = ["gene\tc1\tc2\tp1\tp2"] + [f"{g}\t50\t50\t50\t50" for g in genes[:9]] + ["g9\t50\t50\t5000\t5000"]
    (tmp_path / "counts.tsv").write_text("\n".join(rows) + "\n")
    (tmp_path / "design.tsv").write_text("sample\tcondition\nc1\tcontrol\nc2\tcontrol\np1\tPEC\np2\tPEC\n")
    got = {}
    for rule in ("sample_mean", "size_factor_mean"):
        out = tmp_path / f"{rule}.tsv"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "gene_directions.py"),
                str(tmp_path / "counts.tsv"),
                str(tmp_path / "design.tsv"),
                str(out),
                rule,
            ],
            check=True,
            capture_output=True,
        )
        got[rule] = dict(line.replace('"', "").split("\t")[::4] for line in out.read_text().splitlines()[1:])
    assert got["sample_mean"]["g0"] == "down" and got["sample_mean"]["g9"] == "up"  # CPM: 9 false calls
    assert got["size_factor_mean"]["g0"] == "" and got["size_factor_mean"]["g9"] == "up"
