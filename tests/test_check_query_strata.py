"""check_query_strata.py's long-table check: bins must add up to the whole set.

Runs the checker's sum logic on a made-up results table: one row group whose bins add
up, Pfam-N protein-level axes that hold only queries (listed apart), then one bin
changed so the group no longer adds up.
"""

import importlib.util
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cqs", ROOT / "scripts/query_strata/check_query_strata.py"
)
cqs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cqs)


def rows(truth, axis_bins, whole):
    base = dict(
        experiment="qfo_region_midi_plus",
        tool="t",
        setting="s",
        query_set="q",
        target_db="QfO mouse",
        truth_source=truth,
        dedup_or_raw="raw",
        metric="fmax",
        value=0.5,
    )
    out = [base | dict(stratum_axis="all", stratum_bin="all", n_instances=whole)]
    for axis, bins in axis_bins.items():
        out += [
            base | dict(stratum_axis=axis, stratum_bin=b, n_instances=n)
            for b, n in bins
        ]
    return out


def run(tmp_path, table):
    p = tmp_path / "results.parquet"
    pl.DataFrame(table).write_parquet(p)
    strata = tmp_path / "strata"
    strata.mkdir(exist_ok=True)
    return cqs.long_table_check(p)


def test_bins_that_add_up_pass_and_pfamn_queries_only_is_apart(tmp_path):
    t = rows(
        "Pfam-A domain instances",
        {
            "membrane": [("neither", 7), ("alpha-helical TM", 3)],
            "region_length": [("<30", 4), (">100", 6)],
        },
        10,
    )
    t += rows(
        "Pfam-N domain instances",
        {
            "membrane": [("neither", 2), ("alpha-helical TM", 1)],
            "busco": [("complete", 3)],
            "region_length": [("<30", 40), (">100", 60)],
        },
        100,
    )
    bad, _ = run(tmp_path, t)
    assert bad == 0


def test_a_bin_that_does_not_add_up_fails(tmp_path):
    t = rows(
        "Pfam-A domain instances",
        {"membrane": [("neither", 7), ("alpha-helical TM", 2)]},
        10,
    )
    bad, _ = run(tmp_path, t)
    assert bad == 1


def test_pfamn_protein_axes_that_disagree_fail(tmp_path):
    t = rows(
        "Pfam-N domain instances",
        {
            "membrane": [("neither", 2), ("alpha-helical TM", 1)],
            "busco": [("complete", 4)],
        },
        100,
    )
    bad, _ = run(tmp_path, t)
    assert bad == 1
