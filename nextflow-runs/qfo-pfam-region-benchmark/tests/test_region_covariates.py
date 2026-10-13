"""Region-level pLDDT and disorder: the slice, and what happens when it cannot be taken.

The per-protein columns these sit beside cannot answer "was this DOMAIN confident /
disordered". An ordered domain inside a mostly-disordered protein takes the protein's
number and reads as disordered, which is the failure these tables exist to remove.
"""
import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

BIN = Path(__file__).resolve().parent.parent / "bin"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bqc = _load("build_query_covariates")
pdm = _load("predict_disorder_metapredict")


def truth_of(rows):
    return pl.DataFrame(
        rows, schema=["accession", "domain_start", "domain_end"], orient="row")


def test_region_plddt_takes_the_domains_own_residues_not_the_proteins():
    # 1-100 confident, 101-200 not. The protein mean is 50, which is neither.
    track = [100.0] * 100 + [0.0] * 100
    got = bqc.domain_plddt(
        truth_of([("P1", 1, 100), ("P1", 101, 200)]), {"P1": track}
    ).sort("domain_start")
    assert got["mean_plddt_region"].to_list() == [100.0, 0.0]
    assert got["frac_plddt_lt70_region"].to_list() == [0.0, 1.0]
    assert got["n_residues_region_plddt"].to_list() == [100, 100]


def test_region_plddt_coordinates_are_one_based_and_inclusive():
    """Truth carries Pfam envelope coordinates, which every other table joins on as 1-based
    inclusive. Off by one here would shift every region by a residue and never error."""
    track = [1.0, 2.0, 3.0, 4.0, 5.0]
    got = bqc.domain_plddt(truth_of([("P1", 2, 4)]), {"P1": track})
    assert got["n_residues_region_plddt"][0] == 3
    assert got["mean_plddt_region"][0] == pytest.approx(3.0)   # residues 2,3,4
    assert got["min_plddt_region"][0] == pytest.approx(2.0)


@pytest.mark.parametrize("case,tracks,rows", [
    ("no model at all", {}, [("P1", 1, 10)]),
    ("interval past the modelled length", {"P1": [50.0] * 5}, [("P1", 1, 10)]),
])
def test_region_plddt_keeps_unmeasurable_domains_as_nulls(case, tracks, rows):
    """Dropped rows would silently shrink the denominator of any figure on this axis, and
    a shrinking denominator is exactly the defect the identity panel had to print n for."""
    got = bqc.domain_plddt(truth_of(rows), tracks)
    assert got.height == 1, case
    assert got["mean_plddt_region"][0] is None, case
    assert got["n_residues_region_plddt"][0] == 0, case


def test_region_disorder_separates_two_domains_of_one_protein(tmp_path):
    truth = tmp_path / "truth.parquet"
    truth_of([("P1", 1, 4), ("P1", 5, 8)]).write_parquet(truth)
    out = tmp_path / "dom.parquet"
    n = pdm.write_domain_disorder(
        truth, out, {"P1": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]}, thr=0.5)
    got = pl.read_parquet(out).sort("domain_start")
    assert n == 2
    assert got["mean_disorder_region"].to_list() == [0.0, 1.0]
    assert got["disorder_fraction_region"].to_list() == [0.0, 1.0]


def test_region_disorder_joins_on_the_same_key_as_identity(tmp_path):
    """(accession, domain_start, domain_end) -- the key attach_identity already uses, so
    the two region axes can be put on one figure without a second join convention."""
    truth = tmp_path / "truth.parquet"
    truth_of([("P1", 1, 4)]).write_parquet(truth)
    out = tmp_path / "dom.parquet"
    pdm.write_domain_disorder(truth, out, {"P1": [0.2] * 4}, thr=0.5)
    cols = pl.read_parquet(out).columns
    for k in ("accession", "domain_start", "domain_end"):
        assert k in cols


# --- the scoring side: region covariates become their own stratum axis -------------

ed = _load("evaluate_domain_calls")


def truth_frame():
    """Two domains of ONE protein, one confident and ordered, one not. The whole point of
    the region axes is that these must land in different strata."""
    return pl.DataFrame({
        "accession": ["P1", "P1"],
        "pfam_id": ["PF1", "PF2"],
        "domain_start": [1, 101],
        "domain_end": [100, 200],
    })


def region_table(col, values):
    return pl.DataFrame({
        "accession": ["P1", "P1"],
        "domain_start": [1, 101],
        "domain_end": [100, 200],
        col: values,
    })


def test_two_domains_of_one_protein_land_in_different_strata():
    got = ed.attach_region_covariates(truth_frame(), {
        "mean_plddt_region": region_table("mean_plddt_region", [96.0, 33.0]),
        "mean_disorder_region": region_table("mean_disorder_region", [0.02, 0.55]),
    })
    assert got["stratum_plddt_region"].to_list() == ["95-97", "30-45"]
    assert got["stratum_disorder_region"].n_unique() == 2


def test_an_unmeasurable_domain_gets_no_stratum_rather_than_a_zero_one():
    got = ed.attach_region_covariates(truth_frame(), {
        "mean_plddt_region": region_table("mean_plddt_region", [96.0, None]),
        "mean_disorder_region": None,
    })
    assert got["stratum_plddt_region"].to_list() == ["95-97", None]
    # A missing table must not invent a stratum for every domain either.
    assert got["stratum_disorder_region"].null_count() == got.height


def test_region_stratum_mean_averages_instances_not_proteins():
    """The per-protein rule that stratum_value_mean uses everywhere else would dedupe these
    two rows to one and report an arbitrary one of the two values."""
    t = ed.attach_region_covariates(truth_frame(), {
        "mean_plddt_region": region_table("mean_plddt_region", [90.0, 80.0]),
        "mean_disorder_region": None,
    })
    got = ed.stratum_value_mean(t, "plddt_region")
    assert got == pytest.approx(85.0)


def test_protein_level_axes_still_dedupe_by_protein():
    """Guard on the branch: the region exception must not leak into the protein axes, where
    a twelve-domain protein would then count twelve times."""
    t = pl.DataFrame({
        "accession": ["P1", "P1", "P2"],
        "mean_plddt": [90.0, 90.0, 60.0],
    })
    assert ed.stratum_value_mean(t, "plddt") == pytest.approx(75.0)
