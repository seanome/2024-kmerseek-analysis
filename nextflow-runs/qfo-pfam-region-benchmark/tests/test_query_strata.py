"""Query-side strata: the bins match RULES.md, and every axis partitions the whole set.

RULES.md (results/leaderboard/query_strata/) was written before any value was computed.
These tests hold the scorer to it: each edge lands in the bin the rules name, and on a real
scoring run the bins of every axis add up to the ungrouped row.
"""
import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import score, write_perfect_regions

BIN = Path(__file__).resolve().parent.parent / "bin"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


edc = _load("evaluate_domain_calls")

BREADTH = pl.Struct({"pfam_id": pl.String, "n_superkingdoms": pl.Int64,
                     "n_phyla": pl.Int64})


def one_protein(**cols):
    base = {"accession": "P1", "length_aa": 200, "seg_fraction": 0.0,
            "deeptmhmm_type": "GLOB", "busco_status": "complete",
            "nearest_identity_yeast": 95.0, "pfam_family_breadth": None}
    base.update(cols)
    schema = {"accession": pl.String, "length_aa": pl.Int64, "seg_fraction": pl.Float64,
              "deeptmhmm_type": pl.String, "busco_status": pl.String,
              "nearest_identity_yeast": pl.Float64,
              "pfam_family_breadth": pl.List(BREADTH)}
    return pl.DataFrame([base], schema=schema)


def truth_rows(rows):
    return pl.DataFrame(rows, schema=["accession", "pfam_id", "domain_start", "domain_end"],
                        orient="row")


@pytest.mark.parametrize("value,label", [
    (None, "no hit"), (0.0, "<30%"), (29.99, "<30%"), (30.0, "30-50%"), (49.9, "30-50%"),
    (50.0, "50-90%"), (89.99, "50-90%"), (90.0, ">=90%"), (100.0, ">=90%")])
def test_nearest_identity_edges(value, label):
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]),
                                one_protein(nearest_identity_yeast=value), "yeast")
    assert t["stratum_nearest_identity"].to_list() == [label]


def test_nearest_identity_reads_this_runs_target_only():
    cov = one_protein(nearest_identity_yeast=95.0).with_columns(
        pl.lit(10.0).alias("nearest_identity_ecoli"))
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]), cov, "ecoli")
    assert t["stratum_nearest_identity"].to_list() == ["<30%"]


@pytest.mark.parametrize("value,label", [
    (0.0, "<5%"), (0.0499, "<5%"), (0.05, "5-20%"), (0.20, "5-20%"), (0.2001, ">20%"),
    (1.0, ">20%")])
def test_low_complexity_edges(value, label):
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]),
                                one_protein(seg_fraction=value), "yeast")
    assert t["stratum_low_complexity"].to_list() == [label]


@pytest.mark.parametrize("value,label", [
    (1, "<100"), (99, "<100"), (100, "100-300"), (299, "100-300"), (300, "300-1000"),
    (1000, "300-1000"), (1001, ">1000")])
def test_protein_length_edges(value, label):
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]),
                                one_protein(length_aa=value), "yeast")
    assert t["stratum_protein_length"].to_list() == [label]


def test_region_length_is_inclusive_residue_count():
    # 1-29 is 29 residues, 1-30 is 30, 1-100 is 100, 1-101 is 101.
    t = edc.attach_query_strata(
        truth_rows([("P1", "PF1", 1, 29), ("P1", "PF1", 1, 30), ("P1", "PF1", 1, 100),
                    ("P1", "PF1", 1, 101)]), one_protein(), "yeast")
    assert t["stratum_region_length"].to_list() == ["<30", "30-100", "30-100", ">100"]


@pytest.mark.parametrize("dtm,label", [
    ("TM", "alpha-helical TM"), ("SP+TM", "alpha-helical TM"), ("BETA", "beta-barrel"),
    ("SP", "neither"), ("GLOB", "neither")])
def test_membrane_classes(dtm, label):
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]),
                                one_protein(deeptmhmm_type=dtm), "yeast")
    assert t["stratum_membrane"].to_list() == [label]


def test_family_breadth_is_per_region_and_pfam_only():
    cov = one_protein(pfam_family_breadth=[
        {"pfam_id": "PF1", "n_superkingdoms": 3, "n_phyla": 40},
        {"pfam_id": "PF2", "n_superkingdoms": 1, "n_phyla": 1},
        {"pfam_id": "PF3", "n_superkingdoms": 0, "n_phyla": 0},
    ])
    t = edc.attach_query_strata(
        truth_rows([("P1", "PF1", 1, 50), ("P1", "PF2", 60, 120), ("P1", "PF3", 130, 160),
                    ("P1", "PF9", 170, 200), ("P1", "TRANSMEM", 5, 25)]), cov, "yeast")
    assert t["stratum_family_superkingdoms"].to_list() == ["3", "1", "not found",
                                                           "not found", None]
    assert t["stratum_family_phyla"].to_list() == [">20", "1", "not found", "not found",
                                                   None]


@pytest.mark.parametrize("n,label", [(1, "1"), (2, "2-5"), (5, "2-5"), (6, "6-20"),
                                     (20, "6-20"), (21, ">20")])
def test_family_phyla_edges(n, label):
    cov = one_protein(pfam_family_breadth=[
        {"pfam_id": "PF1", "n_superkingdoms": 1, "n_phyla": n}])
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]), cov, "yeast")
    assert t["stratum_family_phyla"].to_list() == [label]


def test_no_strata_columns_means_no_new_axes():
    t = edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]),
                                pl.DataFrame({"accession": ["P1"]}), "yeast")
    assert [c for c in t.columns if c.startswith("stratum_")] == []
    assert edc.attach_query_strata(truth_rows([("P1", "PF1", 1, 50)]), None,
                                   "yeast").columns == ["accession", "pfam_id",
                                                        "domain_start", "domain_end"]


NEW_AXES = ["nearest_identity", "membrane", "low_complexity", "protein_length",
            "region_length", "family_superkingdoms", "family_phyla", "busco"]


def test_bins_partition_the_whole_set_on_a_real_scoring_run(pfam_truth, tmp_path):
    """Score the fixture's perfect arm with every new axis, then add the bins back up."""
    truth = pl.read_parquet(pfam_truth["truth"])
    accs = truth["accession"].unique().sort().to_list()
    fams = (truth.group_by("accession").agg(pl.col("pfam_id").unique().sort()))
    fam_of = dict(zip(fams["accession"].to_list(), fams["pfam_id"].to_list()))
    # Spread the proteins over every bin, so no axis collapses to one value.
    rows = []
    for i, a in enumerate(accs):
        rows.append({
            "accession": a,
            "length_aa": [50, 150, 500, 2000][i % 4],
            "seg_fraction": [0.0, 0.1, 0.5][i % 3],
            "deeptmhmm_type": ["TM", "BETA", "GLOB", "SP+TM", "SP"][i % 5],
            "busco_status": ["complete", "fragmented", "not a BUSCO gene"][i % 3],
            "nearest_identity_yeast": [None, 10.0, 40.0, 70.0, 99.0][i % 5],
            "pfam_family_breadth": [
                {"pfam_id": f, "n_superkingdoms": 1 + (j % 3), "n_phyla": [1, 3, 9, 50][j % 4]}
                for j, f in enumerate(fam_of.get(a, []))],
        })
    cov = pl.DataFrame(rows, schema={
        "accession": pl.String, "length_aa": pl.Int64, "seg_fraction": pl.Float64,
        "deeptmhmm_type": pl.String, "busco_status": pl.String,
        "nearest_identity_yeast": pl.Float64, "pfam_family_breadth": pl.List(BREADTH)})
    cov_path = tmp_path / "covariates.parquet"
    cov.write_parquet(cov_path)
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], tmp_path / "regions.tsv")
    m = score(pfam_truth["truth"], pfam_truth["map"], tmp_path / "regions.tsv",
              tmp_path / "run", "pfam", "--covariates", cov_path)
    m = m.filter(pl.col("split") == "all")
    whole = m.filter(pl.col("stratum_axis") == "all").select(
        "dedup_transfers", pl.col("n_truth_instances").alias("whole"))
    assert whole.height >= 1
    got = set(m["stratum_axis"].unique().to_list())
    assert set(NEW_AXES) <= got, f"missing axes: {set(NEW_AXES) - got}"
    sums = (m.filter(pl.col("stratum_axis").is_in(NEW_AXES))
            .group_by("dedup_transfers", "stratum_axis")
            .agg(pl.col("n_truth_instances").sum().alias("sum"))
            .join(whole, on="dedup_transfers"))
    bad = sums.filter(pl.col("sum") != pl.col("whole"))
    assert bad.height == 0, bad
