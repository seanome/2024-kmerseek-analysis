"""The DisProt landing reduction, on small hand-made region files.

Each test builds one arm's output the way the pipeline writes it (a kmerseek region
parquet, or an aligner's 8-column TSV.gz) and checks one rule of the reduction: the
coordinate shift, the list cut, the transfer rule, the two label rules, and which score
kmerseek ranks by at each region length.
"""

import gzip
import sys
from pathlib import Path

import polars as pl
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import reduce_disprot_region_landing as rd  # noqa: E402

sys.path.insert(0, str(rd.QFO_BIN))
import evaluate_domain_calls as ev  # noqa: E402


def regions(rows):
    """Human DisProt regions, given 1-based inclusive like DisProt, stored as the script does."""
    return pl.DataFrame(
        [
            dict(query_acc=a, term_id=t, true_start=s - 1, true_end=e)
            for a, t, s, e in rows
        ]
    )


def tmap(rows):
    return pl.DataFrame(
        [
            dict(target_acc=a, t_term=t, t_feat_start=s - 1, t_feat_end=e)
            for a, t, s, e in rows
        ]
    )


def aligner_file(tmp_path, rows, tool="hmmer3_phmmer"):
    """rows: (query, target, qstart, qend, tstart, tend, score), 1-based inclusive."""
    d = tmp_path / "regions" / tool
    d.mkdir(parents=True)
    p = d / f"human_vs_yeast.{tool}.tsv.gz"
    with gzip.open(p, "wt") as fh:
        for q, t, qs, qe, ts, te, sc in rows:
            fh.write(f"sp|{q}|X\tsp|{t}|Y\t{qs}\t{qe}\t{ts}\t{te}\t{sc}\t1e-5\n")
    return dict(path=p, target="yeast", tool=tool, arm=tool)


def kmerseek_file(tmp_path, rows):
    """rows: (query, target, qstart, qend, tstart, tend, mean_idf, evalue), 0-based end-excluded."""
    d = tmp_path / "kmerseek"
    d.mkdir(parents=True)
    p = d / "human_vs_yeast.hp_pbotc_1st_ed2.k19.lctrue.regions.parquet"
    pl.DataFrame(
        [
            dict(
                query_name=f"sp|{q}|X",
                target_name=f"sp|{t}|Y",
                region_start=qs,
                region_end=qe,
                target_start=ts,
                target_end=te,
                region_mean_idf=idf,
                region_evalue=e,
            )
            for q, t, qs, qe, ts, te, idf, e in rows
        ]
    ).write_parquet(p)
    return dict(
        path=p,
        target="yeast",
        tool="kmerseek",
        arm="kmerseek.hp_pbotc_1st_ed2_k19_lcTrue",
    )


def row(out, rule):
    r = out.filter(pl.col("label_rule") == rule)
    assert r.height <= 1
    return r.row(0, named=True) if r.height else None


def test_an_aligner_call_on_the_region_lands_with_iou_one(tmp_path):
    # phmmer reports 11-40 on the query and 101-130 on the target, 1-based inclusive,
    # exactly the DisProt intervals on both sides.
    job = aligner_file(tmp_path, [("Q1", "T1", 11, 40, 101, 130, 50.0)])
    out, *_ = rd.reduce_one(
        job,
        regions([("Q1", "IDPO:1", 11, 40)]),
        tmap([("T1", "IDPO:1", 101, 130)]),
        ev,
        0.5,
        1000,
    )
    r = row(out, "same_term")
    assert r["landed"] and r["best_iou"] == pytest.approx(1.0)
    assert r["land_target_acc"] == "T1"


def test_same_term_needs_the_same_function_and_any_term_does_not(tmp_path):
    job = aligner_file(tmp_path, [("Q1", "T1", 11, 40, 101, 130, 50.0)])
    out, *_ = rd.reduce_one(
        job,
        regions([("Q1", "IDPO:1", 11, 40)]),
        tmap([("T1", "IDPO:2", 101, 130)]),
        ev,
        0.5,
        1000,
    )
    assert row(out, "same_term") is None
    assert row(out, "any_term")["landed"]


def test_a_call_covering_under_half_of_the_target_region_carries_no_label(tmp_path):
    # Target region 101-200 (100 aa); the call covers 101-140, 40% of it.
    job = aligner_file(tmp_path, [("Q1", "T1", 11, 40, 101, 140, 50.0)])
    out, hits, *_ = rd.reduce_one(
        job,
        regions([("Q1", "IDPO:1", 11, 40)]),
        tmap([("T1", "IDPO:1", 101, 200)]),
        ev,
        0.5,
        1000,
    )
    assert out.height == 0
    # ...but the tool did report something on the human region.
    assert hits.row(0, named=True)["n_hits"] == 1


def test_the_list_cut_drops_targets_below_rank_L(tmp_path):
    # T2 is the only labelled target and ranks second; at L = 1 it is cut before transfer.
    job = aligner_file(
        tmp_path,
        [
            ("Q1", "T1", 200, 260, 1, 61, 90.0),
            ("Q1", "T2", 11, 40, 101, 130, 50.0),
        ],
    )
    args = (
        regions([("Q1", "IDPO:1", 11, 40)]),
        tmap([("T2", "IDPO:1", 101, 130)]),
        ev,
        0.5,
    )
    out_l1, _, ranks, _ = rd.reduce_one(job, *args, 1)
    assert out_l1.height == 0
    assert ranks.filter(pl.col("target_acc") == "T2")["rank"].item() == 2
    out_l2, *_ = rd.reduce_one(job, *args, 2)
    assert row(out_l2, "same_term")["landed"]


def test_kmerseek_ranks_short_regions_by_mean_idf_and_long_ones_by_evalue(tmp_path):
    # Two human regions: 20 aa (short) and 60 aa (long). Two target proteins, each carrying
    # the labels for both. T_idf has the higher mean IDF and no E-value; T_ev the reverse.
    # At L = 1 the short region can only be reached through T_idf, the long one through T_ev.
    rows = []
    for t, idf, e in (("T_idf", 9.0, float("inf")), ("T_ev", 1.0, 1e-20)):
        rows += [
            ("Q1", t, 10, 30, 100, 120, idf, e),  # the short region, 0-based 10..30
            ("Q1", t, 100, 160, 300, 360, idf, e),  # the long one, 0-based 100..160
        ]
    job = kmerseek_file(tmp_path, rows)
    labels = []
    for t in ("T_idf", "T_ev"):
        labels += [(t, "IDPO:1", 101, 120), (t, "IDPO:2", 301, 360)]
    out, _, ranks, rho = rd.reduce_one(
        job,
        regions([("Q1", "IDPO:1", 11, 30), ("Q1", "IDPO:2", 101, 160)]),
        tmap(labels),
        ev,
        0.5,
        1,
    )
    same = out.filter(pl.col("label_rule") == "same_term")
    short = same.filter(pl.col("term_id") == "IDPO:1").row(0, named=True)
    long_ = same.filter(pl.col("term_id") == "IDPO:2").row(0, named=True)
    assert short["rank_by"] == "mean_idf" and short["land_target_acc"] == "T_idf"
    assert long_["rank_by"] == "evalue_score" and long_["land_target_acc"] == "T_ev"
    # An infinite E-value has no rank at all, not a bad one.
    assert ranks.filter(
        (pl.col("rank_by") == "evalue_score") & (pl.col("target_acc") == "T_idf")
    ).is_empty()
    assert set(rho["rank_by"]) == {"mean_idf", "evalue_score"}
