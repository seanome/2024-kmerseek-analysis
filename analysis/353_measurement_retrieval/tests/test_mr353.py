"""Tests for mr353.py on SYNTHETIC fixtures only. No result here is data."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mr353  # noqa: E402

_BLASTP = (
    os.environ.get("NB353_BLAST_BIN") and Path(os.environ["NB353_BLAST_BIN"]) / "blastp"
)
_BLASTP = _BLASTP or (Path(shutil.which("blastp")) if shutil.which("blastp") else None)
# Folder holding blastp: NB353_BLAST_BIN if set, else wherever blastp is on PATH.
BLAST_BIN = _BLASTP.parent if _BLASTP else Path("/nonexistent")
PY = sys.executable


def fixture(work, step):
    subprocess.run(
        [
            PY,
            str(HERE / "fixtures" / "make_synthetic.py"),
            "--work",
            str(work),
            "--step",
            step,
        ],
        check=True,
    )


def run(work, *args):
    mr353.main(["--work", str(work), *map(str, args)])


# ---- unit tests ---------------------------------------------------------------------
def pairs(s):
    return Counter(zip(s, s[1:]))


def test_dipeptide_shuffle_keeps_pairs_and_ends():
    rng = random.Random(1)
    s = "MSEYQPSLFALNPMGFSPLDGSKSTNENVSASTSTAKP"
    for _ in range(50):
        t = mr353.dipeptide_shuffle(s, rng)
        assert pairs(t) == pairs(s) and t[0] == s[0] and t[-1] == s[-1]


def test_composition_shuffle_keeps_letters():
    rng = random.Random(2)
    s = "MSEYQPSLFALNPMGFSPLD"
    assert sorted(mr353.composition_shuffle(s, rng)) == sorted(s)


def test_low_diversity_is_flagged():
    rng = random.Random(3)
    # a homopolymer has no shuffle distinct from itself
    assert (
        mr353.distinct_shuffles("A" * 20, mr353.composition_shuffle, 20, rng, 500) == []
    )
    # ACAC... has exactly one other walk with the same pairs, start and end: none distinct
    got = mr353.distinct_shuffles("AC" * 10, mr353.dipeptide_shuffle, 20, rng, 500)
    assert len(got) < 20
    # an ordinary 20-mer gives 20 distinct shuffles of both kinds
    s = "MSEYQPSLFALNPMGFSPLD"
    assert len(mr353.distinct_shuffles(s, mr353.dipeptide_shuffle, 20, rng, 2000)) == 20


def test_overlap_is_end_exclusive():
    assert mr353.overlap(0, 5, 5, 10) == 0
    assert mr353.overlap(0, 6, 5, 10) == 1
    assert mr353.overlap(3, 4, 0, 10) == 1


def test_sanborn_labels_follow_the_papers_z_test():
    plan = mr353.load_plan()
    assert mr353.sanborn_label(None, plan) == "not measured"
    assert mr353.sanborn_label(3.72, plan) == "active"
    assert mr353.sanborn_label(2.0, plan) == "weak"
    assert mr353.sanborn_label(-1.0, plan) == "inactive"


def test_self_records_are_dropped():
    plan = mr353.load_plan()
    ev = pl.DataFrame(
        [
            dict(
                evidence_id="E1",
                protein_md5="m",
                protein_accession="P1",
                start0=10,
                end0=20,
            ),
            dict(
                evidence_id="E2",
                protein_md5="m",
                protein_accession="P1",
                start0=100,
                end0=110,
            ),
        ]
    )
    q = pl.DataFrame(
        [dict(query_id="Q", source_accession="P1", source_start="5", sequence="X" * 20)]
    )
    hits = pl.DataFrame(
        [
            dict(
                workflow="B",
                setting="s",
                method="m",
                query_id=qid,
                target="m|P1",
                q_start=0,
                q_end=20,
                t_start=t0,
                t_end=t0 + 20,
                q_aln="",
                t_aln="",
                score=1.0,
                score_name="",
                evalue=None,
                evalue_name="",
            )
            for qid, t0 in (("Q", 5), ("Q", 95), ("Q__dipeptide01", 5))
        ]
    )
    cards = mr353.hits_to_cards(hits, ev, q, plan)
    got = set(cards.select("query_id", "evidence_id").iter_rows())
    # E1 is where Q came from (P1 5-25 overlaps 10-20): dropped for Q and for its shuffle.
    assert got == {("Q", "E2")}


def test_kmerseek_reader_on_real_v040_output(tmp_path):
    d = tmp_path / "km"
    d.mkdir()
    shutil.copy(
        HERE / "tests" / "kmerseek_v0.4.0_sample.csv", d / "hp_lehninger2_k12.csv"
    )
    run(tmp_path, "read-kmerseek", "--dir", d)
    h = pl.read_parquet(tmp_path / "hits" / "hits_A.parquet")
    src = pl.read_csv(d / "hp_lehninger2_k12.csv", infer_schema_length=10000)
    assert h.height == src.height == 20
    r, s = h.row(0, named=True), src.row(0, named=True)
    assert (r["q_start"], r["q_end"], r["t_start"], r["t_end"]) == (
        s["region_start"],
        s["region_end"],
        s["target_start"],
        s["target_end"],
    )
    # kmerseek's coordinates are 0-based end-exclusive: the span length is the letters' length
    assert r["q_end"] - r["q_start"] == len(r["q_aln"])
    assert "not calibrated" in r["evalue_name"]


# ---- the whole pipeline on synthetic fixtures ---------------------------------------
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    if not (BLAST_BIN / "blastp").exists():
        pytest.skip("BLAST+ not found")
    w = tmp_path_factory.mktemp("syn")
    fixture(w, "evidence")
    run(w, "import-queries", "--queue", w / "synthetic_queue.csv")
    run(w, "make-controls")
    run(w, "search-b", "--blast-bin", BLAST_BIN, "--threads", 2)
    fixture(w, "stub")
    run(w, "read-kmerseek", "--dir", w / "kmerseek_stub")
    run(w, "cards")
    fixture(w, "dev-labels")
    run(w, "tune", "--dev-labels", w / "dev_labels.csv")
    run(w, "freeze", "--queue", w / "synthetic_queue.csv")
    run(w, "session-pack")
    fixture(w, "fill-session")
    run(w, "review-pack", "--session", w / "session_filled.csv")
    fixture(w, "fill-review")
    return w


def test_copied_query_is_excluded(synthetic):
    ex = pl.read_csv(synthetic / "queries" / "excluded.csv")
    assert "SYNQ_COPY" in ex["query_id"].to_list()


def test_split_is_by_arrival_order(synthetic):
    q = pl.read_csv(synthetic / "queries" / "queries.csv", infer_schema_length=0)
    assert q.filter(pl.col("split") == "development").height == 40
    ev = q.filter(pl.col("split") == "evaluation")
    assert ev.height == 100
    assert (
        q.filter(pl.col("split") == "development")["received_at"].max()
        < ev["received_at"].min()
    )


def test_controls_are_searched_like_queries(synthetic):
    h = pl.concat(
        [pl.read_parquet(p) for p in (synthetic / "hits").glob("hits_*.parquet")],
        how="vertical_relaxed",
    )
    ids = set(h["query_id"])
    assert any("__dipeptide" in i for i in ids) and any(
        "__composition" in i for i in ids
    )


def test_blast_task_follows_query_length(synthetic):
    h = pl.read_parquet(synthetic / "hits" / "hits_B.parquet")
    q = dict(mr353.read_fasta(synthetic / "controls" / "controls.fasta"))
    for part in ("development", "evaluation"):
        q.update(mr353.read_fasta(synthetic / "queries" / f"{part}.fasta"))
    for qid, method in (
        h.filter(pl.col("method") != "exact")
        .select("query_id", "method")
        .unique()
        .iter_rows()
    ):
        assert (method == "blastp-short") == (len(q[qid]) < 30)


def test_both_workflows_get_a_frozen_choice(synthetic):
    lock = json.loads((synthetic / "evaluation_plan.lock.json").read_text())
    assert lock["choice"]["A"]["alphabet"] == "hp_lehninger2"
    assert lock["choice"]["B"]["setting"] == "exact+blast"
    assert (
        lock["choice"]["A"]["max_shuffle_cards"]
        <= mr353.load_plan()["tuning"]["max_cutoff"]
    )


def test_review_is_blind_and_orders_balanced(synthetic):
    rv = pl.read_csv(synthetic / "review" / "blind_review.csv")
    assert not any("workflow" in c for c in rv.columns)
    assert rv.select("query_id", "evidence_id").is_unique().all()
    so = pl.read_csv(synthetic / "review" / "session_order.csv")
    first = so.filter(pl.col("order_position") == 1)["workflow"].value_counts()
    assert sorted(first["count"].to_list()) == [50, 50]
    assert (so.filter(pl.col("order_position") > 1)["prior_exposure"] == "yes").all()


def test_synthetic_inputs_never_pass(synthetic):
    run(
        synthetic,
        "score",
        "--review",
        synthetic / "review_filled.csv",
        "--session",
        synthetic / "session_filled.csv",
    )
    v = json.loads((synthetic / "results" / "verdict.json").read_text())
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert any("synthetic" in r for r in v["reasons"])
    assert pl.read_csv(synthetic / "results" / "per_query_outcomes.csv").height == 100


def test_score_rules_on_unlabelled_copies(synthetic, tmp_path):
    """Strip the synthetic flags from copies (in a separate folder) to check the PASS and
    FAIL branches of the rule. This only checks arithmetic; the numbers are made up."""
    w = tmp_path / "w"
    shutil.copytree(synthetic, w)
    for f in ("review_filled.csv", "session_filled.csv"):
        df = pl.read_csv(w / f, infer_schema_length=0).with_columns(
            pl.lit("no").alias("synthetic")
        )
        df.write_csv(w / f)
    q = pl.read_csv(w / "queries" / "queries.csv", infer_schema_length=0).with_columns(
        pl.lit("no").alias("synthetic")
    )
    q.write_csv(w / "queries" / "queries.csv")
    # the lock must catch the edited queries file
    run(
        w,
        "score",
        "--review",
        w / "review_filled.csv",
        "--session",
        w / "session_filled.csv",
    )
    v = json.loads((w / "results" / "verdict.json").read_text())
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert any("changed after freezing" in r for r in v["reasons"])
    # re-freeze (the lock records the new file) and make A useful and fast on every query
    run(w, "freeze", "--queue", w / "synthetic_queue.csv")
    key = dict(csv.reader(open(w / "review" / "blind_key.csv")))
    rv = pl.read_csv(w / "review_filled.csv", infer_schema_length=0)
    rv = rv.with_columns(
        pl.lit("yes").alias("useful"), pl.lit("made-up reason").alias("reason")
    )
    rv.write_csv(w / "review_filled.csv")
    ss = pl.read_csv(w / "session_filled.csv", infer_schema_length=0).with_columns(
        pl.when(pl.col("workflow") == "A")
        .then(pl.lit("10"))
        .otherwise(pl.lit("100"))
        .alias("search_seconds"),
        pl.lit("0").alias("reading_seconds"),
        pl.lit("0").alias("verification_seconds"),
    )
    ss.write_csv(w / "session_filled.csv")
    run(
        w,
        "score",
        "--review",
        w / "review_filled.csv",
        "--session",
        w / "session_filled.csv",
    )
    v = json.loads((w / "results" / "verdict.json").read_text())
    n_a = len(
        {
            r["query_id"]
            for r in rv.iter_rows(named=True)
            if "A" in key[r["card_id"]].split(";")
        }
    )
    assert v["yields"]["A"]["n"] == n_a
    assert v["median_reduction"] == pytest.approx(0.9)
    expect = "PASS" if (n_a >= 20 and n_a >= v["yields"]["B"]["n"]) else "FAIL"
    assert v["verdict"] == expect, v
    print("unlabelled-copy check:", n_a, v["yields"]["B"]["n"], v["verdict"])
    # useful only where A found the card: B's yield cannot exceed A's, so the rule passes
    # whenever A has 20 useful queries
    only_a = rv.with_columns(
        pl.Series(
            "useful",
            ["yes" if "A" in key[c].split(";") else "no" for c in rv["card_id"]],
        )
    )
    only_a.write_csv(w / "review_filled.csv")
    run(
        w,
        "score",
        "--review",
        w / "review_filled.csv",
        "--session",
        w / "session_filled.csv",
    )
    v = json.loads((w / "results" / "verdict.json").read_text())
    assert v["yields"]["B"]["n"] <= v["yields"]["A"]["n"] == n_a
    assert v["verdict"] == ("PASS" if n_a >= 20 else "FAIL"), v
    print("A-only check:", v["yields"]["A"]["n"], v["yields"]["B"]["n"], v["verdict"])
    rv.write_csv(w / "review_filled.csv")
    # a 'yes' with no answer on whether the analyst saw it before does not count
    rv.with_columns(pl.lit("").alias("previously_seen_by_analyst")).write_csv(
        w / "review_filled.csv"
    )
    run(
        w,
        "score",
        "--review",
        w / "review_filled.csv",
        "--session",
        w / "session_filled.csv",
    )
    v = json.loads((w / "results" / "verdict.json").read_text())
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE" and v["yields"]["A"]["n"] == 0
    # a 'yes' without a reason does not count, and is reported
    rv.with_columns(pl.lit("").alias("reason")).write_csv(w / "review_filled.csv")
    run(
        w,
        "score",
        "--review",
        w / "review_filled.csv",
        "--session",
        w / "session_filled.csv",
    )
    v = json.loads((w / "results" / "verdict.json").read_text())
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE" and v["yields"]["A"]["n"] == 0


def test_workflow_c_retained_items_enter_the_blind_pool(synthetic, tmp_path):
    w = tmp_path / "c"
    shutil.copytree(synthetic, w)
    run(w, "session-pack", "--with-c")
    so = pl.read_csv(w / "review" / "session_order.csv", infer_schema_length=0)
    first = (
        so.filter(pl.col("order_position") == "1")["workflow"]
        .value_counts()["count"]
        .to_list()
    )
    assert sum(first) == 100 and max(first) - min(first) <= 1
    qid = so["query_id"][0]
    so = so.with_columns(
        pl.when((pl.col("query_id") == qid) & (pl.col("workflow") == "C"))
        .then(pl.lit("SYN:E001; Doe 2020 J Synth 1:1"))
        .otherwise(pl.lit(""))
        .alias("retained_evidence_ids")
    )
    so.write_csv(w / "session.csv")
    run(w, "review-pack", "--session", w / "session.csv")
    key = dict(csv.reader(open(w / "review" / "blind_key.csv")))
    rv = pl.read_csv(w / "review" / "blind_review.csv", infer_schema_length=0)
    c_cards = [r for r in rv.iter_rows(named=True) if "C" in key[r["card_id"]]]
    assert {r["query_id"] for r in c_cards} == {qid}
    assert {(r["evidence_id"] or r["references"]) for r in c_cards} == {
        "SYN:E001",
        "Doe 2020 J Synth 1:1",
    }
