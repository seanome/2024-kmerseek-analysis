"""Embedding search calls (bin/embedding_calls.py) and their rows.

tests/fixtures/embedding is hand-built in TM-Vec's, DeepBLAST's and PLMSearch's output
layouts from the committed MMseqs2 fixture (tests/make_embedding_fixture.sh); it tests
the parsers and wiring, not the tools.
"""

import gzip
import json
import shutil
import sys

import polars as pl
import pytest

from conftest import BIN, FIXTURES, run

sys.path.insert(0, str(BIN))
import annotator_tool_rows as atr  # noqa: E402
import embedding_calls as ec  # noqa: E402

EMB = FIXTURES / "embedding"
YEAST_MAP = FIXTURES / "annotations" / "yeast_pfam_domains.parquet"


def regions(path) -> list[list[str]]:
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    d = tmp_path_factory.mktemp("short")
    run(
        "embedding_calls.py",
        "short-headers",
        "--fasta",
        FIXTURES / "qfo" / "human.fasta",
        "--out",
        d / "q.fasta",
    )
    run(
        "embedding_calls.py",
        "short-headers",
        "--fasta",
        FIXTURES / "qfo" / "yeast.fasta",
        "--out",
        d / "t.fasta",
    )
    return d


def test_short_headers_keep_the_accession(short):
    first = (short / "q.fasta").read_text().splitlines()[0]
    assert first == ">A0A024RBG1"


def test_short_headers_refuse_a_repeated_accession(tmp_path):
    fa = tmp_path / "x.fasta"
    fa.write_text(">sp|P1|A_HUMAN\nMA\n>tr|P1|B_HUMAN\nMC\n")
    proc = __import__("subprocess").run(
        [
            sys.executable,
            str(BIN / "embedding_calls.py"),
            "short-headers",
            "--fasta",
            str(fa),
            "--out",
            str(tmp_path / "o.fasta"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "appears twice" in proc.stderr


def test_deepblast_alignments_are_located_in_the_full_sequences():
    qseqs = {"q": "MMMACDEFGHKKK"}
    tseqs = {"t": "PPACDFGHPP"}
    best = pl.DataFrame({"query": ["q"], "target": ["t"], "score": [0.8]})
    pairs = [("q", "ACDEFGH", "t", "ACD-FGH"), ("q", "AC", "other", "AC")]
    hits, unplaced = ec.alignment_hits(pairs, best, qseqs, tseqs)
    assert unplaced == 0 and hits.height == 1
    h = hits.row(0, named=True)
    assert (h["qstart"], h["qend"], h["tstart"], h["tend"]) == (4, 10, 3, 8)
    assert h["bits"] == 0.8


def test_tmvec_without_alignments_claims_no_position(short, tmp_path):
    run(
        "embedding_calls.py",
        "tmvec",
        "--tabular",
        EMB / "tmvec.tabular.tsv",
        "--queries",
        short / "q.fasta",
        "--domain-map",
        YEAST_MAP,
        "--setting",
        "cath",
        "--species",
        "yeast",
        "--out",
        tmp_path / "r.tsv.gz",
        "--qc-out",
        tmp_path / "qc.json",
    )
    lengths = {k: len(v) for k, v in ec.abh.read_fasta(short / "q.fasta").items()}
    r = regions(tmp_path / "r.tsv.gz")
    assert r and all(x[2] == "1" and int(x[3]) == lengths[x[0]] for x in r)
    assert (
        "whole query" in json.loads((tmp_path / "qc.json").read_text())["call_interval"]
    )


def test_both_parsers_reproduce_the_committed_fixture(short, tmp_path):
    run(
        "embedding_calls.py",
        "tmvec",
        "--tabular",
        EMB / "tmvec.tabular.tsv",
        "--alignments",
        EMB / "tmvec.alignments.txt",
        "--targets",
        short / "t.fasta",
        "--queries",
        short / "q.fasta",
        "--domain-map",
        YEAST_MAP,
        "--setting",
        "cath_deepblast",
        "--species",
        "yeast",
        "--out",
        tmp_path / "t.tsv.gz",
        "--qc-out",
        tmp_path / "t.json",
    )
    run(
        "embedding_calls.py",
        "plmsearch",
        "--results",
        EMB / "plmsearch.results.tsv",
        "--queries",
        short / "q.fasta",
        "--domain-map",
        YEAST_MAP,
        "--setting",
        "ss_predictor",
        "--species",
        "yeast",
        "--out",
        tmp_path / "p.tsv.gz",
        "--qc-out",
        tmp_path / "p.json",
    )
    assert regions(tmp_path / "t.tsv.gz") == regions(
        EMB / "tmvec.cath_deepblast.yeast.regions.tsv.gz"
    )
    assert regions(tmp_path / "p.tsv.gz") == regions(
        EMB / "plmsearch.ss_predictor.yeast.regions.tsv.gz"
    )


def test_plmsearch_takes_each_querys_highest_score():
    df = pl.DataFrame(
        {"query": ["q", "q", "q"], "target": ["b", "a", "c"], "score": [0.9, 0.9, 0.2]}
    )
    assert ec.best_pairs(df)["target"].to_list() == ["a"]


def test_rows_cards_and_gpu_hours_per_target(tmp_path):
    res = tmp_path / "results"
    for sub in ("metrics", "calls", "regions"):
        (res / sub).mkdir(parents=True)
    for tool, setting in (("tmvec", "cath_deepblast"), ("plmsearch", "ss_predictor")):
        for species in ("yeast", "mouse"):
            stem = f"pfam.{tool}.{setting}.{species}"
            m = pl.read_parquet(EMB / f"pfam.{tool}.{setting}.yeast.metrics.parquet")
            m.with_columns(pl.lit(species).alias("species")).write_parquet(
                res / "metrics" / f"{stem}.metrics.parquet"
            )
            shutil.copy(
                EMB / f"pfam.{tool}.{setting}.yeast.calls.parquet",
                res / "calls" / f"{stem}.calls.parquet",
            )
            shutil.copy(
                EMB / f"{tool}.{setting}.yeast.calls_qc.json",
                res / "regions" / f"{tool}.{setting}.{species}.calls_qc.json",
            )
    head = (
        "task_id\tprocess\ttag\tstatus\tcpus\trealtime\tpeak_rss\tcontainer\tcomplete\n"
    )
    trace = [
        ("1", "TMVEC:tmvecSearch", "tmvec yeast", "30m", "4"),
        ("2", "TMVEC:tmvecSearch", "tmvec mouse", "1h", "5"),
        ("3", "PLMSEARCH:plmsearchEmbed", "plmsearch embed query", "6m", "6"),
        ("4", "PLMSEARCH:plmsearchEmbed", "plmsearch embed yeast", "6m", "7"),
        ("5", "scoreAnnotatorTool", "tmvec cath_deepblast yeast", "1m", "8"),
        ("6", "scoreAnnotatorTool", "tmvec cath_deepblast mouse", "1m", "9"),
    ]
    (tmp_path / "trace.txt").write_text(
        head
        + "".join(
            f"{i}\t{p}\t{t}\tCOMPLETED\t8\t{w}\t2 GB\timg\t{c}\n"
            for i, p, t, w, c in trace
        )
    )
    out = tmp_path / "lb"
    run(
        "annotator_tool_rows.py",
        "--results",
        res,
        "--trace",
        tmp_path / "trace.txt",
        "--out",
        out,
    )

    rows = pl.read_parquet(out / "rows.parquet")
    assert set(rows.filter(pl.col("tool") == "tmvec")["target_db"]) == {
        "QfO yeast",
        "QfO mouse",
    }
    assert set(rows.filter(pl.col("tool") == "plmsearch")["metric"]) == {"family_fmax"}

    comp = pl.read_parquet(out / "compute.parquet")
    tm = comp.filter(pl.col("step") == "tmvecSearch").row(0, named=True)
    assert tm["setting"] == "cath_deepblast" and tm["n_target_dbs"] == 2
    assert tm["gpu_h"] == pytest.approx(1.5)
    emb = comp.filter(pl.col("step") == "plmsearchEmbed").row(0, named=True)
    assert (
        emb["n_tasks"] == 2 and emb["n_target_dbs"] == 1
    )  # the query set is no target
    score = comp.filter((pl.col("step") == "scoreAnnotatorTool")).row(0, named=True)
    assert score["n_tasks"] == 2 and score["gpu_h"] is None

    card = json.loads((out / "cards" / "tmvec.cath_deepblast.json").read_text())
    assert card["needs_gpu"] is True
    assert set(card["scores_by_target"]) == {"yeast", "mouse"}
    assert card["target_db"] == ["QfO mouse", "QfO yeast"]
    pcard = json.loads((out / "cards" / "plmsearch.ss_predictor.json").read_text())
    assert pcard["scores_by_target"]["yeast"]["median_iou_tp"] == "not applicable"
    pile = pl.read_parquet(out / "pileup_calls.parquet")
    assert set(pile["tool"]) == {"tmvec"}
    assert set(pile["target"]) == {"QfO yeast", "QfO mouse"}
    assert atr.TOOL_INFO["plmsearch"]["has_query_coordinates"] is False
