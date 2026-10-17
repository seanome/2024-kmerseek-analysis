"""A truth set built outside the run must be cut to the run's query proteins before scoring.

Pfam-N truth is built once from the full annotations, so it holds every human Pfam protein:
44_315 regions on 18_724 proteins. The mini, midi and midi-plus runs search 200, 964 and
998 of them, and until this test existed every one of them scored Pfam-N recall against all
44_315. On midi-plus only 2_218 of those regions sit on a searched protein, so recall and
the recall half of Fmax were about 20 times too low for every tool.

Two things are checked:

  * restrict_truth_to_queries.py keeps exactly the rows on proteins in the run's
    human_pfam_domains.parquet, keeps the columns, and reports before and after counts
  * main.nf sends every pre-built truth set through restrictTruthToQueries and never mixes
    a pre-built human truth file into truth_sets directly
"""

import json
import re
from pathlib import Path

import polars as pl
import pytest

from conftest import FIXTURES, run, write_perfect_regions

MAIN_NF = Path(__file__).resolve().parents[1] / "main.nf"


def _truth_with_outsiders(path: Path) -> tuple[set[str], int]:
    """Fixture queries' first 10 proteins, plus 3 proteins the run never searched."""
    queries = (
        pl.read_parquet(FIXTURES / "annotations" / "human_pfam_domains.parquet")[
            "accession"
        ]
        .unique()
        .sort()
        .head(10)
        .to_list()
    )
    outsiders = ["Q00000", "Q00001", "Q00002"]
    rows = [(a, "PF00001", 10, 60) for a in queries] + [
        (a, "PF00002", 5, 90) for a in outsiders for _ in range(2)
    ]
    pl.DataFrame(
        rows,
        schema=["accession", "pfam_id", "domain_start", "domain_end"],
        orient="row",
    ).with_columns(pl.lit(None, dtype=pl.Int32).alias("protein_length")).write_parquet(
        path
    )
    return set(queries), len(queries)


def test_keeps_only_query_proteins(tmp_path):
    full = tmp_path / "full" / "human_pfamn_truth.parquet"
    full.parent.mkdir()
    queries, n_query_rows = _truth_with_outsiders(full)
    out = tmp_path / "human_pfamn_truth.parquet"
    summary = tmp_path / "summary.json"
    run(
        "restrict_truth_to_queries.py",
        "--truth",
        full,
        "--annotations",
        FIXTURES / "annotations",
        "--label",
        "pfamn",
        "--out",
        out,
        "--summary-out",
        summary,
    )

    kept = pl.read_parquet(out)
    assert set(kept["accession"]) == queries
    assert kept.height == n_query_rows
    assert kept.columns == pl.read_parquet(full).columns

    s = json.loads(summary.read_text())
    assert s["before"] == {"n_regions": n_query_rows + 6, "n_proteins": 13}
    assert s["after"] == {"n_regions": n_query_rows, "n_proteins": 10}
    assert s["n_query_proteins"] == 60


def test_no_overlap_gives_empty_truth(tmp_path):
    full = tmp_path / "t.parquet"
    pl.DataFrame(
        {
            "accession": ["Q00000"],
            "pfam_id": ["PF00001"],
            "domain_start": [1],
            "domain_end": [50],
        }
    ).write_parquet(full)
    summary = tmp_path / "s.json"
    run(
        "restrict_truth_to_queries.py",
        "--truth",
        full,
        "--annotations",
        FIXTURES / "annotations",
        "--label",
        "mcsa",
        "--out",
        tmp_path / "o.parquet",
        "--summary-out",
        summary,
    )
    assert json.loads(summary.read_text())["after"]["n_regions"] == 0


def _code(text: str) -> str:
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_main_nf_routes_prebuilt_truth_through_restriction():
    src = _code(MAIN_NF.read_text())
    block = src[src.index("add_prebuilt = {") : src.index('add_prebuilt("pfamn"')]
    # Inside add_prebuilt the human truth goes to prebuilt_truth, never to truth_sets.
    assert "prebuilt_truth = prebuilt_truth.mix(" in block
    assert not re.search(
        r"truth_sets\s*=\s*truth_sets\.mix\(\s*Channel\.of\(\s*tuple\(label,\s*t\)",
        block,
    )
    after = src[src.index('add_prebuilt("mcsa"') :]
    assert re.search(
        r"restrictTruthToQueries\(\s*prebuilt_truth\s*,\s*annotations\s*\)", after
    )


def test_rescore_from_calls_reproduces_scorer_and_fixes_recall(pfam_truth, tmp_path):
    """Score a perfect tool against a truth padded with never-searched proteins, then
    rescore its call file. "before" must equal the scorer's own row (the script exits
    non-zero otherwise); "after" divides the same regions found by the searched proteins'
    regions only. Not 1.0: some fixture regions have no same-family target domain."""
    from conftest import score

    truth = pl.read_parquet(pfam_truth["truth"])
    outsiders = truth.head(6).with_columns(
        pl.format("Q0000{}", pl.int_range(pl.len()) // 2).alias("accession")
    )
    full = tmp_path / "human_pfamn_truth.parquet"
    pl.concat([truth, outsiders]).write_parquet(full)

    regions = tmp_path / "perfect.regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions)
    work = tmp_path / "score"
    m = score(full, pfam_truth["map"], regions, work, "pfamn")
    m.write_parquet(tmp_path / "all_domain_metrics.parquet")
    whole = m.filter(
        (pl.col("split") == "all")
        & (pl.col("stratum_axis") == "all")
        & ~pl.col("dedup_transfers")
    )
    assert whole["recall"].item() < 1.0  # the outsiders were in the denominator

    out = tmp_path / "rescored.csv"
    run(
        "rescore_pfamn_from_calls.py",
        "--calls-dir",
        work,
        "--truth",
        full,
        "--annotations",
        FIXTURES / "annotations",
        "--metrics",
        tmp_path / "all_domain_metrics.parquet",
        "--run",
        "fixture",
        "--out",
        out,
    )
    r = pl.read_csv(out).filter(~pl.col("dedup_transfers"))
    assert r["n_truth_instances_before"].item() == truth.height + 6
    assert r["n_truth_instances_after"].item() == truth.height
    assert r["recall_before"].item() == pytest.approx(whole["recall"].item())
    assert r["n_instances_found_after"].item() == r["n_instances_found_before"].item()
    assert r["recall_after"].item() == pytest.approx(
        r["recall_before"].item() * (truth.height + 6) / truth.height
    )
    assert r["fmax_after"].item() > r["fmax_before"].item()


def test_rescore_keeps_unpublished_call_files_and_stops_on_a_mismatch(
    pfam_truth, tmp_path
):
    """A calls/ folder shared by two runs holds files with no row in this run's metrics.
    They are kept and flagged. A published value that differs still stops the script."""
    import shutil

    from conftest import score

    regions = tmp_path / "perfect.regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions)
    work = tmp_path / "score"
    m = score(pfam_truth["truth"], pfam_truth["map"], regions, work, "pfamn")
    m.write_parquet(tmp_path / "metrics.parquet")
    for f in work.glob("pfamn.*.yeast*.calls.parquet"):
        shutil.copy(f, f.with_name(f.name.replace(".yeast", ".mouse")))

    out = tmp_path / "rescored.csv"
    args = [
        "--calls-dir",
        work,
        "--truth",
        pfam_truth["truth"],
        "--annotations",
        FIXTURES / "annotations",
        "--run",
        "fixture",
        "--out",
        out,
    ]
    run("rescore_pfamn_from_calls.py", *args, "--metrics", tmp_path / "metrics.parquet")
    r = pl.read_csv(out)
    assert r.height == 4
    assert r.filter(pl.col("species") == "yeast")["in_published_metrics"].all()
    assert not r.filter(pl.col("species") == "mouse")["in_published_metrics"].any()

    m.with_columns(pl.col("fmax") + 0.01).write_parquet(tmp_path / "bad.parquet")
    import subprocess
    import sys

    from conftest import BIN

    proc = subprocess.run(
        [
            sys.executable,
            str(BIN / "rescore_pfamn_from_calls.py"),
            *map(str, args),
            "--metrics",
            str(tmp_path / "bad.parquet"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "do not reproduce" in proc.stderr
