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


def _finished_run(pfam_truth, tmp_path):
    """A finished run's layout: calls/, all_domain_metrics.parquet, and a --pfamn_dir
    whose human truth also holds 3 proteins the run never searched."""
    import shutil

    from conftest import score

    truth = pl.read_parquet(pfam_truth["truth"])
    outsiders = truth.head(6).with_columns(
        pl.format("Q0000{}", pl.int_range(pl.len()) // 2).alias("accession")
    )
    pfamn_dir = tmp_path / "pfamn"
    pfamn_dir.mkdir()
    full = pfamn_dir / "human_pfamn_truth.parquet"
    pl.concat([truth, outsiders]).write_parquet(full)
    shutil.copy(pfam_truth["map"], pfamn_dir / "yeast_domain_map.parquet")

    regions = tmp_path / "perfect.regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions, pad=7)
    run_dir = tmp_path / "results"
    m = score(full, pfam_truth["map"], regions, run_dir / "calls", "pfamn")
    m.write_parquet(run_dir / "all_domain_metrics.parquet")
    # A call file from another run sharing the folder: no published row.
    for f in (run_dir / "calls").glob("pfamn.*.yeast*.calls.parquet"):
        shutil.copy(f, f.with_name(f.name.replace(".yeast", ".mouse")))
    return truth, m, run_dir, pfamn_dir


def _rescore_args(run_dir, pfamn_dir):
    return [
        "--run-dir", run_dir,
        "--annotations", FIXTURES / "annotations",
        "--pfamn-dir", pfamn_dir,
        "--jobs", 2,
    ]  # fmt: skip


def test_rescore_reproduces_every_row_and_fixes_recall(pfam_truth, tmp_path):
    """Before: every published row, every stratum, recomputed exactly. After: recall over
    the searched proteins only, the same regions found."""
    truth, m, run_dir, pfamn_dir = _finished_run(pfam_truth, tmp_path)
    run("rescore_pfamn_from_calls.py", *_rescore_args(run_dir, pfamn_dir))

    after = pl.read_parquet(run_dir / "pfamn_rescored_metrics.parquet")
    assert after["before_matches_published"].all()
    assert set(after["species"]) == {"yeast"}  # the unpublished mouse files are skipped
    assert after.height == m.height
    assert after["stratum_axis"].n_unique() > 1

    keys = ["dedup_transfers", "split", "stratum_axis", "stratum"]
    whole = (
        m.filter((pl.col("split") == "all") & (pl.col("stratum_axis") == "all"))
        .join(after, on=keys, suffix="_after")
        .sort("dedup_transfers")
    )
    assert whole.height == 2
    assert (whole["n_truth_instances"] == truth.height + 6).all()
    assert (whole["n_truth_instances_after"] == truth.height).all()
    assert (whole["n_instances_found"] == whole["n_instances_found_after"]).all()
    assert (whole["recall"] < 1.0).all()
    for r in whole.iter_rows(named=True):
        assert r["recall_after"] == pytest.approx(
            r["recall"] * (truth.height + 6) / truth.height
        )
        assert r["fmax_after"] > r["fmax"]
    check = pl.read_csv(run_dir / "pfamn_rescore_check.csv")
    assert (check["n_rows_differ"] == 0).all()


def test_rescore_stops_when_the_whole_set_does_not_reproduce(pfam_truth, tmp_path):
    import subprocess
    import sys

    from conftest import BIN

    _, m, run_dir, pfamn_dir = _finished_run(pfam_truth, tmp_path)
    m.with_columns(
        pl.when(pl.col("stratum_axis") == "all")
        .then(pl.col("fmax") + 0.01)
        .otherwise(pl.col("fmax"))
        .alias("fmax")
    ).write_parquet(run_dir / "all_domain_metrics.parquet")
    proc = subprocess.run(
        [sys.executable, str(BIN / "rescore_pfamn_from_calls.py"),
         *map(str, _rescore_args(run_dir, pfamn_dir))],
        capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode != 0
    assert "do not reproduce" in proc.stderr
    assert not (run_dir / "pfamn_rescored_metrics.parquet").exists()
