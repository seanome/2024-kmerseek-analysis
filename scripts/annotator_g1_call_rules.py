#!/usr/bin/env python3
"""How much of the best-hit rows' Fmax comes from the call rule, not the search.

PR 160's rows place each best-hit domain on the query through the alignment. The main
run's rows (PR 148: phmmer, MMseqs2 against one QfO proteome each) instead call the
whole aligned part of the query, once per target domain the hit covers, for every hit.
Those are two different rules, so their Fmax cannot be compared directly.

This rescores the SAME best hits two ways, with evaluate_domain_calls.py, on the same
truth:
    placed        the PR 160 rule (annotator_best_hit.py place, --direct-annotation)
    whole_region  the main run's rule: the best hit's whole query interval, labelled with
                  every target domain it covers by half or more (the script's own
                  transfer step, with the hit proteins' hmmscan domains as the map)

Inputs are copies of the Sherlock run's files (run folder in the PR body):
    <data>/hits/<tool>.<setting>.hits.tsv.gz
    <data>/hit_targets/hit_target_domains.tsv.gz
    <data>/human_domain_truth.parquet, <data>/human_query_covariates.parquet
    <data>/UP000005640_9606.fasta

    python scripts/annotator_g1_call_rules.py --data <folder> \
        --out results/leaderboard/annotator_tools/call_rules.parquet
"""

import argparse
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

BIN = (
    Path(__file__).resolve().parents[1] / "nextflow-runs/qfo-pfam-region-benchmark/bin"
)
sys.path.insert(0, str(BIN))
import annotator_best_hit as abh  # noqa: E402

SETTINGS = [
    ("diamond_besthit", "sensitive"),
    ("diamond_besthit", "ultra-sensitive"),
    ("mmseqs2_besthit", "s7.5"),
]


def run(*args) -> None:
    proc = subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True
    )
    if proc.returncode:
        sys.exit(proc.stderr[-3000:])


def score(
    regions: Path, tool: str, setting: str, d: Path, tmp: Path, extra: list
) -> dict:
    stem = tmp / f"{tool}.{setting}.{len(extra)}"
    run(
        BIN / "evaluate_domain_calls.py",
        "--regions",
        regions,
        "--tool",
        tool,
        "--variant",
        setting,
        "--species",
        "swissprot_minus_mammalia",
        "--truth",
        d / "human_domain_truth.parquet",
        "--covariates",
        d / "human_query_covariates.parquet",
        "--min-overlap",
        0.5,
        "--strict-iou",
        0.8,
        *extra,
        "--calls-out",
        f"{stem}.calls.parquet",
        "--metrics-out",
        f"{stem}.metrics.parquet",
        "--curve-out",
        f"{stem}.curve.parquet",
    )
    m = pl.read_parquet(f"{stem}.metrics.parquet").filter(
        (pl.col("split") == "all")
        & (pl.col("stratum_axis") == "all")
        & ~pl.col("dedup_transfers").fill_null(False)
    )
    r = m.row(0, named=True)
    return {
        k: r[k]
        for k in (
            "fmax",
            "fmax_precision",
            "fmax_recall",
            "n_truth_instances",
            "n_calls",
        )
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = a.data
    rows = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        domains = abh.read_domains(d / "hit_targets/hit_target_domains.tsv.gz")
        domain_map = tmp / "hit_domain_map.parquet"
        domains.select(
            pl.col("target").alias("accession"),
            "pfam_id",
            pl.col("dstart").alias("domain_start"),
            pl.col("dend").alias("domain_end"),
        ).write_parquet(domain_map)
        for tool, setting in SETTINGS:
            hits = d / f"hits/{tool}.{setting}.hits.tsv.gz"
            placed = tmp / f"{tool}.{setting}.placed.tsv.gz"
            run(
                BIN / "annotator_best_hit.py",
                "place",
                "--hits",
                hits,
                "--target-domains",
                d / "hit_targets/hit_target_domains.tsv.gz",
                "--queries",
                d / "UP000005640_9606.fasta",
                "--tool",
                tool,
                "--setting",
                setting,
                "--out",
                placed,
                "--qc-out",
                tmp / "qc.json",
            )
            rows.append(
                {
                    "tool": tool,
                    "setting": setting,
                    "call_rule": "placed",
                    **score(placed, tool, setting, d, tmp, ["--direct-annotation"]),
                }
            )
            best = abh.best_hits(abh.read_hits(hits))
            whole = tmp / f"{tool}.{setting}.whole.tsv.gz"
            with gzip.open(whole, "wt") as fh:
                fh.write(
                    best.select(
                        "query",
                        "target",
                        "qstart",
                        "qend",
                        "tstart",
                        "tend",
                        "bits",
                        "evalue",
                    ).write_csv(separator="\t", include_header=False)
                )
            rows.append(
                {
                    "tool": tool,
                    "setting": setting,
                    "call_rule": "whole_region",
                    **score(whole, tool, setting, d, tmp, ["--domain-map", domain_map]),
                }
            )
    out = pl.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    with pl.Config(tbl_rows=20, tbl_cols=10, float_precision=3):
        print(out)


if __name__ == "__main__":
    main()
