#!/usr/bin/env python3
"""Run `kmerseek pair` on every notebook-244 case and draw it with visualize_pair.py.

One case is one row of tables/244_hero_candidates.csv (case_id = its 0-based row index).
For each case this runs `kmerseek pair` on the human protein and the target protein, with
the alphabet and k of the kmerseek arm chosen for the case's feature type, and renders the
JSON with kmerseek's scripts/visualize_pair.py: a dot plot of every shared k-mer with both
proteins' Swiss-Prot features along the axes, and one alignment block per matched region.

Writes figures/244_pairs/<case_id>_<gene>_<feature_type>_<species>/ (JSON, PNG, SVG, HTML)
and tables/244_case_pairs.csv: per case, how many k-mers and regions `pair` found and
whether one of its regions is the run's call (same start and end on both proteins).

`kmerseek pair` has no low-complexity mask and no significance cutoff, so it shows every
shared k-mer and every region, including ones the run's search masked or filtered out.

Usage:
    python scripts/pair_244_cases.py --kmerseek-repo ~/code/kmerseek-244-pairs [--case 7]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
OUT = ROOT / "figures" / "244_pairs"
FASTA = TAB / "244_case_sequences.fasta"
NOTES = Path(
    "/Users/olga/data/qfo-pfam-region-midi-plus/extract/244_swissprot_feature_notes.parquet"
)
ARM_RE = re.compile(r"^kmerseek\.(?P<alpha>.+)_k(?P<k>\d+)_lc(?P<lc>True|False)$")


def case_stem(
    case_id: int, gene: str | None, query: str, ftype: str, species: str
) -> str:
    return f"{case_id:03d}_{gene or query}_{ftype}_{species}"


def write_domains(cases: pl.DataFrame, path: Path) -> None:
    """Swiss-Prot features of every case protein, as the table visualize_pair.py reads:
    accession, domain_start, domain_end (1-based, inclusive), name = the feature type.
    """
    accs = set(cases["query"]) | set(cases["target"])
    (
        pl.read_parquet(NOTES)
        .filter(pl.col("accession").is_in(accs))
        .select(
            "accession", "domain_start", "domain_end", pl.col("pfam_id").alias("name")
        )
        .unique()
        .sort("accession", "domain_start", "domain_end", "name")
        .write_csv(path, separator="\t")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kmerseek-repo", type=Path, required=True)
    ap.add_argument("--case", type=int, action="append", help="only these case_ids")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    binary = args.kmerseek_repo / "target" / "release" / "kmerseek"
    viz = args.kmerseek_repo / "scripts" / "visualize_pair.py"
    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", args.kmerseek_repo, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    cases = pl.read_csv(TAB / "244_hero_candidates.csv", infer_schema_length=None)
    cases = cases.with_row_index("case_id").with_columns(
        pl.col("case_id").cast(pl.Int64)
    )
    calls = pl.read_csv(TAB / "244_case_calls.csv", infer_schema_length=None).filter(
        pl.col("tool") == "kmerseek chosen arm"
    )
    if args.case:
        cases = cases.filter(pl.col("case_id").is_in(args.case))

    OUT.mkdir(parents=True, exist_ok=True)
    domains = OUT / "swissprot_features.tsv"
    write_domains(cases, domains)

    rows = []
    for r in cases.iter_rows(named=True):
        m = ARM_RE.match(r["kmerseek_chosen_arm"])
        alpha, k = m["alpha"], int(m["k"])
        stem = case_stem(
            r["case_id"], r["gene"], r["query"], r["feature_type"], r["species"]
        )
        d = OUT / stem
        d.mkdir(exist_ok=True)
        js = d / f"{stem}.pair.json"
        subprocess.run(
            [
                binary,
                "pair",
                "-q",
                FASTA,
                "--query-name",
                r["query"],
                "-t",
                FASTA,
                "--target-name",
                r["target"],
                "-k",
                str(k),
                "-a",
                alpha,
                "-o",
                js,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                viz,
                "--pair",
                js,
                "--output-dir",
                d,
                "--domains",
                domains,
                "--html",
                "--dpi",
                str(args.dpi),
            ],
            check=True,
            capture_output=True,
        )
        pj = json.loads(js.read_text())
        call = calls.filter(pl.col("case_id") == r["case_id"]).row(0, named=True)
        # The run's call is 1-based inclusive; pair regions are 0-based, end-exclusive.
        want = (
            call["query_start"] - 1,
            call["query_end"],
            call["target_start"] - 1,
            call["target_end"],
        )
        found = [
            (g["query_start"], g["query_end"], g["target_start"], g["target_end"])
            for g in pj["regions"]
        ]
        # The pair region on the call's diagonal that contains the call, if any.
        diag = want[2] - want[0]
        holder = [
            g
            for g in pj["regions"]
            if g["target_start"] - g["query_start"] == diag
            and g["query_start"] <= want[0]
            and g["query_end"] >= want[1]
        ]
        holder = min(
            holder, key=lambda g: g["query_end"] - g["query_start"], default=None
        )
        png = next(d.glob("*.png"), None)
        rows.append(
            {
                "case_id": r["case_id"],
                "gene": r["gene"],
                "query": r["query"],
                "feature_type": r["feature_type"],
                "species": r["species"],
                "target": r["target"],
                "alphabet": alpha,
                "ksize": k,
                "n_shared_kmers": len(pj["shared_kmers"]),
                "n_regions": len(pj["regions"]),
                "run_call_query": f"{call['query_start']}-{call['query_end']}",
                "run_call_target": f"{call['target_start']}-{call['target_end']}",
                "run_call_is_a_pair_region": want in found,
                "pair_region_holding_call_query": (
                    f"{holder['query_start'] + 1}-{holder['query_end']}"
                    if holder
                    else None
                ),
                "pair_region_holding_call_target": (
                    f"{holder['target_start'] + 1}-{holder['target_end']}"
                    if holder
                    else None
                ),
                "png": png.name if png else None,
                "folder": f"figures/244_pairs/{stem}",
                "kmerseek": f"{version} ({commit})",
            }
        )
        print(
            f"{stem}: {rows[-1]['n_regions']} regions, run call found: {want in found}"
        )

    out = pl.DataFrame(rows)
    if not args.case:
        out.write_csv(TAB / "244_case_pairs.csv")
        print(f"wrote {TAB / '244_case_pairs.csv'}: {out.height} cases")


if __name__ == "__main__":
    main()
