#!/usr/bin/env python3
"""One kmerseek hits page per notebook-244 case, from the run's own search output.

For each row of tables/244_hero_candidates.csv (case_id = 0-based row index), takes every
region the midi-plus search reported for the case's human protein with the kmerseek arm
chosen for its feature type, against the case's species, and renders it with kmerseek's
scripts/visualize_search.py: the human protein with its Swiss-Prot features, a histogram of
how many target proteins have a region over each residue, and one row per target protein,
which opens into that pair's dot plot and alignments (`kmerseek pair`).

Input is the cut of the region tables made by scripts/extract_244_case_hits.py. Cases that
share a human protein, arm and species share one page.

Writes figures/244_hits/<human acc>.<alphabet>.k<k>.<species>.kmerseek_hits.html and
tables/244_case_hits_index.csv, with, per case, where the case's target protein sits:
  * page_rank: its row on the page, which orders proteins by kmerseek's page statistic
    (Benjamini-Hochberg corrected tail probability of the best region). A page shows the
    top --max-rows proteins (100), so a target ranked lower is not on it (on_page);
    a page that ran down to every target reached 70 MB for one case;
  * run_rank: its rank under the run's own rule (evaluate_domain_calls.load_regions:
    Bonferroni-corrected region tail probability < 0.05, then best region_enrichment).

Usage:
    python scripts/hits_244_cases.py --hits-dir <cut tables> --kmerseek-repo ~/code/kmerseek-244-pairs
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import hero_example_utils as he  # noqa: E402

TAB = ROOT / "tables"
OUT = ROOT / "figures" / "244_hits"
SPROT = he.MIDI / "truth_swissprot"
sys.path.insert(0, str(ROOT / "scripts"))
from extract_244_case_hits import table_name  # noqa: E402
from kmerseek_run_rank import ranked_targets  # noqa: E402


def accession(col: str) -> pl.Expr:
    return pl.col(col).str.split("|").list.get(1)


def domains_tsv(species: str, path: Path) -> Path:
    """Swiss-Prot features of human and of the species, in visualize_pair's table form."""
    parts = [
        pl.read_parquet(SPROT / "human_swissprot_truth.parquet"),
        pl.read_parquet(SPROT / f"{species}_domain_map.parquet"),
    ]
    pl.concat(
        [p.select("accession", "pfam_id", "domain_start", "domain_end") for p in parts]
    ).select(
        "accession", "domain_start", "domain_end", pl.col("pfam_id").alias("name")
    ).unique().sort(
        "accession", "domain_start"
    ).write_csv(
        path, separator="\t"
    )
    return path


def with_residues(rows: pl.DataFrame, species: str) -> pl.DataFrame:
    """Add region_subseq / target_subseq, which `kmerseek search -o` writes and the run's
    parquet tables do not: residues [start, end) of each protein, from the QfO FASTAs.
    """
    q = he.sequences("human", set(rows["query_acc"]))
    t = he.sequences(species, set(rows["target_acc"]))
    return rows.with_columns(
        region_subseq=pl.struct("query_acc", "region_start", "region_end").map_elements(
            lambda s: q[s["query_acc"]][s["region_start"] : s["region_end"]],
            return_dtype=pl.String,
        ),
        target_subseq=pl.struct(
            "target_acc", "target_start", "target_end"
        ).map_elements(
            lambda s: t[s["target_acc"]][s["target_start"] : s["target_end"]],
            return_dtype=pl.String,
        ),
    )


def run_rank(rows: pl.DataFrame, target: str) -> int | None:
    """Rank of ``target`` under the run's rule (kmerseek_run_rank), or None if none of its
    regions passes the Bonferroni cut."""
    hit = ranked_targets(rows).filter(pl.col("target_acc") == target).collect()
    return int(hit["rank"][0]) if hit.height else None


def page_rank(csv_path: Path, target: str, viz_dir: Path) -> int | None:
    """Row of ``target`` on the page: visualize_search's own ordering, all rows."""
    sys.path.insert(0, str(viz_dir))
    from visualize_search import rank_proteins  # noqa: E402

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    for i, (name, others, _) in enumerate(rank_proteins(rows, 10**9), start=1):
        if any(n.split("|")[1] == target for n in [name, *others]):
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hits-dir", type=Path, required=True)
    ap.add_argument("--kmerseek-repo", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=100)
    args = ap.parse_args()
    binary = args.kmerseek_repo / "target" / "release" / "kmerseek"
    viz = args.kmerseek_repo / "scripts" / "visualize_search.py"

    cases = pl.read_csv(TAB / "244_hero_candidates.csv", infer_schema_length=None)
    cases = cases.with_row_index("case_id").with_columns(
        pl.col("case_id").cast(pl.Int64)
    )
    OUT.mkdir(parents=True, exist_ok=True)
    # The per-page CSVs are hundreds of MB; they stay next to the cut tables, not in git.
    work = args.hits_dir / "csv"
    work.mkdir(exist_ok=True)

    index, done = [], {}
    for r in cases.iter_rows(named=True):
        arm, sp, q = r["kmerseek_chosen_arm"], r["species"], r["query"]
        m = he._ARM_RE.match(arm)
        stem = f"{q}.{m['alpha']}.k{m['k']}.{sp}"
        src = args.hits_dir / table_name(arm, sp)
        row = {
            "case_id": r["case_id"],
            "gene": r["gene"],
            "query": q,
            "feature_type": r["feature_type"],
            "species": sp,
            "target": r["target"],
            "arm": arm,
        }
        if not src.exists():
            index.append(
                {
                    **row,
                    "page": None,
                    "n_target_proteins": None,
                    "page_rank": None,
                    "run_rank": None,
                    "note": f"region table not cut yet: {src.name}",
                }
            )
            continue
        if stem not in done:
            rows = (
                pl.read_parquet(src)
                .with_columns(
                    query_acc=accession("query_name"),
                    target_acc=accession("target_name"),
                )
                .filter(pl.col("query_acc") == q)
            )
            rows = with_residues(rows, sp)
            csv_path = work / f"{stem}.csv"
            rows.drop("query_acc", "target_acc").write_csv(csv_path)
            dom = domains_tsv(sp, work / f"swissprot_features.human_{sp}.tsv")
            page_dir = OUT / stem
            subprocess.run(
                [
                    sys.executable,
                    viz,
                    "--csv",
                    csv_path,
                    "--query-fasta",
                    he.proteome_fasta("human"),
                    "--target-fasta",
                    he.proteome_fasta(sp),
                    "--output-dir",
                    page_dir,
                    "--domains",
                    dom,
                    "--kmerseek",
                    binary,
                    "--max-rows",
                    str(args.max_rows),
                ],
                check=True,
                capture_output=True,
            )
            page = next(page_dir.glob("*.kmerseek_hits.html"))
            done[stem] = (rows, csv_path, page)
        rows, csv_path, page = done[stem]
        index.append(
            {
                **row,
                "page": str(page.relative_to(ROOT)),
                "n_target_proteins": rows["target_acc"].n_unique(),
                "page_rank": (pr := page_rank(csv_path, r["target"], viz.parent)),
                "on_page": pr is not None and pr <= args.max_rows,
                "run_rank": run_rank(rows, r["target"]),
                "note": None,
            }
        )
        print(
            f"{r['case_id']:>3} {stem}: page row {index[-1]['page_rank']}, run rank {index[-1]['run_rank']}"
        )

    pl.DataFrame(index, infer_schema_length=None).write_csv(
        TAB / "244_case_hits_index.csv"
    )
    print(
        f"pages: {len(done)}; cases with a page: {sum(i['page'] is not None for i in index)} of {len(index)}"
    )


if __name__ == "__main__":
    main()
