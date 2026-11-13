#!/usr/bin/env python3
"""Collect the branchwater manysearch results into one gene x run containment table.

Inputs (from slurm/23_sra_search.sbatch): ms.ge<cutoff>.<alphabet>.k<k>.csv (every query x run
pair, threshold 0 with -A, so zero overlap is a row), queries.describe.csv and
runs.ge<cutoff>.describe.csv (hashes per signature).

Containment = matched hashes / the query protein's hashes at that setting. A protein with no
hash at a setting (too short for k at scaled 10) is "unsearchable" there: its containment is
left empty, not 0. A run whose sketch has no hash is likewise unsearchable.

Usage: sra_collect.py <search_dir> <query_map.tsv> <runs.tsv> <out_dir> [cutoff_for_main_table=0.1]
Writes:
- gene_run_containment_all.tsv.gz: every cutoff, setting, protein and run (stays on Sherlock);
- gene_run_containment.tsv.gz: the main cutoff only (committed);
- containment_summary.tsv: per cutoff, setting, dataset, condition and query set, the median and
  mean containment and the share of protein-run pairs with any matched hash;
- containment_by_setting.tsv: the same per cutoff, setting, species (Botryllus or human) and set;
- unsearchable.tsv: proteins with no hash and runs with an empty sketch, per setting.
"""

import gzip
import re
import sys
from pathlib import Path

import polars as pl

search, qmap_p, runs_p, out = sys.argv[1:5]
main_cut = sys.argv[5] if len(sys.argv) > 5 else "0.1"
search, out = Path(search), Path(out)
out.mkdir(parents=True, exist_ok=True)

qmap = pl.read_csv(qmap_p, separator="\t", infer_schema_length=0)
# one row per protein: its lists joined, so the table is not duplicated per list
qprot = (
    qmap.with_columns(
        paper_list=pl.when(pl.col("set") == "paper_list")
        .then(pl.col("contrast") + ":" + pl.col("direction"))
        .otherwise(pl.lit("random_length_matched"))
    )
    .group_by("query", "original_id", "gene", "set", "length")
    .agg(
        pl.col("paper_list").unique().sort().str.join(";"),
        pl.col("gene_name").first(),
    )
)
runs = pl.read_csv(runs_p, separator="\t", infer_schema_length=0).select(
    pl.col("run"),
    pl.col("sample"),
    pl.col("dataset"),
    pl.col("condition"),
    pl.col("contrast").alias("run_contrast"),
)
# One row per md5. Signatures with the same md5 hold the same hashes, so the same n_hashes; all
# empty sketches share one md5 (80 human runs at HP k27 and k30), which without this made the
# joins below many-to-many and ran out of memory.
qdesc = (
    pl.read_csv(search / "queries.describe.csv")
    .select(pl.col("md5").alias("query_md5"), pl.col("n_hashes").alias("query_hashes"))
    .unique("query_md5")
)
parts = []
for f in sorted(search.glob("ms.ge*.csv")):
    m = re.match(r"ms\.ge([\d.]+)\.(\w+)\.k(\d+)\.csv$", f.name)
    cut, alpha, k = m.group(1), m.group(2), int(m.group(3))
    rdesc = (
        pl.read_csv(search / f"runs.ge{cut}.describe.csv")
        .select(
            pl.col("md5").alias("match_md5"), pl.col("n_hashes").alias("run_hashes")
        )
        .unique("match_md5")
    )
    d = pl.read_csv(f).select(
        "query_name",
        "query_md5",
        "match_name",
        "match_md5",
        "containment",
        "intersect_hashes",
    )
    d = d.join(qdesc, on="query_md5", how="left").join(
        rdesc, on="match_md5", how="left"
    )
    parts.append(
        d.with_columns(cutoff=pl.lit(cut), alphabet=pl.lit(alpha), k=pl.lit(k))
    )
d = pl.concat(parts).rename(
    {"query_name": "query", "match_name": "run", "intersect_hashes": "matched_hashes"}
)
d = d.with_columns(
    searchable=(pl.col("query_hashes") > 0) & (pl.col("run_hashes") > 0),
).with_columns(
    containment=pl.when(pl.col("searchable"))
    .then(pl.col("containment"))
    .otherwise(None)
)
d = d.join(qprot, on="query", how="left").join(runs, on="run", how="left")
cols = [
    "gene",
    "query",
    "original_id",
    "gene_name",
    "set",
    "paper_list",
    "length",
    "run",
    "sample",
    "dataset",
    "condition",
    "run_contrast",
    "alphabet",
    "k",
    "cutoff",
    "containment",
    "matched_hashes",
    "query_hashes",
    "run_hashes",
    "searchable",
]
d = d.select(cols).sort("cutoff", "alphabet", "k", "set", "gene", "query", "run")
with gzip.open(out / "gene_run_containment_all.tsv.gz", "wb") as fh:
    d.write_csv(fh, separator="\t")
with gzip.open(out / "gene_run_containment.tsv.gz", "wb") as fh:
    d.filter(pl.col("cutoff") == main_cut).write_csv(fh, separator="\t")
summ = (
    d.filter(pl.col("searchable"))
    .group_by("cutoff", "alphabet", "k", "dataset", "condition", "set")
    .agg(
        pairs=pl.len(),
        proteins=pl.col("query").n_unique(),
        runs=pl.col("run").n_unique(),
        median_containment=pl.col("containment").median(),
        mean_containment=pl.col("containment").mean(),
        share_any_hash=(pl.col("matched_hashes") > 0).mean(),
    )
    .sort("cutoff", "alphabet", "k", "dataset", "condition", "set")
)
unsearch = d.group_by("cutoff", "alphabet", "k", "set").agg(
    proteins=pl.col("query").n_unique(),
    unsearchable_proteins=pl.col("query")
    .filter(pl.col("query_hashes") == 0)
    .n_unique(),
    empty_runs=pl.col("run").filter(pl.col("run_hashes") == 0).n_unique(),
)
summ.write_csv(out / "containment_summary.tsv", separator="\t")
by_setting = (
    d.filter(pl.col("searchable"))
    .with_columns(
        species=pl.when(pl.col("dataset") == "PRJNA1219309")
        .then(pl.lit("Botryllus"))
        .otherwise(pl.lit("human"))
    )
    .group_by("cutoff", "alphabet", "k", "species", "set")
    .agg(
        pairs=pl.len(),
        median_containment=pl.col("containment").median(),
        mean_containment=pl.col("containment").mean(),
        share_any_hash=(pl.col("matched_hashes") > 0).mean(),
    )
    .sort("cutoff", "alphabet", "k", "species", "set")
)
by_setting.write_csv(out / "containment_by_setting.tsv", separator="\t")
print(by_setting)
unsearch.sort("cutoff", "alphabet", "k", "set").write_csv(
    out / "unsearchable.tsv", separator="\t"
)
print(summ.filter(pl.col("dataset") == "PRJNA1219309"))
print(unsearch.sort("cutoff", "alphabet", "k", "set"))
