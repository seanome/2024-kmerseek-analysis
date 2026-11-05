#!/usr/bin/env python3
"""Score gene-counts directions against the paper's lists, with paper_recovery.py's test (notebook 361).

Usage: cheap_route_score.py <pec_root> <counts.tsv> <design_dir> <id_map.tsv> <out_dir> [sample_mean]

For each contrast: gene_directions.py on the contrast's samples, then each gene with a direction
is written as one fake search hit, so that paper_recovery.py applies its own list, random gene sets
and direction matching without change. Its channel name kamrat_deseq2_p05 is only a label here.

Writes <out_dir>/recovery.tsv, <out_dir>/directions/<contrast>.tsv, and prints the pass rules of
exports/cheap_route/README.md.
"""
import subprocess
import sys
from pathlib import Path

import polars as pl

root, counts, design_dir, id_map, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
rule = sys.argv[6:]  # passed on to gene_directions.py
SCRIPTS = Path(__file__).resolve().parent
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
CHANNEL = "kamrat_deseq2_p05"
(out / "directions").mkdir(parents=True, exist_ok=True)
protein = (pl.read_csv(id_map, separator="\t").filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
           .with_columns(pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene"))
           .group_by("gene").agg(pl.col("stable_id").min().alias("target")))
view = out / "view"
for c in CONTRASTS:
    tsv = out / "directions" / f"{c}.tsv"
    design = design_dir / f"contrast_{c}.design.tsv"
    samples = pl.read_csv(design, separator="\t")["sample"].to_list()
    sub = out / "directions" / f"{c}.counts.tsv"
    pl.read_csv(counts, separator="\t").select("gene", *samples).write_csv(sub, separator="\t")
    subprocess.run([sys.executable, str(SCRIPTS / "gene_directions.py"), str(sub), str(design), str(tsv), *rule], check=True)
    sub.unlink()
    d = pl.read_csv(tsv, separator="\t").filter(pl.col("direction").is_in(["up", "down"])).join(protein, on="gene")
    s = view / c / f"search_{CHANNEL}"
    s.mkdir(parents=True, exist_ok=True)
    d.select(pl.col("gene").alias("query"), "target", pl.lit(True).alias("supported")).write_parquet(s / "align.botznik.parquet")
    d.select(pl.col("gene").alias("peptide_id"), "direction").write_csv(s / "pep.peptides.tsv", separator="\t")
rec = out / "recovery_all.tsv"
subprocess.run([sys.executable, str(SCRIPTS / "paper_recovery.py"), str(view), str(root / "paper/paper_deseq_all.parquet"),
                str(root / "paper/paper_named_lists.tsv"), id_map, str(rec)], check=True, stdout=subprocess.DEVNULL)
r = pl.read_csv(rec, separator="\t").filter(pl.col("channel") == CHANNEL)
r.write_csv(out / "recovery.tsv", separator="\t")
rec.unlink()
with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=160):
    print(r.select("contrast", "list", "list_genes_with_protein", "reached", "reached_either_way", "null_median", "share_null_at_or_above"))
n, lists = r["reached"].sum(), (r["share_null_at_or_above"] <= 0.05).sum()
print(f"right direction: {n} of {r['list_genes_with_protein'].sum()}, need 439: {'pass' if n >= 439 else 'FAIL'}")
print(f"lists beating random: {lists} of {r.height}, need 9: {'pass' if lists >= 9 else 'FAIL'}")
