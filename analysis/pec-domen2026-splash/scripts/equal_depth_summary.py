#!/usr/bin/env python3
"""Counts and reached genes for the equal-depth test (notebook 361).

Usage: equal_depth_summary.py <pec_root> <botznik.id_map.tsv> <out_dir>

Arms: `all_samples` is the original run (<pec_root>/contrasts/<c>, every sample at full depth);
`full`, `2000000`, `1000000` are under <pec_root>/equal_depth/<c>/ and use only the kept pairs.
A gene is reached when one of the arm's peptides aligns to one of its old gene-model proteins
with `supported` true (E <= 1e-5, query coverage >= 50 %), the rule of reached_genes.py.

Writes, in <out_dir>:
  equal_depth_counts.tsv    per contrast and arm: samples, read pairs, tested and significant
                            anchors, compactors, peptides, supported peptides, genes reached
  equal_depth_reached.tsv   contrast, arm, gene, peptides
  equal_depth_control.tsv   simulated control: SPLASH significant anchors and recall per planted
                            class at each depth (full depth from <pec_root>/control_eval/with_kamrat)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

pec, id_map, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
ED = pec / "equal_depth"
CONTRASTS = ["young_24h", "young_2h"]
ARMS = ["all_samples", "full", "2000000", "1000000"]


def lines(p):
    return int(subprocess.run(["wc", "-l", str(p)], capture_output=True, text=True, check=True).stdout.split()[0])


genes = (pl.read_csv(id_map, separator="\t", columns=["stable_id", "original_id"])
         .with_columns(pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
rows, reached = [], []
for c in CONTRASTS:
    for arm in ARMS:
        d = pec / "contrasts" / c if arm == "all_samples" else ED / c / arm
        row = dict(contrast=c, arm=arm)
        inp = d / "input.txt"
        if inp.exists():
            samples = [ln.split()[0] for ln in inp.read_text().splitlines() if ln.strip()]
            row["samples"] = len(samples)
            row["pairs_per_sample"] = arm if arm.isdigit() else "full"
        sp = d / "splash_out"
        if (sp / "result.after_correction.scores.tsv").exists():
            row["tested_anchors"] = lines(sp / "result.after_correction.all_anchors.tsv") - 1
            row["significant_anchors"] = lines(sp / "result.after_correction.scores.tsv") - 1
        if (d / "compactors/compactors.tsv").exists():
            row["compactors"] = lines(d / "compactors/compactors.tsv") - 1
        tl = d / "search_compactors/translate.log"
        if tl.exists():
            m = re.search(r"peptides: (\d+)", tl.read_text())
            row["peptides"] = int(m.group(1)) if m else None
        al = d / "search_compactors/align.botznik.parquet"
        if al.exists():
            sup = pl.scan_parquet(al).filter(pl.col("supported")).select("query", "target")
            row["supported_peptides"] = sup.select(pl.col("query").n_unique()).collect().item()
            r = (sup.join(genes.lazy(), left_on="target", right_on="stable_id")
                 .group_by("gene").agg(pl.col("query").n_unique().alias("peptides")).collect())
            row["genes_reached"] = r.height
            reached.append(r.with_columns(pl.lit(c).alias("contrast"), pl.lit(arm).alias("arm"))
                           .select("contrast", "arm", "gene", "peptides"))
        elif (d / "search.done").exists() or (d / "search_compactors/search.done").exists():
            row["supported_peptides"] = 0
            row["genes_reached"] = 0
        rows.append(row)
        print(row)
counts = pl.DataFrame(rows, infer_schema_length=None)
counts.write_csv(out / "equal_depth_counts.tsv", separator="\t")
if reached:
    pl.concat(reached).sort("contrast", "arm", "gene").write_csv(out / "equal_depth_reached.tsv", separator="\t")

ctrl = []
for depth, f in [("full (about 1000000)", pec / "control_eval/with_kamrat/control_truth_summary.json"),
                 *[(str(d), ED / "control" / str(d) / "control_eval/control_truth_summary.json")
                   for d in (500000, 250000, 125000)]]:
    if not f.exists():
        print("missing", f)
        continue
    s = json.loads(f.read_text())["channels"]["splash"]
    for cls, n in s["genes_in_class"].items():
        ctrl.append(dict(pairs_per_sample=depth, truth_class=cls, genes_in_class=n,
                         genes_found=s["genes_found"][cls], recall=s["recall"][cls],
                         significant_anchors=s["significant_anchors"]))
if ctrl:
    pl.DataFrame(ctrl).write_csv(out / "equal_depth_control.tsv", separator="\t")
    with pl.Config(tbl_rows=40):
        print(pl.DataFrame(ctrl))
with pl.Config(tbl_rows=40, tbl_cols=20):
    print(counts)
