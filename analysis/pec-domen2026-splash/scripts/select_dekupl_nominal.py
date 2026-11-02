#!/usr/bin/env python3
"""DE-kupl k-mers at uncorrected p < 0.05, no fold-change cutoff, in DE-kupl's diff-counts format.

Usage: select_dekupl_nominal.py <DEkupl_result dir> <design.tsv> <out_dir> [all-kmers diff-counts.tsv.gz]

Added 2026-11-02 (Olga: "use nominal p < 0.05 for kamrat and de-kupl"). DE-kupl keeps only k-mers
with BH-corrected p <= 0.05 and |log2FC| >= 1 in diff-counts.tsv.gz (its config), which is empty
in every real contrast. But it writes the uncorrected p of every tested k-mer to
control_vs_PEC_kmer_counts/raw_pvals.txt.gz (no header: tag, p), so no rerun is needed.

meanA, meanB and log2FC are rebuilt the way DE-kupl writes them: meanA = mean over control
samples of count / normalization_factor (kmer_counts/normalization_factors.tsv), meanB the same
over PEC, log2FC = log2(meanB / meanA). Condition A is control and B is PEC (dekupl_config.py),
so log2FC > 0 = higher in PEC = up. The script checks this rebuild against every k-mer DE-kupl
itself kept in diff-counts.tsv.gz (only the simulated control has any) and stops on a mismatch.
That rebuild is right for limma, whose log2FC DE-kupl writes as log2(meanB / meanA). With DESeq2,
DE-kupl writes DESeq2's own log2 fold change, which can differ in sign from the ratio of means; so
for DESeq2 pass a DE-kupl diff-counts.tsv.gz written with its filters opened up (corrected p <= 1,
|log2FC| >= 0, so every tested k-mer is in it; 19_dekupl_deseq2_p05.sbatch), and meanA, meanB and
log2FC are taken from it. The check against DE-kupl's own rows is then skipped (it would compare
the table with itself).

Writes <out_dir>/up.diff-counts.tsv and down.diff-counts.tsv (gzipped by the caller; columns tag, pvalue, meanA,
meanB, log2FC, samples; the input of DE-kupl's mergeTags) and <out_dir>/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

res, design, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
data_all = sys.argv[4] if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
dsg = pl.read_csv(design, separator="\t")
ctrl = dsg.filter(pl.col("condition") == "control")["sample"].to_list()
pec = dsg.filter(pl.col("condition") == "PEC")["sample"].to_list()
assert ctrl and pec
nf = dict(pl.read_csv(res / "kmer_counts/normalization_factors.tsv", separator="\t").iter_rows())
pv = pl.read_csv(res / "control_vs_PEC_kmer_counts/raw_pvals.txt.gz", separator="\t", has_header=False,
                 new_columns=["tag", "pvalue"], schema_overrides={"pvalue": pl.Float64}, null_values=["NA"])
counts = pl.read_csv(res / "kmer_counts/raw-counts.tsv.gz", separator="\t")
n_tested, n_counted = pv.height, counts.height
d = pv.filter(pl.col("pvalue") < 0.05).join(counts, on="tag", how="inner")
if data_all is None:
    d = (d.with_columns(pl.mean_horizontal([pl.col(s) / nf[s] for s in ctrl]).alias("meanA"),
                        pl.mean_horizontal([pl.col(s) / nf[s] for s in pec]).alias("meanB"))
         .with_columns((pl.col("meanB") / pl.col("meanA")).log(2).alias("log2FC")))
else:
    da = pl.read_csv(data_all, separator="\t", columns=["tag", "meanA", "meanB", "log2FC"], null_values=["NA"],
                     schema_overrides={"meanA": pl.Float64, "meanB": pl.Float64, "log2FC": pl.Float64})
    d = d.join(da, on="tag", how="inner")
d = d.select("tag", "pvalue", "meanA", "meanB", "log2FC", *ctrl, *pec)
n_p05 = pv.filter(pl.col("pvalue") < 0.05).height
assert d.height == n_p05, f"{n_p05 - d.height} k-mers with p < 0.05 have no counts"

# check the rebuild against DE-kupl's own rows
if data_all is None:
    own = pl.read_csv(res / "control_vs_PEC_kmer_counts/diff-counts.tsv.gz", separator="\t")
    check = {"dekupl_kept_kmers": own.height}
else:
    own = pl.DataFrame()
    check = {"skipped": "fold change taken from DE-kupl's own all-k-mer table"}
if own.height:
    j = own.join(d, on="tag", suffix="_ours")
    check["compared"] = j.height
    for c in ("meanA", "meanB", "log2FC"):
        diff = (j[c] - j[f"{c}_ours"]).abs().max()
        check[f"max_abs_diff_{c}"] = float(diff)
        assert diff < 1e-3, f"{c} rebuild differs from DE-kupl by up to {diff}"
    assert j.height == own.height, "some DE-kupl-kept k-mers are not at p < 0.05 here"
    check["same_sign_log2FC"] = bool(((j["log2FC"] > 0) == (j["log2FC_ours"] > 0)).all())

up = d.filter(pl.col("log2FC") > 0)
down = d.filter(pl.col("log2FC") < 0)
for name, t in (("up", up), ("down", down)):
    # mergeTags reads gzipped TSV; sorted by p so the k-mer with the smallest p represents a contig
    t.sort("pvalue").write_csv(out / f"{name}.diff-counts.tsv", separator="\t")
s = {"kmers_counted": n_counted, "kmers_with_p": n_tested,
     "p_missing": int(pv["pvalue"].is_null().sum()), "p_lt_0.05": n_p05,
     "up": up.height, "down": down.height, "equal_means": d.height - up.height - down.height,
     "check_against_dekupl": check}
(out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
print(json.dumps(s))
