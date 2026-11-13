"""Gene-level lengthScaledTPM-style aggregation of kallisto output.

Same method as the package's de-reproduction/aggregate_baseline.py (copied in
sources/package_aggregate_baseline.py), with paths as arguments. Transcripts
map to genes through the oyster_v9 feature table (mRNA product_accession ->
GeneID); transcripts without an mRNA row (for example XR_ noncoding RNAs) are
counted as unmapped and reported.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--feature-table", required=True)
p.add_argument("--quant-dir", required=True, help="one subdirectory per sample with abundance.tsv")
p.add_argument("--samples", required=True, help="TSV with sample and condition columns")
p.add_argument("--out", required=True)
a = p.parse_args()

mapping = {}
with gzip.open(a.feature_table, "rt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["# feature"] == "mRNA" and row["GeneID"]:
            mapping[row["product_accession"]] = row["GeneID"]

samples = pd.read_csv(a.samples, sep="\t")["sample"]
tpms, lengths, totals, status = {}, {}, {}, {}
for sample in samples:
    quant = Path(a.quant_dir) / sample
    d = pd.read_csv(quant / "abundance.tsv", sep="\t")
    d["gene"] = d["target_id"].map(mapping)
    unmapped = float(d.loc[d.gene.isna(), "est_counts"].sum())
    d = d.dropna(subset=["gene"])
    d["weighted_length"] = d.tpm * d.eff_length
    group = d.groupby("gene")
    tpm = group.tpm.sum()
    length = (group.weighted_length.sum() / tpm).replace([np.inf, -np.inf], np.nan)
    tpms[sample], lengths[sample] = tpm, length
    totals[sample] = float(d.est_counts.sum())
    info = json.loads((quant / "run_info.json").read_text())
    status[sample] = dict(mapped_gene_est_counts=totals[sample], unmapped_est_counts=unmapped,
                          n_processed=info["n_processed"], n_pseudoaligned=info["n_pseudoaligned"],
                          p_pseudoaligned=info["p_pseudoaligned"])

tpm = pd.DataFrame(tpms).fillna(0)
length = pd.DataFrame(lengths).mean(axis=1)
# Missing lengths occur only in all-zero genes; their output stays zero.
weighted = tpm.mul(length.fillna(1), axis=0)
counts = weighted.div(weighted.sum(axis=0), axis=1).mul(pd.Series(totals), axis=1)
counts.index = counts.index.astype(str)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
counts.to_csv(out / "gene_counts.tsv", sep="\t", index_label="gene_id")
(out / "aggregation.json").write_text(json.dumps(dict(
    method="gene TPM times across-sample mean TPM-weighted effective gene length; "
           "per-sample scaling to total mapped transcript estimated counts; rounded in fit_de",
    transcripts_with_gene=len(mapping), sample_counts=status), indent=2))
print("Gene count matrix", counts.shape)
