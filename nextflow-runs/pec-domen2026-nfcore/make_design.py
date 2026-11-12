#!/usr/bin/env python3
"""Inputs for nf-core/rnaseq and nf-core/differentialabundance on the 47 PEC samples (notebook 361).

Usage: make_design.py <analysis/pec-domen2026-splash/design> <trimmed_reads_dir> <out_dir>

Uses the six contrasts and designs of the k-mer routes, so each can be scored against these genes.
Writes samplesheet.csv for nf-core/rnaseq, and <contrast>/samples.csv and <contrast>/contrasts.csv
for nf-core/differentialabundance, one run per contrast: its input check tests the design on every
sample in the sheet before any cut to a contrast, and pairs nested inside contrasts fail that test.
Reads are the fastp-trimmed reads the routes used. Blocking is pair (young_132d: genotype), the
routes' DESeq2 design.
"""
import re
import sys
from pathlib import Path

import pandas as pd

design, reads, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
samples = []
for c in CONTRASTS:
    d = pd.read_csv(design / f"contrast_{c}.design.tsv", sep="\t", dtype=str)
    samples.append(pd.DataFrame({"sample": d["sample"], "contrast": c, "group": c + "_" + d["condition"],
                                 "pair": d["pair"].map(lambda x: re.sub(r"\W", "_", x)), "genotype": d["genotype"].map(lambda x: re.sub(r"\W", "_", x))}))
    (out / c).mkdir(exist_ok=True)
    samples[-1].to_csv(out / c / "samples.csv", index=False)
    pd.DataFrame([dict(id=c, variable="group", reference=f"{c}_control", target=f"{c}_PEC",
                       blocking="genotype" if c == "young_132d" else "pair")]).to_csv(out / c / "contrasts.csv", index=False)
s = pd.concat(samples)
assert s["sample"].is_unique and len(s) == 47, len(s)
pd.DataFrame({"sample": s["sample"], "fastq_1": reads + "/" + s["sample"] + "_1.fastq.gz",
              "fastq_2": reads + "/" + s["sample"] + "_2.fastq.gz", "strandedness": "unstranded"}).to_csv(out / "samplesheet.csv", index=False)
