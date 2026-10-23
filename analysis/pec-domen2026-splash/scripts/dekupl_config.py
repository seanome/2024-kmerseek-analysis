#!/usr/bin/env python3
"""Write a dekupl-run config.json for one contrast, with the settings fixed in the analysis plan.

Usage: dekupl_config.py <design.tsv> <fastq_dir> <tmp_dir> <out config.json> [diff_method]

design.tsv: header with sample and condition (control|PEC). Condition A = control,
B = PEC, so DE-kupl's log2FC is log2(PEC / control), the same direction as the other channels.
"""
import csv
import json
import sys

design, fastq_dir, tmp_dir, out = sys.argv[1:5]
method = sys.argv[5] if len(sys.argv) > 5 else "limma"
if method not in ("limma", "DESeq2", "Ttest"):
    raise SystemExit(f"unknown diff_method {method}")
rows = list(csv.DictReader(open(design), delimiter="\t"))
conds = {r["condition"] for r in rows}
if conds != {"control", "PEC"}:
    raise SystemExit(f"need control and PEC, got {conds}")
cfg = {
    "fastq_dir": fastq_dir,
    "output_dir": "DEkupl_result",
    "tmp_dir": tmp_dir,
    "kmer_length": 31,
    "diff_method": method,
    "lib_type": "unstranded",
    "data_type": "WGS",
    "masking": "nomask",
    "r1_suffix": "_1.fastq.gz",
    "r2_suffix": "_2.fastq.gz",
    "seed": "fixed",
    "dekupl_counter": {"min_recurrence": 2, "min_recurrence_abundance": 5},
    "diff_analysis": {
        "condition": {"A": "control", "B": "PEC"},
        "pvalue_threshold": 0.05,
        "log2fc_threshold": 1,
    },
    "samples": [{"name": r["sample"], "condition": r["condition"]} for r in rows],
}
json.dump(cfg, open(out, "w"), indent=1)
print(f"{len(rows)} samples written to {out}")
