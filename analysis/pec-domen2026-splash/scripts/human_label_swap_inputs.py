#!/usr/bin/env python3
"""Study lists and gene tables under each swapped contrast's name (notebook 361).

Usage: human_label_swap_inputs.py <label_swap/contrasts.tsv> <published_genes.tsv> <gene_basemean.tsv[.gz]> <out_dir>

human_compare.py joins lists, random gene pools and reached genes by contrast name. A swapped
contrast is scored against its real contrast's list and pool, so each table is repeated once per
swap under the swap's name. Writes <out_dir>/published_genes.tsv and gene_basemean.tsv.
"""
import sys
from pathlib import Path

import polars as pl

swaps_p, pub_p, base_p, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
swaps = pl.read_csv(swaps_p, separator="\t", columns=["contrast", "base_contrast"]).rename({"contrast": "swap"})
for path, name in ((pub_p, "published_genes.tsv"), (base_p, "gene_basemean.tsv")):
    t = pl.read_csv(path, separator="\t")
    r = (t.rename({"contrast": "base_contrast"}).join(swaps, on="base_contrast")
         .rename({"swap": "contrast"}).select(t.columns))
    assert r.height == sum(t.filter(pl.col("contrast") == b).height for b in swaps["base_contrast"]), name
    r.write_csv(out / name, separator="\t")
    print(name, r.height)
