from typing import Literal

import pandas as pd

SCOP_LINEAGES = Literal["family", "superfamily", "fold", "class"]


FOLDSEEK_SCOP_FIXED = pd.read_csv(
    # Originally from:
    # https://raw.githubusercontent.com/steineggerlab/foldseek-analysis/refs/heads/main/scopbenchmark/data/scop_lookup.fix.tsv
    "s3://seanome-kmerseek/scope-benchmark/steineggerlab_foldseek-analysis_scop_fixed.csv",
    index_col=0,
).squeeze()
