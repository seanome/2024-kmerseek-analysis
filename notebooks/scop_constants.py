from typing import Literal, get_args

import pandas as pd

SCOP_LINEAGES = Literal["family", "superfamily", "fold", "class"]

lineage_cols = get_args(SCOP_LINEAGES)
query_scop_cols = [f"query_{x}" for x in lineage_cols]
match_scop_cols = [f"match_{x}" for x in lineage_cols]
same_scop_cols = [f"same_{x}" for x in lineage_cols]
n_scop_cols = [f"n_{x}" for x in lineage_cols]


FOLDSEEK_SCOP_FIXED = pd.read_csv(
    # Originally from:
    # https://raw.githubusercontent.com/steineggerlab/foldseek-analysis/refs/heads/main/scopbenchmark/data/scop_lookup.fix.tsv
    "s3://seanome-kmerseek/scope-benchmark/steineggerlab_foldseek-analysis_scop_fixed.csv",
    index_col=0,
).squeeze()
