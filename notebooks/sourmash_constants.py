from typing import Literal, get_args

MOLTYPES = Literal["protein", "dayhoff", "hp"]
SOURMASH_SCORE_COLS = Literal[
    "containment",
    "tf_idf_score",
    "containment_adjusted_log10",
    "log10_prob_overlap_adjusted",
    "log10_max_containment",
    "log10_jaccard",
    "log10_tf_idf_score",
    "intersect_hashes",
]

sourmash_score_cols = get_args(SOURMASH_SCORE_COLS)
