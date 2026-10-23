#!/usr/bin/env python3
"""Pick notebook 353's one alphabet and k from the development half, by the rule in
notebooks/353_prereg.md, and freeze it.

    select_setting.py <dev results dir> <out dir>

<dev results dir> holds one folder per setting (<alphabet>.k<k>/) with real_scores.tsv.gz and
shuffle_scores.tsv.gz (score.py output, gzipped by pull_results.sh). Writes <out dir>/dev_selection.tsv (every setting's score,
shuffle rate and whether it was excluded) and <out dir>/frozen_setting.json.

Rule: score = mean over the 36 (window length x substitution level) combinations of the
fraction found at E* = 0.1. Excluded when the fraction of dipeptide shuffles with at least
one region at E <= 0.1 against a record of a different protein is above 3 x (1 - e^-0.1).
Highest score wins; ties go to the smaller k, then to the alphabet with fewer classes. If
every setting is excluded, the smallest ratio of shuffle rate to (1 - e^-0.1) is kept.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl

E_STAR = 0.1
PREDICTED = 1 - math.exp(-E_STAR)
MAX_RATIO = 3.0
N_CLASSES = {"dayhoff6": 6, "sdm12": 12, "mmseqs12": 12}
N_CELLS = 36


def main(dev_dir: str, out_dir: str) -> int:
    rows = []
    for d in sorted(Path(dev_dir).iterdir()):
        if not (d / "real_scores.tsv.gz").exists():
            continue
        alphabet, k = d.name.rsplit(".k", 1)
        real = pl.read_csv(d / "real_scores.tsv.gz", separator="\t")
        shuf = pl.read_csv(d / "shuffle_scores.tsv.gz", separator="\t")
        cells = real.group_by("length", "n_within", "n_cross").agg(
            pl.col(f"found_{E_STAR:g}").mean().alias("recall")
        )
        assert cells.height == N_CELLS, (d, cells.height)
        rate = (shuf[f"n_other_{E_STAR:g}"] >= 1).mean()
        rows.append(
            {
                "alphabet": alphabet,
                "k": int(k),
                "n_classes": N_CLASSES[alphabet],
                "n_real_queries": real.height,
                "score_mean_recall_e0.1": cells["recall"].mean(),
                "n_shuffles": shuf.height,
                "shuffle_rate_e0.1": rate,
                "shuffle_rate_over_predicted": rate / PREDICTED,
            }
        )
    table = pl.DataFrame(rows).with_columns(
        (pl.col("shuffle_rate_over_predicted") > MAX_RATIO).alias("excluded")
    )
    eligible = table.filter(~pl.col("excluded"))
    if eligible.height:
        pick = eligible.sort(
            ["score_mean_recall_e0.1", "k", "n_classes"],
            descending=[True, False, False],
        ).row(0, named=True)
        how = "highest score among settings not excluded"
    else:
        pick = table.sort("shuffle_rate_over_predicted").row(0, named=True)
        how = "every setting excluded; smallest shuffle rate over prediction"
    table = table.with_columns(
        ((pl.col("alphabet") == pick["alphabet"]) & (pl.col("k") == pick["k"])).alias(
            "picked"
        )
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.sort("alphabet", "k").write_csv(out / "dev_selection.tsv", separator="\t")
    frozen = {
        "alphabet": pick["alphabet"],
        "k": pick["k"],
        "rule": how,
        "score_mean_recall_e0.1": pick["score_mean_recall_e0.1"],
        "shuffle_rate_e0.1": pick["shuffle_rate_e0.1"],
    }
    (out / "frozen_setting.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(table.sort("alphabet", "k"))
    print(frozen)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
