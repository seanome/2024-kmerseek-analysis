"""Keep query_id, framework, slack and target_accession of each regex hit file, as parquet.

The slack-2 regex hit files repeat the regex on every row (Mollusca: 659 MB); notebook 297
needs only the pairs. Run on Sherlock as job 48306700 (2026-10-17):

    sbatch -A ayeletv -p hns -c 2 --mem=16G --time=0:20:00 -J cys-regex --wrap \
        'apptainer exec --bind /scratch $IMG python3 regex_pairs.py $RUN/results/regex $RUN/reduced'
"""

import sys
from pathlib import Path

import polars as pl

for p in sorted(Path(sys.argv[1]).glob("*.regex.d*.hits.csv")):
    t = (
        pl.read_csv(p)
        .select("query_id", "framework", "slack", "target_accession")
        .unique()
        .sort("query_id", "target_accession")
    )
    t.write_parquet(Path(sys.argv[2]) / p.name.replace(".hits.csv", ".pairs.parquet"))
    print(p.name, t.height)
