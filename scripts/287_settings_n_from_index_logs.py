"""Notebook 287: the database size N and the E-value fit of every one of notebook 241's 152
settings, for the sum statistic's search-space count.

N is the "total k-mers found" line of each human index's build log, the number kmerseek's
E-value uses (db_n_kmers). The logs are notebook 283's rebuild of 241's indexes
($SCRATCH/283-bhf-low-complexity/indexes/logs on Sherlock, the indexes 284's shuffled
search read), copied to INDEX_LOGS below on 2026-10-20. `fitted` is notebook 241's
arms.csv: whether the setting has a Karlin-Altschul fit, so that its calls get an E-value.

Why this exists: 287 first took its settings from the calls that had an E-value on the
three real queries. That gave 69, but 71 settings have a fit; two (gbmr4 k21,
hp_lehninger_hpc3 k25) never gave a real query a finite E-value, and a shuffled copy of
Ced9 did, which stopped scripts/287_shuffle_chains.py (job 48677545, 2026-10-20).

Output: notebooks/287_settings_N_human_index_logs.csv
"""

import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
INDEX_LOGS = D / "287/index_logs_283"


def main() -> None:
    rows = []
    for p in sorted(INDEX_LOGS.glob("*.index.log")):
        m = re.search(r"(\d+) total k-mers found", p.read_text())
        if m is None:
            raise SystemExit(f"{p}: no 'total k-mers found' line")
        rows.append(
            dict(setting=p.name.removesuffix(".index.log"), n_db=int(m.group(1)))
        )
    arms = pl.read_csv(D / "arms.csv").select(
        setting=pl.col("alphabet") + ".k" + pl.col("ksize").cast(pl.String),
        fitted=pl.col("fitted"),
    )
    out = (
        pl.DataFrame(rows)
        .join(arms, on="setting", how="full", coalesce=True)
        .sort("setting")
    )
    if out["n_db"].null_count() or out["fitted"].null_count() or out.height != 152:
        raise SystemExit(
            f"logs and arms.csv disagree: {out.filter(pl.any_horizontal(pl.all().is_null()))}"
        )
    # the 69 settings whose N 287_chain_scores.py rebuilt from the calls must agree exactly
    rebuilt = pl.read_csv(ROOT / "notebooks/287_settings_N_human.csv").select(
        "setting", rebuilt="n_db"
    )
    bad = out.join(rebuilt, on="setting").filter(pl.col("n_db") != pl.col("rebuilt"))
    if bad.height:
        raise SystemExit(f"index logs and rebuilt N differ: {bad}")
    out.write_csv(ROOT / "notebooks/287_settings_N_human_index_logs.csv")
    print(
        f"{out.height} settings, {out['fitted'].sum()} with an E-value fit; "
        f"{rebuilt.height} rebuilt from calls, all equal to the logs"
    )


if __name__ == "__main__":
    main()
