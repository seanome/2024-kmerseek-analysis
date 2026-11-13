#!/usr/bin/env python3
"""Pick the kmerseek settings of the full time-split run: one alphabet-k pair per alphabet.

The leaderboard needs one row per tool and setting, not a sweep (Olga, 2026-10-16). For each
alphabet, the pair is the k whose **extend** search found the most tune-half instances in
the pilot, each at its own matched cutoff; the full run then searches that index twice,
extend and exact (exact scored on region_tfidf). 19 alphabets, 38 searches per chunk.

Rule, written before this script was run (the per-setting counts of 186 of the pilot's
settings had been seen, in tables/pilot_partial_2026-10-14_tfidf/per_setting_found.csv):
  1. Only the pilot's tune half: sample queries with half == "tune". The pilot scorer
     dropped the test half before any call was joined, so nothing here can read it.
  2. Candidates: the alphabet's extend settings with a search output for every pilot chunk
     and no chunk refused for lack of a Karlin-Altschul fit.
  3. Best = most tune-half instances found (a landing call of the right family at or above
     the setting's matched cutoff, as in score_transfers.found_at_cutoff).
  4. Ties: more of those found instances are P4 instances (Motif, Disordered, Repeat, or
     under 60 aa); then the larger k, whose search needs less memory.
  5. A setting named in --skip is never a candidate (it was not searched in the pilot).

Writes <out>/leaderboard_settings.tsv (alphabet, ksize, c_opt; the full run's --settings)
and <out>/leaderboard_settings_evidence.csv (every candidate with its counts and rank).
"""

import argparse
from pathlib import Path

import polars as pl

from score_transfers import INSTANCE_KEY, found_at_cutoff, is_p4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="pilot scores dir")
    ap.add_argument("--inputs", type=Path, required=True, help="pilot inputs dir")
    ap.add_argument(
        "--settings", type=Path, required=True, help="all-setting settings.tsv"
    )
    ap.add_argument("--skip", default="", help="alphabet:k,... left out of the pilot")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    q = pl.read_csv(a.inputs / "queries.tsv", separator="\t")
    inst = pl.read_csv(a.inputs / "instances.tsv", separator="\t")
    tune_q = q.filter((pl.col("role") == "sample") & (pl.col("half") == "tune"))
    tune = is_p4(
        inst.filter(pl.col("role") == "sample").join(
            tune_q.select("query"), on="query", how="semi"
        )
    )
    print(f"tune-half instances: {tune.height} on {tune['query'].n_unique()} queries")

    summ = pl.read_csv(a.scores / "setting_summary.csv")
    if "found_test" in summ.columns:
        raise SystemExit(
            "this scores dir was made with --read-test-half; refusing to read it"
        )
    cov = pl.read_csv(
        a.scores / "setting_coverage.csv", schema_overrides={"ksize": pl.String}
    )
    n_chunks = cov["chunks"].max()
    cutoffs = dict(zip(summ["setting"], summ["cutoff_tune"]))
    ic = pl.read_parquet(a.scores / "instance_calls.parquet")
    found = found_at_cutoff(ic, cutoffs).join(
        tune.select(INSTANCE_KEY + ["p4"]), on=INSTANCE_KEY, how="inner"
    )
    per = found.group_by("setting").agg(found=pl.len(), found_p4=pl.col("p4").sum())

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    st = pl.read_csv(a.settings, separator="\t", schema_overrides={"ksize": pl.String})
    cand = (
        st.join(
            cov.filter(pl.col("arm") == "extend"), on=["alphabet", "ksize"], how="left"
        )
        .with_columns(
            setting=pl.format("kmerseek {} k{} extend", "alphabet", "ksize"),
            skipped=pl.concat_str("alphabet", pl.lit(":"), "ksize").is_in(list(skip)),
        )
        .join(per, on="setting", how="left")
        .with_columns(
            pl.col("found").fill_null(0),
            pl.col("found_p4").fill_null(0),
            k=pl.col("ksize").cast(pl.Int64),
            candidate=(~pl.col("skipped"))
            & (pl.col("chunks") == n_chunks)
            & (pl.col("unfitted_chunks") == 0)
            & pl.col("setting").is_in(list(cutoffs)),
        )
        .sort(
            ["alphabet", "candidate", "found", "found_p4", "k"],
            descending=[False, True, True, True, True],
        )
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("alphabet"))
    )
    chosen = cand.filter(pl.col("candidate") & (pl.col("rank") == 1))
    missing = set(st["alphabet"]) - set(chosen["alphabet"])
    if missing:
        raise SystemExit(f"no candidate extend setting for {sorted(missing)}")
    bad = chosen.filter(pl.col("skipped"))
    assert bad.height == 0, bad

    a.out.mkdir(parents=True, exist_ok=True)
    cand.select(
        "alphabet",
        "ksize",
        "c_opt",
        "setting",
        "chunks",
        "unfitted_chunks",
        "skipped",
        "candidate",
        "found",
        "found_p4",
        "rank",
    ).write_csv(a.out / "leaderboard_settings_evidence.csv")
    chosen.sort("alphabet").select("alphabet", "ksize", "c_opt").write_csv(
        a.out / "leaderboard_settings.tsv", separator="\t"
    )
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print(
            chosen.sort("alphabet").select(
                "alphabet", "ksize", "c_opt", "found", "found_p4"
            )
        )
    print(f"{chosen.height} alphabet-k pairs, {2 * chosen.height} kmerseek settings")


if __name__ == "__main__":
    main()
