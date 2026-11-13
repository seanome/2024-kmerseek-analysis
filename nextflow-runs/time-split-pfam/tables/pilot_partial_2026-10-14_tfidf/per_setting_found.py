"""Tune-half instances found at each setting's matched cutoff, nb 280 pilot (placed domains,
one-round MMseqs2), exact arm on Poisson vs tf-idf. Writes per_setting_found.csv."""
import sys
R = "/scratch/users/olgabot/280-time-split"; P = f"{R}/results-pilot"
sys.path.insert(0, f"{R}/repo/nextflow-runs/time-split-pfam/bin")
import polars as pl
from score_transfers import is_p4, found_at_cutoff, INSTANCE_KEY
q = pl.read_csv(f"{R}/inputs/pilot/queries.tsv", separator="\t")
inst = pl.read_csv(f"{R}/inputs/pilot/instances.tsv", separator="\t")
hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))["query"]
inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
tune = is_p4(inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query").filter(pl.col("half") == "tune"))
common = set(pl.read_csv(f"{P}/scores-partial-projected/setting_summary.csv")["setting"]) & set(pl.read_csv(f"{P}/scores-partial-tfidf/setting_summary.csv")["setting"])
rows = []
for score, O in (("poisson", "scores-partial-projected"), ("tfidf", "scores-partial-tfidf")):
    s = pl.read_csv(f"{P}/{O}/setting_summary.csv"); cut = dict(zip(s["setting"], s["cutoff_tune"]))
    f = found_at_cutoff(pl.read_parquet(f"{P}/{O}/instance_calls.parquet"), cut).join(tune.select(INSTANCE_KEY + ["p4"]), on=INSTANCE_KEY)
    for st in sorted(common):
        g = f.filter(pl.col("setting") == st)
        rows.append({"exact_score": score, "setting": st, "found": g.height, "found_p4": g.filter("p4").height})
out = pl.DataFrame(rows).with_columns(
    tool=pl.when(pl.col("setting").str.starts_with("kmerseek")).then(pl.lit("kmerseek")).otherwise(pl.col("setting")),
    arm=pl.col("setting").str.split(" ").list.last())
out.write_csv(f"{P}/per_setting_found.csv")
print(out.filter(~pl.col("setting").str.starts_with("kmerseek")))
print(out.filter(pl.col("setting").str.starts_with("kmerseek")).group_by("exact_score", "arm").agg(n=pl.len(), zero=(pl.col("found") == 0).sum(), median=pl.col("found").median(), max=pl.col("found").max()).sort("exact_score", "arm"))
