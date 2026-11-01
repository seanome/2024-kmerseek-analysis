import sys
sys.path.insert(0, "/scratch/users/olgabot/280-time-split/repo/nextflow-runs/time-split-pfam/bin")
import polars as pl
from score_transfers import is_p4, found_at_cutoff, INSTANCE_KEY
R = "/scratch/users/olgabot/280-time-split"; O = f"{R}/results-pilot/scores-partial"
q = pl.read_csv(f"{R}/inputs/pilot/queries.tsv", separator="\t")
inst = pl.read_csv(f"{R}/inputs/pilot/instances.tsv", separator="\t")
hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))["query"]
q = q.filter(~pl.col("query").is_in(hidden.implode())); inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
inst_s = inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query")
p4 = is_p4(inst_s).filter(pl.col("p4") & (pl.col("half") == "tune"))
ic = pl.read_parquet(f"{O}/instance_calls.parquet")
summ = pl.read_csv(f"{O}/setting_summary.csv")
cut = dict(zip(summ["setting"], summ["cutoff_tune"]))
found = found_at_cutoff(ic, cut)
km = [s for s in cut if s.startswith("kmerseek ")]; comp = [s for s in cut if not s.startswith("kmerseek ")]
by_comp = found.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique()
u = (found.filter(pl.col("setting").is_in(km)).join(p4.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
     .join(by_comp, on=INSTANCE_KEY, how="anti"))
# check against the scorer's own counts
chk = pl.read_csv(f"{O}/unique_finds_tune.csv").select("setting", "unique_finds").join(
    u.group_by("setting").agg(n=pl.len()), on="setting", how="left").with_columns(pl.col("n").fill_null(0))
print("counts match unique_finds_tune.csv:", chk.filter(pl.col("n") != pl.col("unique_finds")).height == 0)
# best call details, the instance's facts, and what each comparator did with it at any score
det = u.join(ic, on=["setting"] + INSTANCE_KEY, how="left").join(
    p4.select(INSTANCE_KEY + ["pfam_id", "pfam_type", "length", "kingdom", "uncharacterized_t0", "clan_t1"]), on=INSTANCE_KEY)
det = det.join(q.select("query", pl.col("length").alias("query_length"), "source"), on="query").with_columns(cutoff=pl.col("setting").replace_strict(cut, default=None))
anyc = ic.filter(pl.col("setting").is_in(comp)).group_by(INSTANCE_KEY).agg(
    comparator_any_score=pl.concat_str(pl.col("setting"), pl.lit(" "), pl.col("score").round(2)).str.join("; "))
det = det.join(anyc, on=INSTANCE_KEY, how="left")
det.write_csv(f"{O}/unique_finds_tune_instances.csv")
per = det.group_by(INSTANCE_KEY + ["pfam_id", "pfam_type", "length", "query_length", "kingdom"]).agg(
    n_settings=pl.len(), settings=pl.col("setting").str.replace("kmerseek ", "").sort().str.join(", "),
    comparator_any_score=pl.col("comparator_any_score").first()).sort("n_settings", descending=True)
per.write_csv(f"{O}/unique_finds_tune_by_instance.csv")
with pl.Config(tbl_rows=60, tbl_width_chars=260, fmt_str_lengths=200, tbl_cols=20):
    print(per.select("query", "pfam_acc", "pfam_id", "pfam_type", "t1_start", "t1_end", "length", "query_length", "kingdom", "n_settings", "comparator_any_score"))
    print(per.select("query", "pfam_id", "settings"))
