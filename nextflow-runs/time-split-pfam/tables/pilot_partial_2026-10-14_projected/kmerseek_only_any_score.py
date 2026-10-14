"""nb 280 pilot tune half (placed-domain scoring, MMseqs2 one round): instances kmerseek
finds for which MMseqs2, phmmer and hmmscan have no landing call of that family at any score."""
import sys
R = "/scratch/users/olgabot/280-time-split"; O = f"{R}/results-pilot/scores-partial-projected"
sys.path.insert(0, f"{R}/repo/nextflow-runs/time-split-pfam/bin")
import polars as pl
from score_transfers import is_p4, found_at_cutoff, INSTANCE_KEY
pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(60)
q = pl.read_csv(f"{R}/inputs/pilot/queries.tsv", separator="\t")
inst = pl.read_csv(f"{R}/inputs/pilot/instances.tsv", separator="\t")
hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))["query"]
inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
tune = is_p4(inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query").filter(pl.col("half") == "tune"))
ic = pl.read_parquet(f"{O}/instance_calls.parquet")
s = pl.read_csv(f"{O}/setting_summary.csv"); cut = dict(zip(s["setting"], s["cutoff_tune"]))
comp = ["MMseqs2", "phmmer", "hmmscan Pfam 33.1"]
other_any = ic.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique()
km = ic.filter(pl.col("setting").str.starts_with("kmerseek ")).join(tune.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
km_cut = found_at_cutoff(km, cut)
print("tune-half instances:", tune.height, "| with any landing call from another tool:", tune.join(other_any, on=INSTANCE_KEY, how="semi").height)
for label, k in (("kmerseek at its cutoff", km_cut), ("kmerseek at any score", km.select(["setting"] + INSTANCE_KEY))):
    only = k.join(other_any, on=INSTANCE_KEY, how="anti")
    per = (only.group_by(INSTANCE_KEY).agg(n_settings=pl.len(), settings=pl.col("setting").str.replace("kmerseek ", "").sort().head(4).str.join(", "))
           .join(tune.select(INSTANCE_KEY + ["pfam_id", "pfam_type", "length", "kingdom", "p4"]), on=INSTANCE_KEY)
           .join(q.select("query", pl.col("length").alias("query_len")), on="query")
           .sort("n_settings", descending=True))
    print(f"\n== {label}, no other tool at any score: {per.height} instances")
    print(per.select("query", "pfam_acc", "pfam_id", "pfam_type", "t1_start", "t1_end", "length", "query_len", "kingdom", "p4", "n_settings", "settings"))
    if label.endswith("any score"):
        best = (km.join(only.select(INSTANCE_KEY).unique(), on=INSTANCE_KEY, how="semi")
                .with_columns(margin=pl.col("score") - pl.col("setting").replace_strict(cut, default=None))
                .sort("margin", descending=True).unique(INSTANCE_KEY, keep="first"))
        print("best margin to the setting's cutoff (score minus cutoff; >= 0 means found):")
        print(best.select("query", "pfam_acc", "t1_start", "setting", "qstart", "qend", "score", "margin", "landing").sort("margin", descending=True))
        per.write_csv(f"{O}/kmerseek_only_any_score.csv")
