"""nb 280 pilot tune half: exact arm scored on region_poisson_score vs region_tfidf
(both with domains placed through the hit and one-round MMseqs2), on the same settings."""
import sys
R = "/scratch/users/olgabot/280-time-split"; P = f"{R}/results-pilot"
sys.path.insert(0, f"{R}/repo/nextflow-runs/time-split-pfam/bin")
import polars as pl
from score_transfers import is_p4, found_at_cutoff, INSTANCE_KEY
pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(70)
q = pl.read_csv(f"{R}/inputs/pilot/queries.tsv", separator="\t")
inst = pl.read_csv(f"{R}/inputs/pilot/instances.tsv", separator="\t")
hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))["query"]
inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
tune = is_p4(inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query").filter(pl.col("half") == "tune"))
p4 = tune.filter("p4")
comp = ["MMseqs2", "phmmer", "hmmscan Pfam 33.1"]
runs = {"poisson": "scores-partial-projected", "tfidf": "scores-partial-tfidf"}
common = None
for O in runs.values():
    st = set(pl.read_csv(f"{P}/{O}/setting_summary.csv")["setting"])
    common = st if common is None else common & st
km = sorted(x for x in common if x.startswith("kmerseek "))
print("settings in both:", len(km), "kmerseek (", sum(x.endswith(" exact") for x in km), "exact,", sum(x.endswith(" extend") for x in km), "extend ); tune instances", tune.height, "of which P4", p4.height)
for name, O in runs.items():
    s = pl.read_csv(f"{P}/{O}/setting_summary.csv"); cut = dict(zip(s["setting"], s["cutoff_tune"]))
    ic = pl.read_parquet(f"{P}/{O}/instance_calls.parquet")
    found = found_at_cutoff(ic, cut)
    f_all = found.join(tune.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
    f_p4 = found.join(p4.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
    byc = f_all.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique()
    print(f"\n== exact scored on {name}")
    for arm in ("exact", "extend"):
        sets = [x for x in km if x.endswith(" " + arm)]
        per = f_all.filter(pl.col("setting").is_in(sets)).group_by("setting").len()
        per_p4 = f_p4.filter(pl.col("setting").is_in(sets)).group_by("setting").len()
        n_any = f_all.filter(pl.col("setting").is_in(sets)).select(INSTANCE_KEY).unique().height
        print(f"  {arm}: settings finding anything {per.height}/{len(sets)}; best setting finds {per['len'].max() if per.height else 0} instances "
              f"({per_p4['len'].max() if per_p4.height else 0} P4); any {arm} setting {n_any}")
    print("  comparators together:", byc.height, "instances;", f_p4.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique().height, "P4")
    un = f_all.filter(pl.col("setting").is_in(km)).join(byc, on=INSTANCE_KEY, how="anti")
    per = (un.group_by(INSTANCE_KEY).agg(n_settings=pl.len(), settings=pl.col("setting").str.replace("kmerseek ", "").sort().head(3).str.join(", "))
           .join(tune.select(INSTANCE_KEY + ["pfam_id", "pfam_type", "length", "p4"]), on=INSTANCE_KEY).sort("n_settings", descending=True))
    print(f"  instances a kmerseek setting finds at its cutoff and no comparator finds at its cutoff: {per.height}")
    if per.height:
        print(per.select("query", "pfam_id", "pfam_type", "t1_start", "t1_end", "length", "p4", "n_settings", "settings"))
        per.write_csv(f"{P}/{O}/kmerseek_unique_finds_tune_all_types.csv")
    u = pl.read_csv(f"{P}/{O}/unique_finds_tune.csv").filter(pl.col("setting").is_in(km))
    print("  P4 unique finds: settings with any", u.filter(pl.col("unique_finds") > 0).height, "| top:",
          u.sort("unique_minus_expected", descending=True).head(3).select("setting", "unique_finds", "expected").rows())
