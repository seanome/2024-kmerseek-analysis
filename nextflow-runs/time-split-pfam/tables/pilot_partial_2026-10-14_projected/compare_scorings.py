"""Compare three scorings of the nb 280 pilot tune half on the same kmerseek settings."""
import sys
R = "/scratch/users/olgabot/280-time-split"; P = f"{R}/results-pilot"
sys.path.insert(0, f"{R}/repo/nextflow-runs/time-split-pfam/bin")
import polars as pl
from score_transfers import is_p4, found_at_cutoff, INSTANCE_KEY
pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(80)
old = set(pl.read_csv(f"{P}/scores-partial/unique_finds_tune.csv")["setting"])
q = pl.read_csv(f"{R}/inputs/pilot/queries.tsv", separator="\t")
inst = pl.read_csv(f"{R}/inputs/pilot/instances.tsv", separator="\t")
hidden = q.filter((pl.col("role") == "sample") & (pl.col("half") == "test"))["query"]
q = q.filter(~pl.col("query").is_in(hidden.implode()))
inst = inst.filter(~pl.col("query").is_in(hidden.implode()))
inst_s = inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query")
p4 = is_p4(inst_s).filter(pl.col("p4") & (pl.col("half") == "tune"))
print("tune P4 instances:", p4.height, "on", p4["query"].n_unique(), "queries; kmerseek settings compared:", len(old))
comp = ["MMseqs2", "phmmer", "hmmscan Pfam 33.1"]
for name, O in (("whole hit, 3 rounds", "scores-partial"), ("whole hit, 1 round", "scores-partial-mmseqs-one-round"),
                ("placed domain, 1 round", "scores-partial-projected")):
    s = pl.read_csv(f"{P}/{O}/setting_summary.csv")
    u = pl.read_csv(f"{P}/{O}/unique_finds_tune.csv").filter(pl.col("setting").is_in(list(old)))
    ic = pl.read_parquet(f"{P}/{O}/instance_calls.parquet")
    cut = dict(zip(s["setting"], s["cutoff_tune"]))
    found = found_at_cutoff(ic, cut)
    fp4 = found.join(p4.select(INSTANCE_KEY), on=INSTANCE_KEY, how="semi")
    kmf = fp4.filter(pl.col("setting").is_in(list(old)))
    print(f"\n== {name}")
    for t in comp:
        r = s.filter(pl.col("setting") == t).row(0, named=True)
        print(f"  {t}: cutoff E {10 ** -r['cutoff_tune']:.3g}; controls found at any score "
              f"{r['controls_found_any_score']}/{r['controls']}; tune P4 instances found {fp4.filter(pl.col('setting') == t).height}")
    km_best = kmf.group_by("setting").len()["len"].max()
    print("  tune P4 instances found by any kmerseek setting:", kmf.select(INSTANCE_KEY).unique().height,
          "| best single setting:", km_best,
          "| by any comparator:", fp4.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique().height)
    print("  settings with a unique find:", u.filter(pl.col("unique_finds") > 0).height, "of", u.height,
          "| max:", u["unique_finds"].max(),
          "| top:", u.sort("unique_minus_expected", descending=True).head(3).select("setting", "unique_finds", "expected").rows())
    if O == "scores-partial-projected":
        byc = found.filter(pl.col("setting").is_in(comp)).select(INSTANCE_KEY).unique()
        un = kmf.join(byc, on=INSTANCE_KEY, how="anti")
        per = (un.group_by(INSTANCE_KEY).agg(n_settings=pl.len())
               .join(p4.select(INSTANCE_KEY + ["pfam_id", "pfam_type", "length"]), on=INSTANCE_KEY)
               .sort("n_settings", descending=True))
        print(per)
        per.write_csv(f"{P}/{O}/unique_finds_tune_by_instance.csv")
        print(open(f"{P}/{O}/p4.json").read())
