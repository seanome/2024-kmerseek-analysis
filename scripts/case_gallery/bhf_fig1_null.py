"""For BHF: each Figure 1 draft gene's best rank over notebook 241's 152 alphabet/k values,
under E-value, mean IDF and tf-idf, against notebook 241's random-protein null: at each
alphabet/k where the gene was ranked, draw one human protein from that hit list at random,
keep the best of those ranks, repeat 20,000 times. p = share of draws as good or better.
A protein's score is its best region; rank = 1 + number strictly better (notebook 241).
Writes tables/case_gallery/bhf_fig1_ranks.csv."""
import numpy as np
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[2]
from sources import D241 as D
FIG1 = ["ZNF292", "RSF1", "TSHZ1", "TSHZ2", "TSHZ3", "RNMT", "SFI1", "TRAPPC10", "NDNF"]
MET = [("E", "region_evalue", True), ("idf", "region_mean_idf", False), ("tfidf", "region_tfidf", False)]
reg = pl.read_parquet(D / "regions.parquet", columns=["alphabet", "ksize_arm", "query_name", "gene",
      "region_evalue", "region_mean_idf", "region_tfidf", "target_name"]).filter(pl.col("query_name") == "BHF")
reg = reg.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for _, c, _ in MET])
reg = reg.with_columns(pl.col("gene").alias("gene"))
per = reg.group_by("alphabet", "ksize_arm", "target_name", "gene").agg(
    pl.col("region_evalue").min().alias("E"), pl.col("region_mean_idf").max().alias("idf"), pl.col("region_tfidf").max().alias("tfidf"))
per = per.with_columns([pl.col(k).rank("min", descending=not low).over("alphabet", "ksize_arm").cast(pl.Int64).alias("rk_" + k) for k, _, low in MET])
rng = np.random.default_rng(241)
NDRAW = 20_000
arm_ranks = {}
for k, _, _ in MET:
    for (al, ks), sub in per.filter(pl.col("rk_" + k).is_not_null()).group_by("alphabet", "ksize_arm"):
        arm_ranks[(k, al, ks)] = sub["rk_" + k].to_numpy()
rows = []
for gene in FIG1:
    mine = per.filter(pl.col("gene") == gene)
    rec = {"gene": gene, "n_arms": mine.height}
    for k, _, _ in MET:
        m = mine.filter(pl.col("rk_" + k).is_not_null())
        if m.height == 0:
            rec.update({"best_" + k: None, "n_arms_" + k: 0, "at_" + k: "", "null_median_" + k: None, "p_random_as_good_" + k: None}); continue
        b = int(m["rk_" + k].min())
        draws = np.full(NDRAW, np.iinfo(np.int64).max)
        for al, ks in m.select("alphabet", "ksize_arm").iter_rows():
            r = arm_ranks[(k, al, ks)]
            draws = np.minimum(draws, r[rng.integers(0, len(r), NDRAW)])
        at = m.filter(pl.col("rk_" + k) == b).sort("alphabet", "ksize_arm")
        rec.update({"best_" + k: b, "n_arms_" + k: m.height,
                    "at_" + k: "; ".join(f"{x['alphabet']} k{x['ksize_arm']} (of {len(arm_ranks[(k, x['alphabet'], x['ksize_arm'])]):,})" for x in at.iter_rows(named=True)),
                    "null_median_" + k: float(np.median(draws)), "p_random_as_good_" + k: round(float((draws <= b).mean()), 3)})
    rows.append(rec)
out = pl.DataFrame(rows)
out.write_csv(ROOT / "tables" / "case_gallery" / "bhf_fig1_ranks.csv")
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
print(out.select("gene", *[c for k in ["E","idf","tfidf"] for c in ("best_"+k, "n_arms_"+k, "null_median_"+k, "p_random_as_good_"+k)]))
print(out.select("gene", "at_idf", "at_tfidf").head(4))

