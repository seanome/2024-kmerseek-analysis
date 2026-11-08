"""kmerseek against MMseqs2 on every domain metric, for all proteins and for protein subsets.

Same inputs as summarize.py. For each kmerseek setting, each metric is divided by
MMseqs2's on the same species and subset, and the median over species is taken
(searches with no calls are left out; a subset needs at least 5 species). Ratios are
turned so that above 1 always means kmerseek is better (boundary error is lower-is-better,
so it is MMseqs2 / kmerseek). "Typical" is the median over the 101 settings; "best" is the
single best setting, which is picked after looking and so is an upper bound.
Run: python other_metrics.py <metrics_dir> <all_domain_metrics.parquet>
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig  # noqa: E402

NINE = ["arabidopsis", "chicken", "ciona", "ecoli", "fly", "mouse", "worm", "yeast", "zebrafish"]
# metric -> (label, higher is better)
METRICS = {
    "fmax": ("Fmax", True),
    "family_fmax": ("Fmax, family level", True),
    "recall_reachable": ("Recall, all calls", True),
    "fmax_precision": ("Precision at Fmax", True),
    "precision": ("Precision, all calls", True),
    "domain_count_accuracy": ("Domain count right", True),
    "median_iou_tp": ("Overlap with true domain (IoU)", True),
    "dbd_median": ("Boundary error, flipped", False),
}
SUBSETS = [
    ("all", "all", "All proteins"),
    ("geneset", "zinc_finger_c2h2", "C2H2 zinc fingers"),
    ("feature_length_bin", "16-30", "Domain 16-30 aa"),
    ("feature_length_bin", "31-60", "Domain 31-60 aa"),
    ("feature_length_bin", "61-120", "Domain 61-120 aa"),
    ("feature_length_bin", "121-250", "Domain 121-250 aa"),
    ("feature_length_bin", "251+", "Domain 251+ aa"),
    ("identity", "0-20%", "Identity 0-20%"),
    ("identity", "20-30%", "Identity 20-30%"),
    ("identity", "30-40%", "Identity 30-40%"),
    ("identity", "40-60%", "Identity 40-60%"),
    ("identity", "60-100%", "Identity 60-100%"),
]

metrics_dir, other_tools = Path(sys.argv[1]), Path(sys.argv[2])
k = pl.concat(
    [pl.read_parquet(f) for f in sorted(metrics_dir.glob("pfam.kmerseek.*_lcFalse.*.metrics.parquet"))
     if ".dedup." not in f.name],
    how="diagonal_relaxed",
)
ref = (
    pl.read_parquet(other_tools)
    .filter(pl.col("truth_set") == "pfam", ~pl.col("dedup_transfers"), pl.col("tool") == "mmseqs2_seqseq")
)
keys = ["species", "stratum_axis", "stratum"]
k = k.filter(pl.col("split") == "all", pl.col("species").is_in(NINE), pl.col("n_calls") > 0)
ref = ref.filter(pl.col("split") == "all", pl.col("species").is_in(NINE)).select(
    *keys, *[pl.col(m).alias(m + "_mmseqs2") for m in METRICS]
)
j = k.join(ref, on=keys).with_columns(pl.col("variant").str.replace("_lcFalse", "").alias("setting"))

rows = []
for m, (label, higher) in METRICS.items():
    r = (pl.col(m) / pl.col(m + "_mmseqs2")) if higher else (pl.col(m + "_mmseqs2") / pl.col(m))
    g = (
        j.with_columns(r.alias("ratio"))
        .filter(pl.col("ratio").is_finite())
        .group_by("setting", "stratum_axis", "stratum")
        .agg(pl.col("ratio").median(), pl.col(m).median().alias("kmerseek"),
             pl.col(m + "_mmseqs2").median().alias("mmseqs2"), pl.len().alias("n_species"))
        .filter(pl.col("n_species") >= 5)
    )
    rows.append(g.with_columns(pl.lit(m).alias("metric")))
per_setting = pl.concat(rows)
summary = (
    per_setting.group_by("metric", "stratum_axis", "stratum")
    .agg(
        pl.col("ratio").median().alias("typical_ratio"),
        pl.col("ratio").max().alias("best_ratio"),
        pl.col("setting").sort_by("ratio").last().alias("best_setting"),
        pl.col("kmerseek").sort_by("ratio").last().alias("best_setting_value"),
        pl.col("mmseqs2").first().alias("mmseqs2_value"),
        pl.len().alias("n_settings"),
    )
    .sort("metric", "stratum_axis", "stratum")
)
summary.write_csv(HERE / "kmerseek_vs_mmseqs2_all_metrics.csv")


def get(metric, axis, stratum):
    row = summary.filter(pl.col("metric") == metric, pl.col("stratum_axis") == axis, pl.col("stratum") == stratum)
    return (row["typical_ratio"][0], row["best_ratio"][0]) if row.height else (np.nan, np.nan)


pubfig.use_style()
fig, (ax_a, ax_b) = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=62, ncols=2)
TYP, BEST = pubfig.OKABE_ITO["blue"], pubfig.OKABE_ITO["vermillion"]


FLOOR = 0.01  # ratios of 0 (kmerseek found nothing) are drawn at the left edge


def dots(ax, labels, pairs):
    pairs = [(max(t, FLOOR), max(b, FLOOR)) for t, b in pairs]
    y = np.arange(len(labels))[::-1]
    for yi, (t, b) in zip(y, pairs):
        ax.plot([t, b], [yi, yi], color=pubfig.GREY, lw=0.5, zorder=1)
    ax.scatter([p[0] for p in pairs], y, color=TYP, s=12, zorder=2, label="Typical setting (median of 101)")
    ax.scatter([p[1] for p in pairs], y, color=BEST, s=12, zorder=2, label="Best single setting (chosen after looking)")
    ax.axvline(1, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks([FLOOR, 0.03, 0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["0.01\nor less", "0.03", "0.1", "0.3", "1", "3", "10"])
    ax.set_xlim(FLOOR * 0.8, 25)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)


dots(ax_a, [v[0] for v in METRICS.values()], [get(m, "all", "all") for m in METRICS])
ax_a.set_xlabel("kmerseek / MMseqs2, all proteins\n(above 1 = kmerseek better)")
dots(ax_b, [s[2] for s in SUBSETS], [get("fmax", a, s) for a, s, _ in SUBSETS])
ax_b.set_xlabel("kmerseek / MMseqs2, Fmax\n(above 1 = kmerseek better)")
pubfig.shared_legend(fig, ncol=2)
for ax, letter in zip((ax_a, ax_b), "ab"):
    pubfig.panel_label(ax, letter, dx_pt=-90)
fig.savefig(HERE / "kmerseek_vs_mmseqs2_all_metrics.pdf")
pubfig.save(fig, str(HERE / "kmerseek_vs_mmseqs2_all_metrics"), formats=("png",))

with pl.Config(tbl_rows=200, tbl_width_chars=200, fmt_str_lengths=40):
    print(summary.filter(pl.col("stratum_axis") == "all").select("metric", "typical_ratio", "best_ratio", "best_setting", "best_setting_value", "mmseqs2_value"))
    print(summary.filter(pl.col("metric") == "fmax", pl.col("stratum_axis").is_in(["geneset", "feature_length_bin", "identity"])).select("stratum_axis", "stratum", "typical_ratio", "best_ratio", "best_setting", "best_setting_value", "mmseqs2_value", "n_settings"))
