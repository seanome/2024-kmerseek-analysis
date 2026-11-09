"""kmerseek / MMseqs2 Fmax for each species, for C2H2 zinc fingers, domains of 16-30 aa and all proteins.

other_metrics.py reports the median over 9 species. This prints and draws the 9 values behind
that median, so one species cannot carry the result unseen. "Typical" is the median Fmax ratio
over all kmerseek settings with calls in that species; dayhoff6 k9 is the best setting overall.
Same inputs as other_metrics.py.
Run: python per_species.py <metrics_dir> <all_domain_metrics.parquet>
"""

import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import shrink_png  # noqa: E402

NINE = [
    "arabidopsis",
    "chicken",
    "ciona",
    "ecoli",
    "fly",
    "mouse",
    "worm",
    "yeast",
    "zebrafish",
]
SUBSETS = [
    ("all", "all", "All proteins"),
    ("geneset", "zinc_finger_c2h2", "C2H2 zinc fingers"),
    ("feature_length_bin", "16-30", "Domains 16-30 aa"),
]
BEST = "dayhoff6_k9"

metrics_dir, other_tools = Path(sys.argv[1]), Path(sys.argv[2])
keys = ["species", "stratum_axis", "stratum"]
cells = pl.DataFrame(
    [{"stratum_axis": a, "stratum": s, "subset": label} for a, s, label in SUBSETS]
)
k = (
    pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(
                metrics_dir.glob("pfam.kmerseek.*_lcFalse.*.metrics.parquet")
            )
            if ".dedup." not in f.name
        ],
        how="diagonal_relaxed",
    )
    .filter(
        pl.col("split") == "all", pl.col("species").is_in(NINE), pl.col("n_calls") > 0
    )
    .join(cells, on=["stratum_axis", "stratum"])
    .with_columns(pl.col("variant").str.replace("_lcFalse", "").alias("setting"))
)
ref = (
    pl.read_parquet(other_tools)
    .filter(
        pl.col("truth_set") == "pfam",
        ~pl.col("dedup_transfers"),
        pl.col("tool") == "mmseqs2_seqseq",
        pl.col("split") == "all",
        pl.col("species").is_in(NINE),
    )
    .join(cells, on=["stratum_axis", "stratum"])
    .select(
        *keys,
        "subset",
        pl.col("fmax").alias("mmseqs2_fmax"),
        "n_stratum_proteins",
        "n_truth_instances",
    )
)
j = k.join(ref.drop("subset"), on=keys).with_columns(
    (pl.col("fmax") / pl.col("mmseqs2_fmax")).alias("ratio")
)
per_species = (
    j.filter(pl.col("ratio").is_finite())
    .group_by("subset", "species")
    .agg(
        pl.col("ratio").median().alias("typical_ratio"),
        pl.col("ratio")
        .filter(pl.col("setting") == BEST)
        .first()
        .alias(f"{BEST}_ratio"),
        pl.col("fmax").median().alias("kmerseek_typical_fmax"),
        pl.col("fmax").filter(pl.col("setting") == BEST).first().alias(f"{BEST}_fmax"),
        pl.len().alias("n_settings"),
    )
    .join(ref, on=["subset", "species"], how="right")
    .select(
        "subset",
        "species",
        "n_stratum_proteins",
        "n_truth_instances",
        "mmseqs2_fmax",
        "kmerseek_typical_fmax",
        f"{BEST}_fmax",
        "typical_ratio",
        f"{BEST}_ratio",
        "n_settings",
    )
    .sort("subset", "species")
)
per_species.write_csv(HERE / "per_species_ratios.csv")
with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200, float_precision=3):
    print(per_species)

# One row per subset, one dot per species: is the median carried by a few species?
pubfig.use_style()
fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=48)
colours = {
    "typical_ratio": pubfig.OKABE_ITO["blue"],
    f"{BEST}_ratio": pubfig.OKABE_ITO["vermillion"],
}
labels = {
    "typical_ratio": "Typical setting (median over settings)",
    f"{BEST}_ratio": "dayhoff6 k9",
}
FLOOR = 0.01
for i, (_, _, label) in enumerate(reversed(SUBSETS)):
    sub = per_species.filter(pl.col("subset") == label)
    for col, dy in (("typical_ratio", 0.12), (f"{BEST}_ratio", -0.12)):
        vals = sub[col].drop_nulls().to_list()
        ax.scatter(
            [max(v, FLOOR) for v in vals],
            [i + dy] * len(vals),
            s=8,
            color=colours[col],
            alpha=0.7,
            linewidths=0,
            label=labels[col] if i == 0 else None,
        )
ax.axvline(1, color="black", lw=0.5)
ax.set_xscale("log")
ax.set_xlim(FLOOR * 0.8, 20)
ax.set_xticks([0.01, 0.1, 1, 10], ["0.01 or less", "0.1", "1", "10"])
ax.set_yticks(range(len(SUBSETS)), [s[2] for s in reversed(SUBSETS)])
ax.set_ylim(-0.6, len(SUBSETS) - 0.4)
ax.set_xlabel(
    "kmerseek / MMseqs2 Fmax, one dot per species\n(above 1 = kmerseek better)"
)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
fig.savefig(HERE / "per_species_ratios.png", dpi=300)
shrink_png.process(HERE / "per_species_ratios.png", check=False)
fig.savefig(HERE / "per_species_ratios.pdf")
