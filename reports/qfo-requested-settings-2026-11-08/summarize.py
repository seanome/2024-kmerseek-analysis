"""Pfam domain-finding Fmax for the kmerseek settings asked about on 2026-11-08.

Inputs (copied from Sherlock, qfo-pfam-region-benchmark/data/midi-plus/results/):
  metrics/pfam.kmerseek.<setting>.<species>.metrics.parquet  (one per search)
  all_domain_metrics.parquet                                   (other tools)
Run: python summarize.py <metrics_dir> <all_domain_metrics.parquet>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig  # noqa: E402

NINE = ["arabidopsis", "chicken", "ciona", "ecoli", "fly", "mouse", "worm", "yeast", "zebrafish"]
ASKED = {"protein20": range(5, 11), "dayhoff6": [9, 11, 13, 15, 17, 19, 21]}
# First ask: odd k 19-33. Follow-up the same day: k 25, 30, 35, 40, so H/P k-mers carry the
# same 25-40 bits as protein20 k6-9 and dayhoff6 k10-15 (one H/P letter is 1 bit).
HP_K = [19, 21, 23, 25, 27, 29, 30, 31, 33, 35, 40]
import math  # noqa: E402


def bits_per_letter(alphabet):
    """log2 of the alphabet size: 20 for protein20, 6 for dayhoff6, 3 for hp_lehninger_hpc3, 2 for other H/P."""
    return math.log2({"protein20": 20, "dayhoff6": 6, "hp_lehninger_hpc3": 3}.get(alphabet, 2))

metrics_dir, other_tools = Path(sys.argv[1]), Path(sys.argv[2])
files = [f for f in metrics_dir.glob("pfam.kmerseek.*_lcFalse.*.metrics.parquet") if ".dedup." not in f.name]
d = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").filter(
    (pl.col("split") == "all") & (pl.col("stratum_axis") == "all") & pl.col("species").is_in(NINE)
)
d = d.with_columns(
    pl.col("variant").str.extract(r"^(.*)_k(\d+)_", 1).alias("alphabet"),
    pl.col("variant").str.extract(r"_k(\d+)_", 1).cast(int).alias("k"),
)
is_hp = pl.col("alphabet").str.starts_with("hp_") & (pl.col("alphabet") != "hp_thomas_dill2_ext2")
d = d.filter(
    (is_hp & pl.col("k").is_in(HP_K))
    | ((pl.col("alphabet") == "protein20") & pl.col("k").is_in(list(ASKED["protein20"])))
    | ((pl.col("alphabet") == "dayhoff6") & pl.col("k").is_in(ASKED["dayhoff6"]))
)

b = pl.read_parquet(other_tools).filter(
    (pl.col("truth_set") == "pfam") & (pl.col("split") == "all") & (pl.col("stratum_axis") == "all")
    & pl.col("species").is_in(NINE) & (pl.col("tool") != "kmerseek") & ~pl.col("dedup_transfers")
)
mm = b.filter(pl.col("tool") == "mmseqs2_seqseq").select("species", pl.col("fmax").alias("mmseqs2_fmax"))
d = d.join(mm, on="species", how="left")

empty = d.filter(pl.col("n_calls") == 0).select("alphabet", "k", "species").sort("alphabet", "k", "species")
empty.write_csv(HERE / "empty_searches.csv")
summary = (
    d.filter(pl.col("n_calls") > 0)
    .group_by("alphabet", "k")
    .agg(
        pl.len().alias("n_species_with_calls"),
        pl.col("fmax").median().round(3).alias("median_fmax"),
        pl.col("recall_reachable").median().round(3).alias("median_recall_reachable"),
        pl.col("fmax_precision").median().round(3).alias("median_precision_at_fmax"),
        (pl.col("fmax") / pl.col("mmseqs2_fmax")).median().round(2).alias("median_fmax_over_mmseqs2"),
    )
    .sort("alphabet", "k")
    .with_columns(
        (pl.col("k") * pl.col("alphabet").map_elements(bits_per_letter, return_dtype=pl.Float64))
        .round(1).alias("bits")
    )
)
summary.write_csv(HERE / "kmerseek_requested_settings.csv")
tools = b.group_by("tool", "variant").agg(
    pl.len().alias("n_species"), pl.col("fmax").median().round(3).alias("median_fmax"),
    pl.col("recall_reachable").median().round(3).alias("median_recall_reachable"),
).sort("median_fmax", descending=True)
tools.write_csv(HERE / "other_tools.csv")

# Figure: median Fmax against k-mer bits (k x log2 alphabet size), one line per alphabet; other tools as gray reference lines.
pubfig.use_style()
fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=62)
colors = {"protein20": "#1b7837", "dayhoff6": "#762a83", "hp_thomas_dill2": "#e08214", "hp_pbotc_1st_ed2": "#2166ac"}
for (alpha,), g in summary.group_by("alphabet", maintain_order=True):
    g = g.sort("k")
    c = colors.get(alpha, "#bdbdbd")
    lw, z = (1.0, 3) if alpha in colors else (0.5, 2)
    ax.plot(g["bits"], g["median_fmax"], marker="o", ms=2, lw=lw, color=c, zorder=z,
            label=alpha if alpha in colors else None)
ax.plot([], [], color="#bdbdbd", lw=0.5, marker="o", ms=2, label="other H/P alphabets")
for row in tools.filter(pl.col("tool").is_in(["hmmer3_phmmer", "mmseqs2_seqseq"])).iter_rows(named=True):
    ax.axhline(row["median_fmax"], color="black", lw=0.5, ls="--")
    # phmmer (0.321) sits just above MMseqs2 (0.310): label one above its line, one below.
    above = row["tool"] == "hmmer3_phmmer"
    ax.text(summary["bits"].max(), row["median_fmax"] + (0.003 if above else -0.003),
            row["tool"].replace("hmmer3_", ""), fontsize=5, va="bottom" if above else "top", ha="right")
ax.set_xlabel("bits per k-mer (k × log2 of alphabet size)")
ax.set_ylabel("Pfam domain Fmax\n(median over species with output)")
ax.set_ylim(0, 0.36)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.8), frameon=False, fontsize=5)
ax.set_title("kmerseek reaches 1/5 to 1/16 of MMseqs2's Pfam domain Fmax", fontsize=6)
fig.savefig(HERE / "fmax_by_k.png", dpi=300)
fig.savefig(HERE / "fmax_by_k.pdf")
print(summary)
print(tools)
print(f"{empty.height} empty searches")
