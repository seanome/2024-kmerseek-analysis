#!/usr/bin/env python3
"""Per run: how many reads have a chosen frame that matches Swiss-Prot, and how well.

Reads the per-run outputs of `pec-tools best-frame` ($S/sra_search/runs/<run>/<run>.hist.tsv and
<run>.summary.tsv) and writes:
- <out_stem>.tsv: one row per run with the read counts at each step (all reads, reads with any
  Swiss-Prot match in the chosen frame, reads at each cutoff) and the per-run CPU time;
- <out_stem>_hist.tsv: per run, reads by the share of the chosen frame's k-mers in Swiss-Prot
  (20 bins) and by the raw number of matching k-mers;
- the figure <out_stem>.{png,svg,pdf}.

Usage: sra_frame_matches.py <runs.tsv> <runs_dir> <setting label> <chosen_cutoff> <out_stem>
"""

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

runs_p, runs_dir, label, chosen, out_stem = sys.argv[1:]
chosen = float(chosen)
runs = pl.read_csv(runs_p, separator="\t", infer_schema_length=0)
steps, hists = [], []


def seconds(time_file: Path) -> float | None:
    if not time_file.exists():
        return None
    t = time_file.read_text()
    u = re.search(r"User time \(seconds\): ([\d.]+)", t)
    s = re.search(r"System time \(seconds\): ([\d.]+)", t)
    return float(u.group(1)) + float(s.group(1)) if u and s else None


for r in runs.iter_rows(named=True):
    d = Path(runs_dir) / r["run"]
    sm = d / f"{r['run']}.summary.tsv"
    if not sm.exists():
        continue
    s = dict(pl.read_csv(sm, separator="\t").iter_rows())
    row = {
        "run": r["run"],
        "sample": r["sample"],
        "dataset": r["dataset"],
        "condition": r["condition"],
        "contrast": r["contrast"],
        **s,
        "best_frame_cpu_s": seconds(d / "best_frame.time"),
    }
    for t in sorted(d.glob("manysketch.ge*.time")):
        row[f"manysketch_cpu_s_{t.name[len('manysketch.'):-len('.time')]}"] = seconds(t)
    steps.append(row)
    h = pl.read_csv(d / f"{r['run']}.hist.tsv", separator="\t").filter(
        pl.col("matches") > 0
    )
    frac = (
        h.with_columns(
            bin=(
                ((pl.col("matches") / pl.col("windows")) * 20).floor().clip(0, 19) / 20
            ).round(2)
        )
        .group_by("bin")
        .agg(pl.col("reads").sum())
    )
    hists.append(
        frac.with_columns(
            run=pl.lit(r["run"]), kind=pl.lit("share of k-mers in Swiss-Prot")
        )
    )
    cnt = h.group_by(bin=pl.col("matches").cast(pl.Float64)).agg(pl.col("reads").sum())
    hists.append(cnt.with_columns(run=pl.lit(r["run"]), kind=pl.lit("matching k-mers")))
st = pl.DataFrame(steps, infer_schema_length=None)
st = st.with_columns(gbp=pl.col("bases") / 1e9)
st.write_csv(f"{out_stem}.tsv", separator="\t")
hs = pl.concat(hists).select("run", "kind", "bin", "reads").sort("run", "kind", "bin")
hs.write_csv(f"{out_stem}_hist.tsv", separator="\t")
print(
    st.select(
        "run",
        "condition",
        "all_reads",
        "any_match",
        *[c for c in st.columns if c.startswith("fraction_ge")],
        "best_frame_cpu_s",
        "gbp",
    )
)

import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

CONTRASTS = [
    "young_2h",
    "young_24h",
    "old_24h",
    "young_13_49d",
    "young_132d",
    "old_41_93d",
    "not_in_SI",
]
order = st.with_columns(
    o0=(pl.col("dataset") != "PRJNA1219309").cast(pl.Int8),
    o1=pl.col("contrast").replace_strict(
        {c: i for i, c in enumerate(CONTRASTS)}, default=99
    ),
    o2=(pl.col("condition") == "PEC").cast(pl.Int8),
).sort("o0", "dataset", "o1", "o2", "all_reads")
runs_o = order["run"].to_list()
n = len(runs_o)
share_bins = [round(i / 20, 2) for i in range(20)]
count_bins = list(range(1, 51))


def matrix(kind, bins):
    m = np.full((n, len(bins)), np.nan)
    x = hs.filter(pl.col("kind") == kind)
    tot = dict(zip(order["run"], order["any_match"]))
    idx = {b: j for j, b in enumerate(bins)}
    ridx = {r: i for i, r in enumerate(runs_o)}
    for r, b, v in x.select("run", "bin", "reads").iter_rows():
        if r in ridx and b in idx and tot[r]:
            m[ridx[r], idx[b]] = 100 * v / tot[r]
    return np.nan_to_num(m, nan=0.0)


pf.use_style()
H = min(170, 40 + 0.9 * n)
fig, axs = pf.figure(
    pf.TWO_COLUMN_MM,
    H,
    ncols=3,
    sharey=True,
    gridspec_kw={"width_ratios": [1, 1.2, 0.8]},
)
a, b, c = axs
cmap = sns.color_palette("mako_r", as_cmap=True)
A = matrix("share of k-mers in Swiss-Prot", share_bins)
B = matrix("matching k-mers", [float(x) for x in count_bins])
vmax = float(np.percentile(np.concatenate([A.ravel(), B.ravel()]), 99))
ima = a.imshow(
    A,
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    extent=(0, 1, n - 0.5, -0.5),
    interpolation="nearest",
)
b.imshow(
    B,
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=vmax,
    extent=(0.5, 50.5, n - 0.5, -0.5),
    interpolation="nearest",
)
cb = fig.colorbar(ima, ax=[a, b], location="top", fraction=0.04, pad=0.02, shrink=0.6)
cb.set_label(
    f"Reads in the bin (% of the run's reads with any Swiss-Prot match; darkest = {vmax:.0f} % or more)"
)
a.axvline(0.5, color="white", ls=":", lw=0.8)
a.axvline(chosen, color="white", ls="-", lw=0.8)
a.set_xlabel(
    f"Share of chosen frame's k-mers in Swiss-Prot\n(white: cutoff {chosen:g} solid, 0.5 dotted)"
)
b.set_xlabel("Matching k-mers in chosen frame")
# group labels and separators
groups = order.with_columns(
    g=pl.when(pl.col("dataset") == "PRJNA1219309")
    .then(pl.col("contrast") + " " + pl.col("condition"))
    .otherwise(pl.lit("human ") + pl.col("dataset"))
)["g"].to_list()
ticks, labels, start = [], [], 0
for i in range(1, n + 1):
    if i == n or groups[i] != groups[start]:
        ticks.append((start + i - 1) / 2)
        labels.append(f"{groups[start].replace('_', ' ')} ({i - start})")
        if i < n:
            for ax in axs:
                ax.axhline(i - 0.5, color="white" if ax is not c else pf.GREY, lw=0.6)
        start = i
a.set_yticks(ticks, labels)
a.set_ylabel("Run (group, number of runs)")
y = np.arange(n)
c.scatter(
    100 * order["any_match"] / order["all_reads"],
    y,
    s=5,
    marker="o",
    facecolors="white",
    edgecolors="black",
    lw=0.5,
    label="any Swiss-Prot match",
)
fr = [col for col in order.columns if col.startswith("fraction_ge")]
for i, col in enumerate(fr):
    c.scatter(
        100 * order[col] / order["all_reads"],
        y,
        s=5,
        marker=["s", "^"][i % 2],
        color="black" if i == 0 else pf.OKABE_ITO["reddish_purple"],
        lw=0,
        label=f"share >= {col[len('fraction_ge_'):]}",
    )
c.set_xlim(0, 100)
c.set_xlabel("Reads kept (% of all reads)")
c.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 12 / H), ncol=1, fontsize=5)
a.set_title(f"Frame choice: {label}", loc="left")
for ax, letter in zip(axs, "abc"):
    pf.panel_label(ax, letter)
pf.save(fig, out_stem)
