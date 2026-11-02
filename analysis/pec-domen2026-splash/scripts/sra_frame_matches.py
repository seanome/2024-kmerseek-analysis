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

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3)
a, b, c = axs
COND = {"PEC": pf.OKABE_ITO["reddish_purple"], "control": "black"}
for r in st.iter_rows(named=True):
    col = COND.get(r["condition"], pf.GREY)
    for ax, kind in [(a, "share of k-mers in Swiss-Prot"), (b, "matching k-mers")]:
        x = hs.filter((pl.col("run") == r["run"]) & (pl.col("kind") == kind)).sort(
            "bin"
        )
        if x.height == 0:
            continue
        y = 100 * x["reads"] / r["any_match"]
        if kind.startswith("share"):
            ax.step(x["bin"] + 0.05, y, where="pre", color=col, lw=0.5, alpha=0.6)
        else:
            ax.plot(x["bin"], y, color=col, lw=0.5, alpha=0.6)
for cond, col in COND.items():
    a.plot([], [], color=col, label=f"{cond} run")
a.plot([], [], color=pf.GREY, label="human run (paper's exercise series)")
a.axvline(0.5, color=pf.GREY, ls=":", lw=0.8, label="cutoff 0.5 (Olga's)")
a.axvline(
    chosen,
    color=pf.GREY,
    ls="-",
    lw=0.8,
    label=f"cutoff {chosen} (from the Botryllus test)",
)
a.set_xlim(0, 1)
a.set_xlabel("Share of chosen frame's k-mers in Swiss-Prot")
a.set_ylabel("Reads (% of reads with any match)")
b.set_xlabel("Matching k-mers in chosen frame")
b.set_ylabel("Reads (% of reads with any match)")
b.set_xlim(0, 50)
cols = [c for c in st.columns if c.startswith("fraction_ge")]
order = st.sort("dataset", "condition", "all_reads")
xs = range(order.height)
c.scatter(
    xs,
    100 * order["any_match"] / order["all_reads"],
    s=6,
    marker="o",
    facecolors="white",
    edgecolors="black",
    lw=0.5,
    label="any match",
)
marks = ["s", "^", "v"]
for i, col in enumerate(cols):
    c.scatter(
        xs,
        100 * order[col] / order["all_reads"],
        s=6,
        marker=marks[i % 3],
        color="black",
        lw=0,
        label=f"share >= {col[len('fraction_ge_'):]}",
    )
c.set_xlabel("Run (grouped by dataset and condition)")
c.set_ylabel("Reads kept (% of all reads)")
c.set_xticks([])
c.legend(loc="upper right", fontsize=5)
a.legend(loc="upper right", fontsize=5)
a.set_title(f"Frame choice: {label}", loc="left")
for ax, letter in zip(axs, "abc"):
    pf.panel_label(ax, letter)
pf.save(fig, out_stem)
