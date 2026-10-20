"""How much the Pfam-N recall denominator bug moved each tool, on the mini, midi and
midi-plus runs.

Before restrictTruthToQueries (main.nf), every QfO region run scored Pfam-N against all
44,315 human Pfam-N regions on 18,724 proteins. The searched proteins carry far fewer:
midi-plus 2,218 regions on 966 proteins, midi 2,137 on 932, mini 2,701 on 196.

The numbers come from data/<run>/results/pfamn_rescored_whole_set.csv, written on
Sherlock by `make rescore-pfamn-from-calls`. That job recomputed both the old and the new
whole-set numbers from each run's call files, and stopped if the old ones differed from
the run's own all_domain_metrics.parquet. Only call files with a row there are used.

Two checks against values computed here without the call files:

  recall  regions found do not change (every call is on a searched protein), so the new
          recall must be the old one times regions before / regions after.
  Fmax    a never-searched protein has recall 0, so at every score cut-off the new recall
          is the old one times proteins before / proteins after, and precision does not
          change. F1 at the old best cut-off with that recall is a point on the new curve,
          so the exact new Fmax must be at least that.

Writes results/leaderboard/pfamn_denominator_before_after.csv (one row per run, tool,
setting, species, with dedup on and off), results/leaderboard/
pfamn_denominator_before_after_by_tool.csv (medians over species), and the figure
figures/pfamn_denominator_before_after_qfo_midi_plus.{pdf,svg,png}.

Run: python scripts/pfamn_denominator_before_after.py --data $LEADERBOARD_DATA
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

QFO = "2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data"
RUNS = ["midi-plus", "midi", "mini"]
TOOL_ORDER = [
    "kmerseek",
    "mmseqs2_seqseq",
    "mmseqs2_iterative",
    "hmmer3_phmmer",
    "hmmer3_jackhmmer",
    "hhblits",
    "foldseek",
    "prostt5",
    "folddisco",
    "reseek",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/leaderboard"))
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    rows = []
    for run in RUNS:
        r = pl.read_csv(
            args.data / QFO / run / "results/pfamn_rescored_whole_set.csv"
        ).filter(pl.col("in_published_metrics"))
        print(
            f"{run}: {r.height} call files; Pfam-N "
            f"{r['n_truth_instances_before'][0]:,} regions on "
            f"{r['n_stratum_proteins_before'][0]:,} proteins -> "
            f"{r['n_truth_instances_after'][0]:,} on "
            f"{r['n_stratum_proteins_after'][0]:,}"
        )
        rc = pl.col("fmax_recall_before") * (
            pl.col("n_stratum_proteins_before") / pl.col("n_stratum_proteins_after")
        )
        pr = pl.col("fmax_precision_before")
        rows.append(
            r.select(
                "run",
                "tool",
                pl.col("variant").alias("setting"),
                "species",
                "dedup_transfers",
                pl.col("n_instances_found_after").alias("regions_found"),
                pl.col("n_truth_instances_before").alias("n_regions_before"),
                pl.col("n_truth_instances_after").alias("n_regions_after"),
                pl.col("n_stratum_proteins_before").alias("n_proteins_before"),
                pl.col("n_stratum_proteins_after").alias("n_proteins_after"),
                "recall_before",
                "recall_after",
                "fmax_before",
                "fmax_after",
                pl.when(pr + rc > 0)
                .then(2 * pr * rc / (pr + rc))
                .otherwise(0.0)
                .alias("fmax_after_lower_bound"),
            )
        )
    df = pl.concat(rows).sort("run", "tool", "setting", "species", "dedup_transfers")

    ratio = df["recall_after"] / (
        df["recall_before"] * df["n_regions_before"] / df["n_regions_after"]
    )
    assert ratio.drop_nans().is_between(1 - 1e-9, 1 + 1e-9).all(), "recall check"
    assert (
        df["fmax_after"] >= df["fmax_after_lower_bound"] - 1e-12
    ).all(), "Fmax check"
    print(f"both checks hold on all {df.height} rows")

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.outdir / "pfamn_denominator_before_after.csv")

    # One row per (run, tool): the median over species of the setting with the highest
    # median Fmax after the fix, raw calls (dedup off). Every other tool has one setting.
    raw = df.filter(~pl.col("dedup_transfers"))
    per_setting = raw.group_by("run", "tool", "setting").agg(
        pl.len().alias("n_species"),
        pl.col("n_regions_before").first(),
        pl.col("n_regions_after").first(),
        pl.col("n_proteins_before").first(),
        pl.col("n_proteins_after").first(),
        pl.col("recall_before").median(),
        pl.col("recall_after").median(),
        pl.col("fmax_before").median(),
        pl.col("fmax_after").median(),
    )
    best = (
        per_setting.sort(["fmax_after", "setting"], descending=[True, False])
        .group_by("run", "tool", maintain_order=True)
        .first()
        .with_columns(
            pl.col("run")
            .replace_strict({r: i for i, r in enumerate(RUNS)})
            .alias("_r"),
            pl.col("tool")
            .replace_strict({t: i for i, t in enumerate(TOOL_ORDER)}, default=99)
            .alias("_o"),
        )
        .sort("_r", "_o")
        .drop("_r", "_o")
    )
    best.write_csv(args.outdir / "pfamn_denominator_before_after_by_tool.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, float_precision=4):
        print(best)

    # Every setting of every tool, not only the best: 406 kmerseek settings on midi, 15 on
    # midi-plus, 2 on mini. Same medians over species, raw calls.
    every = per_setting.with_columns(
        pl.col("setting")
        .str.extract(r"^(.*)_k\d+_lc(?:True|False)$", 1)
        .alias("alphabet"),
        pl.col("setting").str.extract(r"_k(\d+)_lc", 1).cast(pl.Int32).alias("k"),
        (pl.col("setting").str.extract(r"_lc(True|False)$", 1) == "True").alias(
            "low_complexity_masked"
        ),
    ).sort("run", "tool", "alphabet", "k", "low_complexity_masked", "setting")
    every.write_csv(args.outdir / "pfamn_denominator_before_after_by_setting.csv")
    print(
        every.group_by("run", "tool")
        .agg(pl.col("setting").n_unique().alias("settings"))
        .sort("run", "tool")
    )
    draw_every_kmerseek_setting(
        every.filter(pl.col("run") == "midi"),
        args.figdir / "pfamn_fmax_every_kmerseek_setting_qfo_midi",
    )

    draw(
        raw.filter(pl.col("run") == "midi-plus"),
        best.filter(pl.col("run") == "midi-plus"),
        args.figdir / "pfamn_denominator_before_after_qfo_midi_plus",
    )


# Letters per alphabet, from the number at the end of its name (hp_thomas_dill2 has 2,
# protein20 has 20), grouped by size with a gap between groups (clear-figures 7d).
SIZE_GROUPS = [
    ("2-3 letters (hydrophobic-polar)", 2, 3),
    ("4-8 letters", 4, 8),
    ("12-18 letters", 12, 18),
    ("20 letters (no reduction)", 20, 20),
]


def draw_every_kmerseek_setting(every: pl.DataFrame, stem: Path):
    """Median Pfam-N Fmax after the fix for every kmerseek setting on one run, against k.

    One panel per alphabet-size group. Each alphabet is one colour, with its k values
    joined by a line (k is ordered); filled = low-complexity regions not masked, open =
    masked. The other tools are grey horizontal lines, labelled, in every panel.
    """
    pf.use_style()
    km = every.filter(pl.col("tool") == "kmerseek").with_columns(
        pl.col("alphabet").str.extract(r"(\d+)$", 1).cast(pl.Int32).alias("letters")
    )
    others = every.filter(pl.col("tool") != "kmerseek").sort("fmax_after")
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 120, nrows=2, ncols=2, sharey=True)
    colours = list(pf.OKABE_ITO.values())
    ymax = max(km["fmax_after"].max(), others["fmax_after"].max()) * 1.08
    for ax, (title, lo, hi), letter in zip(axs.flat, SIZE_GROUPS, "abcd"):
        g = km.filter(pl.col("letters").is_between(lo, hi))
        for (alph,), c in zip(
            g.select("alphabet").unique().sort("alphabet").iter_rows(), colours
        ):
            a = g.filter(pl.col("alphabet") == alph)
            # Masked first, as a larger ring, so an unmasked setting with the same score
            # still shows as the filled dot inside it.
            for masked, face, size in ((True, "white", 16), (False, c, 5)):
                s = a.filter(pl.col("low_complexity_masked") == masked).sort("k")
                ax.plot(s["k"], s["fmax_after"], color=c, lw=0.6, alpha=0.7)
                ax.scatter(
                    s["k"], s["fmax_after"], s=size, facecolor=face, edgecolor=c,
                    linewidth=0.6, zorder=3,
                    label=alph if not masked else None,
                )  # fmt: skip
        x0, x1 = ax.get_xlim()
        for r in others.iter_rows(named=True):
            ax.axhline(r["fmax_after"], color=pf.GREY, lw=0.4, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{title}: {g['setting'].n_unique()} settings", loc="left")
        ax.set_xlabel("k (letters per k-mer)")
        ax.legend(
            loc="center right",
            ncol=1,
            handletextpad=0.2,
            columnspacing=0.6,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1,
        )
        pf.panel_label(ax, letter, dx_pt=-30)
    for ax in axs[:, 0]:
        ax.set_ylabel("Pfam-N Fmax after the fix\n(median over 9 target proteomes)")
    # Other tools named once, at the right edge of the top-right panel. Tools within 0.015
    # of each other share one label, so the names do not print on top of each other.
    groups = []
    for r in others.sort("fmax_after", descending=True).iter_rows(named=True):
        if groups and groups[-1][-1]["fmax_after"] - r["fmax_after"] < 0.015:
            groups[-1].append(r)
        else:
            groups.append([r])
    ax = axs[0, 1]
    for g in groups:
        y = sum(r["fmax_after"] for r in g) / len(g)
        ax.annotate(
            ", ".join(r["tool"] for r in g), (1.0, y),
            xycoords=("axes fraction", "data"), xytext=(3, 0),
            textcoords="offset points", va="center", fontsize=5, color="#555555",
            annotation_clip=False,
        )  # fmt: skip
    fig.suptitle(
        "QfO midi: 964 human query proteins, 9 target proteomes, raw calls; Fmax over "
        "the 2,137 Pfam-N regions on searched proteins. Filled dot: low-complexity regions "
        "kept; ring: masked. Grey lines: the other tools. Higher is better.",
        fontsize=6,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    pf.save(fig, str(stem))


def draw(raw: pl.DataFrame, best: pl.DataFrame, stem: Path):
    """Two panels, recall and Fmax, one column per tool; each species is one point."""
    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 75, ncols=2)
    before_c, after_c = pf.GREY, pf.OKABE_ITO["blue"]
    tools = [t for t in TOOL_ORDER if t in set(best["tool"])]
    pick = dict(zip(best["tool"], best["setting"]))
    n_reg, n_prot = best["n_regions_after"][0], best["n_proteins_after"][0]
    for ax, (b, a, ylab) in zip(
        axs,
        [
            (
                "recall_before",
                "recall_after",
                "Pfam-N recall (regions found / regions)",
            ),
            ("fmax_before", "fmax_after", "Pfam-N Fmax"),
        ],
    ):
        for i, t in enumerate(tools):
            s = raw.filter((pl.col("tool") == t) & (pl.col("setting") == pick[t]))
            n = s.height
            xs = [i - 0.15 + 0.3 * j / max(n - 1, 1) for j in range(n)]
            ax.scatter(
                [x - 0.2 for x in xs],
                s[b],
                s=6,
                color=before_c,
                label=(
                    "before: all 44,315 regions on 18,724 proteins" if i == 0 else None
                ),
            )
            ax.scatter(
                [x + 0.2 for x in xs],
                s[a],
                s=6,
                color=after_c,
                label=(
                    f"after: {n_reg:,} regions on the {n_prot:,} searched proteins "
                    "that have Pfam-N"
                    if i == 0
                    else None
                ),
            )
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(
            [t if t != "kmerseek" else f"kmerseek\n{pick[t]}" for t in tools],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.6, len(tools) - 0.4)
    pf.shared_legend(
        fig,
        ncol=1,
        title=(
            "QfO midi-plus: 998 human query proteins searched against 9 QfO proteomes. "
            "One point per target proteome, raw calls. Higher is better."
        ),
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter, dx_pt=-30)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pf.save(fig, str(stem))


if __name__ == "__main__":
    main()
