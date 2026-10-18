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

    draw(
        raw.filter(pl.col("run") == "midi-plus"),
        best.filter(pl.col("run") == "midi-plus"),
        args.figdir / "pfamn_denominator_before_after_qfo_midi_plus",
    )


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
