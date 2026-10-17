"""How much the Pfam-N recall denominator bug moved each tool, on the midi and midi-plus runs.

Before restrictTruthToQueries (main.nf), every QfO region run scored Pfam-N against all
44,315 human Pfam-N regions on 18,724 proteins. midi-plus searched 998 proteins, which
carry 2,218 of those regions on 966 proteins; midi searched 964 proteins, carrying 2,137
regions on 932 proteins.

What can be computed exactly from the metrics already on disk, and what cannot:

  recall  = regions found / regions in the truth. Calls exist only on searched proteins,
            so the regions found do not change. After = n_instances_found / n_after.
            Exact.
  Fmax    = best F1 over score cut-offs, with precision averaged over proteins that got a
            call and recall averaged over proteins in the truth. A protein never searched
            has recall 0, so at EVERY cut-off the new recall is the old one times
            18,724 / (proteins after). Precision does not change. The metrics keep only
            the old best cut-off, so F1 there with the scaled recall is a value the new
            curve reaches: new Fmax is at least that. The exact value needs the rescore.

Writes results/leaderboard/pfamn_denominator_before_after.csv (one row per run, tool,
setting, species, with dedup on and off) and
results/leaderboard/pfamn_denominator_before_after_by_tool.csv (medians over species), and
the figure figures/pfamn_denominator_before_after_qfo_midi_plus.{pdf,svg,png}.

Run: python scripts/pfamn_denominator_before_after.py --data $LEADERBOARD_DATA
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

QFO = "2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data"
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


def query_proteins(data: Path) -> dict[str, set[str]]:
    """midi-plus queries are its Pfam-A truth proteins; midi's are the chr6 subset of them."""
    mp = data / QFO / "midi-plus/results/truth"
    plus = set(pl.read_parquet(mp / "human_domain_truth.parquet")["accession"])
    cov = pl.read_parquet(mp / "human_query_covariates.parquet")
    chr6 = set(cov.filter(pl.col("query_set") == "chr6")["accession"])
    return {"midi-plus": plus, "midi": chr6}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/leaderboard"))
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    pfamn = pl.read_parquet(args.data / QFO / "pfamn/human_pfamn_truth.parquet")
    n_regions_before = pfamn.height
    n_proteins_before = pfamn["accession"].n_unique()

    rows = []
    for run, queries in query_proteins(args.data).items():
        kept = pfamn.filter(pl.col("accession").is_in(queries))
        n_regions_after = kept.height
        n_proteins_after = kept["accession"].n_unique()
        print(
            f"{run}: {len(queries):,} query proteins; Pfam-N {n_regions_before:,} regions "
            f"on {n_proteins_before:,} proteins -> {n_regions_after:,} on "
            f"{n_proteins_after:,}"
        )
        m = pl.read_parquet(
            args.data / QFO / run / "results/all_domain_metrics.parquet"
        )
        m = m.filter(
            (pl.col("truth_set") == "pfamn")
            & (pl.col("stratum_axis") == "all")
            & (pl.col("split") == "all")
        )
        assert set(m["n_truth_instances"]) == {n_regions_before}, "already restricted?"
        scale = n_proteins_before / n_proteins_after
        rc = pl.col("fmax_recall") * scale
        pr = pl.col("fmax_precision")
        rows.append(
            m.select(
                pl.lit(run).alias("run"),
                "tool",
                pl.col("variant").alias("setting"),
                "species",
                "dedup_transfers",
                pl.col("n_instances_found").alias("regions_found"),
                pl.col("n_truth_instances").alias("n_regions_before"),
                pl.lit(n_regions_after).alias("n_regions_after"),
                pl.lit(n_proteins_before).alias("n_proteins_before"),
                pl.lit(n_proteins_after).alias("n_proteins_after"),
                pl.col("recall").alias("recall_before"),
                (pl.col("n_instances_found") / n_regions_after).alias("recall_after"),
                pl.col("fmax").alias("fmax_before"),
                pl.when(pr + rc > 0)
                .then(2 * pr * rc / (pr + rc))
                .otherwise(0.0)
                .alias("fmax_after_at_least"),
            )
        )
    df = pl.concat(rows).sort("run", "tool", "setting", "species", "dedup_transfers")
    # The exact recall must equal the old one times the region ratio; a mismatch would mean
    # found regions were not all on query proteins.
    ratio = df["recall_after"] / (
        df["recall_before"] * df["n_regions_before"] / df["n_regions_after"]
    )
    assert ratio.drop_nans().is_between(1 - 1e-9, 1 + 1e-9).all()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.outdir / "pfamn_denominator_before_after.csv")

    # One row per (run, tool): the median over species of the setting with the highest
    # median recall after the fix, raw calls (dedup off). Every other tool has one setting.
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
        pl.col("fmax_after_at_least").median(),
    )
    best = (
        per_setting.sort(["recall_after", "setting"], descending=[True, False])
        .group_by("run", "tool", maintain_order=True)
        .first()
        .with_columns(
            pl.col("tool")
            .replace_strict({t: i for i, t in enumerate(TOOL_ORDER)}, default=99)
            .alias("_o")
        )
        .sort("run", "_o")
        .drop("_o")
    )
    best.write_csv(args.outdir / "pfamn_denominator_before_after_by_tool.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, float_precision=4):
        print(best)

    draw(
        df.filter((pl.col("run") == "midi-plus") & ~pl.col("dedup_transfers")),
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
    for ax, (b, a, ylab) in zip(
        axs,
        [
            (
                "recall_before",
                "recall_after",
                "Pfam-N recall (regions found / regions)",
            ),
            ("fmax_before", "fmax_after_at_least", "Pfam-N Fmax"),
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
                marker="o",
                label=(
                    "before: all 44,315 regions on 18,724 proteins" if i == 0 else None
                ),
            )
            mark = "o" if a == "recall_after" else "^"
            lab = (
                "after: 2,218 regions on the 966 searched proteins that have Pfam-N"
                if a == "recall_after"
                else None
            )
            ax.scatter(
                [x + 0.2 for x in xs],
                s[a],
                s=6,
                color=after_c,
                marker=mark,
                label=lab if i == 0 else None,
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
    # The triangle is the one mark that means "at least": the exact Fmax needs the rescore.
    axs[1].scatter(
        [],
        [],
        s=6,
        color=after_c,
        marker="^",
        label="after, Fmax at the old best cut-off (true Fmax is at least this)",
    )
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
