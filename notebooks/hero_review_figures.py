"""Review figures drawn only from the committed notebook 244 tables."""

from pathlib import Path
import polars as pl
import pubfig as pf

ROOT = Path(__file__).resolve().parents[1]


def finish_figure(section):
    pf.use_style()
    candidates = pl.read_csv(ROOT / "tables/244_hero_candidates.csv")
    strict = candidates.filter(pl.col("admitted_under") == "all criteria")
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 75)
    if section == "stage0":
        values = (
            pl.read_csv(ROOT / "tables/244_arm_by_feature_type.csv")
            .group_by("feature_type", "arm")
            .agg(
                pl.col("n_landed_report_half").sum().alias("landed"),
                pl.col("n_instances_report_half").sum().alias("total"),
            )
            .with_columns((pl.col("landed") / pl.col("total")).alias("fraction"))
            .group_by("feature_type")
            .agg(pl.col("fraction").max())
            .sort("fraction")
        )
        ax.barh(
            values["feature_type"].to_list(),
            values["fraction"].to_list(),
            color=pf.OKABE_ITO["blue"],
        )
        ax.set_xlabel("Largest report-half fraction across all arms (descriptive)")
        ax.set_title(
            "Report-half maximum across arms; not an independent selected-arm score"
        )
    elif section == "cases":
        values = strict.group_by("feature_type").len().sort("len")
        ax.barh(
            values["feature_type"].to_list(),
            values["len"].to_list(),
            color=pf.OKABE_ITO["blue"],
        )
        ax.set_xlabel("Cases passing all criteria; species counted separately")
        ax.set_title("87 positional benchmark cases; one was also found by the scan")
    else:
        values = (
            strict.filter(
                pl.col("gene").is_in(["COL9A1", "PHACTR2"])
                & (pl.col("species") == "chicken")
            )
            .sort("species_mya", descending=True)
            .unique("gene", keep="first", maintain_order=True)
        )
        tools = ["kmerseek", "Foldseek", "ProstT5", "Reseek", "phmmer", "MMseqs2"]
        x = __import__("numpy").arange(len(tools))
        for i, row in enumerate(values.iter_rows(named=True)):
            ys = [
                row["kmerseek_iou"] if t == "kmerseek" else row[t + "_iou"]
                for t in tools
            ]
            ax.bar(
                x + (i - 0.5) * 0.35,
                ys,
                0.35,
                label=row["gene"],
                color=[pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"]][i],
            )
        ax.set_xticks(x, tools)
        ax.set_ylabel("Intersection over union with the scored human feature")
        pf.shared_legend(fig)
        ax.set_title(
            "Hero and backup: local interval placement, not whole-protein homology"
        )
    pf.save(fig, ROOT / "figures" / ("244_review_" + section + "_committed_tables"))
    return fig
