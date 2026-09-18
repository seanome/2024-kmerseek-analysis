#!/usr/bin/env python3
"""
make_multiqc_input.py

Convert DisProt benchmark parquets to MultiQC custom-content files.

Outputs (all written to <outdir>/):
  sweep_recall_<stratum>_mqc.yaml  — heatmap: alphabet x ksize, mean recall@FDR5%
  sweep_best_ksize_mqc.tsv         — each alphabet at its best ksize, per stratum
  auc_pr_<stratum>_mqc.tsv         — AUC-PR table (rows = species, cols = headline tools)
  recall_fdr5_<stratum>_mqc.tsv    — recall@FDR5% table
  *_vs_mya_mqc.yaml                — line plots against divergence
  multiqc_config.yaml              — section order, titles, colours

The per-tool tables and line plots show the HEADLINE tools only: the baselines plus the
--headline-n kmerseek combos with the best mean recall on the disordered stratum. A
1_647-column table is not a table. The sweep sections show every combo.

Usage:
    make_multiqc_input.py <all_disprot_metrics.parquet> <outdir> [--headline-n 3]
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disprot_tool_names import alphabet_classes, annotate, headline_tools, is_kmerseek  # noqa: E402


MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

SPECIES_ORDER = sorted(MYA, key=MYA.get)

# One hue per tool family. Several kmerseek combos on one plot share the blue and are
# told apart by their labels; a second shade would be a colour with no meaning.
FAMILY_COLORS = {"mmseqs2": "#795548", "foldseek": "#E53935", "kmerseek": "#1f77b4"}


def tool_color(tool: str) -> str:
    return FAMILY_COLORS["kmerseek"] if is_kmerseek(tool) else FAMILY_COLORS.get(tool, "#888888")

DISORDER_CATEGORIES = ["ordered", "partial", "disordered", "all"]


def pivot_metric(df: pl.DataFrame, metric: str, disorder_cat: str, tools: list[str]):
    """Return wide table: rows=species, cols=tool values."""
    sub = df.filter(pl.col("disorder_category") == disorder_cat)
    tools = [t for t in tools if t in sub["tool"].unique().to_list()]
    rows = {}
    for sp in SPECIES_ORDER:
        if sp not in MYA:
            continue
        row = {"Sample": sp}
        for tool in tools:
            val = sub.filter(
                (pl.col("species") == sp) & (pl.col("tool") == tool)
            )[metric].to_list()
            row[tool] = round(val[0], 4) if val and val[0] is not None and val[0] == val[0] else ""
        rows[sp] = row
    return rows, tools


def write_table_mqc(path: Path, rows: dict, tools: list[str],
                    section_name: str, description: str,
                    scale: str = "RdYlGn") -> None:
    with open(path, "w") as f:
        f.write(f"# plot_type: 'table'\n")
        f.write(f"# section_name: '{section_name}'\n")
        f.write(f"# description: '{description}'\n")
        f.write(f"# pconfig:\n")
        f.write(f"#   namespace: 'DisProt Benchmark'\n")
        header = "Sample\t" + "\t".join(tools)
        f.write(header + "\n")
        for sp in SPECIES_ORDER:
            if sp not in rows:
                continue
            r = rows[sp]
            vals = "\t".join(str(r.get(t, "")) for t in tools)
            f.write(f"{sp}\t{vals}\n")


def write_linegraph_mqc(path: Path, df: pl.DataFrame, metric: str,
                        section_name: str, description: str, tools: list[str],
                        disorder_cat: str = "all") -> None:
    """Write a MultiQC custom linegraph YAML (x=MYA, lines=tools)."""
    sub = df.filter(pl.col("disorder_category") == disorder_cat)
    tools = [t for t in tools if t in sub["tool"].unique().to_list()]

    datasets = []
    for tool in tools:
        t_sub = sub.filter(pl.col("tool") == tool).sort("mya")
        points = {}
        for row in t_sub.iter_rows(named=True):
            if row[metric] is not None and row[metric] == row[metric]:
                points[row["mya"]] = round(row[metric], 4)
        if points:
            datasets.append({tool: points})

    color_lines = "\n".join(
        f"  - '{t}': '{tool_color(t)}'"
        for t in tools if any(t in d for d in datasets)
    )

    lines = [
        f"id: '{path.stem.removesuffix('_mqc')}'",
        f"section_name: '{section_name}'",
        f"description: '{description}'",
        f"plot_type: 'linegraph'",
        f"pconfig:",
        f"  id: '{path.stem}_plot'",
        f"  title: '{section_name}'",
        f"  xlab: 'Evolutionary distance (Mya)'",
        f"  ylab: '{metric}'",
        f"  xlog: true",
        f"  ymin: 0",
        f"  ymax: 1",
        f"data:",
    ]
    for ds in datasets:
        for tool, pts in ds.items():
            color = tool_color(tool)
            lines.append(f"  '{tool}':")
            for x, y in sorted(pts.items()):
                lines.append(f"    {x}: {y}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_sweep_heatmap_mqc(path: Path, sweep: pl.DataFrame, disorder_cat: str, lowcomp: bool,
                            mask_suffix: str) -> None:
    """One heatmap per stratum (and per mask setting when both ran): rows = alphabets from
    finest to coarsest, columns = ksize, cell = mean recall@FDR5% across species."""
    sub = sweep.filter((pl.col("disorder_category") == disorder_cat) & (pl.col("lowcomp") == lowcomp))
    if len(sub) == 0:
        return
    alphabets = (
        sub.select(["alphabet", "classes"]).unique()
        .sort(["classes", "alphabet"], descending=[True, False])["alphabet"].to_list()
    )
    ksizes = list(range(int(sub["ksize"].min()), int(sub["ksize"].max()) + 1))
    cell = {(a, k): r for a, k, r in sub.select(["alphabet", "ksize", "mean_recall_fdr05"]).iter_rows()}
    mask_note = "low-complexity k-mers removed" if lowcomp else "low-complexity k-mers kept"
    lines = [
        f"id: '{path.stem.removesuffix('_mqc')}'",
        f"section_name: 'kmerseek sweep: recall at 5% FDR, {disorder_cat} queries{mask_suffix}'",
        f"description: 'Every alphabet over its ksize range, {mask_note}. Cell = recall at 5% FDR "
        f"averaged over the 9 target species; blank = not run. Rows run from the finest alphabet "
        f"(20 classes) to the coarsest (2).'",
        "plot_type: 'heatmap'",
        "pconfig:",
        f"  id: '{path.stem}_plot'",
        f"  title: 'Recall at 5% FDR, {disorder_cat} queries ({mask_note})'",
        "  xlab: 'k-mer size k (residues)'",
        "  ylab: 'alphabet (classes)'",
        "  min: 0",
        "  max: 1",
        "  colstops: [[0, '#f7fbff'], [0.5, '#6baed6'], [1, '#08306b']]",
        "  square: false",
        "xcats: [" + ", ".join(str(k) for k in ksizes) + "]",
        "ycats: [" + ", ".join(f"'{a} ({alphabet_classes(a)})'" for a in alphabets) + "]",
        "data:",
    ]
    for a in alphabets:
        row = []
        for k in ksizes:
            r = cell.get((a, k))
            row.append("null" if r is None else f"{r:.4f}")
        lines.append("  - [" + ", ".join(row) + "]")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_best_ksize_table(path: Path, sweep: pl.DataFrame) -> None:
    """Each alphabet at the ksize with the best mean recall@FDR5% on the disordered
    stratum, with that combo's recall on the other strata beside it."""
    ranked = (
        sweep.filter(pl.col("disorder_category") == "disordered")
        .sort("mean_recall_fdr05", descending=True, nulls_last=True)
        .group_by(["alphabet", "lowcomp"], maintain_order=True).first()
        .select(["tool", "alphabet", "classes", "lowcomp", "ksize", "mean_recall_fdr05", "mean_auc_pr"])
    )
    if len(ranked) == 0:
        return
    other = {
        (t, c): r for t, c, r in
        sweep.filter(pl.col("disorder_category").is_in(["ordered", "partial"]))
        .select(["tool", "disorder_category", "mean_recall_fdr05"]).iter_rows()
    }
    with open(path, "w") as f:
        f.write("# plot_type: 'table'\n")
        f.write("# section_name: 'kmerseek sweep: each alphabet at its best k'\n")
        f.write("# description: 'The ksize with the highest mean recall at 5% FDR on the disordered "
                "queries, per alphabet and mask setting, with that combo read off the other strata. "
                "Rows are ordered by recall on the disordered queries.'\n")
        f.write("# pconfig:\n#   namespace: 'kmerseek sweep'\n")
        f.write("Sample\talphabet\tclasses\tlow-complexity removed\tbest k\trecall disordered\t"
                "recall partial\trecall ordered\tAUC-PR disordered\n")
        for row in ranked.sort("mean_recall_fdr05", descending=True, nulls_last=True).iter_rows(named=True):
            t = row["tool"]
            def v(x):
                return "" if x is None else f"{x:.4f}"
            f.write("\t".join([
                t, row["alphabet"], str(row["classes"]), str(row["lowcomp"]).lower(), str(row["ksize"]),
                v(row["mean_recall_fdr05"]), v(other.get((t, "partial"))), v(other.get((t, "ordered"))),
                v(row["mean_auc_pr"]),
            ]) + "\n")


def sweep_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("alphabet").is_not_null())
        .group_by(["tool", "alphabet", "ksize", "lowcomp", "disorder_category"])
        .agg(pl.col("recall_at_fdr05").mean().alias("mean_recall_fdr05"),
             pl.col("auc_pr").mean().alias("mean_auc_pr"))
        .with_columns(pl.col("alphabet").map_elements(alphabet_classes, return_dtype=pl.Int32).alias("classes"))
    )


def write_multiqc_config(path: Path, tools: list[str], section_ids: list[str]) -> None:
    # Every section id, on one scale. A partial report_section_order competes with
    # MultiQC's own 10/20/30 defaults and the sweep would land wherever it fell.
    order_block = "\n".join(f"  {sid}:\n    order: {1000 - 10 * i}" for i, sid in enumerate(section_ids))
    content = f"""\
title: "DisProt Benchmark"
subtitle: "kmerseek alphabet x ksize sweep against MMseqs2 on disordered human proteins"
intro_text: >
  Human DisProt proteins searched against nine QfO species proteomes (100 to 2000 Mya),
  scored against Pfam domain sharing and split by each query's predicted disorder.
  kmerseek runs every alphabet over its ksize range (the same matrix as the QfO region
  benchmark and the invertebrate dark set); the per-tool tables show the baselines and
  the best few combos, the sweep sections show all of them.

report_header_info:
  - Ground truth: "Pfam domain sharing"
  - Species: "9 (mouse → E. coli, 100–2000 Mya)"
  - Query sets: "DisProt + MobiDB"

custom_plot_config:
  sample_names_rename:
    - ["arabidopsis", "A. thaliana (1500 Mya)"]
    - ["chicken", "G. gallus (300 Mya)"]
    - ["ciona", "C. intestinalis (550 Mya)"]
    - ["ecoli", "E. coli (2000 Mya)"]
    - ["fly", "D. melanogaster (600 Mya)"]
    - ["mouse", "M. musculus (100 Mya)"]
    - ["worm", "C. elegans (650 Mya)"]
    - ["yeast", "S. cerevisiae (900 Mya)"]
    - ["zebrafish", "D. rerio (430 Mya)"]

table_columns_placement:
{chr(10).join(f"  {t}: {{ placement: {500 + i * 10} }}" for i, t in enumerate(tools))}

report_section_order:
{order_block}

section_comments:
  sweep_recall_disordered: >
    The sweep's headline panel. Each cell is one alphabet at one ksize, searched by the
    DisProt queries whose mean predicted disorder is above 0.5, averaged over the nine
    target species. Blank cells were not run: each alphabet's ksize range starts where
    its k-mers carry about 18 bits, so coarse alphabets start at larger k.
  sweep_best_ksize: >
    One row per alphabet (and mask setting), at the ksize that scored best on the
    disordered queries. The other columns read that same combo off the other strata.
  auc_pr_all: >
    Area under the precision-recall curve, all query proteins, per species. Headline
    tools only: the baselines and the best few kmerseek combos on the disordered queries.
  recall_fdr5_all: >
    Recall at 5% FDR, all query proteins, per species. Headline tools only.
"""
    with open(path, "w") as f:
        f.write(content)


def main(metrics_parquet: str, outdir: str, headline_n: int = 3) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(metrics_parquet)
    df = df.with_columns(
        [pl.col(c).fill_nan(None) for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    )
    if "alphabet" not in df.columns:
        df = annotate(df)
    if "mya" not in df.columns:
        df = df.with_columns(pl.col("species").replace_strict(MYA, default=None).cast(pl.Int32).alias("mya"))

    headline = headline_tools(df, headline_n)
    all_tools = headline
    section_ids = []

    # ── The sweep: every combo ───────────────────────────────────────────────
    sweep = sweep_summary(df)
    masks = sorted(sweep["lowcomp"].unique().to_list()) if len(sweep) else []
    for lc in masks:
        mask_suffix = (f", low-complexity {'removed' if lc else 'kept'}" if len(masks) > 1 else "")
        file_suffix = (f"_lc{str(lc).lower()}" if len(masks) > 1 else "")
        for cat in ["disordered", "partial", "ordered", "all"]:
            write_sweep_heatmap_mqc(out / f"sweep_recall_{cat}{file_suffix}_mqc.yaml", sweep, cat, lc, mask_suffix)
            section_ids.append(f"sweep_recall_{cat}{file_suffix}")
    if len(sweep):
        write_best_ksize_table(out / "sweep_best_ksize_mqc.tsv", sweep)
        section_ids.append("sweep_best_ksize")
    section_ids += ["auc_pr_vs_mya", "recall_fdr5_vs_mya"]
    section_ids += [f"{m}_{c}" for c in DISORDER_CATEGORIES for m in ("auc_pr", "recall_fdr5")]

    # ── Per-disorder-category tables, headline tools ─────────────────────────
    for cat in DISORDER_CATEGORIES:
        suffix = cat.replace(" ", "_")

        # AUC-PR
        rows, tools = pivot_metric(df, "auc_pr", cat, headline)
        write_table_mqc(
            out / f"auc_pr_{suffix}_mqc.tsv", rows, tools,
            section_name=f"AUC-PR — {cat} proteins",
            description=f"Area under precision-recall curve for {cat} proteins (rows = species ordered by divergence).",
        )

        # Recall@FDR5
        rows, tools = pivot_metric(df, "recall_at_fdr05", cat, headline)
        write_table_mqc(
            out / f"recall_fdr5_{suffix}_mqc.tsv", rows, tools,
            section_name=f"Recall @ FDR 5% — {cat} proteins",
            description=f"Fraction of true homologs found at ≤5% FDR for {cat} proteins.",
        )

    # ── Line graph: AUC-PR vs Mya (all proteins) ─────────────────────────────
    write_linegraph_mqc(
        out / "auc_pr_vs_mya_mqc.yaml", df,
        metric="auc_pr",
        section_name="AUC-PR vs evolutionary distance",
        description="AUC-PR against divergence time (Mya), headline tools.",
        tools=headline, disorder_cat="all",
    )

    # ── Line graph: Recall@FDR5 vs Mya ───────────────────────────────────────
    write_linegraph_mqc(
        out / "recall_fdr5_vs_mya_mqc.yaml", df,
        metric="recall_at_fdr05",
        section_name="Recall @ FDR 5% vs evolutionary distance",
        description="Recall at 5% FDR against divergence time (Mya), headline tools.",
        tools=headline, disorder_cat="all",
    )

    # ── MultiQC config ────────────────────────────────────────────────────────
    write_multiqc_config(out / "multiqc_config.yaml", all_tools, section_ids)

    print(f"Wrote MultiQC input files to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_parquet")
    ap.add_argument("outdir")
    ap.add_argument("--headline-n", type=int, default=3)
    a = ap.parse_args()
    main(a.metrics_parquet, a.outdir, a.headline_n)
