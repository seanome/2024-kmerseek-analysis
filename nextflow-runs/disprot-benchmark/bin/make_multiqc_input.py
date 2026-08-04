#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
make_multiqc_input.py

Convert DisProt benchmark parquets to MultiQC custom-content files.

Outputs (all written to <outdir>/):
  auc_pr_mqc.tsv         — AUC-PR table (sample = species, cols = tools)
  recall_fdr5_mqc.tsv    — Recall@FDR5% table
  n_found_mqc.tsv        — Number of pairs found per tool/species
  mya_auc_pr_mqc.yaml    — Line plot: AUC-PR vs evolutionary distance
  multiqc_config.yaml    — Section order, titles, colors

Usage:
    make_multiqc_input.py <all_disprot_metrics.parquet> <outdir>
"""

import sys
from pathlib import Path

import polars as pl


MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

SPECIES_ORDER = sorted(MYA, key=MYA.get)

TOOL_COLORS = {
    "mmseqs2":      "#795548",
    "foldseek":     "#E53935",
    "kmerseek_k26": "#AB47BC",
    "kmerseek_k25": "#CE93D8",
    "kmerseek_k24": "#9C27B0",
    "kmerseek_k20": "#E1BEE7",
}

DISORDER_CATEGORIES = ["ordered", "partial", "disordered", "all"]


def pivot_metric(df: pl.DataFrame, metric: str, disorder_cat: str) -> pl.DataFrame:
    """Return wide table: rows=species, cols=tool values."""
    sub = df.filter(pl.col("disorder_category") == disorder_cat)
    tools = sorted(sub["tool"].unique().to_list())
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
                        section_name: str, description: str,
                        disorder_cat: str = "all") -> None:
    """Write a MultiQC custom linegraph YAML (x=MYA, lines=tools)."""
    sub = df.filter(pl.col("disorder_category") == disorder_cat)
    tools = sorted(sub["tool"].unique().to_list())

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
        f"  - '{t}': '{TOOL_COLORS.get(t, '#888888')}'"
        for t in tools if any(t in d for d in datasets)
    )

    lines = [
        f"id: '{path.stem}'",
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
            color = TOOL_COLORS.get(tool, "#888888")
            lines.append(f"  '{tool}':")
            for x, y in sorted(pts.items()):
                lines.append(f"    {x}: {y}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_multiqc_config(path: Path, tools: list[str]) -> None:
    tool_color_block = "\n".join(
        f"  '{t}': '{TOOL_COLORS.get(t, '#888888')}'"
        for t in tools
    )
    content = f"""\
title: "DisProt Benchmark"
subtitle: "Kmerseek vs MMseqs2 — IDR homology detection"
intro_text: >
  Benchmarks kmerseek (hp-thomas-dill encoding, k=26) against MMseqs2 for detecting
  protein homology through intrinsically disordered regions (IDRs).
  Ground truth: Pfam domain sharing across 9 QfO species (100–2000 Mya).

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

section_comments:
  auc_pr_all: >
    Area under precision-recall curve (all query proteins, mean across 9 species).
    Higher is better. Random baseline ≈ fraction of positives.
  recall_fdr5_all: >
    Recall at 5% FDR threshold (all query proteins). Reflects operating-point performance.
    Kmerseek scores are not calibrated at this threshold — AUC-PR is a fairer comparison.
"""
    with open(path, "w") as f:
        f.write(content)


def main(metrics_parquet: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(metrics_parquet)
    # Add mya column
    df = df.with_columns(
        pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
    )

    all_tools = sorted(df["tool"].unique().to_list())

    # ── Per-disorder-category tables ─────────────────────────────────────────
    for cat in DISORDER_CATEGORIES:
        suffix = cat.replace(" ", "_")

        # AUC-PR
        rows, tools = pivot_metric(df, "auc_pr", cat)
        write_table_mqc(
            out / f"auc_pr_{suffix}_mqc.tsv", rows, tools,
            section_name=f"AUC-PR — {cat} proteins",
            description=f"Area under precision-recall curve for {cat} proteins (rows = species ordered by divergence).",
        )

        # Recall@FDR5
        rows, tools = pivot_metric(df, "recall_at_fdr05", cat)
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
        description="AUC-PR as a function of divergence time (Mya). Steeper drop = worse performance at large evolutionary distances.",
        disorder_cat="all",
    )

    # ── Line graph: Recall@FDR5 vs Mya ───────────────────────────────────────
    write_linegraph_mqc(
        out / "recall_fdr5_vs_mya_mqc.yaml", df,
        metric="recall_at_fdr05",
        section_name="Recall @ FDR 5% vs evolutionary distance",
        description="Recall at 5% FDR vs divergence time. Kmerseek recall is low due to score calibration, not ranking failure.",
        disorder_cat="all",
    )

    # ── MultiQC config ────────────────────────────────────────────────────────
    write_multiqc_config(out / "multiqc_config.yaml", all_tools)

    print(f"Wrote MultiQC input files to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
