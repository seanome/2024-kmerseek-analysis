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
    make_multiqc_input.py <all_disprot_metrics.parquet> <outdir> [benchmark_stats.txt]

The stats file is what buildDisprotGroundTruth wrote; it carries the query count the
overview quotes. Without it the overview says n/a for that one number.
"""

import json
import re
import sys
from pathlib import Path

import polars as pl

# The data-flow diagram in the overview is drawn by the module every report in this
# repository shares. Under Nextflow it is staged into the task directory and found through
# PYTHONPATH; run by hand from bin/, it is found at ../../shared.
try:
    import flow_diagram as fd
except ImportError:  # pragma: no cover - the by-hand path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import flow_diagram as fd


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

# Every section id this script writes, in reading order. report_section_order is one scale:
# a partial list competes with MultiQC's own defaults for the ids left out, so every id is
# listed. Table ids are the TSV stems minus _mqc; the line graphs carry their own id.
SECTION_ORDER = (
    ["overview"]
    + [f"{m}_{cat}" for cat in DISORDER_CATEGORIES for m in ("auc_pr", "recall_fdr5")]
    + ["auc_pr_vs_mya_mqc", "recall_fdr5_vs_mya_mqc"]
)

TOOL_WORDS = {
    "mmseqs2": "MMseqs2 (sequence alignment)",
    "foldseek": "Foldseek (structure, AlphaFold models)",
}


def tool_word(tool: str) -> str:
    if tool.startswith("kmerseek_k"):
        return f"kmerseek (hp-thomas-dill alphabet, k={tool[len('kmerseek_k'):]})"
    return TOOL_WORDS.get(tool, tool)


def read_query_count(stats_path: Path | None) -> int | None:
    """'Total DisProt query proteins: 271' out of benchmark_stats.txt."""
    if stats_path is None or not stats_path.exists():
        return None
    m = re.search(r"Total DisProt query proteins:\s*(\d+)", stats_path.read_text())
    return int(m.group(1)) if m else None


def overview_facts(df: pl.DataFrame, n_queries: int | None) -> dict:
    """This run's numbers for the overview, read off the metrics and nowhere else."""
    f = {"n_queries": n_queries, "tools": sorted(df["tool"].unique().to_list())}
    f["species"] = [sp for sp in SPECIES_ORDER if sp in set(df["species"].to_list())]
    mya = [MYA[sp] for sp in f["species"]]
    f["mya_min"], f["mya_max"] = (min(mya), max(mya)) if mya else (None, None)
    # The answer key is the same for every tool, so read it off one of them.
    one = df.filter((pl.col("tool") == f["tools"][0]) & (pl.col("disorder_category") == "all"))
    f["pairs_min"], f["pairs_max"] = (one["n_pairs"].min(), one["n_pairs"].max()) if one.height else (None, None)
    f["pairs_total"] = int(one["n_pairs"].sum()) if one.height else None
    f["pos_total"] = int(one["n_positives"].sum()) if one.height else None
    return f


def overview_flow_svg(f: dict) -> str:
    """The benchmark as a picture: query and targets in, one arm per tool, scores joined to
    the Pfam pair labels and to each query's disorder, then metrics and the report."""
    F = fd.Flow()
    arm_colour = {t: TOOL_COLORS.get(t, "#888888") for t in f["tools"]}
    y = F.legend([
        [dict(text="sequences or results, with a count"), dict(text="a step", arrow=True),
         dict(text="an arm this run did not do", dashed=True)],
        [dict(text=tool_word(t).split(" (")[0], stroke=arm_colour[t]) for t in f["tools"]],
        [dict(text="query proteins", icon="genetics"), dict(text="target proteomes", icon="database"),
         dict(text="a search tool", icon="search")],
        [dict(text="scores and labels", icon="table_rows"), dict(text="disorder", icon="waves"),
         dict(text="the report", icon="summarize")],
    ])
    half = 350
    lx, rx = 20, 410
    lmid, rmid = lx + half // 2, rx + half // 2
    F.label(lmid, y + 8, ["QUERY: what is searched"], bold=True, size=14)
    F.label(rmid, y + 8, ["TARGET: what is searched against"], bold=True, size=14)
    ya = y + 18
    tw = half - fd.ICON_PX - fd.SVG_PAD
    a_l = F.box(lx, ya, half, ["human proteins with a DisProt entry", f"{fd.num(f['n_queries'])} proteins"]
                + fd.wrap("the subset of the Pfam pair benchmark's human queries that DisProt "
                          "annotates as having a disordered region", tw),
                bold_first=True, icon="genetics")
    a_r = F.box(rx, ya, half, [f"{len(f['species'])} QfO proteomes"]
                + fd.wrap(f"{fd.num(f['mya_min'])} to {fd.num(f['mya_max'])} million years from human", tw),
                bold_first=True, icon="database")
    y_from = max(a_l[1] + a_l[3], a_r[1] + a_r[3])

    # One arm per tool that ran; Foldseek is the one a run may skip.
    known = ["kmerseek", "foldseek", "mmseqs2"]
    ran = {t.split("_k")[0]: t for t in f["tools"]}
    cols = fd.columns(3)
    mids = [fd.mid(c) for c in cols]
    yb = y_from + 18 + fd.SVG_LINE_PX + 26
    F.fan([lmid, rmid], y_from, mids, yb, ["every query against every target proteome"])
    boxes = []
    aw = cols[0][1] - fd.ICON_PX - fd.SVG_PAD
    for name, (x, w) in zip(known, cols):
        tool = ran.get(name)
        word = tool_word(tool) if tool else tool_word(name)
        head, _, rest = word.partition(" (")
        lines = [head] + fd.wrap(rest.rstrip(")"), aw)
        boxes.append(F.box(x, yb, w, lines, stroke=arm_colour.get(tool, "#888888"), stroke_w=2.5,
                           bold_first=True, icon="search", dashed=tool is None))
    y_arms = max(b[1] + b[3] for b in boxes)
    for m in mids:
        F.line(m, y_arms, m, y_arms + 26, arrow=True)
    sc = F.box(20, y_arms + 26, 740,
               ["a score for every human protein x target protein pair the tool reported"],
               icon="table_rows")

    # The two side inputs to scoring, with the pair scores passing between them.
    y_side = sc[1] + sc[3] + 30
    xm = fd.SVG_W // 2
    sw = 300
    key = F.box(20, y_side, sw, ["answer key: Pfam pair labels"]
                + fd.wrap(f"a pair is positive when the two proteins share a Pfam family, negative "
                          f"when they share none; {fd.num(f['pairs_min'])} to {fd.num(f['pairs_max'])} "
                          f"pairs per proteome, {fd.num(f['pairs_total'])} in all, "
                          f"{fd.num(f['pos_total'])} positive", sw - fd.ICON_PX - fd.SVG_PAD),
                bold_first=True, icon="table_rows")
    dis = F.box(fd.SVG_W - 20 - sw, y_side, sw, ["disorder of each query protein"]
                + fd.wrap("metapredict, mean over the protein: ordered below 0.2, partial 0.2 to 0.5, "
                          "disordered above 0.5", sw - fd.ICON_PX - fd.SVG_PAD),
                bold_first=True, icon="waves")
    y_side_end = max(key[1] + key[3], dis[1] + dis[3])
    y_sc = y_side_end + 30
    F.step(xm, sc[1] + sc[3], y_sc, ["join on the pair,", "then on the query"])
    F.line(20 + sw // 2, key[1] + key[3], 20 + sw // 2, y_sc, arrow=True)
    F.line(fd.SVG_W - 20 - sw // 2, dis[1] + dis[3], fd.SVG_W - 20 - sw // 2, y_sc, arrow=True)
    m = F.box(20, y_sc, 740, ["AUC-PR, AUC-ROC and recall at 5% FDR"]
              + fd.wrap("per tool, per target proteome, per disorder bin; a pair the tool did not "
                        "report scores 0", 740 - 60), icon="table_rows", bold_first=True)
    F.line(xm, m[1] + m[3], xm, m[1] + m[3] + 26, arrow=True)
    F.box(20, m[1] + m[3] + 26, 740, ["this report: one table per metric and bin, then the two "
                                      "curves against divergence"], icon="summarize")
    return F.render()


def write_overview(out: Path, df: pl.DataFrame, n_queries: int | None) -> None:
    f = overview_facts(df, n_queries)
    arms = "; ".join(tool_word(t) for t in f["tools"])
    steps = [
        f"<b>Query: human proteins with a DisProt entry.</b> {fd.num(f['n_queries'])} proteins: "
        f"the human queries of the Pfam pair benchmark that DisProt annotates as carrying a "
        f"disordered region. Nothing else is rebuilt; this is a subset of that benchmark.",
        f"<b>Targets: {len(f['species'])} Quest-for-Orthologs proteomes</b> "
        f"({', '.join(f['species'])}), {fd.num(f['mya_min'])} to {fd.num(f['mya_max'])} million "
        f"years from human.",
        f"<b>Search.</b> Every query against every target proteome, one arm per tool: {arms}. "
        f"Each arm reports a score for every human x target pair it found.",
        f"<b>Answer key: the Pfam pair labels.</b> A pair is positive when the two proteins "
        f"share a Pfam family and negative when they share none. {fd.num(f['pairs_min'])} to "
        f"{fd.num(f['pairs_max'])} pairs per proteome, {fd.num(f['pairs_total'])} in all, "
        f"{fd.num(f['pos_total'])} of them positive. A pair a tool did not report scores 0.",
        "<b>Disorder of each query.</b> metapredict scores every residue 0 to 1; the mean over "
        "the protein puts it in a bin: ordered below 0.2, partial 0.2 to 0.5, disordered "
        "above 0.5.",
        "<b>Score.</b> AUC-PR, AUC-ROC and recall at 5% FDR (the recall reached while "
        "precision is still at least 0.95), per tool, per target proteome, per disorder bin.",
        "<b>Report.</b> One table per metric and bin, rows ordered by divergence, then the "
        "two curves of metric against divergence.",
    ]
    why = "".join(f"<li>{w}</li>" for w in [
        "<b>The question is whether a disordered region can carry a homology signal that a "
        "structure search cannot see.</b> A region with no stable fold has nothing for "
        "Foldseek to encode; a k-mer method reads the sequence regardless. The disorder bins "
        "are what make the comparison: the same tools, the same pairs, split by how much of "
        "the query is disordered.",
        "<b>The answer key is reused, not rebuilt.</b> The Pfam pair labels are the ones the "
        "whole-proteome pair benchmark already uses, restricted to DisProt queries, so a "
        "difference here is a difference in the proteins, not in the labelling.",
        "<b>AUC-PR before recall at a threshold.</b> The tools' scores are on different "
        "scales and kmerseek's is not calibrated at 5% FDR, so a threshold metric mixes "
        "ranking with calibration. AUC-PR reads the ranking alone.",
        "<b>The target proteome is the divergence axis.</b> The same queries against mouse "
        "and against E. coli ask how far each signal reaches back in time.",
    ])
    cfg = {
        "id": "overview",
        "section_name": "What was done, and why",
        "description": (
            "<p>Human proteins with a DisProt entry searched against Quest-for-Orthologs "
            "proteomes by three tools, every reported pair scored against the Pfam pair "
            "labels and split by how disordered the query is. The steps, this run's numbers, "
            "and the flow of data from the inputs to the tables below.</p>"),
        "plot_type": "html",
        "data": (
            "<h4 style='margin-top:0.4em'>What was done</h4><ol>"
            + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
            "<h4>Why</h4><ul>" + why + "</ul>"
            "<h4>Data flow</h4>"
            "<p>Read top to bottom. A box is a set of sequences or results with its count in "
            "this run; an arrow is the step that makes the next one. Arm boxes take the colour "
            "their tool has in the curves below.</p>"
            + overview_flow_svg(f)),
    }
    (out / "overview_mqc.json").write_text(json.dumps(cfg, indent=1))


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
    tool_list = ", ".join(tool_word(t).split(" (")[0] for t in tools)
    # Each file here is its own MultiQC module (no parent_id), and MODULES sort with the
    # largest order first, the reverse of sections inside one module. So the first id in
    # SECTION_ORDER gets the largest number.
    order_block = "\n".join(f"  {sid}: {{ order: {len(SECTION_ORDER) - i} }}"
                            for i, sid in enumerate(SECTION_ORDER))
    content = f"""\
title: "DisProt Benchmark"
subtitle: "{tool_list}: homology detection through intrinsically disordered regions"
intro_text: >
  Human proteins with a DisProt entry, searched against nine QfO proteomes (100 to 2000
  million years from human) by {tool_list}, scored against the Pfam pair labels and split
  by how disordered the query protein is. The first section says what was run, with this
  run's numbers, why, and how the data flows to the tables.

report_section_order:
{order_block}

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


def main(metrics_parquet: str, outdir: str, stats_txt: str | None = None) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(metrics_parquet)
    # Add mya column
    df = df.with_columns(
        pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
    )

    all_tools = sorted(df["tool"].unique().to_list())

    # ── Overview: what was done, why, and the data flow ──────────────────────
    write_overview(out, df, read_query_count(Path(stats_txt) if stats_txt else None))

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
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
