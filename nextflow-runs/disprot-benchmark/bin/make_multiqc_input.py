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
    import metric_explainers as mx
except ImportError:  # pragma: no cover - the by-hand path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import flow_diagram as fd
    import metric_explainers as mx


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

# Reading order: every protein first, then the three disorder bins from least to most
# disordered, so the split is read against the whole it splits.
DISORDER_CATEGORIES = ["all", "ordered", "partial", "disordered"]

# Every section id this script writes, in reading order. report_section_order is one scale:
# a partial list competes with MultiQC's own defaults for the ids left out, so every id is
# listed. Table ids are the TSV stems minus _mqc; the line graphs carry their own id.
SECTION_ORDER = (
    ["overview", "metric_explainers"]
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
    # A tool that ran but reported no pair on any proteome has no metric to show. Said
    # in words rather than drawn as a normal arm: on the 2026-08 DisProt run Foldseek's
    # nine result files were all empty and its column vanished from every table.
    f["found"] = {t: int(df.filter(pl.col("tool") == t)["n_found"].sum()) for t in f["tools"]}
    # AUC-PR per tool per bin, as the mean over the proteomes that have a value.
    f["aucpr"] = {}
    for cat in DISORDER_CATEGORIES:
        sub = df.filter(pl.col("disorder_category") == cat)
        f["aucpr"][cat] = {}
        for t in f["tools"]:
            vals = sub.filter(pl.col("tool") == t)["auc_pr"].drop_nulls().drop_nans()
            f["aucpr"][cat][t] = (float(vals.mean()), int(vals.len())) if vals.len() else (None, 0)
    return f


def overview_flow_spec(f: dict) -> dict:
    """The benchmark as a box-and-arrow spec: query and targets meet at a junction and one
    bus fans out to the three arms; the pair scores drop to the metrics box while the
    answer key and the disorder bins enter it from the sides."""
    ran = {t.split("_k")[0]: t for t in f["tools"]}
    colour = {n: TOOL_COLORS.get(t, "#888888") for n, t in ran.items()}
    kinds = {n: {"color": colour[n], "label": tool_word(t).split(" (")[0]} for n, t in ran.items()}
    arms = {}
    for name, x in [("kmerseek", 20), ("foldseek", 270), ("mmseqs2", 520)]:
        tool = ran.get(name)
        word = tool_word(tool or name)
        head, _, rest = word.partition(" (")
        sub = rest.rstrip(")")
        if tool and f["found"].get(tool) == 0:
            sub = "reported 0 pairs on every proteome"
        arms[name] = {"x": x, "y": 200, "w": 220, "h": 66, "icon": "search", "kind": name if tool else None,
                      "title": head, "sub": sub, "bar": True, "dashed": tool is None,
                      "samples": [name] + ([tool] if tool and tool != name else [])}
    nodes = {
        "q0": {"x": 20, "y": 66, "w": 320, "h": 52, "icon": "genetics",
               "title": "human proteins with a DisProt entry", "sub": f"{fd.num(f['n_queries'])} proteins"},
        "t0": {"x": 420, "y": 66, "w": 320, "h": 52, "icon": "database",
               "title": f"{len(f['species'])} QfO proteomes",
               "sub": f"{fd.num(f['mya_min'])} to {fd.num(f['mya_max'])} million years from human"},
        "J1": {"junction": True, "x": 380, "y": 92},
        **arms,
        "scores": {"x": 20, "y": 340, "w": 720, "h": 44, "icon": "table_rows",
                   "title": "a score for every human x target protein pair the tool reported"},
        "key": {"x": 20, "y": 440, "w": 340, "h": 60, "icon": "table_rows",
                "title": "answer key: Pfam pair labels",
                "sub": f"{fd.num(f['pairs_total'])} pairs, {fd.num(f['pos_total'])} positive"},
        "disorder": {"x": 400, "y": 440, "w": 340, "h": 60, "icon": "waves",
                     "title": "disorder of each query protein", "sub": "3 bins by mean metapredict score"},
        "metrics": {"x": 20, "y": 580, "w": 720, "h": 56, "icon": "table_rows",
                    "title": "AUC-PR, AUC-ROC and recall at 5% FDR",
                    "sub": "per tool, per target proteome, per disorder bin"},
        "report": {"x": 20, "y": 700, "w": 720, "h": 44, "icon": "summarize",
                   "title": "this report: one table per metric and bin, then the two curves against divergence"},
    }
    edges = [
        {"from": "q0", "to": "J1"}, {"from": "t0", "to": "J1"},
        {"from": "J1", "to": "kmerseek", "bus": 60, "label": "every query against every target proteome",
         "lx": 392, "ly": 146, "anchor": "start"},
        {"from": "J1", "to": "foldseek", "bus": 60}, {"from": "J1", "to": "mmseqs2", "bus": 60},
        {"from": "kmerseek", "to": "scores"}, {"from": "foldseek", "to": "scores"}, {"from": "mmseqs2", "to": "scores"},
        {"from": "scores", "to": "metrics", "label": "join on the pair, then on the query", "ly": 414},
        {"from": "key", "to": "metrics", "label": "a pair the tool did not report scores 0", "ly": 560},
        {"from": "disorder", "to": "metrics"}, {"from": "metrics", "to": "report"},
    ]
    return {"height": 780, "kinds": kinds,
            "barLegend": "bar along the bottom of an arm: its AUC-PR, mean over the proteomes with a value (full width = 1.0)",
            "lanes": [[30, 130, "inputs"], [170, 290, "search"], [320, 400, "scores"], [420, 530, "labels"],
                      [560, 660, "metrics"], [680, 760, "report"]],
            "headers": [[190, 52, "QUERY: what is searched"], [570, 52, "TARGET: what is searched against"]],
            "nodes": nodes, "edges": edges}


def overview_details(f: dict) -> dict:
    ran = {t.split("_k")[0]: t for t in f["tools"]}
    tool_text = {
        "kmerseek": "Reads the sequence whether or not it folds. The question is whether a "
                    "disordered region can carry a homology signal a structure search cannot see.",
        "foldseek": "Structure search over AlphaFold models. A region with no stable fold has "
                    "nothing for Foldseek to encode, which is the contrast the benchmark is built on.",
        "mmseqs2": "Sequence alignment: the conventional sequence baseline.",
    }
    d = {
        "q0": {"title": "human proteins with a DisProt entry", "count": f"{fd.num(f['n_queries'])} proteins",
               "text": "Nothing else is rebuilt: the queries are a subset of the whole-proteome pair "
                       "benchmark, so a difference here is a difference in the proteins, not in the "
                       "labelling.",
               "facts": [["proteins", fd.num(f["n_queries"])],
                         ["source", "DisProt-annotated human queries of the Pfam pair benchmark"]],
               "links": [["What was done, and why", "overview"]]},
        "t0": {"title": f"{len(f['species'])} QfO proteomes",
               "count": f"{fd.num(f['mya_min'])} to {fd.num(f['mya_max'])} million years from human",
               "text": "The same targets the pair benchmark uses; each proteome is a point on the "
                       "divergence axis.",
               "facts": [["proteomes", ", ".join(f["species"])]],
               "links": [["AUC-PR vs evolutionary distance", "auc_pr_vs_mya_mqc"]]},
        "scores": {"title": "pair scores", "count": "",
                   "text": "Every pair each tool reported, with the tool's own score. Scores are on "
                           "different scales, so no metric here compares a score across tools: each "
                           "one walks a tool's own ranking from the top.",
                   "facts": [[f"{tool_word(t).split(' (')[0]}: pairs reported, all proteomes", fd.num(n)]
                             for t, n in f["found"].items()],
                   "links": [["What was done, and why", "overview"]]},
        "key": {"title": "answer key: Pfam pair labels",
                "count": f"{fd.num(f['pairs_total'])} pairs, {fd.num(f['pos_total'])} positive",
                "text": "Pfam pair labels reused from the whole-proteome pair benchmark, restricted "
                        "to DisProt queries. A pair is positive when the two proteins share a Pfam "
                        "family and negative when they share none.",
                "facts": [["pairs per proteome", f"{fd.num(f['pairs_min'])} to {fd.num(f['pairs_max'])}"],
                          ["pairs in all", fd.num(f["pairs_total"])], ["positive pairs", fd.num(f["pos_total"])]],
                "links": [["What was done, and why", "overview"]]},
        "disorder": {"title": "disorder of each query protein", "count": "metapredict, mean over the protein",
                     "text": "metapredict scores every residue 0 to 1; the mean over the protein puts "
                             "the query in a bin. The bins are what make the comparison: same tools, "
                             "same pairs, split by how disordered the query is.",
                     "facts": [["ordered", "mean below 0.2"], ["partial", "0.2 to 0.5"], ["disordered", "above 0.5"]],
                     "links": [["AUC-PR: disordered proteins", "auc_pr_disordered"]]},
        "metrics": {"title": "AUC-PR, AUC-ROC and recall at 5% FDR", "count": "",
                    "text": "One value per tool, per target proteome, per bin. A pair the tool did not "
                            "report scores 0, so a tool that reports few pairs loses recall rather "
                            "than being excused.",
                    "facts": [], "links": [["AUC-PR: all proteins", "auc_pr_all"],
                                           ["Recall @ FDR 5%: all proteins", "recall_fdr5_all"]]},
        "report": {"title": "this report", "count": "",
                   "text": "One table per metric and bin, rows ordered by divergence, then the two "
                           "curves of metric against divergence.",
                   "facts": [], "links": [["AUC-PR vs evolutionary distance", "auc_pr_vs_mya_mqc"],
                                          ["Recall @ FDR 5% vs evolutionary distance", "recall_fdr5_vs_mya_mqc"]]},
    }
    for name in ["kmerseek", "foldseek", "mmseqs2"]:
        tool = ran.get(name)
        d[name] = {"title": tool_word(tool or name).split(" (")[0],
                   "count": tool_word(tool).partition(" (")[2].rstrip(")") if tool else "did not run",
                   "text": tool_text[name]
                           + (" On this run it reported no pair on any proteome, so it has no metric "
                              "and no column in the tables." if tool and f["found"].get(tool) == 0 else ""),
                   "facts": [], "links": []}
    return d


def overview_control(f: dict) -> dict:
    """The disorder-bin control: each tool's AUC-PR in the bin, as the mean over the
    proteomes with a value, and links to that bin's two tables."""
    ran = {t.split("_k")[0]: t for t in f["tools"]}
    labels = {"all": "all", "ordered": "ordered, below 0.2", "partial": "partial, 0.2 to 0.5",
              "disordered": "disordered, above 0.5"}
    options = []
    for cat in ["all", "ordered", "partial", "disordered"]:
        subs, bars, facts, links = {}, {}, {}, {}
        for name, tool in ran.items():
            v, n = f["aucpr"][cat][tool]
            if v is None:
                subs[name] = ("reported 0 pairs on every proteome" if f["found"].get(tool) == 0
                              else "no metric in this bin")
                bars[name] = 0.0
                facts[name] = [["bin", labels[cat]], ["AUC-PR", "no value"]]
            else:
                subs[name] = f"AUC-PR {v:.2f}, {n} proteomes"
                bars[name] = v
                facts[name] = [["bin", labels[cat]], [f"AUC-PR, mean over {n} proteomes", f"{v:.3f}"]]
            links[name] = [[f"AUC-PR: {cat} proteins", f"auc_pr_{cat}"],
                           [f"Recall @ FDR 5%: {cat} proteins", f"recall_fdr5_{cat}"]]
        options.append({"id": cat, "label": labels[cat], "subs": subs, "bars": bars, "facts": facts,
                        "links": links})
    return {"label": "Disorder bin of the query (mean metapredict score)", "options": options}


def write_metric_explainers(out: Path) -> None:
    """How to read the metrics: the threshold sweep read as AUC-PR, recall at 5% FDR, and
    AUC-ROC, with a widget for the first two."""
    roc = ("<div class='mx'><h5>AUC-ROC</h5><p class='def'>The same sweep of the cutoff, read on "
           "different axes: the share of true pairs found against the share of false pairs let "
           "through. 0.5 is a coin toss, 1 is perfect. With many more negative pairs than positive "
           "ones it moves less than AUC-PR does, which is why AUC-PR is the headline here.</p></div>")
    cfg = {
        "id": "metric_explainers",
        "section_name": "How to read the metrics",
        "description": ("<p>The three metrics in the tables below are one threshold sweep read three "
                        "ways. The example is a toy; the tables carry the numbers.</p>"),
        "plot_type": "html",
        "data": mx.bundle(mx.threshold(floor=0.95, title="AUC-PR and recall at 5% FDR over a score threshold"), roc),
    }
    (out / "metric_explainers_mqc.json").write_text(json.dumps(cfg, indent=1))


def write_overview(out: Path, df: pl.DataFrame, n_queries: int | None) -> None:
    f = overview_facts(df, n_queries)
    arms = "; ".join(
        tool_word(t) + (" (ran, but reported no pair on any proteome, so it has no metric)"
                        if f["found"].get(t) == 0 else "") for t in f["tools"])
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
        "<b>Two readings of one ranking.</b> Both metrics walk each tool's own ranking from "
        "the top; no cutoff is chosen by the tool. AUC-PR is the whole curve. Recall at 5% "
        "FDR is how far down the list a reader gets before one pair in twenty is wrong, "
        "the operating point a curated annotation needs; a tool can rank well overall and "
        "still score near zero there if a few false pairs sit at the very top of its list.",
        "<b>The target proteome is the divergence axis.</b> The same queries against mouse "
        "and against E. coli ask how far each signal reaches back in time.",
    ])
    block = fd.block(dict(
        overview_flow_spec(f), title="DisProt benchmark", details=overview_details(f),
        control=overview_control(f),
        footnote="AUC-PR in the bars and the panel is the mean over the proteomes that have a "
                 "value in that bin; a proteome with no pair in a bin has no row in that table."),
        uid="disprot-flow")
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
            "<p>Read top to bottom; the label on the left says what kind of thing each row "
            "holds. A box holds a name and one number; an arrow is the step that makes the "
            "next box. Arm boxes take the colour their tool has in the curves below. Hover a "
            "box to see what feeds it; click it for what it is, its numbers, and the tables "
            "that show it. Switch the disorder bin and the AUC-PR bars in the arm boxes "
            "move.</p>"
            + block),
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
        f.write(f"# description: '{description} <a href=\"#overview\">&uarr; back to the data flow</a>'\n")
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
        f"description: '{description} <a href=\"#overview\">&uarr; back to the data flow</a>'",
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


def write_multiqc_config(path: Path, tools: list[str], header: str = "") -> None:
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

{header}

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
    Area under the precision-recall curve, every query protein, one row per target
    proteome. Higher is better; a random ranking scores about the share of positive pairs.
  recall_fdr5_all: >
    Recall reached while precision stays at or above 0.95, walking each tool's own ranking
    from the top; no cutoff is chosen by the tool. Near zero means false pairs sit at the
    top of that tool's list, whatever its AUC-PR.
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
    write_metric_explainers(out)

    # ── Per-disorder-category tables ─────────────────────────────────────────
    for cat in DISORDER_CATEGORIES:
        suffix = cat.replace(" ", "_")

        # AUC-PR
        rows, tools = pivot_metric(df, "auc_pr", cat)
        write_table_mqc(
            out / f"auc_pr_{suffix}_mqc.tsv", rows, tools,
            section_name=f"AUC-PR — {cat} proteins",
            description=(f"Area under the precision-recall curve, {cat} query proteins, one row per "
                         f"target proteome in divergence order."),
        )

        # Recall@FDR5
        rows, tools = pivot_metric(df, "recall_at_fdr05", cat)
        write_table_mqc(
            out / f"recall_fdr5_{suffix}_mqc.tsv", rows, tools,
            section_name=f"Recall @ FDR 5% — {cat} proteins",
            description=(f"Recall reached while precision stays at or above 0.95, {cat} query "
                         f"proteins, one row per target proteome in divergence order."),
        )

    # ── Line graph: AUC-PR vs Mya (all proteins) ─────────────────────────────
    write_linegraph_mqc(
        out / "auc_pr_vs_mya_mqc.yaml", df,
        metric="auc_pr",
        section_name="AUC-PR vs evolutionary distance",
        description=("AUC-PR against divergence time in million years, every query protein. A "
                     "line that falls is a tool that finds fewer of the pairs the further the "
                     "proteome is from human."),
        disorder_cat="all",
    )

    # ── Line graph: Recall@FDR5 vs Mya ───────────────────────────────────────
    write_linegraph_mqc(
        out / "recall_fdr5_vs_mya_mqc.yaml", df,
        metric="recall_at_fdr05",
        section_name="Recall @ FDR 5% vs evolutionary distance",
        description=("Recall at precision 0.95 or better against divergence time, every query "
                     "protein. Read it against the AUC-PR line above: a tool low here but not "
                     "there ranks well overall and has false pairs near the top of its list."),
        disorder_cat="all",
    )

    # ── MultiQC config ────────────────────────────────────────────────────────
    f = overview_facts(df, read_query_count(Path(stats_txt) if stats_txt else None))
    arms = "; ".join(tool_word(t) + (" (ran, reported no pair)" if f["found"].get(t) == 0 else "")
                     for t in f["tools"])
    write_multiqc_config(out / "multiqc_config.yaml", all_tools, header=fd.header_yaml([
        ("Query", f"human proteins with a DisProt entry, {fd.num(f['n_queries'])} proteins"),
        ("Targets", f"{len(f['species'])} Quest-for-Orthologs proteomes, {fd.num(f['mya_min'])} to "
                    f"{fd.num(f['mya_max'])} million years from human"),
        ("Ground truth", f"Pfam pair labels: {fd.num(f['pairs_total'])} pairs, {fd.num(f['pos_total'])} "
                         f"positive (the two proteins share a Pfam family)"),
        ("Arms", arms),
        ("Disorder bins", "mean metapredict score: ordered below 0.2, partial 0.2 to 0.5, disordered above 0.5"),
    ]))

    print(f"Wrote MultiQC input files to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
