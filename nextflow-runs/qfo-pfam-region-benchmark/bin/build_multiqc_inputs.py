#!/usr/bin/env python3
"""Turn the benchmark's metrics, curves and Nextflow trace into MultiQC custom content.

One report for the whole run: what each tool found, where it degrades, and what it cost
in wall time, CPU and memory. Everything is written as `*_mqc.json` files that MultiQC's
custom-content module picks up, plus a general-statistics table.

JSON rather than YAML on purpose -- the scoring container carries polars and numpy but no
PyYAML, and the section files are machine-written either way. The one hand-edited file,
assets/multiqc_config.yaml, stays YAML.

Two conventions run through every section:

  heldout, not all      113 alphabet x ksize combos get compared, so the winner is picked
                        by selection. Reporting it on the half of the families that chose
                        it is optimistically biased. Falls back to `all` when a run has no
                        split column, and says so in the section.
  never pool truth sets Pfam is circular with the profile baselines, Swiss-Prot is not.
                        A number averaged across them has no interpretation, so each truth
                        set gets its own section.
"""

import argparse
import json
import math
from itertools import cycle
from pathlib import Path

import polars as pl

import mqc_trace as mt

# Threshold-free, so a tool that ships a lenient default cutoff is not rewarded for it.
HEADLINE = ["fmax", "auprc", "roc_auc", "recall_reachable", "precision", "coverage",
            "smin", "ndo", "sens_first_fp_mean"]

# The four method classes the paper's framing turns on: what a tool needs before it can
# answer at all. Colors are reused by every plot so a class keeps one identity.
CLASSES = {
    "kmerseek":   ("kmerseek (this work)", "#0f9d76"),
    "structure":  ("needs 3D structure",   "#c9528f"),
    "plm":        ("needs language model", "#2b7bba"),
    "alignment":  ("sequence alignment",   "#7f7f7f"),
    "ceiling":    ("annotation ceiling",   "#c99a00"),
}
TOOL_CLASS = {
    "kmerseek": "kmerseek",
    "foldseek": "structure", "reseek": "structure", "folddisco": "structure",
    "prostt5": "plm",
    "hmmer3_phmmer": "alignment", "hmmer3_jackhmmer": "alignment",
    "mmseqs2_seqseq": "alignment", "mmseqs2_iterative": "alignment",
    "hhblits": "alignment",
    "hmmscan": "ceiling",
}
# What each tool needs on the input side, for the capability table beside the frontier.
# Written out rather than derived from the class so ProstT5's "no structures, but a 3B
# parameter model" case is stated rather than implied.
NEEDS_3D = {"foldseek": "Yes", "reseek": "Yes", "folddisco": "Yes"}
ALIGNMENT_FREE = {
    "kmerseek": "Yes", "folddisco": "Yes (motif)",
    "foldseek": "No (3D aln)", "reseek": "No (3D aln)", "prostt5": "No (3Di aln)",
    "hmmer3_phmmer": "No (seq aln)", "hmmer3_jackhmmer": "No (seq aln)",
    "mmseqs2_seqseq": "No (seq aln)", "mmseqs2_iterative": "No (seq aln)",
    "hhblits": "No (profile aln)", "hmmscan": "No (profile aln)",
}
# Identity bins below the twilight-zone boundary. The central claim is stated here.
GRAY_ZONE = ["0-20%", "20-30%", "30-40%"]

# For scatters grouped by something with no fixed class (alphabet, process). Okabe-Ito,
# which stays distinguishable for the common forms of color blindness.
SERIES_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9",
                 "#f0e442", "#000000"]
# The bins are an ordered axis, so they get a ramp rather than the default categorical
# palette, which put "20-30%" in near-black next to a pastel "30-40%". Dark is hard.
IDENTITY_COLORS = {
    "0-20%": "#08306b", "20-30%": "#2b7bba", "30-40%": "#6baed6",
    "40-60%": "#bdd7e7", "60-100%": "#eff3ff", "no_homolog": "#d9b3b3",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def clean(value):
    """JSON has no NaN or Infinity. Plotly reads null as a gap, which is what these are."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write_section(outdir: Path, section_id: str, cfg: dict) -> None:
    cfg.setdefault("parent_id", "qfo_region")
    cfg.setdefault("parent_name", "QfO Pfam region benchmark")
    (outdir / f"{section_id}_mqc.json").write_text(json.dumps(clean(cfg), indent=1))


def tool_class(tool: str) -> str:
    return TOOL_CLASS.get(tool, "alignment")


def tool_color(tool: str) -> str:
    return CLASSES[tool_class(tool)][1]


def pick_split(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """The heldout half if the run produced one, otherwise everything, named either way."""
    if "split" not in df.columns:
        return df, "all"
    held = df.filter(pl.col("split") == "heldout")
    if held.height:
        return held, "heldout"
    return df.filter(pl.col("split") == "all"), "all"


def ungrouped(df: pl.DataFrame) -> pl.DataFrame:
    if "stratum_axis" not in df.columns:
        return df
    return df.filter(pl.col("stratum_axis") == "all")


def best_variants(df: pl.DataFrame) -> pl.DataFrame:
    """One row per tool: its best variant, with every headline metric averaged.

    Averaged over species before ranking, never summed -- summing would let the species
    with the most annotated proteins pick the winner. kmerseek brings 113 alphabet x ksize
    combos against every baseline's one, so each tool is reduced to its own best variant
    rather than letting the sweep bury the baselines in a shared ranking.
    """
    cols = [c for c in HEADLINE if c in df.columns]
    if not cols or df.height == 0:
        return df.head(0)
    per_variant = (
        df.group_by("tool", "variant")
        .agg([pl.col(c).mean() for c in cols]
             + [pl.col("species").n_unique().alias("n_species")])
        .sort("fmax", descending=True, nulls_last=True)
    )
    return (
        per_variant.group_by("tool")
        .agg([pl.col("variant").first().alias("best_variant"),
              pl.col("n_species").first()] + [pl.col(c).first() for c in cols])
        .sort("fmax", descending=True, nulls_last=True)
    )


def fmt_metric_headers(cols: list[str]) -> dict:
    """Column formatting for the leaderboard tables, keyed by metric name."""
    spec = {
        "best_variant": dict(title="Best variant", description="Highest mean Fmax over species"),
        "fmax":         dict(title="Fmax", description="CAFA protein-centric Fmax", min=0, max=1,
                             scale="RdYlGn", format="{:,.3f}"),
        "auprc":        dict(title="AUPRC", description="Area under precision / reachable-recall",
                             min=0, max=1, scale="Blues", format="{:,.3f}"),
        "roc_auc":      dict(title="ROC AUC", min=0, max=1, scale="Purples", format="{:,.3f}"),
        "recall_reachable": dict(title="Recall (reachable)", min=0, max=1, scale="Greens",
                                 format="{:,.3f}",
                                 description="Of the domains transferable from this target "
                                             "proteome at all, the fraction recovered"),
        "precision":    dict(title="Precision", min=0, max=1, scale="Oranges", format="{:,.3f}",
                             description="Gray-zone calls excluded from the denominator"),
        "coverage":     dict(title="Scoreable", min=0, max=1, scale="Greys", format="{:,.3f}",
                             description="Fraction of calls that could be judged at all"),
        "smin":         dict(title="Smin", scale="RdYlGn-rev", format="{:,.2f}",
                             description="CAFA semantic distance; lower is better"),
        "ndo":          dict(title="nDO", min=0, max=1, scale="BuGn", format="{:,.3f}",
                             description="Residue-level normalized domain overlap"),
        "sens_first_fp_mean": dict(title="Sens. to 1st FP", min=0, max=1, scale="YlGn",
                                   format="{:,.3f}",
                                   description="Domains recovered above a query's first "
                                               "false positive"),
        "n_species":    dict(title="Species", format="{:,.0f}", scale=False),
    }
    return {c: spec[c] for c in cols if c in spec}


# ---------------------------------------------------------------------------
# sections: accuracy
# ---------------------------------------------------------------------------

def section_leaderboards(out: Path, metrics: pl.DataFrame) -> None:
    cut, split = pick_split(ungrouped(metrics))
    for ts in sorted(cut["truth_set"].unique().to_list()):
        board = best_variants(cut.filter(pl.col("truth_set") == ts))
        if board.height == 0:
            continue
        cols = ["best_variant", "n_species"] + [c for c in HEADLINE if c in board.columns]
        data = {row["tool"]: {c: row[c] for c in cols} for row in board.to_dicts()}
        write_section(out, f"qfo_leaderboard_{ts}", {
            "id": f"qfo_leaderboard_{ts}",
            "section_name": f"Leaderboard — {ts} truth",
            "description": (
                f"Best variant per tool, averaged over target species, on the "
                f"<code>{split}</code> split with no stratification. "
                + ("Pfam-A domains are defined by profile HMMs, so this truth set is "
                   "circular with phmmer, jackhmmer, hhblits and hmmscan and flatters "
                   "them. Compare against the Swiss-Prot and Pfam-N boards before "
                   "concluding anything." if ts == "pfam" else
                   "Literature-curated, defined by function, and circular with neither "
                   "the profile baselines nor the structure baselines."
                   if ts == "swissprot" else
                   "Labels that exist precisely where the Pfam-A HMMs failed."
                   if ts == "pfamn" else
                   "Catalytic residues from M-CSA. Coverage is 95 human proteins, so this "
                   "is a vignette and must not carry a headline number." if ts == "mcsa"
                   else "")),
            "plot_type": "table",
            "pconfig": {"id": f"qfo_leaderboard_{ts}_table",
                        "title": f"Domain-finding leaderboard ({ts})",
                        "col1_header": "Tool", "sort_rows": False, "scale": False},
            "headers": fmt_metric_headers(cols),
            "data": data,
        })


def section_frontier(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                     n_queries: int, primary_truth: str) -> None:
    """Sensitivity against speed, with the incumbent frontier drawn in.

    y is Fmax restricted to domain instances whose closest same-family target domain is
    under 40% identical -- the regime the hypothesis is about, not the easy one. x is
    measured throughput from the trace, not a published figure: query proteins divided by
    the search task's own wall time, median over target species.

    A tool missing from either axis is missing from the plot. That is mostly the structure
    arms when no structures were staged, and it is stated in the section rather than left
    as an unexplained gap.
    """
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))

    gray = cut.filter(
        (pl.col("stratum_axis") == "identity") & pl.col("stratum").is_in(GRAY_ZONE)
    )
    if gray.height:
        y_label = "Fmax, &lt;40% identity"
        y_note = ("Fmax over domain instances under 40% identity to their closest "
                  "same-family target domain.")
        # Weight by instances so a bin holding 40 domains does not count as much as one
        # holding 4000.
        ycol = (
            gray.group_by("tool", "variant")
            .agg(((pl.col("fmax") * pl.col("n_truth_instances")).sum()
                  / pl.col("n_truth_instances").sum().clip(lower_bound=1)).alias("y"))
        )
    else:
        y_label = "Fmax (all instances)"
        y_note = ("No identity stratification in this run (<code>--skip_identity</code>), "
                  "so this is Fmax over all instances. The gray-zone cut is the one the "
                  "claim is stated on; re-run without that flag to get it.")
        ycol = ungrouped(cut).group_by("tool", "variant").agg(pl.col("fmax").mean().alias("y"))

    best = (ycol.sort("y", descending=True, nulls_last=True)
                .group_by("tool").agg(pl.col("variant").first(), pl.col("y").first()))

    thr = throughput_per_tool(trace, n_queries)
    joined = best.join(thr, on="tool", how="inner").filter(
        pl.col("y").is_not_null() & (pl.col("queries_per_s") > 0)
    )
    dropped = sorted(set(best["tool"]) - set(joined["tool"]))
    if joined.height == 0:
        return

    rows = joined.sort("queries_per_s").to_dicts()
    # The frontier the incumbents actually reach. Everything above and to the right of
    # this corner is territory no existing tool occupies, which is the claim the figure is
    # making -- so it is computed from the data rather than drawn by hand.
    incumbents = [r for r in rows if r["tool"] != "kmerseek"]
    y_best = max((r["y"] for r in incumbents), default=None)
    x_best = max((r["queries_per_s"] for r in incumbents), default=None)

    points = {}
    for r in rows:
        cls = tool_class(r["tool"])
        label = r["tool"] if r["tool"] != "kmerseek" else f"kmerseek {r['variant']}"
        points[label] = {
            "x": r["queries_per_s"], "y": r["y"],
            "color": CLASSES[cls][1], "group": CLASSES[cls][0], "annotation": label,
            "marker_size": 14 if cls == "kmerseek" else 9,
        }

    xs = [r["queries_per_s"] for r in rows]
    ys = [r["y"] for r in rows]
    pconfig = {
        "id": "qfo_frontier_plot",
        "title": f"Sensitivity x speed ({primary_truth} truth, {split} split)",
        "xlab": "throughput (query proteins / s, log scale)",
        "ylab": y_label, "xlog": True, "height": 560, "marker_line_width": 1,
        # Point labels are drawn beside the marker, so a point at the axis limit loses
        # half its label. A decade of padding either side is enough on a log axis.
        "xmin": min(xs) / 3, "xmax": max(xs) * 3,
        "ymin": 0, "ymax": min(1.0, max(ys) * 1.25),
        # MultiQC copies a "%" out of the axis label into the tick suffix, and this label
        # carries one in "<40% identity" -- which turned every Fmax tick into "0.6%".
        # Setting the suffix explicitly is the documented way to stop that inference.
        "ysuffix": "", "tt_decimals": 3,
    }
    lines = []
    if y_best is not None:
        lines.append({"value": y_best, "color": "#999999", "dash": "dash", "width": 2,
                      "label": "best incumbent sensitivity"})
    if lines:
        pconfig["y_lines"] = lines
    if x_best is not None:
        pconfig["x_lines"] = [{"value": x_best, "color": "#999999", "dash": "dash",
                               "width": 2, "label": "fastest incumbent"}]

    note = ""
    if dropped:
        note = ("<p>No throughput row for " + ", ".join(f"<code>{d}</code>" for d in dropped)
                + " — that arm either did not run or produced no trace record, so it is "
                  "absent from the plot rather than plotted at zero.</p>")
    write_section(out, "qfo_frontier", {
        "id": "qfo_frontier",
        "section_name": "The frontier",
        "description": (
            f"<p>{y_note} Throughput is measured on this run: {n_queries:,} human query "
            "proteins divided by each search task's wall time at the CPU count it was "
            "given, taken as the median over target species. Dashed lines mark the best "
            "and fastest incumbent, so the upper-right quadrant is the part of the space "
            "no existing tool reaches.</p>"
            "<p>Speed here includes indexing, because every arm is timed the same way: "
            "one task, one target proteome, start to finish. A tool that amortises an "
            "index across many searches looks worse here than in steady-state use.</p>"
            + note),
        "plot_type": "scatter",
        "pconfig": pconfig,
        "data": points,
    })

    # The capability table that travels with the figure: sensitivity means little without
    # what a tool needs before it can produce it.
    board = best_variants(ungrouped(cut))
    cap = {}
    for row in board.to_dicts():
        t = row["tool"]
        thr_row = thr.filter(pl.col("tool") == t)
        cap[t] = {
            "cls": CLASSES[tool_class(t)][0],
            "needs_3d": NEEDS_3D.get(t, "No"),
            "alignment_free": ALIGNMENT_FREE.get(t, "No"),
            "fmax": row.get("fmax"),
            "gray_fmax": (best.filter(pl.col("tool") == t)["y"].first()
                          if best.filter(pl.col("tool") == t).height else None),
            "queries_per_s": (thr_row["queries_per_s"].first() if thr_row.height else None),
            "cpu_hours": (thr_row["cpu_hours"].first() if thr_row.height else None),
        }
    write_section(out, "qfo_capability", {
        "id": "qfo_capability",
        "section_name": "What each tool needs",
        "description": "Inputs required, accuracy, and measured cost, side by side.",
        "plot_type": "table",
        "pconfig": {"id": "qfo_capability_table", "title": "Method capabilities and cost",
                    "col1_header": "Tool", "sort_rows": False, "scale": False},
        "headers": {
            "cls": dict(title="Class", scale=False),
            "needs_3d": dict(title="Needs 3D?", scale=False),
            "alignment_free": dict(title="Alignment-free?", scale=False),
            "fmax": dict(title="Fmax (all)", min=0, max=1, scale="RdYlGn", format="{:,.3f}"),
            "gray_fmax": dict(title="Fmax (<40% id)", min=0, max=1, scale="RdYlGn",
                              format="{:,.3f}"),
            "queries_per_s": dict(title="Queries/s", scale="Blues", format="{:,.1f}"),
            "cpu_hours": dict(title="CPU-hours", scale="Reds", format="{:,.2f}",
                              description="Summed over every search task in this arm"),
        },
        "data": cap,
    })


def throughput_per_tool(trace: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """Median query proteins per second per tool, plus that arm's total CPU-hours.

    Median over target species rather than mean: one species dominating the sweep's wall
    time should not move a rate that is meant to describe the method.
    """
    empty = pl.DataFrame(schema={"tool": pl.String, "queries_per_s": pl.Float64,
                                 "cpu_hours": pl.Float64})
    if trace.height == 0:
        return empty
    searches = trace.filter(
        pl.col("is_search") & pl.col("realtime_s").is_not_null() & (pl.col("realtime_s") > 0)
    )
    if searches.height == 0:
        return empty
    per_task = searches.with_columns(
        (n_queries / pl.col("realtime_s")).alias("qps")
    )
    rate = per_task.group_by("tool").agg(
        pl.col("qps").median().alias("queries_per_s"),
        pl.col("cpu_hours").sum().alias("cpu_hours"),
    )
    # mmseqs2 runs two variants under one process name, so the trace cannot separate
    # them; the metrics table can. Emit both metric-table spellings from the one rate
    # rather than dropping mmseqs2 out of the join entirely.
    mm = rate.filter(pl.col("tool") == "mmseqs2")
    if mm.height:
        rate = pl.concat([
            rate.filter(pl.col("tool") != "mmseqs2"),
            mm.with_columns(pl.lit("mmseqs2_seqseq").alias("tool")),
            mm.with_columns(pl.lit("mmseqs2_iterative").alias("tool")),
        ])
    return rate


def section_curves(out: Path, curves: pl.DataFrame, metrics: pl.DataFrame,
                   primary_truth: str, max_lines: int) -> None:
    """Precision / recall and ROC, one line per tool at its best variant."""
    if curves.height == 0:
        return
    cut, split = pick_split(curves.filter(pl.col("truth_set") == primary_truth))
    if cut.height == 0:
        return
    mcut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    board = best_variants(mcut)
    keep = {r["tool"]: r["best_variant"] for r in board.head(max_lines).to_dicts()}

    for kind, xcol, ycol, xlab, ylab, title in (
        ("pr", "recall_reachable", "precision", "recall (reachable instances)",
         "precision", "Precision / recall"),
        ("roc", "fpr", "tpr", "false positive rate", "true positive rate", "ROC"),
    ):
        if xcol not in cut.columns or ycol not in cut.columns:
            continue
        data = {}
        for tool, variant in keep.items():
            sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
            if sub.height == 0:
                continue
            # Pooled over species by binning x, because one line per tool per species
            # would be 100+ lines. The per-species curves stay in all_domain_curves.parquet.
            binned = (
                sub.with_columns((pl.col(xcol) * 100).round(0).cast(pl.Int64).alias("bin"))
                .group_by("bin").agg(pl.col(ycol).mean().alias("y"))
                .sort("bin")
            )
            series = {str(r["bin"] / 100): r["y"] for r in binned.to_dicts()}
            if series:
                data[tool if tool != "kmerseek" else f"kmerseek {variant}"] = series
        if not data:
            continue
        write_section(out, f"qfo_{kind}_curve", {
            "id": f"qfo_{kind}_curve",
            "section_name": f"{title} curves",
            "description": (f"{primary_truth} truth, <code>{split}</code> split, each "
                            "tool at its best variant, pooled over target species. "
                            "Recall counts distinct true domain instances; precision "
                            "counts calls."),
            "plot_type": "linegraph",
            "pconfig": {"id": f"qfo_{kind}_curve_plot", "title": f"{title} ({primary_truth})",
                        "xlab": xlab, "ylab": ylab, "xmin": 0, "xmax": 1,
                        "ymin": 0, "ymax": 1, "height": 500},
            "data": data,
        })


def section_identity(out: Path, metrics: pl.DataFrame, primary_truth: str,
                     max_tools: int) -> None:
    """Fmax by percent-identity bin: the twilight-zone axis the claim is stated on."""
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    ident = cut.filter(pl.col("stratum_axis") == "identity")
    if ident.height == 0:
        return
    board = best_variants(ungrouped(cut)).head(max_tools)
    keep = {r["tool"]: r["best_variant"] for r in board.to_dicts()}
    order = [b for b in ["0-20%", "20-30%", "30-40%", "40-60%", "60-100%", "no_homolog"]
             if b in ident["stratum"].unique().to_list()]
    data = {}
    for tool, variant in keep.items():
        sub = ident.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        by_bin = sub.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()
        lookup = {r["stratum"]: r["fmax"] for r in by_bin}
        data[tool if tool != "kmerseek" else f"kmerseek {variant}"] = {
            b: lookup.get(b) for b in order
        }
    if not data:
        return
    write_section(out, "qfo_identity", {
        "id": "qfo_identity",
        "section_name": "Twilight zone",
        "description": (
            f"Fmax by percent identity between a domain instance and its closest "
            f"same-family domain in the target proteome ({primary_truth} truth, "
            f"<code>{split}</code> split). <code>no_homolog</code> is instances with no "
            "same-family target domain at all: unreachable by transfer, kept as their own "
            "bin so they never contaminate the &lt;20% bin the hypothesis cares about."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_identity_plot", "title": "Fmax by percent identity",
                    "ylab": "Fmax", "cpswitch": False, "stacking": "group", "height": 500},
        "categories": {b: {"name": b, "color": IDENTITY_COLORS.get(b)} for b in order},
        "data": data,
    })


def section_divergence(out: Path, metrics: pl.DataFrame, primary_truth: str,
                       max_tools: int) -> None:
    """Fmax against divergence time. The species IS the divergence axis here."""
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cut = cut.filter(pl.col("species") != "all")
    if cut.height == 0 or "species_mya" not in cut.columns:
        return
    board = best_variants(cut).head(max_tools)
    keep = {r["tool"]: r["best_variant"] for r in board.to_dicts()}
    data, recall = {}, {}
    for tool, variant in keep.items():
        sub = (cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
                  .group_by("species_mya")
                  .agg(pl.col("fmax").mean(),
                       pl.col("recall").mean(),
                       pl.col("recall_reachable").mean())
                  .sort("species_mya"))
        name = tool if tool != "kmerseek" else f"kmerseek {variant}"
        data[name] = {str(r["species_mya"]): r["fmax"] for r in sub.to_dicts()}
        recall[name] = {str(r["species_mya"]): r["recall_reachable"] for r in sub.to_dicts()}
    write_section(out, "qfo_divergence", {
        "id": "qfo_divergence",
        "section_name": "Divergence",
        "description": (
            f"Fmax and reachable recall against divergence time from human, in millions "
            f"of years ({primary_truth} truth, <code>{split}</code> split). Raw recall is "
            "deliberately absent: a human family that does not exist in the target "
            "proteome cannot be transferred by any search, and E. coli holds 971 of "
            "human's 8,909 families against mouse's 8,805. Comparing tools on raw recall "
            "would mostly compare proteomes."),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_divergence_plot", "title": "Accuracy vs divergence time",
                    "xlab": "divergence from human (Mya)", "ylab": "score",
                    "ymin": 0, "ymax": 1, "height": 500,
                    "data_labels": [{"name": "Fmax", "ylab": "Fmax"},
                                    {"name": "Recall (reachable)",
                                     "ylab": "recall_reachable"}]},
        "data": [data, recall],
    })


def section_alphabet_matrix(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """The sweep itself: Fmax over alphabet x ksize, one heatmap per low-complexity arm."""
    cut, split = pick_split(ungrouped(metrics.filter(
        (pl.col("truth_set") == primary_truth) & (pl.col("tool") == "kmerseek"))))
    if cut.height == 0:
        return
    parsed = cut.with_columns(
        pl.col("variant").str.extract(r"^(.*)_k\d+_lc(?:True|False)$", 1).alias("alphabet"),
        pl.col("variant").str.extract(r"_k(\d+)_lc", 1).cast(pl.Int64).alias("ksize"),
        pl.col("variant").str.extract(r"_lc(True|False)$", 1).alias("lowcomp"),
    ).filter(pl.col("alphabet").is_not_null())
    if parsed.height == 0:
        return

    for lc in ["False", "True"]:
        sub = parsed.filter(pl.col("lowcomp") == lc)
        if sub.height == 0:
            continue
        grid = sub.group_by("alphabet", "ksize").agg(pl.col("fmax").mean()).sort("ksize")
        ks = sorted(grid["ksize"].unique().to_list())
        # Ranked by each alphabet's best cell, so the top rows are the ones worth reading.
        alphas = (grid.group_by("alphabet").agg(pl.col("fmax").max())
                      .sort("fmax", descending=True)["alphabet"].to_list())
        lookup = {(r["alphabet"], r["ksize"]): r["fmax"] for r in grid.to_dicts()}
        rows = [[lookup.get((a, k)) for k in ks] for a in alphas]
        write_section(out, f"qfo_alphabet_lc{lc}", {
            "id": f"qfo_alphabet_lc{lc}",
            "section_name": f"Alphabet x ksize — low-complexity filter {lc.lower()}",
            "description": (
                f"Mean Fmax over target species for every alphabet and k-mer size in the "
                f"sweep ({primary_truth} truth, <code>{split}</code> split), with "
                f"low-complexity k-mer removal <b>{lc.lower()}</b>. Blank cells are combos "
                "outside that alphabet's k range — the floor is set from measured bits per "
                "symbol, not from class count, so a 2-letter alphabet starts at k=18 while "
                "protein20 starts at k=4."),
            "plot_type": "heatmap",
            "pconfig": {"id": f"qfo_alphabet_lc{lc}_plot",
                        "title": f"Fmax by alphabet and ksize (lc={lc.lower()})",
                        "xlab": "k", "ylab": "alphabet", "min": 0, "max": 1,
                        "square": False, "height": 500},
            "xcats": [str(k) for k in ks],
            "ycats": alphas,
            "data": rows,
        })

    # Does dropping low-complexity k-mers help? It depends on the alphabet, which is why
    # the toggle is swept rather than fixed -- so the answer belongs in the report.
    both = (parsed.group_by("alphabet", "lowcomp").agg(pl.col("fmax").max())
                  .pivot(on="lowcomp", index="alphabet", values="fmax"))
    if {"True", "False"}.issubset(set(both.columns)):
        delta = both.with_columns((pl.col("True") - pl.col("False")).alias("delta")).sort("delta")
        write_section(out, "qfo_lowcomplexity", {
            "id": "qfo_lowcomplexity",
            "section_name": "Low-complexity filter",
            "description": (
                "Change in best-combo Fmax from removing low-complexity k-mers, per "
                "alphabet. Positive means the filter helped. A 2-letter alphabet generates "
                "far more homopolymer-like k-mers than a 20-letter one, so this is expected "
                "to split by alphabet rather than point one way."),
            "plot_type": "bargraph",
            "pconfig": {"id": "qfo_lowcomplexity_plot",
                        "title": "Fmax change from low-complexity removal",
                        "ylab": "Fmax(filtered) - Fmax(unfiltered)", "cpswitch": False,
                        "height": 400},
            "categories": {"delta": {"name": "delta Fmax", "color": "#0f9d76"}},
            "data": {r["alphabet"]: {"delta": r["delta"]} for r in delta.to_dicts()},
        })


def section_boundary(out: Path, metrics: pl.DataFrame, primary_truth: str,
                     max_tools: int) -> None:
    """Right family in the wrong place is the failure mode this benchmark exists to catch."""
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    if cut.height == 0:
        return
    board = best_variants(cut).head(max_tools)
    keep = {r["tool"]: r["best_variant"] for r in board.to_dicts()}
    cols = ["ndo", "residue_precision", "residue_recall", "residue_f1", "median_iou_tp",
            "precision_iou80", "recall_iou80", "n_tp_strict",
            "dbd_median", "dbd_mean",
            "nterm_offset_median", "nterm_offset_mean", "nterm_offset_iqr",
            "cterm_offset_median", "cterm_offset_mean", "cterm_offset_iqr",
            "domain_count_mcc", "domain_count_accuracy"]
    cols = [c for c in cols if c in cut.columns]
    if not cols:
        return
    data = {}
    for tool, variant in keep.items():
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        agg = sub.select([pl.col(c).mean() for c in cols]).to_dicts()[0]
        name = tool if tool != "kmerseek" else f"kmerseek {variant}"
        agg["semantics"] = (sub["interval_semantics"].first()
                            if "interval_semantics" in sub.columns else "alignment")
        data[name] = agg
    write_section(out, "qfo_boundary", {
        "id": "qfo_boundary",
        "section_name": "Boundary accuracy",
        "description": (
            f"Where a call lands, not just whether the family is right ({primary_truth} "
            f"truth, <code>{split}</code> split). Offsets are in residues, signed: negative "
            "N-terminal means the call starts before the true domain, so a systematic "
            "bias shows as a median away from zero rather than as a wider IQR. Rows marked "
            "<code>motif</code> report the envelope of a discontinuous residue set rather "
            "than an alignment, so their boundary numbers measure a different thing and "
            "should not be ranked against the alignment rows."),
        "plot_type": "table",
        "pconfig": {"id": "qfo_boundary_table", "title": "Residue-level and boundary metrics",
                    "col1_header": "Tool", "sort_rows": False, "scale": False},
        "headers": {
            "semantics": dict(title="Semantics", scale=False),
            "ndo": dict(title="nDO", min=0, max=1, scale="BuGn", format="{:,.3f}",
                        description="Normalized domain overlap: correctly labelled "
                                    "residues over true domain residues"),
            "residue_precision": dict(title="Residue prec.", min=0, max=1, scale="BuGn",
                                      format="{:,.3f}",
                                      description="Measured from the prediction side"),
            "residue_recall": dict(title="Residue rec.", min=0, max=1, scale="BuGn",
                                   format="{:,.3f}",
                                   description="Measured from the truth side; equals nDO"),
            "residue_f1": dict(title="Residue F1", min=0, max=1, scale="BuGn",
                               format="{:,.3f}"),
            "median_iou_tp": dict(title="Median IoU (TP)", min=0, max=1, scale="Greens",
                                  format="{:,.3f}",
                                  description="How well correct calls overlay the true "
                                              "domain interval"),
            "precision_iou80": dict(title="Precision @ IoU>=0.8", min=0, max=1,
                                    scale="Oranges", format="{:,.3f}",
                                    description="The strict correctly-parsed criterion "
                                                "used by structure domain parsers"),
            "recall_iou80": dict(title="Recall @ IoU>=0.8", min=0, max=1, scale="Oranges",
                                 format="{:,.3f}"),
            "n_tp_strict": dict(title="Strict TP", format="{:,.0f}", scale="Greens",
                                description="Calls at IoU >= 0.8"),
            "dbd_median": dict(title="DBD median", scale="Reds-rev", format="{:,.1f}",
                               description="Domain boundary distance in residues, over "
                                           "correct calls only"),
            "dbd_mean": dict(title="DBD mean", scale="Reds-rev", format="{:,.1f}"),
            "nterm_offset_median": dict(title="N-term median", scale=False, format="{:,.1f}",
                                        description="Signed: positive means the call "
                                                    "starts after the true domain"),
            "nterm_offset_mean": dict(title="N-term mean", scale=False, format="{:,.1f}",
                                      hidden=True),
            "nterm_offset_iqr": dict(title="N-term IQR", scale="Reds-rev", format="{:,.1f}",
                                     hidden=True),
            "cterm_offset_median": dict(title="C-term median", scale=False, format="{:,.1f}",
                                        description="Signed: positive means the call ends "
                                                    "after the true domain"),
            "cterm_offset_mean": dict(title="C-term mean", scale=False, format="{:,.1f}",
                                      hidden=True),
            "cterm_offset_iqr": dict(title="C-term IQR", scale="Reds-rev", format="{:,.1f}",
                                     hidden=True),
            "domain_count_mcc": dict(title="Count MCC", scale="RdYlGn", format="{:,.3f}",
                                     description="Single- vs multi-domain call, scored by "
                                                 "Matthews correlation"),
            "domain_count_accuracy": dict(title="Count acc.", min=0, max=1, scale="Greys",
                                          format="{:,.3f}", hidden=True),
        },
        "data": data,
    })


def _per_tool_table(cut: pl.DataFrame, cols: list[str], max_tools: int) -> dict:
    """Mean of each column over species, one row per tool at its best variant."""
    board = best_variants(cut).head(max_tools)
    cols = [c for c in cols if c in cut.columns]
    data = {}
    for row in board.to_dicts():
        tool, variant = row["tool"], row["best_variant"]
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0 or not cols:
            continue
        name = tool if tool != "kmerseek" else f"kmerseek {variant}"
        data[name] = sub.select([pl.col(c).mean() for c in cols]).to_dicts()[0]
    return data


def section_cafa(out: Path, metrics: pl.DataFrame, primary_truth: str,
                 max_tools: int) -> None:
    """The CAFA-derived scalars, with the threshold each one is reached at.

    Fmax and Smin are protein-centric: precision is averaged over proteins the tool made a
    prediction on, recall over proteins that have a true annotation. A tool that stays
    silent on a hard protein is therefore not scored on it for precision but is for
    recall, which is the asymmetry CAFA introduced on purpose.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cols = ["fmax", "fmax_threshold", "fmax_precision", "fmax_recall", "wfmax",
            "smin", "smin_threshold", "smin_ru", "smin_mi"]
    data = _per_tool_table(cut, cols, max_tools)
    if not data:
        return
    write_section(out, "qfo_cafa", {
        "id": "qfo_cafa",
        "section_name": "CAFA-style metrics",
        "description": (
            f"{primary_truth} truth, <code>{split}</code> split, averaged over target "
            "species. <b>Fmax</b> is the maximum F-score over score thresholds; the "
            "precision and recall columns are the operating point where it is reached. "
            "<b>wFmax</b> weights each family by its information content, "
            "IC = -log<sub>2</sub> P(family), so recovering a rare family counts for more "
            "than recovering a common one. <b>Smin</b> is the minimum of "
            "sqrt(remaining uncertainty<sup>2</sup> + misinformation<sup>2</sup>) in bits, "
            "and lower is better; <code>smin_ru</code> is information still missing (false "
            "negatives) and <code>smin_mi</code> is information invented (false positives) "
            "at that threshold, so the two say which way a tool is failing.<br>"
            "The weighting here is <b>not</b> CAFA's information accretion, which is "
            "defined over an ontology's parent-child structure. Pfam is flat, so plain IC "
            "is used and the metric is reported under that narrower definition."),
        "plot_type": "table",
        "pconfig": {"id": "qfo_cafa_table", "title": f"CAFA-style metrics ({primary_truth})",
                    "col1_header": "Tool", "sort_rows": False, "scale": False},
        "headers": {
            "fmax": dict(title="Fmax", min=0, max=1, scale="RdYlGn", format="{:,.3f}"),
            "fmax_threshold": dict(title="@ threshold", scale=False, format="{:,.2f}",
                                   description="Score cutoff where Fmax is reached"),
            "fmax_precision": dict(title="Prec. @ Fmax", min=0, max=1, scale="Oranges",
                                   format="{:,.3f}"),
            "fmax_recall": dict(title="Rec. @ Fmax", min=0, max=1, scale="Greens",
                                format="{:,.3f}"),
            "wfmax": dict(title="wFmax", min=0, max=1, scale="PuBuGn", format="{:,.3f}",
                          description="Fmax weighted by family information content"),
            "smin": dict(title="Smin (bits)", scale="RdYlGn-rev", format="{:,.2f}",
                         description="Lower is better"),
            "smin_threshold": dict(title="@ threshold", scale=False, format="{:,.2f}",
                                   hidden=True),
            "smin_ru": dict(title="RU (bits)", scale="Reds", format="{:,.2f}",
                            description="Information remaining: false negatives"),
            "smin_mi": dict(title="MI (bits)", scale="Reds", format="{:,.2f}",
                            description="Information invented: false positives"),
        },
        "data": data,
    })


def section_threshold_metrics(out: Path, metrics: pl.DataFrame, primary_truth: str,
                              max_tools: int) -> None:
    """The tool's own operating point beside the best one it could have chosen."""
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cols = ["precision", "precision_strict", "recall", "recall_reachable", "f1",
            "f1_reachable", "roc_auc", "auprc", "best_f1", "best_f1_threshold",
            "best_f1_precision", "best_f1_recall_reachable", "median_iou_tp"]
    data = _per_tool_table(cut, cols, max_tools)
    if not data:
        return
    write_section(out, "qfo_threshold", {
        "id": "qfo_threshold",
        "section_name": "Threshold-based and threshold-free",
        "description": (
            f"{primary_truth} truth, <code>{split}</code> split, averaged over target "
            "species. The left block is what each tool reported at its own default cutoff, "
            "which differs between tools and is not a property of the method. The right "
            "block is threshold-free, and is the comparable one: <b>ROC AUC</b> is the "
            "probability a correct call outranks an incorrect one, <b>AUPRC</b> is average "
            "precision over score-ranked calls, and <b>best F1</b> is the optimum at any "
            "threshold with the operating point it sits at.<br>"
            "Recall is against <i>reachable</i> instances throughout — those whose family "
            "exists in the target proteome and could be transferred at all. "
            "<code>precision_strict</code> is the same precision with gray-zone calls "
            "charged as errors, kept visible so the gray-zone convention can never be "
            "mistaken for a free improvement."),
        "plot_type": "table",
        "pconfig": {"id": "qfo_threshold_table",
                    "title": f"Threshold metrics ({primary_truth})",
                    "col1_header": "Tool", "sort_rows": False, "scale": False},
        "headers": {
            "precision": dict(title="Precision", min=0, max=1, scale="Oranges",
                              format="{:,.3f}"),
            "precision_strict": dict(title="Prec. (strict)", min=0, max=1, scale="Oranges",
                                     format="{:,.3f}",
                                     description="Gray-zone calls counted as errors"),
            "recall": dict(title="Recall (raw)", min=0, max=1, scale="Greens",
                           format="{:,.3f}", hidden=True),
            "recall_reachable": dict(title="Recall", min=0, max=1, scale="Greens",
                                     format="{:,.3f}",
                                     description="Of transferable instances only"),
            "f1": dict(title="F1", min=0, max=1, scale="RdYlGn", format="{:,.3f}",
                       hidden=True),
            "f1_reachable": dict(title="F1", min=0, max=1, scale="RdYlGn",
                                 format="{:,.3f}"),
            "roc_auc": dict(title="ROC AUC", min=0, max=1, scale="Purples",
                            format="{:,.3f}"),
            "auprc": dict(title="AUPRC", min=0, max=1, scale="Blues", format="{:,.3f}"),
            "best_f1": dict(title="Best F1", min=0, max=1, scale="RdYlGn",
                            format="{:,.3f}"),
            "best_f1_threshold": dict(title="@ threshold", scale=False, format="{:,.2f}"),
            "best_f1_precision": dict(title="Prec. @ best", min=0, max=1, scale="Oranges",
                                      format="{:,.3f}", hidden=True),
            "best_f1_recall_reachable": dict(title="Rec. @ best", min=0, max=1,
                                             scale="Greens", format="{:,.3f}", hidden=True),
            "median_iou_tp": dict(title="Median IoU (TP)", min=0, max=1, scale="Greens",
                                  format="{:,.3f}"),
        },
        "data": data,
    })


# Where each truth set comes from and what it is circular with. Scale is measured off the
# run rather than written here; these are the properties a measurement cannot recover.
TRUTH_PROVENANCE = {
    "pfam": ("Pfam-A database", "Profile HMM baselines: phmmer, jackhmmer, hhblits, hmmscan",
             "Primary. Domains are DEFINED by the profile HMMs those baselines run, so "
             "their scores here are inflated, and a region Pfam never annotated is "
             "labelled absent — which charges every correct cryptic-domain rescue as a "
             "false positive."),
    "swissprot": ("UniProt literature curation", "Nothing here",
                  "Feature-based, not HMM-based. Circular with neither the profile "
                  "baselines nor the structure baselines, which is why it is the default "
                  "primary truth set for the frontier and curve sections."),
    "pfamn": ("Pfam 35.0 explicit gaps", "Nothing here",
              "The label set that exists precisely where the Pfam-A HMMs failed. Turns a "
              "slice of Pfam-silent territory back into scoreable true positives."),
    "mcsa": ("M-CSA catalytic site database", "Nothing here",
             "Function defined by mechanism. Coverage is a hundred-odd human proteins, so "
             "this is a VIGNETTE: it must not carry a headline number."),
}


def section_truth_provenance(out: Path, metrics: pl.DataFrame) -> None:
    if "truth_set" not in metrics.columns:
        return
    base = ungrouped(metrics)
    if "split" in base.columns:
        base = base.filter(pl.col("split") == "all")
    data = {}
    for ts in sorted(base["truth_set"].unique().to_list()):
        sub = base.filter(pl.col("truth_set") == ts)
        built, circular, note = TRUTH_PROVENANCE.get(ts, ("", "unknown", ""))
        data[ts] = {
            "built_by": built,
            "circular_with": circular,
            # Max over tools, not sum: every tool is scored against the same answer key,
            # so this is the size of that key, not a total over rows.
            "instances": (sub["n_truth_instances"].max()
                          if "n_truth_instances" in sub.columns else None),
            "reachable": (sub["n_reachable_instances"].max()
                          if "n_reachable_instances" in sub.columns else None),
            "note": note,
        }
    if not data:
        return
    write_section(out, "qfo_truth_provenance", {
        "id": "qfo_truth_provenance",
        "section_name": "Truth sets and circularity",
        "description": (
            "Instance counts are measured off this run — the largest answer key any tool "
            "was scored against, which is the size of the key itself rather than a total "
            "over rows. For reference, the full sets are 50,185 human Pfam domain "
            "instances, 142,857 human Swiss-Prot features, Pfam-N streamed from EBI, and "
            "106 human proteins in M-CSA. A run scoped to fewer species or a mini test set "
            "will show less than that.<br>"
            "No number in this report is ever averaged across truth sets. Pfam is circular "
            "with the profile baselines and Swiss-Prot is not, so a mean over the two has "
            "no interpretation."),
        "plot_type": "table",
        "pconfig": {"id": "qfo_truth_provenance_table", "title": "Truth set provenance",
                    "col1_header": "Truth set", "sort_rows": False, "scale": False},
        "headers": {
            "built_by": dict(title="Built by", scale=False),
            "circular_with": dict(title="Circular with", scale=False),
            "instances": dict(title="Instances scored", format="{:,.0f}", scale="Blues"),
            "reachable": dict(title="Max reachable", format="{:,.0f}", scale="Greens",
                              description="Best single target proteome in this run"),
            "note": dict(title="Reading", scale=False),
        },
        "data": data,
    })


CITATIONS = [
    ("Reseek — the structure-search baseline that scales the alphabet up where kmerseek "
     "scales it down to two letters",
     "Edgar, R. C. (2024). Reseek: structure search with a mega-alphabet. "
     "<i>Bioinformatics</i>, btae687.",
     "https://doi.org/10.1093/bioinformatics/btae687"),
    ("CAFA — Fmax, wFmax and Smin are adapted from the Critical Assessment of Functional "
     "Annotation challenge",
     "Friedberg, I., et al. Critical Assessment of Functional Annotation (CAFA), periodic "
     "challenges.",
     "https://www.biofunctionprediction.org/"),
    ("CASP — nDO and the domain-boundary metrics come from the structure-assessment "
     "lineage (CASP, and the Chainsaw / Merizo parsers that follow it)",
     "Critical Assessment of Protein Structure Prediction (CASP) challenge metrics.",
     "https://predictioncenter.org/"),
    ("Pfam — the primary truth set, and the annotation ceiling hmmscan is run against",
     "El-Gebali, S., et al. (2019). The Pfam protein families database in 2019. "
     "<i>Nucleic Acids Research</i>, 47(D1), D427-D432.",
     "https://doi.org/10.1093/nar/gky995"),
    ("Foldseek — the structure-structure baseline, and the 3Di alphabet ProstT5 predicts",
     "van Kempen, M., et al. (2024). Fast and accurate protein structure search with "
     "Foldseek. <i>Nature Biotechnology</i>, 42, 243-246.",
     "https://doi.org/10.1038/s41587-023-01773-0"),
    ("HMMER — phmmer, jackhmmer and hmmscan",
     "Eddy, S. R. (2011). Accelerated profile HMM searches. "
     "<i>PLoS Computational Biology</i>, 7(10), e1002195.",
     "https://doi.org/10.1371/journal.pcbi.1002195"),
    ("MMseqs2 — the fast sequence-search baseline",
     "Steinegger, M., &amp; Soding, J. (2017). MMseqs2 enables sensitive protein sequence "
     "searching for the analysis of massive data sets. <i>Nature Biotechnology</i>, 35, "
     "1026-1028.",
     "https://doi.org/10.1038/nbt.3988"),
    ("HH-suite — the profile-profile baseline",
     "Steinegger, M., et al. (2019). HH-suite3 for fast remote homology detection and deep "
     "protein annotation. <i>BMC Bioinformatics</i>, 20, 473.",
     "https://doi.org/10.1186/s12859-019-3019-7"),
]


def section_citations(out: Path) -> None:
    """Where the metrics and the baselines come from, next to the numbers they produced."""
    rows = "".join(
        f"<li style='margin-bottom:0.7em'><b>{what}.</b><br>{cite} "
        f"<a href='{url}' target='_blank' rel='noopener'>{url}</a></li>"
        for what, cite, url in CITATIONS
    )
    write_section(out, "qfo_citations", {
        "id": "qfo_citations",
        "section_name": "Methods and citations",
        "description": "What the metrics and the baselines in this report are taken from.",
        "plot_type": "html",
        "data": (
            "<ul style='margin-top:0.5em'>" + rows + "</ul>"
            "<p>Metric definitions as implemented here live in "
            "<code>bin/cafa_metrics.py</code> and <code>bin/evaluate_domain_calls.py</code>; "
            "two of them depart from the source deliberately and say so at the point of "
            "use. wFmax weights by plain information content rather than CAFA's "
            "ontology-based information accretion, because Pfam is flat. nDO is the "
            "residue-level overlap CASP's NDO score is built from, not CASP's full scoring "
            "matrix.</p>"
        ),
    })


def section_grayzone(out: Path, metrics: pl.DataFrame, primary_truth: str,
                     max_tools: int) -> None:
    """How much of each tool's output could be judged at all."""
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    need = {"n_tp_calls", "n_fp_calls", "n_gray_calls"}
    if cut.height == 0 or not need.issubset(set(cut.columns)):
        return
    board = best_variants(cut).head(max_tools)
    keep = {r["tool"]: r["best_variant"] for r in board.to_dicts()}
    data = {}
    for tool, variant in keep.items():
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        agg = sub.select([pl.col(c).sum() for c in ("n_tp_calls", "n_fp_calls",
                                                    "n_gray_calls")]).to_dicts()[0]
        data[tool if tool != "kmerseek" else f"kmerseek {variant}"] = agg
    write_section(out, "qfo_grayzone", {
        "id": "qfo_grayzone",
        "section_name": "Gray-zone accounting",
        "description": (
            f"Every call this run produced, split three ways ({primary_truth} truth, "
            f"<code>{split}</code> split, summed over species). Gray calls land in "
            "territory the annotation never covered: they are excluded from the precision "
            "denominator rather than charged as errors, because a region Pfam never "
            "annotated is not evidence the tool was wrong. The size of the gray slice is "
            "how much that convention is worth to each tool, so it is shown rather than "
            "folded away."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_grayzone_plot", "title": "Calls by outcome",
                    "ylab": "calls", "height": 450},
        "categories": {"n_tp_calls": {"name": "true positive", "color": "#0f9d76"},
                       "n_fp_calls": {"name": "false positive", "color": "#c9528f"},
                       "n_gray_calls": {"name": "gray (unscoreable)", "color": "#c8c8c8"}},
        "data": data,
    })


def section_truthsets(out: Path, metrics: pl.DataFrame, max_tools: int) -> None:
    """The circularity check, in one plot."""
    cut, split = pick_split(ungrouped(metrics))
    sets = sorted(cut["truth_set"].unique().to_list())
    if len(sets) < 2:
        return
    ranked = best_variants(cut).head(max_tools)["tool"].to_list()
    data = {}
    for tool in ranked:
        sub = cut.filter(pl.col("tool") == tool)
        row = {}
        for ts in sets:
            s = sub.filter(pl.col("truth_set") == ts)
            row[ts] = s["fmax"].max() if s.height else None
        data[tool] = row
    write_section(out, "qfo_truthsets", {
        "id": "qfo_truthsets",
        "section_name": "Truth sets side by side",
        "description": (
            f"Best Fmax per tool against each truth set (<code>{split}</code> split). "
            "Profile methods should score highest against Pfam, which is defined by the "
            "same HMMs they run — the gap between a tool's Pfam bar and its Swiss-Prot or "
            "Pfam-N bar is the size of that circularity, and a method that keeps its score "
            "across all three is the one making a claim about biology rather than about "
            "Pfam."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_truthsets_plot", "title": "Fmax by truth set",
                    "ylab": "Fmax", "cpswitch": False, "stacking": "group", "height": 450},
        "categories": {ts: {"name": ts} for ts in sets},
        "data": data,
    })


def section_reachability(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """The per-species ceiling: what could have been transferred at all."""
    cut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cut = cut.filter(pl.col("species") != "all")
    need = {"n_truth_instances", "n_reachable_instances"}
    if cut.height == 0 or not need.issubset(set(cut.columns)):
        return
    per = (cut.group_by("species", "species_mya")
              .agg(pl.col("n_truth_instances").max(),
                   pl.col("n_reachable_instances").max())
              .sort("species_mya"))
    data = {
        r["species"]: {
            "reachable": r["n_reachable_instances"],
            "unreachable": max(0, r["n_truth_instances"] - r["n_reachable_instances"]),
        } for r in per.to_dicts()
    }
    write_section(out, "qfo_reachability", {
        "id": "qfo_reachability",
        "section_name": "Recall ceiling per species",
        "description": (
            "Human domain instances whose family exists somewhere in the target proteome "
            "(reachable) against those whose family does not (unreachable). No search of "
            "any kind can transfer a family the target does not have, so every "
            "recall_reachable in this report divides by the reachable bar only. Species "
            "are ordered by divergence time."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_reachability_plot",
                    "title": "Transferable domain instances by target species",
                    "ylab": "human domain instances", "height": 400},
        "categories": {"reachable": {"name": "reachable", "color": "#0f9d76"},
                       "unreachable": {"name": "no family in target", "color": "#d9d9d9"}},
        "data": data,
    })


# ---------------------------------------------------------------------------
# sections: resources
# ---------------------------------------------------------------------------

def section_resources(out: Path, trace: pl.DataFrame, n_queries: int) -> None:
    if trace.height == 0:
        write_section(out, "qfo_resources_missing", {
            "id": "qfo_resources_missing",
            "section_name": "Resource usage",
            "description": "No Nextflow trace file was found for this run.",
            "plot_type": "html",
            "data": "<p>Resource plots need the trace file named in "
                    "<code>params.trace_file</code>. It is written incrementally by "
                    "Nextflow, so it exists from the first completed task onwards.</p>",
        })
        return

    done = trace.filter(pl.col("realtime_s").is_not_null())

    # --- run summary ---
    total_cpu_h = float(done["cpu_hours"].sum() or 0)
    peak = done["peak_rss_b"].max()
    summary = {
        "run": {
            "tasks": trace.height,
            "completed": int((trace["status"] == "COMPLETED").sum()),
            "cached": int((trace["status"] == "CACHED").sum()),
            "failed": int((trace["status"] == "FAILED").sum()),
            "retried": int((trace["attempt"] > 1).sum()) if "attempt" in trace.columns else 0,
            "cpu_hours": total_cpu_h,
            "wall_hours": float((done["realtime_s"].sum() or 0) / 3600),
            "peak_rss_gb": float(peak / 1024**3) if peak else None,
            "read_gb": float((done["read_b"].sum() or 0) / 1024**3),
            "write_gb": float((done["write_b"].sum() or 0) / 1024**3),
        }
    }
    write_section(out, "qfo_run_summary", {
        "id": "qfo_run_summary",
        "section_name": "Run totals",
        "description": (
            "Every task in the trace, including cached ones — a cached task keeps the "
            "resource figures from the execution that filled the cache, so a "
            "<code>-resume</code> run still reports honest totals for work it did not "
            "repeat. Wall hours is the sum of per-task run times, not elapsed clock time; "
            "the two differ by however much ran in parallel."),
        "plot_type": "table",
        "pconfig": {"id": "qfo_run_summary_table", "title": "Run totals",
                    "col1_header": "", "sort_rows": False, "scale": False},
        "headers": {
            "tasks": dict(title="Tasks", format="{:,.0f}", scale=False),
            "completed": dict(title="Completed", format="{:,.0f}", scale=False),
            "cached": dict(title="Cached", format="{:,.0f}", scale=False),
            "failed": dict(title="Failed", format="{:,.0f}", scale=False),
            "retried": dict(title="Retried", format="{:,.0f}", scale=False),
            "cpu_hours": dict(title="CPU-hours", format="{:,.1f}", scale="Reds"),
            "wall_hours": dict(title="Task-hours", format="{:,.1f}", scale="Blues"),
            "peak_rss_gb": dict(title="Peak RSS (GB)", format="{:,.1f}", scale="Purples"),
            "read_gb": dict(title="Read (GB)", format="{:,.1f}", scale="Greens"),
            "write_gb": dict(title="Written (GB)", format="{:,.1f}", scale="Greens"),
        },
        "data": summary,
    })

    # --- CPU-hours and wall time by process ---
    by_proc = (done.group_by("process")
                   .agg(pl.col("cpu_hours").sum().alias("cpu_hours"),
                        (pl.col("realtime_s").sum() / 3600).alias("task_hours"),
                        pl.len().alias("n_tasks"))
                   .sort("cpu_hours", descending=True))
    write_section(out, "qfo_res_cpu", {
        "id": "qfo_res_cpu",
        "section_name": "CPU time by process",
        "description": ("CPU-hours is run time times allotted cores, so a task that "
                        "requested 16 cores and used one still bills 16. The gap between "
                        "the two bars is exactly that waste."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_res_cpu_plot", "title": "CPU-hours and task-hours by process",
                    "ylab": "hours", "cpswitch": False, "stacking": "group", "height": 450},
        "categories": {"cpu_hours": {"name": "CPU-hours", "color": "#c9528f"},
                       "task_hours": {"name": "task-hours (wall)", "color": "#2b7bba"}},
        "data": {r["process"]: {"cpu_hours": r["cpu_hours"], "task_hours": r["task_hours"]}
                 for r in by_proc.to_dicts()},
    })

    # --- distribution of task run times, per process ---
    times = {}
    for proc in by_proc["process"].to_list():
        vals = (done.filter(pl.col("process") == proc)["realtime_s"] / 60).to_list()
        if vals:
            times[proc] = [round(v, 3) for v in vals]
    if times:
        write_section(out, "qfo_res_walltime", {
            "id": "qfo_res_walltime",
            "section_name": "Task run times",
            "description": ("Minutes per task. The spread within a process is what decides "
                            "the SLURM <code>time</code> request: sizing for the median "
                            "means the tail gets killed and requeued."),
            "plot_type": "box",
            "pconfig": {"id": "qfo_res_walltime_plot", "title": "Run time per task",
                        "xlab": "minutes", "height": 500},
            "data": times,
        })

    # --- memory: what was asked for against what was used ---
    mem = done.filter(pl.col("peak_rss_b").is_not_null()
                      & pl.col("requested_mem_b").is_not_null())
    if mem.height:
        points = {}
        for i, r in enumerate(mem.to_dicts()):
            label = f"{r['process']} [{r['tag']}]" if r["tag"] and r["tag"] != "-" else r["process"]
            points[f"{label} #{i}"] = {
                "x": r["requested_mem_b"] / 1024**3,
                "y": r["peak_rss_b"] / 1024**3,
                "color": tool_color(r["tool"]),
                "group": r["process"],
                "name": label,
                "annotation": "",
            }
        lo = float(mem["requested_mem_b"].min()) / 1024**3
        hi = float(mem["requested_mem_b"].max()) / 1024**3
        diagonal = [{"x": lo + (hi - lo) * i / 80, "y": lo + (hi - lo) * i / 80,
                     "color": "#cccccc", "marker_size": 3, "name": "peak = requested",
                     "group": "peak = requested", "annotation": ""}
                    for i in range(81)]
        write_section(out, "qfo_res_memory", {
            "id": "qfo_res_memory",
            "section_name": "Memory: requested against used",
            "description": (
                "Peak RSS against the memory the task asked SLURM for, one point per task. "
                "Points far below the diagonal are queue time paid for nothing; points at "
                "or above it are the shape that gets OOM-killed on the next combo. The "
                "HP-family alphabets at low k are sized separately in the pipeline "
                "(<code>params.kmerseek_memory_hp_lowk</code>) for exactly this reason, and "
                "this plot is how that rule gets checked against reality."),
            "plot_type": "scatter",
            "pconfig": {"id": "qfo_res_memory_plot",
                        "title": "Peak RSS vs requested memory",
                        "xlab": "requested (GB)", "ylab": "peak RSS (GB)", "height": 520,
                        "marker_size": 7,
                        # Wrapped in a list on purpose: MultiQC 1.35 mis-handles the bare
                        # list-of-points form of extra_series and appends the list itself
                        # as a single point, which fails pydantic validation. The
                        # list-of-lists form takes the branch that works.
                        "extra_series": [diagonal]},
            "data": points,
        })

    # --- kmerseek memory against ksize: the sizing rule, measured ---
    ks = done.filter(pl.col("process") == "kmerseekIndexAndSearch")
    if ks.height:
        ks = ks.with_columns(
            pl.col("tag").str.extract(r"_k(\d+)_lc", 1).cast(pl.Int64).alias("ksize"),
            pl.col("tag").str.extract(r"^[^_]+_(.+)_k\d+_lc", 1).alias("alphabet"),
        ).filter(pl.col("ksize").is_not_null())
        if ks.height:
            alphas = sorted(set(ks["alphabet"].drop_nulls().to_list()))
            color_of = dict(zip(alphas, cycle(SERIES_COLORS)))
            points = {}
            for i, r in enumerate(ks.to_dicts()):
                alpha = r["alphabet"] or "unknown"
                points[f"{r['tag']} #{i}"] = {
                    "x": r["ksize"],
                    "y": (r["peak_rss_b"] or 0) / 1024**3,
                    "color": color_of.get(alpha, "#888888"),
                    "group": alpha,
                    "name": r["tag"],
                    "annotation": "",
                }
            write_section(out, "qfo_res_kmerseek_memory", {
                "id": "qfo_res_kmerseek_memory",
                "section_name": "kmerseek memory against k",
                "description": (
                    "Peak RSS of each index-and-search task against its k-mer size, grouped "
                    "by alphabet. The inverted index scales with the most degenerate k-mer's "
                    "occurrence count, so a 2-letter alphabet at low k is the expensive "
                    "corner. Anything flat here means the per-combo memory rule is "
                    "over-provisioning."),
                "plot_type": "scatter",
                "pconfig": {"id": "qfo_res_kmerseek_memory_plot",
                            "title": "kmerseek peak RSS by k and alphabet",
                            "xlab": "k", "ylab": "peak RSS (GB)", "height": 500,
                            "marker_size": 7},
                "data": points,
            })

    # --- CPU efficiency ---
    eff = done.filter(pl.col("pct_cpu").is_not_null() & pl.col("cpus").is_not_null())
    if eff.height:
        data = {}
        for proc in by_proc["process"].to_list():
            sub = eff.filter(pl.col("process") == proc)
            if sub.height == 0:
                continue
            data[proc] = {
                "cpu_efficiency": float((sub["pct_cpu"] / (100 * sub["cpus"])).mean() * 100),
                "mem_efficiency": (float(sub["mem_used_frac"].mean() * 100)
                                   if sub["mem_used_frac"].null_count() < sub.height else None),
            }
        write_section(out, "qfo_res_efficiency", {
            "id": "qfo_res_efficiency",
            "section_name": "Resource efficiency",
            "description": (
                "Mean fraction of the requested CPU and memory a process actually used. "
                "Both are worth watching for opposite reasons: low CPU efficiency means "
                "cores are reserved and idle, and high memory efficiency means the next "
                "slightly larger input gets OOM-killed. Note that on macOS without "
                "containers Nextflow records neither, so a local run shows this section "
                "empty; the cluster run is the one with real numbers."),
            "plot_type": "bargraph",
            "pconfig": {"id": "qfo_res_efficiency_plot", "title": "Requested resources used",
                        "ylab": "% of request used", "cpswitch": False, "stacking": "group",
                        "ymax": 130, "height": 450},
            "categories": {"cpu_efficiency": {"name": "CPU", "color": "#2b7bba"},
                           "mem_efficiency": {"name": "memory", "color": "#c9528f"}},
            "data": data,
        })

    # --- I/O ---
    io = done.filter(pl.col("read_b").is_not_null() | pl.col("write_b").is_not_null())
    if io.height:
        agg = (io.group_by("process")
                 .agg((pl.col("read_b").sum() / 1024**3).alias("read_gb"),
                      (pl.col("write_b").sum() / 1024**3).alias("write_gb"))
                 .sort("write_gb", descending=True))
        write_section(out, "qfo_res_io", {
            "id": "qfo_res_io",
            "section_name": "Disk I/O",
            "description": (
                "Bytes read and written per process. The search arms dominate: HP "
                "alphabets at low k produce enormous match volume by design, since the "
                "p-value filter is left lenient so Bonferroni correction can happen "
                "downstream. This plot is where that cost shows up."),
            "plot_type": "bargraph",
            "pconfig": {"id": "qfo_res_io_plot", "title": "Disk I/O by process",
                        "ylab": "GB", "cpswitch": False, "stacking": "group", "height": 450},
            "categories": {"read_gb": {"name": "read", "color": "#2b7bba"},
                           "write_gb": {"name": "written", "color": "#c9528f"}},
            "data": {r["process"]: {"read_gb": r["read_gb"], "write_gb": r["write_gb"]}
                     for r in agg.to_dicts()},
        })

    # --- task outcomes ---
    status = (trace.group_by("process", "status").agg(pl.len().alias("n"))
                   .pivot(on="status", index="process", values="n").fill_null(0))
    cats = [c for c in status.columns if c != "process"]
    palette = {"COMPLETED": "#0f9d76", "CACHED": "#9ecae1", "FAILED": "#c9528f",
               "ABORTED": "#bdbdbd", "RUNNING": "#c99a00", "SUBMITTED": "#dddddd"}
    write_section(out, "qfo_res_status", {
        "id": "qfo_res_status",
        "section_name": "Task outcomes",
        "description": (
            "Tasks per process by final status. A FAILED kmerseek task matters more than "
            "it looks: the process uses <code>errorStrategy 'finish'</code> rather than "
            "<code>'ignore'</code>, because a combo that dies and gets skipped leaves an "
            "empty result that reads downstream as \"this alphabet found nothing\", which "
            "is indistinguishable from a real negative."),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_res_status_plot", "title": "Task status by process",
                    "ylab": "tasks", "height": 450},
        "categories": {c: {"name": c.title(), "color": palette.get(c, "#888888")}
                       for c in cats},
        "data": {r["process"]: {c: r[c] for c in cats} for r in status.to_dicts()},
    })

    # --- throughput per search task ---
    searches = done.filter(pl.col("is_search") & (pl.col("realtime_s") > 0))
    if searches.height:
        points = {}
        for i, r in enumerate(searches.to_dicts()):
            points[f"{r['process']} {r['tag']} #{i}"] = {
                "x": n_queries / r["realtime_s"],
                "y": (r["cpu_hours"] or 0),
                "color": tool_color(r["tool"]),
                "group": CLASSES[tool_class(r["tool"])][0],
                "name": f"{r['tool']} {r['tag']}",
                "annotation": "",
            }
        write_section(out, "qfo_res_throughput", {
            "id": "qfo_res_throughput",
            "section_name": "Throughput per search",
            "description": (
                f"One point per search task: query proteins per second of wall time "
                f"against the CPU-hours that task billed, at {n_queries:,} human queries. "
                "Indexing is inside the measurement for every arm, because each task "
                "builds what it needs and searches once. A tool that would amortise an "
                "index over many searches is undersold here, and that is the honest "
                "reading of a benchmark that searches each target proteome once."),
            "plot_type": "scatter",
            "pconfig": {"id": "qfo_res_throughput_plot",
                        "title": "Search throughput against cost",
                        "xlab": "query proteins / s", "ylab": "CPU-hours",
                        "xlog": True, "ylog": True, "height": 520, "marker_size": 8},
            "data": points,
        })


# ---------------------------------------------------------------------------
# general statistics
# ---------------------------------------------------------------------------

def section_general_stats(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                          n_queries: int, primary_truth: str) -> None:
    cut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    board = best_variants(cut)
    if board.height == 0:
        return
    thr = throughput_per_tool(trace, n_queries)
    data = {}
    for row in board.to_dicts():
        t = row["tool"]
        tr = thr.filter(pl.col("tool") == t)
        data[t] = {
            "fmax": row.get("fmax"),
            "auprc": row.get("auprc"),
            "recall_reachable": row.get("recall_reachable"),
            "precision": row.get("precision"),
            "queries_per_s": tr["queries_per_s"].first() if tr.height else None,
            "cpu_hours": tr["cpu_hours"].first() if tr.height else None,
        }
    write_section(out, "qfo_general_stats", {
        "id": "qfo_general_stats",
        "plot_type": "generalstats",
        "pconfig": [
            {"fmax": dict(title="Fmax", min=0, max=1, scale="RdYlGn", format="{:,.3f}",
                          description=f"Best variant, {primary_truth} truth")},
            {"auprc": dict(title="AUPRC", min=0, max=1, scale="Blues", format="{:,.3f}")},
            {"recall_reachable": dict(title="Recall", min=0, max=1, scale="Greens",
                                      format="{:,.3f}",
                                      description="Against transferable instances only")},
            {"precision": dict(title="Prec.", min=0, max=1, scale="Oranges",
                               format="{:,.3f}")},
            {"queries_per_s": dict(title="Q/s", scale="Purples", format="{:,.1f}",
                                   description="Median over target species")},
            {"cpu_hours": dict(title="CPU-h", scale="Reds", format="{:,.2f}")},
        ],
        "data": data,
    })


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--curves", type=Path)
    p.add_argument("--trace", type=Path,
                   help="Nextflow trace file. Missing or unparsable is not an error; the "
                        "resource sections say so instead.")
    p.add_argument("--n-queries", type=int, required=True,
                   help="Human query proteins searched, for throughput")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--primary-truth", default=None,
                   help="Truth set the frontier and curve sections use. Defaults to "
                        "swissprot when present, since Pfam is circular with the profile "
                        "baselines, else the first set found.")
    p.add_argument("--max-tools", type=int, default=14,
                   help="Tools per grouped plot, ranked by Fmax")
    p.add_argument("--max-lines", type=int, default=12,
                   help="Curves per PR/ROC plot")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics = pl.read_parquet(args.metrics)
    curves = (pl.read_parquet(args.curves)
              if args.curves and args.curves.exists() else pl.DataFrame())
    trace = mt.load_trace(args.trace) if args.trace else mt.load_trace(None)

    sets = metrics["truth_set"].unique().to_list() if "truth_set" in metrics.columns else []
    primary = args.primary_truth or ("swissprot" if "swissprot" in sets
                                     else (sets[0] if sets else "pfam"))

    section_frontier(args.outdir, metrics, trace, args.n_queries, primary)
    section_leaderboards(args.outdir, metrics)
    section_cafa(args.outdir, metrics, primary, args.max_tools)
    section_threshold_metrics(args.outdir, metrics, primary, args.max_tools)
    section_truth_provenance(args.outdir, metrics)
    section_curves(args.outdir, curves, metrics, primary, args.max_lines)
    section_identity(args.outdir, metrics, primary, args.max_tools)
    section_divergence(args.outdir, metrics, primary, args.max_tools)
    section_truthsets(args.outdir, metrics, args.max_tools)
    section_alphabet_matrix(args.outdir, metrics, primary)
    section_boundary(args.outdir, metrics, primary, args.max_tools)
    section_grayzone(args.outdir, metrics, primary, args.max_tools)
    section_reachability(args.outdir, metrics, primary)
    section_resources(args.outdir, trace, args.n_queries)
    section_general_stats(args.outdir, metrics, trace, args.n_queries, primary)
    section_citations(args.outdir)

    written = sorted(f.name for f in args.outdir.glob("*_mqc.json"))
    print(f"primary truth set: {primary}")
    print(f"{metrics.height} metric rows, {curves.height} curve points, "
          f"{trace.height} trace records")
    print(f"wrote {len(written)} sections to {args.outdir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
