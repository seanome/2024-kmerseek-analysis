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
import gzip
import io
import json
import math
import re
from itertools import cycle
from pathlib import Path

import numpy as np
import polars as pl

import mqc_trace as mt

# Threshold-free, so a tool that ships a lenient default cutoff is not rewarded for it.
# fmax and family_fmax are the same CAFA machinery read at two levels -- interval placement
# and (protein, family) set membership -- and both are headline because either alone is
# ambiguous about which half of the task a tool failed.
#
# `ndo` is deliberately not here. The column the pipeline writes under that name is a
# literal alias of residue_recall -- identical to the last bit on every row of this run --
# so a report drawing both was drawing one measurement twice and calling the second one a
# normalized domain overlap. Every figure and table in this file drops it; the metrics
# still carry it, and it comes back here when it is computing something of its own.
HEADLINE = ["fmax", "family_fmax", "auprc", "roc_auc", "recall_reachable", "precision",
            "coverage", "smin", "sens_first_fp_mean"]

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
# Colour marks the method class, and every tool in a class shares it on purpose -- the
# class identity is the argument the report makes. The cost is that a static export cannot
# tell foldseek from reseek, both pink. Marker shape splits them: one shape per tool, fixed
# here so a tool keeps the same shape wherever it is plotted. Plotly symbol names.
TOOL_SYMBOL = {
    "kmerseek": "circle",
    "foldseek": "diamond", "reseek": "square", "folddisco": "triangle-up",
    "prostt5": "diamond",
    "hmmer3_phmmer": "circle", "hmmer3_jackhmmer": "square",
    "mmseqs2": "triangle-up",
    "mmseqs2_seqseq": "triangle-up", "mmseqs2_iterative": "triangle-down",
    "hhblits": "x",
    "hmmscan": "star",
}

# Identity bins below the twilight-zone boundary. The central claim is stated here.
GRAY_ZONE = ["0-20%", "20-30%", "30-40%"]

# --- what the Swiss-Prot truth set measures about a proteome --------------------------
#
# The Swiss-Prot answer key is built from manually curated entries, and curation depth is
# not distributed the way protein families are. Ciona intestinalis has 23 curated proteins
# against fly's 3_174, E. coli's 2_999 and mouse's 15_634, so on that truth set a tunicate
# ranks below a bacterium. That ordering is a property of who has been curated, not of what
# is shared with human.
#
# Its Pfam annotation is fine: 20_234 domain rows over 10_658 proteins in 5_542 families,
# and on the Pfam truth set it reaches 82.9% of the answer key, above fly and chicken, with
# E. coli last at 10.8% -- exactly where biology puts them. So Ciona is NOT an annotation
# failure and nothing here should imply its data is broken.
#
# What it needs is a footnote on the Swiss-Prot panels, saying which quantity that arm is
# ranking on and where to read the biological one. Panels on any other truth set need
# nothing.
CURATION_CAVEAT_SPECIES = "ciona"
CURATION_CAVEAT_MARK = "*"
# The truth sets built from manually curated entries, where a species' rank tracks how much
# of it has been curated.
CURATED_TRUTH_SETS = {"swissprot"}


def curation_caveat_applies(species: str, truth_set: str | None) -> bool:
    return species == CURATION_CAVEAT_SPECIES and truth_set in CURATED_TRUTH_SETS


def species_tick(species: str, truth_set: str | None = None, **extra) -> str:
    """A species' tick label: its name, a footnote mark where one is warranted, then the n.

    The mark is a pointer to the caveat below, and it is only drawn where that caveat
    applies -- a Swiss-Prot panel. On a Pfam panel Ciona is an ordinary proteome and
    marking it would say something about the data that is not true.
    """
    mark = CURATION_CAVEAT_MARK if curation_caveat_applies(species, truth_set) else ""
    parts = [species + mark]
    for key, value in extra.items():
        if value is None:
            continue
        shown = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
        parts.append(f"{key} {shown}")
    return " · ".join(parts)


def curation_caveat_note(truth_set: str | None,
                         reach: dict[str, float] | None = None) -> str:
    """The footnote a Swiss-Prot panel carrying ciona prints. Empty on every other panel."""
    if truth_set not in CURATED_TRUTH_SETS:
        return ""
    if reach is not None and CURATION_CAVEAT_SPECIES not in reach:
        return ""
    measured = ""
    if reach and CURATION_CAVEAT_SPECIES in reach:
        measured = (f" Here it reads "
                    f"{100 * reach[CURATION_CAVEAT_SPECIES]:.1f}%.")
    return (f"<b><code>{CURATION_CAVEAT_SPECIES}</code> is marked "
            f"{CURATION_CAVEAT_MARK} because this truth set understates it.</b> Swiss-Prot "
            f"is manually curated and Ciona has 23 curated proteins, against 3,174 for "
            f"fly, 2,999 for E. coli and 15,634 for mouse, so on this answer key a species "
            f"ranks partly by how much of it has been curated." + measured
            + " Its Pfam annotation is complete — 20,234 domain rows over 10,658 proteins "
              "in 5,542 families — and on the Pfam truth set it sits above fly and "
              "chicken. Read the Pfam panel for the biological ordering.")

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

# MultiQC's default heatmap ramp is RdYlBu reversed: a DIVERGING scale, blue at the low end
# through pale yellow in the middle to red at the top. It is the right ramp for a quantity
# with a meaningful midpoint and the wrong one for everything else here. Fmax, best F1 and
# coverage have no midpoint -- 0.5 is not a neutral value, it is just a number between 0
# and 1 -- so the pale-yellow band in the middle reads as "nothing here" over a range where
# plenty is happening. These two are stated explicitly rather than left to the default.
SEQUENTIAL_COLSTOPS = [
    [0.0, "#f7fbff"], [0.125, "#deebf7"], [0.25, "#c6dbef"], [0.375, "#9ecae1"],
    [0.5, "#6baed6"], [0.625, "#4292c6"], [0.75, "#2171b5"], [0.875, "#08519c"],
    [1.0, "#08306b"],
]
# Kept for the one quantity that genuinely diverges: family Fmax minus Fmax, centred at
# zero, where the sign is the reading and the midpoint is a real neutral value.
DIVERGING_COLSTOPS = [
    [0.0, "#2166ac"], [0.25, "#92c5de"], [0.5, "#f7f7f7"], [0.75, "#f4a582"],
    [1.0, "#b2182b"],
]


def heat_max(*grids: list[list]) -> float | None:
    """Largest value across every cell of every grid passed, ignoring blanks.

    Heatmaps here were pinned to 0..1 because the quantities are bounded there. Bounded is
    not the same as occupied: an Fmax grid whose largest cell is 0.28 drawn on 0..1 spends
    72% of its colour range on values that do not exist, and every cell that does exist
    lands in the first two stops of the ramp. Callers that want two panels comparable pass
    both grids in one call and get one number back.
    """
    vals = [v for grid in grids for row in grid for v in row
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(vals) if vals else None


def heat_range_note(vmax: float | None, shared_with: str = "") -> str:
    """The bullet that says what the colour range is, so nobody cross-reads two panels."""
    if vmax is None:
        return ""
    also = f" It is shared with {shared_with}, which is why the two can be read against "\
           "each other." if shared_with else ""
    return (f"<b>Colour runs 0 to {vmax:.2f}</b>, the largest value on this grid rather "
            f"than the 0-to-1 the metric is bounded by, because a grid that tops out at "
            f"{vmax:.2f} drawn on 0-to-1 has no readable ordering.{also} Do not compare "
            f"colours with a panel on a different range; read the numbers.")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

# --- how MultiQC 1.35 labels a scatter point, which is not obvious from the outside ----
#
# Two mechanisms, and the resource scatters were accidentally disabling both.
#
# `annotation` is the text drawn beside the marker. MultiQC also has an automatic pass
# that labels the ~10 biggest outliers, but it runs only `if n_annotated < 10`, where
# n_annotated counts points that already have an "annotation" KEY -- present but empty
# counts. Every resource scatter here set `"annotation": ""` on every point, which both
# suppressed the automatic pass and drew nothing, so those plots rendered as bare dots.
# The fix is to omit the key entirely when there is nothing to draw.
#
# The legend is built from `(color, marker_size, marker_line_width, group)`: one entry per
# unique combination, labelled "{group}: {the names in it, joined}" and cropped at 60
# characters. So `group` is what a reader sees the colour standing for, and `name` has to
# identify the individual point on its own, because it is also the hover label and the
# text the automatic annotation pass draws.
#
# That legend is off by default and cannot be turned on from inside the scatter module.
# MultiQC builds the base layout with `showlegend = True if flat else False`, and
# scatter.create_figure()'s `layout.showlegend = True` mutates the layout object AFTER
# `go.Figure(layout=layout)` has already copied it, so it never reaches the figure. Setting
# it in the pconfig is what actually works, and every scatter here does.
ANNOTATE_EVERY_POINT_BELOW = 20

# Durations a reader already has a feel for, in seconds. The device is borrowed from
# metapredict's Figure 1B (Emenecker, Griffith & Holehouse 2021), which puts dashed lines
# at named durations across a log axis so a reader can place a tool in the minutes band or
# the days band without doing arithmetic on the axis. metapredict plots time itself, with
# fast at the bottom; this report keeps throughput on y so that up and to the right stays
# better, which is the direction every reader arrives with. Only the labelled lines cross
# over. "1 month" is 30 days, the one entry here that needs a convention.
DURATION_BANDS = [(60.0, "1 minute"), (3600.0, "1 hour"), (86400.0, "1 day"),
                  (604800.0, "1 week"), (2_592_000.0, "1 month")]


def throughput_reference_lines(rates: list[float],
                               n_queries: int) -> tuple[list[dict], float, float]:
    """Duration reference lines for a log throughput axis, and the range they sit in.

    A rate of R queries per second finishes the run's N queries in N/R seconds, so the
    band for a duration goes at N/duration on the rate axis. The line positions therefore
    depend on N: a midi run and a full run put "in 1 hour" at different heights. That is
    correct rather than a bug, and the section says so, because a reader comparing two
    reports would otherwise read a moved line as a moved tool.

    The range is computed here rather than left to Plotly for two reasons already paid for
    in this file. y_lines are layout shapes and autorange ignores shapes, so nothing widens
    the axis to fit a band. And the scatter DROPS points outside ymin/ymax instead of
    clipping the axis, so the range is derived from the data and only ever widened past it.
    Bands outside that padded range are left out: a "1 week" line three decades under every
    point is clutter, not a reference.
    """
    lo, hi = min(rates), max(rates)
    ymin, ymax = lo / 3, hi * 3
    # Snapped out to whole decades. Plotly labels a log axis with MINOR ticks whenever the
    # range covers less than about a decade and a half, and its minor ticks are the log10
    # series 2,3,...,9 -- which on this figure produced a y axis reading
    # "2, 100, 5, 10, 5, 2, 1", unreadable and easy to mistake for data. A range that ends
    # on powers of ten gets major ticks only, one per decade.
    ymin = 10.0 ** math.floor(math.log10(ymin)) if ymin > 0 else ymin
    ymax = 10.0 ** math.ceil(math.log10(ymax)) if ymax > 0 else ymax
    lines = []
    for seconds, name in DURATION_BANDS:
        value = n_queries / seconds
        if ymin < value < ymax:
            lines.append({"value": value, "color": "#aaaaaa", "dash": "dash", "width": 1,
                          "label": f"whole query set in {name}"})
    return lines, ymin, ymax


def scatter_point(x, y, *, name, group, color, annotation=None, **extra):
    """One scatter point, labelled so a reader can tell which tool and variant it is."""
    point = {"x": x, "y": y, "name": name, "group": group, "color": color}
    if annotation:
        point["annotation"] = annotation
    point.update(extra)
    return point


def suppress_auto_labels(points: dict) -> dict:
    """Stop MultiQC labelling ten arbitrary outliers on a scatter that wants no labels.

    See the note above ANNOTATE_EVERY_POINT_BELOW: the automatic pass runs only while fewer
    than ten points carry an "annotation" KEY, and a key present with an empty value counts
    against that limit. So a panel whose points are identified by their x category, or one
    that labels a chosen few on purpose, gets an empty annotation on every other point.
    Without it the pass picks the ten largest values and prints their full hover names,
    which on a categorical scatter ran off the left edge of the plot.
    """
    for point in points.values():
        point.setdefault("annotation", "")
    return points


def clean(value):
    """JSON has no NaN or Infinity. Plotly reads null as a gap, which is what these are."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def bullets(*items: str) -> str:
    """A section description's points as a list rather than a wall of paragraphs.

    Every section here has to explain what its columns mean, and prose made a reader hunt
    through four sentences for the one about the column they were looking at. One point
    per <li>, and a point about a specific column opens with that column's name in <b>.
    Empty strings are dropped so a caller can pass a fragment that a run may not have.
    """
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items if i) + "</ul>"


# --- one primary truth set, the rest supplementary -----------------------------------
#
# Nearly every section here was built once per truth set, so the report carried three
# parallel copies of the leaderboard, the per-species winners, the encoding panels, the
# retention panels and the cardinality panels. Three copies of a figure is not three
# results: the truth sets are not independent measurements of the same thing, and a reader
# scrolling past the same plot three times reads the third one as new evidence.
#
# Swiss-Prot leads because it is the one truth set that is NOT circular with the profile
# baselines -- Pfam is built from the same HMMs hmmscan and phmmer run. Pfam and Pfam-N
# keep every panel they had; they move into a supplementary group so the main report is one
# pass. The one place all three stay side by side in the main report is the ratio
# comparison, which is about the difference between them and needs them together.
SUPP_PARENT_ID = "qfo_region_supp"
SUPP_PARENT_NAME = "Supplementary — QfO Pfam region benchmark"
SUPP_PARENT_DESC = (
    "The same panels as the main report, built on the truth sets it does not lead on. "
    "Pfam is circular with the profile baselines, which is why the main report leads on "
    "Swiss-Prot; Pfam-N is neither the circular set nor the curated one. Read these to "
    "check that a result survives the choice of answer key, not as separate results."
)

# Set from the CLI in main(), the same way CANONICAL is. Sections that build one panel per
# truth set read it to decide which copy is the main one.
PRIMARY_TRUTH = "swissprot"


def is_primary_truth(truth_set: str) -> bool:
    return truth_set == PRIMARY_TRUTH


def write_section(outdir: Path, section_id: str, cfg: dict, *,
                  supplementary: bool = False) -> None:
    if supplementary:
        cfg["parent_id"] = SUPP_PARENT_ID
        cfg["parent_name"] = SUPP_PARENT_NAME
        cfg["parent_description"] = SUPP_PARENT_DESC
        cfg["section_name"] = "Supp: " + cfg.get("section_name", section_id)
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


def split_per_truth_set(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    """Each truth set's own best split, kept together, with what was picked for each.

    Truth sets do not all have the same splits. Only Pfam is swept over alphabet x ksize
    and so only Pfam has a selection/heldout partition; Swiss-Prot and Pfam-N are scored
    whole. Running pick_split over the pooled frame therefore found Pfam's heldout rows,
    kept them, and dropped every other truth set from the section entirely, which took the
    two non-circular truth sets out of the leaderboard and made the circularity plot
    disappear because it was left with one set to compare.
    """
    picked = {}
    frames = []
    for ts in sorted(df["truth_set"].unique().to_list()):
        sub, split = pick_split(df.filter(pl.col("truth_set") == ts))
        if sub.height:
            frames.append(sub)
            picked[ts] = split
    return (pl.concat(frames) if frames else df.head(0)), picked


def ungrouped(df: pl.DataFrame) -> pl.DataFrame:
    if "stratum_axis" not in df.columns:
        return df
    return df.filter(pl.col("stratum_axis") == "all")


# How many of the sweep's alphabet x ksize x low-complexity combos to carry into the
# comparison plots. One would hide the shape of the sweep -- whether the winner is a lone
# spike or the top of a plateau of near-identical combos is the interesting part, and it
# changes what the result means. Every baseline still contributes its single variant, so
# the sweep cannot bury them.
TOP_KMERSEEK = 5


# --- pinning one arm across the whole report ----------------------------------------
#
# Which kmerseek is shown depends on the section, and the report never says so. Every
# per-tool section runs best_variants over its own rows, so the arms it draws are whatever
# topped THAT selection: the primary truth set's heldout half for the covariate and
# divergence sections, a gray-zone-weighted Fmax for the frontier and the curves, each
# truth set's own half for its leaderboard, and a rank taken across all three sets at once
# for the side-by-side plot. Those are four different row sets, and on the midi run they
# ranged from kmerseek behind every baseline to kmerseek ahead of every baseline. Nothing
# in the report is wrong; what is missing is that the arm changed under the reader.
#
# The fix here is a mechanism, not a choice. --canonical-variant pins one (tool, variant),
# forces it into every board best_variants builds, marks its row key wherever it is drawn,
# and names it in a section of its own. Off by default, so no existing number moves unless
# somebody asks for it, and no arm is hard-coded as the one this project ships.
CANONICAL_MARK = " ★"
CANONICAL: tuple[str, str] | None = None


def parse_canonical(spec: str | None) -> tuple[str, str] | None:
    """`tool:variant`, or a bare variant, which means kmerseek -- the only swept tool."""
    if not spec:
        return None
    tool, _, variant = spec.rpartition(":")
    return (tool or "kmerseek", variant)


def is_canonical(tool: str, variant: str) -> bool:
    return CANONICAL is not None and CANONICAL == (tool, variant)


def label_of(tool: str, variant: str) -> str:
    """Row key for the comparison tables. Only kmerseek has more than one variant here."""
    base = f"kmerseek {variant}" if tool == "kmerseek" else tool
    return base + CANONICAL_MARK if is_canonical(tool, variant) else base


def short_label(tool: str, variant: str) -> str:
    """Label drawn next to a point on the frontier, where five combos share the space.

    The full row key is kept as the point's name, so hovering and the exported data still
    carry it; this is only what is printed on the plot. Dropping the "kmerseek " prefix and
    spelling the low-complexity arm as a sign saves about fifteen characters per label,
    which is the difference between readable and overlapping.
    """
    if tool != "kmerseek":
        return tool
    return (variant.replace("_k", " k", 1)
                   .replace("_lcTrue", " lc+")
                   .replace("_lcFalse", " lc-"))


# The second headline metric, and the one the remote-homology baselines this benchmark
# cites actually report. Fmax is threshold-optimised but precision-dominated: phmmer on
# mouse Swiss-Prot finds 1_745 true calls against 94_992 false ones, so Fmax is pinned near
# 0.1 for every tool and small precision differences decide the ranking. Sensitivity to the
# first false positive is threshold-free and per-query, so it measures ranking quality
# instead. The two disagree hard enough here to reorder the board, which is why the
# leaderboard selects on both and shows both rather than picking one silently.
RANKING_METRICS = ["fmax", "sens_first_fp_mean"]

# --- aggregation vs total -------------------------------------------------------------
#
# Every row of all_domain_metrics.parquet is one (tool, variant, TARGET SPECIES, truth set,
# split, stratum). So every number in a leaderboard has been collapsed over the species
# axis, and there are two ways to do that which mean different things:
#
#   AGGREGATION  a per-species rate -- Fmax, precision, ROC AUC -- has no meaningful sum.
#                Nine proteomes give nine values and the summary is mean / median / min /
#                max / SD over them. Suffix `__mean`, `__median`, `__min`, `__max`, `__sd`.
#
#   TOTAL        a per-species count -- calls made, instances found -- does sum, and the
#                sum is a property of the ARM's whole run. Suffix `__total`.
#
# The distinction is not pedantry. `n_truth_instances__total` is roughly nine times the
# size of the human answer key, because the same human instance is scored once per target
# proteome; reading it as "the number of domains in the benchmark" is wrong by 9x. The
# suffix is there so that misreading has to survive a column name that contradicts it.
#
# Species are weighted equally in every aggregation. That is deliberate and it is the
# reason the spread columns exist: mouse carries 17_228 reviewed Swiss-Prot entries and
# Ciona 28, so a size-weighted mean would be a mean over mouse.
SPECIES_AGGREGATIONS = ["mean", "median", "min", "max", "sd"]

# Per-species counts. These sum to an arm-level total; they are also aggregated, because
# "median calls per proteome" answers a different question from "calls in total".
COUNT_COLS = [
    "n_calls", "n_tp_calls", "n_fp_calls", "n_gray_calls", "n_tp_strict",
    "n_truth_instances", "n_reachable_instances", "n_instances_found",
    "n_proteins_scored", "n_proteins_ranked",
]


def species_aggregations(cols: list[str], available: set[str]) -> list[pl.Expr]:
    """mean / median / min / max / SD across target species, for each column present.

    SD is null on a single species rather than 0. polars returns null for std() over one
    element, which is the honest answer -- a run against one proteome has no spread to
    report, and a 0 would read as perfect consistency.
    """
    how = {"mean": lambda c: c.mean(), "median": lambda c: c.median(),
           "min": lambda c: c.min(), "max": lambda c: c.max(), "sd": lambda c: c.std()}
    return [how[a](pl.col(c)).alias(f"{c}__{a}")
            for c in cols if c in available for a in SPECIES_AGGREGATIONS]


def species_totals(available: set[str]) -> list[pl.Expr]:
    """Sums over target species, named so they cannot be mistaken for per-species values."""
    return [pl.col(c).sum().alias(f"{c}__total")
            for c in COUNT_COLS if c in available]



# Below this a call-level ROC AUC is worse than a coin flip. Counted per species rather
# than judged off the mean, because the mean hides how many species it happened in.
CHANCE = 0.5


def _guard_single_truth_set(df: pl.DataFrame, across_truth_sets: bool) -> None:
    """The "never pool truth sets" convention, enforced instead of only documented.

    Pfam is circular with the profile baselines and Swiss-Prot is not, so a mean over the
    two has no interpretation. The report's Truth sets section asserts that no number is
    ever averaged across them; this is the check that makes the assertion true. Any section
    that hands mixed rows to the aggregator fails loudly at build time rather than emitting
    a plausible-looking number nobody can trace.

    Opt out only where crossing sets is the point and nothing displayed is a cross-set
    mean, which is the side-by-side circularity plot and nothing else.
    """
    # The same failure shape as truth sets, on a different axis: every arm is scored twice,
    # once with redundant transfers collapsed and once not, so a frame carrying both has two
    # rows per arm and every mean over it is halfway between two different measurements.
    # Not opt-out-able -- the comparison section filters to one mode before it aggregates.
    if "dedup_transfers" in df.columns and df.height:
        modes = df["dedup_transfers"].unique().to_list()
        if len(modes) > 1:
            raise ValueError(
                "best_variants got both dedup_transfers settings in one frame. Those are "
                "two measurements of the same arm -- redundant transfers collapsed or not "
                "-- and a mean over the pair is not either of them. Filter to one."
            )
    if across_truth_sets or "truth_set" not in df.columns:
        return
    found = df["truth_set"].unique().to_list()
    if len(found) > 1:
        raise ValueError(
            f"best_variants got {len(found)} truth sets ({', '.join(sorted(found))}) in "
            "one frame. Averaging across truth sets has no interpretation: Pfam is "
            "circular with the profile baselines and Swiss-Prot is not. Filter to one "
            "truth set, or pass across_truth_sets=True if only the row selection crosses "
            "sets and no displayed number does."
        )


def label_column() -> pl.Expr:
    """label_of() as an expression, so any frame can carry the row key.

    Factored out rather than written twice. section_hgnc read a `label` column that only
    best_variants ever created, so a metrics table reaching that section by any other route
    died with ColumnNotFoundError instead of drawing the plot. That is the shape a
    covariate-free run produces: attach_strata labels every protein "all" when no
    covariates are given, which populates the hgnc axis without a leaderboard ever running.
    """
    return (pl.struct("tool", "variant")
              .map_elements(lambda r: label_of(r["tool"], r["variant"]),
                            return_dtype=pl.String)
              .alias("label"))


def best_variants(df: pl.DataFrame, top_kmerseek: int = TOP_KMERSEEK, *,
                  across_truth_sets: bool = False) -> pl.DataFrame:
    """Each tool's best variant, plus kmerseek's top `top_kmerseek` under each ranking
    metric, ranked by Fmax.

    Averaged over species before ranking, never summed -- summing would let the species
    with the most annotated proteins pick the winner. The mean is not enough on its own,
    so `fmax_sd`, `fmax_min` and `fmax_max` come back beside it: Fmax varies several-fold
    across the nine target proteomes and the size of that variation differs between tools,
    which is a result rather than noise to be averaged away.

    Selection is the union of the top combos under every metric in RANKING_METRICS, so a
    variant that ranks first on one and nowhere on the other still appears. Ranking on Fmax
    alone made the sensitivity column unreadable: the row it belonged to was not in the
    table. Every baseline still contributes exactly one variant because none of them has
    more than one, so the union only ever widens the kmerseek block.

    `rank_fmax` and `rank_sens_first_fp_mean` are ranks over every variant scored, not over the
    rows kept, so they say where a row sits in the whole sweep rather than in the table.

    Columns: tool, variant, label, n_species, the Fmax spread, both ranks, the sub-chance
    ROC count, and every headline metric.
    """
    cols = [c for c in HEADLINE if c in df.columns]
    if not cols or df.height == 0:
        return df.head(0)
    _guard_single_truth_set(df, across_truth_sets)

    have = set(df.columns)
    # Every headline metric now carries the same five-number summary across species that
    # only Fmax used to, plus a total for anything that is a count. See the block above
    # SPECIES_AGGREGATIONS for why the two are named differently.
    extra = ([pl.col("species").n_unique().alias("n_species")]
             + species_aggregations(cols + [c for c in COUNT_COLS if c in have], have)
             + species_totals(have)
             # Legacy names for the three Fmax spread columns. Same values as
             # fmax__sd/min/max; kept because LEADERBOARD_COLS, widest_spread and the
             # section prose all name them, and renaming those in the same change that
             # introduces the suffix would make both harder to review.
             + [pl.col("fmax").std().alias("fmax_sd"),
                pl.col("fmax").min().alias("fmax_min"),
                pl.col("fmax").max().alias("fmax_max")])
    # A sensitivity averaged over 144 ranked proteins is not the same measurement as one
    # averaged over 542, so the denominator travels with the metric. Mean rather than sum:
    # it is a per-species count and the other columns are per-species means too.
    if "n_proteins_ranked" in df.columns:
        # Bare name stays the MEAN over species, which is what every existing caller and
        # every section description means by it. n_proteins_ranked__total is the sum, and
        # the two differ by a factor of nine -- which is the whole reason for the suffix.
        extra.append(pl.col("n_proteins_ranked").mean().alias("n_proteins_ranked"))
        # The mean hides the worst case, and the worst case is severe: wwmj5 k17 ranks 677
        # human proteins against mouse and 1 against Ciona, and that single protein scored
        # 0.333 where the nine-species average is 0.116. Species are weighted equally here
        # on purpose, so the guard is to show the denominator rather than to reweight.
        extra.append(pl.col("n_proteins_ranked").min().alias("n_proteins_ranked_min"))
    if "sens_first_fp_median" in df.columns:
        extra.append(pl.col("sens_first_fp_median").mean().alias("sens_first_fp_median"))
    if "roc_auc" in df.columns:
        extra.append((pl.col("roc_auc") < CHANCE).sum().alias("roc_auc_sub_chance"))

    per_variant = (
        df.group_by("tool", "variant")
        .agg([pl.col(c).mean() for c in cols] + extra)
        .sort("fmax", descending=True, nulls_last=True)
    )
    if per_variant.height == 0:
        return per_variant

    ranked_by = [m for m in RANKING_METRICS if m in per_variant.columns]
    per_variant = per_variant.with_columns(
        [pl.col(m).rank(method="min", descending=True).cast(pl.Int64).alias(f"rank_{m}")
         for m in ranked_by]
        # Constant, so any caller can quote the denominator the ranks are out of without
        # regrouping the frame. A rank of 1 among a handful of baselines and a rank of 1
        # across the whole sweep are different claims.
        + [pl.lit(per_variant.height, dtype=pl.Int64).alias("n_variants_ranked")]
    )

    kept = pl.concat([
        group.sort(metric, descending=True, nulls_last=True)
             .head(top_kmerseek if tool == "kmerseek" else 1)
        for metric in ranked_by
        for (tool,), group in per_variant.group_by("tool", maintain_order=True)
    ]).unique(subset=["tool", "variant"], keep="first", maintain_order=True)

    # The pinned arm rides along whether or not it won anything, which is the entire point
    # of pinning it: an arm that only appears where it happens to rank cannot be followed
    # across sections. It still sorts on Fmax with everything else, so a section that then
    # trims to --max-tools can drop it if it ranks below the cut -- raise that limit rather
    # than expecting the pin to override it.
    if CANONICAL is not None:
        pin = per_variant.filter((pl.col("tool") == CANONICAL[0])
                                 & (pl.col("variant") == CANONICAL[1]))
        if pin.height:
            kept = pl.concat([kept, pin]).unique(
                subset=["tool", "variant"], keep="first", maintain_order=True)

    return (
        kept.with_columns(label_column())
            .sort("fmax", descending=True, nulls_last=True)
    )


AGG_TITLE = {"mean": "mean", "median": "median", "min": "min", "max": "max", "sd": "SD"}

# Which way is better, per aggregation, so a colour ramp does not say the opposite of the
# metric. `min` on a higher-is-better metric is still higher-is-better (a high worst case
# is good); SD has no direction and gets a neutral single-hue ramp.
AGG_SCALE = {"mean": None, "median": None, "min": "Blues", "max": "Blues", "sd": "Reds"}


# Human titles for the per-species counts. The metric specs cover the rates; these are
# raw column names that would otherwise reach a reader as "n_tp_calls (total)".
COUNT_TITLES = {
    "n_calls": "Calls", "n_tp_calls": "True calls", "n_fp_calls": "False calls",
    "n_gray_calls": "Gray calls", "n_tp_strict": "True calls (strict)",
    "n_truth_instances": "Answer-key instances",
    "n_reachable_instances": "Reachable instances",
    "n_instances_found": "Instances found",
    "n_proteins_scored": "Proteins scored", "n_proteins_ranked": "Proteins ranked",
}


def derived_header(col: str, base_spec: dict) -> dict | None:
    """Header for a `metric__agg` or `count__total` column, built off the base metric's.

    Generated rather than written out: ten headline metrics times five aggregations plus
    the totals is more columns than anyone will hand-maintain correctly, and a stale
    hand-written header that says "mean" over a max column is worse than none.
    """
    if "__" not in col:
        return None
    base, kind = col.rsplit("__", 1)
    spec = dict(base_spec.get(base, {}))
    # The base title may already carry an aggregation -- "Fmax (mean)" -- and appending to
    # it produced "Fmax (mean) (median)". Strip one trailing parenthetical so the suffix
    # this function adds is the only one.
    title = spec.get("title") or COUNT_TITLES.get(base, base)
    # Only an AGGREGATION parenthetical is stripped, not any parenthetical: "Fmax (mean)"
    # must lose its "(mean)" before gaining "(median)", but a title whose parenthetical
    # carries meaning keeps it.
    title = re.sub(r"\s*\((mean|median|min|max|SD)\)\s*$", "", title)
    if kind == "total":
        return dict(title=f"{title} (total)", format="{:,.0f}", scale="Greys",
                    hidden=True,
                    description=f"{title} summed over every target species. A TOTAL for "
                                "this arm's whole run, not a per-species value -- an "
                                "instance counted here once per proteome is one human "
                                "instance scored nine times, so this is not the size of "
                                "the answer key")
    if kind not in AGG_TITLE:
        return None
    spec.pop("description", None)
    spec["title"] = f"{title} ({AGG_TITLE[kind]})"
    # SD is a spread, not a score, so it leaves the metric's own 0-1 bounds and ramp behind.
    if kind == "sd":
        spec.pop("min", None), spec.pop("max", None)
    if AGG_SCALE[kind]:
        spec["scale"] = AGG_SCALE[kind]
    spec["hidden"] = True
    what = ("Standard deviation over target species; high means the mean describes no "
            "single species well" if kind == "sd" else
            f"{AGG_TITLE[kind].capitalize()} over target species")
    spec["description"] = (f"{what}. An AGGREGATION across species, never a sum -- "
                           "species are weighted equally, so this is not tilted toward "
                           "the best-annotated proteome")
    return spec


def fmt_metric_headers(cols: list[str]) -> dict:
    """Column formatting for the leaderboard tables, keyed by metric name."""
    spec = {
        "variant": dict(title="Variant", description="Ranked by mean Fmax over species"),
        "fmax":         dict(title="Fmax (mean)",
                             description="CAFA protein-centric Fmax, interval-aware: the "
                                         "call must also land on the annotated interval. "
                                         "Mean over target species -- read it with the SD "
                                         "and range beside it, never on its own",
                             min=0, max=1, scale="RdYlGn", format="{:,.3f}"),
        "family_fmax":  dict(title="Family Fmax", min=0, max=1, scale="RdPu",
                             format="{:,.3f}",
                             description="Same curve on the SET of families called per "
                                         "protein, placement ignored. Fmax minus this is "
                                         "what boundary placement costs"),
        # The spread columns are the point of the block, so none of them is hidden. A
        # several-fold difference in SD between tools is a result about consistency across
        # divergence, and it is invisible in the mean.
        "fmax_sd":      dict(title="Fmax SD", scale="Reds", format="{:,.3f}",
                             description="Standard deviation over target species; high "
                                         "means the mean describes no single species well"),
        "fmax_min":     dict(title="Fmax min", min=0, max=1, scale="Blues",
                             format="{:,.3f}",
                             description="Worst target species"),
        "fmax_max":     dict(title="Fmax max", min=0, max=1, scale="Blues",
                             format="{:,.3f}",
                             description="Best target species"),
        "rank_fmax":    dict(title="Rank (Fmax)", format="{:,.0f}", scale="Greys-rev",
                             description="Rank among every variant scored, not among the "
                                         "rows shown"),
        "rank_sens_first_fp_mean": dict(title="Rank (sens.)", format="{:,.0f}",
                                        scale="Greys-rev",
                                        description="Rank among every variant scored by "
                                                    "sensitivity to first false positive. "
                                                    "A large gap from the Fmax rank means "
                                                    "the two metrics disagree on this row"),
        "auprc":        dict(title="AUPRC", description="Area under precision / reachable-recall",
                             min=0, max=1, scale="Blues", format="{:,.3f}"),
        "roc_auc":      dict(title="ROC AUC", min=0, max=1, scale="Purples", format="{:,.3f}",
                             description="Call-level, pooled across query proteins. Read "
                                         "the sub-chance count beside it before using this "
                                         "column"),
        # A mean ROC AUC just under 0.5 reads as "slightly weak" on a 0-1 ramp when it
        # actually means correct calls rank below incorrect ones. The count says in how
        # many species that happened, which separates a real ordering failure from one tiny
        # proteome dragging an unweighted mean down.
        "roc_auc_sub_chance": dict(title="ROC < 0.5", format="{:,.0f}", scale="Reds",
                                   description="Target species where a correct call was "
                                               "LESS likely to outrank an incorrect one "
                                               "than a coin flip"),
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
        "sens_first_fp_mean": dict(title="Sens. to 1st FP", min=0, max=1, scale="YlGn",
                                   format="{:,.3f}",
                                   description="Domains recovered above a query's first "
                                               "false positive, averaged over query "
                                               "proteins. Threshold-free, so it measures "
                                               "ranking rather than cutoff choice"),
        "sens_first_fp_median": dict(title="Sens. (median)", min=0, max=1, scale="YlGn",
                                     format="{:,.3f}",
                                     description="Median over query proteins. Far below "
                                                 "the mean means a few queries carry the "
                                                 "score"),
        # Without this the sensitivity column is unreadable: it is averaged only over
        # queries a tool produced a ranking for, so a tool that answers on a sixth of the
        # proteins is scored on an easier sixth.
        "n_proteins_ranked": dict(title="Proteins ranked", format="{:,.0f}", scale="Blues",
                                  description="Query proteins the sensitivity is averaged "
                                              "over, per species. A high sensitivity over "
                                              "few proteins is not comparable to the same "
                                              "number over many"),
        "n_proteins_ranked_min": dict(title="Ranked (min)", format="{:,.0f}", scale="Blues",
                                      hidden=True,
                                      description="Worst target species. Species are "
                                                  "weighted equally, so a species with a "
                                                  "handful of ranked proteins moves the "
                                                  "sensitivity as much as one with "
                                                  "hundreds"),
        "n_species":    dict(title="Species", format="{:,.0f}", scale=False,
                             description="Target proteomes every aggregation on this row "
                                         "was taken over"),
    }
    out = {}
    for c in cols:
        if c in spec:
            out[c] = spec[c]
            continue
        derived = derived_header(c, spec)
        if derived is not None:
            out[c] = derived
    return out


# ---------------------------------------------------------------------------
# sections: accuracy
# ---------------------------------------------------------------------------

# Column order for the leaderboard. Fmax and its spread first, because that block is one
# claim and splitting it lets the mean be read alone. The sensitivity block second with its
# denominator attached, then ROC with its sub-chance count, then the rest of the headline
# metrics in HEADLINE order.
LEADERBOARD_COLS = (
    ["variant", "n_species",
     "fmax", "fmax_sd", "fmax_min", "fmax_max", "rank_fmax",
     "sens_first_fp_mean", "sens_first_fp_median", "n_proteins_ranked",
     "n_proteins_ranked_min", "rank_sens_first_fp_mean",
     "roc_auc", "roc_auc_sub_chance"]
    + [c for c in HEADLINE if c not in {"fmax", "sens_first_fp_mean", "roc_auc"}]
    # Every headline metric's five-number summary across species, then the run totals.
    # All hidden by default via derived_header -- fifty extra columns on by default would
    # make the table unreadable, and the point is that they are THERE when a number looks
    # surprising, one click away in MultiQC's column chooser rather than absent.
    + [f"{m}__{a}" for m in HEADLINE for a in SPECIES_AGGREGATIONS]
    + [f"{c}__{a}" for c in COUNT_COLS for a in SPECIES_AGGREGATIONS]
    + [f"{c}__total" for c in COUNT_COLS]
)


def widest_spread(board: pl.DataFrame) -> str:
    """The board's own worst case for reading a mean as one number, named and quoted.

    Measured off the run rather than written into the prose. An earlier draft hard-coded
    the Swiss-Prot figures, which then appeared verbatim on the Pfam board where they were
    simply wrong.
    """
    if board.height == 0 or "fmax_min" not in board.columns:
        return ""
    row = (board.drop_nulls(["fmax_min", "fmax_max"])
                .sort(pl.col("fmax_max") - pl.col("fmax_min"), descending=True))
    if row.height == 0:
        return ""
    r = row.row(0, named=True)
    return (f"The widest spread here is {r['label']}, which averages {r['fmax']:.3f} but "
            f"runs from {r['fmax_min']:.3f} on its worst target proteome to "
            f"{r['fmax_max']:.3f} on its best.")


def call_imbalance(cut: pl.DataFrame) -> str:
    """False calls per true call, pooled over this truth set, for the precision argument.

    Fmax being precision-dominated is a claim about this run's class balance, so it is
    measured here instead of asserted. Pooled over tools and species: the point is the
    order of magnitude a reader is up against, not any one tool's ratio.
    """
    need = {"n_tp_calls", "n_fp_calls"}
    if not need.issubset(set(cut.columns)):
        return ""
    tp = cut["n_tp_calls"].sum() or 0
    fp = cut["n_fp_calls"].sum() or 0
    if tp <= 0 or fp <= 0:
        return ""
    return (f"Across every tool and target species on this truth set there are "
            f"{fp / tp:,.0f} false calls for every true one, which is why precision "
            f"dominates.")


def section_leaderboards(out: Path, metrics: pl.DataFrame,
                         top_kmerseek: int = TOP_KMERSEEK) -> None:
    # Per truth set, not once for the whole frame: see split_per_truth_set. A single global
    # pick_split kept Pfam's heldout rows and dropped every other truth set, leaving the
    # report with one leaderboard and it was the circular one.
    base, picked = split_per_truth_set(ungrouped(metrics))
    for ts in sorted(picked):
        cut, split = base.filter(pl.col("truth_set") == ts), picked[ts]
        board = best_variants(cut, top_kmerseek)
        if board.height == 0:
            continue
        cols = [c for c in LEADERBOARD_COLS if c in board.columns]
        data = {row["label"]: {c: row[c] for c in cols} for row in board.to_dicts()}
        n_ranked = board["n_variants_ranked"][0] if "n_variants_ranked" in board.columns else 0
        write_section(out, f"qfo_leaderboard_{ts}", {
            "id": f"qfo_leaderboard_{ts}",
            "section_name": f"Leaderboard — {ts} truth",
            "description": (
                f"<p>Each tool's best variant and the sweep's top {top_kmerseek} combos "
                f"under each of the two ranking metrics, on the <code>{split}</code> split "
                f"with no stratification. Ranks are out of the {n_ranked:,} variants "
                f"scored against this truth set.</p>"
                + bullets(
                    "<b>Fmax</b> is a mean over target species, and the species disagree. "
                    "The SD, min and max columns are there so the mean is never read as a "
                    "single number.",
                    "<b>Every headline metric carries the same five-number summary.</b> "
                    "Turn on <code>metric (median)</code>, <code>(min)</code>, "
                    "<code>(max)</code> or <code>(SD)</code> in the column chooser; they "
                    "are hidden rather than absent, because fifty columns on by default "
                    "is unreadable and a surprising mean should be one click from its "
                    "spread.",
                    "<b>An aggregation is not a total.</b> A column reading "
                    "<code>(mean)</code>, <code>(median)</code>, <code>(min)</code>, "
                    "<code>(max)</code> or <code>(SD)</code> collapses the target species "
                    "with each weighted equally. A column reading <code>(total)</code> is "
                    "summed over them and belongs to this arm's whole run — "
                    "<code>Instances (total)</code> is about nine times the size of the "
                    "human answer key, because one human instance is scored once per "
                    "target proteome.",
                    "<b>Rates have no total</b> and counts have both, which is why only "
                    "the count columns offer one. Species are weighted equally throughout: "
                    "mouse carries 17_228 reviewed Swiss-Prot entries and Ciona 28, so a "
                    "size-weighted mean would be a mean over mouse.",
                    "<b>Fmax SD</b> separates tools that hold up across divergence from "
                    "tools that do not. "
                    + widest_spread(board),
                    "The Divergence section plots the same per-species values against time "
                    "since the common ancestor, which is what explains the spread.",
                    "<b>The two headline metrics disagree.</b> Rows are ordered by Fmax; "
                    "the two rank columns say where each row sits under each metric, and a "
                    "row with rank 1 under one and rank 200 under the other is the "
                    "disagreement rather than a mistake.",
                    "<b>Fmax</b> is precision-dominated here, so it mostly ranks tools on "
                    "how little they say. "
                    + call_imbalance(cut),
                    "<b>Sens. to 1st FP</b> is threshold-free and per-query, and is "
                    "what the structure-search baselines this benchmark compares against "
                    "report.",
                    "<b>Proteins ranked</b> belongs beside the sensitivity column: "
                    "sensitivity is averaged only over queries the tool produced a ranking "
                    "for, so a tool that answers on a sixth of the proteins is being "
                    "scored on an easier sixth.",
                    "<b>Ranked (min)</b>, hidden by default, is the worst target species, "
                    "and it can fall to single digits.",
                    "<b>ROC AUC</b> is call-level and pooled across queries, whose scores "
                    "are not on a common scale, so it can and does fall below chance while "
                    "Fmax rises.",
                    "<b>ROC &lt; 0.5</b> counts the target species where that happened, "
                    "and is the number to read: the mean can be pulled under 0.5 by one "
                    "proteome with a handful of calls. Where that count is high the "
                    "ordering claim does not hold and the ROC value should not be quoted.",
                    "Pfam-A domains are defined by profile HMMs, so this truth set is "
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
                        "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
            "headers": fmt_metric_headers(cols),
            "data": data,
        }, supplementary=not is_primary_truth(ts))


# CPU budgets a reader already has a feel for, in CPU-hours, for the cost axis. The device
# is metapredict's Figure 1B (Emenecker, Griffith & Holehouse 2021): named durations drawn
# across a log axis so a tool can be placed in the minutes band or the days band without
# doing arithmetic on the axis. Here the axis is cost rather than rate, so the bands are
# CPU time for one search of the whole query set against one target proteome.
CPU_BANDS = [(1 / 60, "1 CPU-minute"), (1.0, "1 CPU-hour"), (24.0, "1 CPU-day"),
             (24 * 7.0, "1 CPU-week")]

# Accuracy the frontier plots, first available. Residue F1 is the one to lead on: it is
# scored on residues rather than on whether a call cleared an overlap threshold, so an arm
# that finds the right family in roughly the right place is not scored the same as one that
# misses, and it does not inherit the IoU>=0.5 cliff the boundary section documents.
FRONTIER_METRICS = ["residue_f1", "precision_iou80", "fmax"]


def cpu_reference_lines(costs: list[float]) -> tuple[list[dict], float, float]:
    """Named CPU-budget lines for a log cost axis, and the range they sit in.

    Same reasoning as throughput_reference_lines and the same two constraints: x_lines are
    layout shapes and autorange ignores shapes, and the scatter DROPS points outside
    xmin/xmax rather than clipping the axis. So the range comes from the data, is only ever
    widened past it, and is snapped to whole decades because Plotly's minor ticks on a
    short log range render as an unreadable 2,3,...,9 series.
    """
    lo, hi = min(costs), max(costs)
    xmin, xmax = lo / 3, hi * 3
    xmin = 10.0 ** math.floor(math.log10(xmin)) if xmin > 0 else xmin
    xmax = 10.0 ** math.ceil(math.log10(xmax)) if xmax > 0 else xmax
    lines = [{"value": v, "color": "#aaaaaa", "dash": "dash", "width": 1, "label": name}
             for v, name in CPU_BANDS if xmin < v < xmax]
    return lines, xmin, xmax


def repelled_labels(rows: list[dict], xkey: str, ykey: str,
                    min_dx: float = 0.18, min_dy: float = 0.06) -> set[int]:
    """Indices of the rows whose label can be drawn without landing on another one.

    MultiQC has no label-repulsion pass: every `annotation` is drawn beside its marker at a
    fixed offset, so fourteen points inside a narrow band print fourteen overlapping
    strings and none of them is readable. This is the cheap version of repulsion -- walk
    the points in plotting order and keep a label only when nothing already labelled is
    within min_dx in log10 x and min_dy in y. Every point keeps its full name on hover
    whether or not it is labelled, so nothing is lost, only unprinted.
    """
    kept: list[tuple[float, float]] = []
    out = set()
    for i, r in enumerate(rows):
        x, y = r.get(xkey), r.get(ykey)
        if x is None or y is None or x <= 0:
            continue
        lx = math.log10(x)
        if any(abs(lx - px) < min_dx and abs(y - py) < min_dy for px, py in kept):
            continue
        kept.append((lx, y))
        out.add(i)
    return out


def section_frontier(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                     n_queries: int, primary_truth: str,
                     top_kmerseek: int = TOP_KMERSEEK) -> None:
    """Accuracy against cost, with metapredict Figure 1B's named budget lines.

    Both axes changed from the version this replaces, and for the same reason: the old ones
    could not separate the arms. Accuracy was gray-zone Fmax, which put all fourteen points
    into x between 0.10 and 0.18 with their labels printed on top of each other. Cost was
    throughput, which is the reciprocal of a quantity a reader sizing a cluster request
    actually wants. Now y is residue F1, which spreads because it scores residues rather
    than a threshold crossing, and x is CPU-hours on a log axis, which spans decades.

    Better is UP AND LEFT here, not up and right: x is a cost. The named vertical bands say
    what each decade of it buys, so nobody has to convert.

    A tool missing from either axis is missing from the plot. That is mostly the structure
    arms when no structures were staged, and it is stated in the section rather than left
    as an unexplained gap.
    """
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    base = ungrouped(cut)
    metric = next((m for m in FRONTIER_METRICS if m in base.columns
                   and base[m].drop_nulls().len()), None)
    if metric is None:
        return
    y_title = {"residue_f1": "Residue F1",
               "precision_iou80": "Precision at IoU >= 0.8"}.get(
        metric, fmt_metric_headers([metric]).get(metric, {}).get("title", metric))
    Y_NOTES = {
        "residue_f1": ("Residue-level F1: the overlap between the residues an arm labels "
                       "with a family and the residues that family really occupies, so a "
                       "call that lands in roughly the right place scores in proportion "
                       "rather than being marked wrong for missing an overlap threshold."),
        "precision_iou80": ("Precision at IoU >= 0.8: the share of calls that land on at "
                            "least four fifths of the true domain."),
        "fmax": ("Fmax over every domain instance. Residue F1 is not in this run, so the "
                 "axis falls back to the interval-level score, which is the one the "
                 "boundary section shows is a localisation measure rather than a "
                 "detection one."),
    }
    y_note = Y_NOTES.get(metric, y_title)
    acc = base.group_by("tool", "variant").agg(pl.col(metric).mean().alias("acc"))

    # The gray-zone number stays, in the table beside the figure rather than on an axis
    # nothing separates on. It is the regime the hypothesis is about and it is worth
    # reading; it is not worth fourteen overlapping labels.
    gray = cut.filter(
        (pl.col("stratum_axis") == "identity") & pl.col("stratum").is_in(GRAY_ZONE)
    )
    gray_acc = (
        gray.group_by("tool", "variant")
            .agg(((pl.col("fmax") * pl.col("n_truth_instances")).sum()
                  / pl.col("n_truth_instances").sum().clip(lower_bound=1)).alias("acc"))
        if gray.height else acc.head(0)
    )

    # One point per baseline, TOP_KMERSEEK points for the sweep. A single kmerseek point
    # would hide whether the winner is a lone spike or the top of a plateau, which is the
    # part of the figure a reader should be able to check.
    ranked = acc.sort("acc", descending=True, nulls_last=True)
    best = pl.concat([
        group.head(top_kmerseek if tool == "kmerseek" else 1)
        for (tool,), group in ranked.group_by("tool", maintain_order=True)
    ]) if ranked.height else ranked

    joined = attach_throughput(best, trace, n_queries).filter(
        pl.col("acc").is_not_null() & pl.col("cpu_hours").is_not_null()
        & (pl.col("cpu_hours") > 0)
    )
    dropped = sorted(set(best["tool"]) - set(joined["tool"]))
    if joined.height == 0:
        # The headline figure is missing, so say why rather than leaving a hole where a
        # reader has to work out that the report is not just quiet about cost.
        write_section(out, "qfo_frontier", {
            "id": "qfo_frontier",
            "section_name": "The frontier",
            "description": "Accuracy against cost, when both are available.",
            "plot_type": "html",
            "data": (
                "<p>Not built: no tool has both an accuracy number and a measured search "
                "cost in this run. Cost comes from the Nextflow trace, so this "
                "figure needs the trace of the run that produced these metrics — pass it "
                "with <code>--report_trace</code>, or use <code>make multiqc</code>, which "
                "fills it in from the newest trace under <code>run/</code>. kmerseek is "
                "the exception: it is the one arm on <code>storeDir</code>, so a store hit "
                "runs no task and leaves no trace row, and its timings come from the "
                "<code>*.timings.jsonl</code> records beside its results instead.</p>"
            ),
        })
        return

    rows = joined.sort("cpu_hours").to_dicts()
    # The accuracy the incumbents actually reach. An arm above this line beats everything
    # that exists, which is the claim the figure is making, so it is computed from the data
    # rather than drawn by hand.
    #
    # Motif arms are excluded from it and no longer get a reference line of their own. The
    # line calibrates a bar every other arm has to clear, and section_boundary already says
    # in as many words that a tool reporting the envelope of a discontinuous residue set
    # measures a different thing and must not be ranked against the alignment arms. On the
    # midi run folddisco sat far above every other arm, so one motif tool was setting the
    # whole target -- and the same tool is excluded from every IoU comparison elsewhere in
    # this report. Drawing its level as a named dotted line put it back into the comparison
    # by the back door. Its points are still plotted, as open markers, so nothing is
    # hidden; what is gone is its line.
    semantics = {}
    if "interval_semantics" in cut.columns:
        semantics = {r["tool"]: r["interval_semantics"] for r in
                     cut.select("tool", "interval_semantics").unique().to_dicts()}

    def is_motif(tool: str) -> bool:
        return semantics.get(tool, "alignment") == "motif"

    aligned = [r for r in rows if r["tool"] != "kmerseek" and not is_motif(r["tool"])]
    best_row = max(aligned, key=lambda r: r["acc"], default=None)
    y_best = best_row["acc"] if best_row else None
    motif_arms = sorted({r["tool"] for r in rows if is_motif(r["tool"])})

    labelled = repelled_labels(rows, "cpu_hours", "acc")
    points = {}
    for i, r in enumerate(rows):
        cls = tool_class(r["tool"])
        label = label_of(r["tool"], r["variant"])
        is_ours = cls == "kmerseek"
        # group is the method class here rather than the tool, because the class IS the
        # argument this figure makes and the legend is where the colours get named. The
        # per-point label is carried by `annotation`, which is drawn on the plot.
        #
        # Size, outline and opacity are what mark the tool being argued for. Colour cannot
        # do it: colour is spoken for by the method class everywhere else in this report,
        # and a green that means "kmerseek" on one plot and "this work" on another is worse
        # than no highlight.
        points[label] = scatter_point(
            r["cpu_hours"], r["acc"],
            name=f"{label} ({r['acc']:.3f} at {r['cpu_hours']:.3g} CPU-h)",
            group=CLASSES[cls][0] + (" · motif semantics" if is_motif(r["tool"]) else ""),
            color=CLASSES[cls][1],
            annotation=(short_label(r["tool"], r["variant"]) if i in labelled else None),
            marker_size=16 if is_ours else 9,
            marker_line_width=2 if is_ours or is_motif(r["tool"]) else 1,
            opacity=1.0 if is_ours else 0.65,
            # Three tools share the pink of "needs 3D structure", and a static PNG export
            # has no hover to tell them apart. Shape does it. A motif arm is drawn open, so
            # that a point excluded from the report's IoU comparisons is visibly a
            # different kind of point rather than one more filled dot.
            marker_symbol=(TOOL_SYMBOL.get(r["tool"], "circle") + "-open"
                           if is_motif(r["tool"])
                           else TOOL_SYMBOL.get(r["tool"], "circle")),
        )

    # Only the repelled labels are drawn. Every other point gets an empty annotation key,
    # which is what stops MultiQC printing full hover names on ten arbitrary outliers --
    # see suppress_auto_labels.
    suppress_auto_labels(points)
    ys = [r["acc"] for r in rows]
    xs = [r["cpu_hours"] for r in rows]
    lines, xmin, xmax = cpu_reference_lines(xs)
    pconfig = {
        "id": "qfo_frontier_plot",
        "title": f"Accuracy against cost ({primary_truth} truth, {split} split)",
        "xlab": "CPU-hours for one search of the query set against one proteome (log)",
        "ylab": f"{y_title} (mean over target species)",
        "xlog": True, "height": 580,
        "xmin": xmin, "xmax": xmax,
        # Point labels are drawn beside the marker, so a point at the axis limit loses half
        # its label. The accuracy metrics here are bounded at 0 and 1, so the padding is
        # clamped there rather than left to run past the end of the scale.
        "ymin": max(0.0, min(ys) - 0.06), "ymax": min(1.0, max(ys) + 0.06),
        # MultiQC copies a "%" out of an axis label into that axis's tick suffix, so the
        # suffixes are set explicitly rather than inferred.
        "xsuffix": "", "ysuffix": "",
        "x_decimals": 2, "y_decimals": 3,
        # See ANNOTATE_EVERY_POINT_BELOW: without this the legend never renders.
        "showlegend": True,
        "x_lines": lines,
    }
    if y_best is not None:
        pconfig["y_lines"] = [{"value": y_best, "color": "#666666", "dash": "dash",
                               "width": 2,
                               "label": f"best incumbent {y_title} "
                                        f"({best_row['tool']})"}]

    note = ""
    if dropped:
        note = ("<b>Missing arms</b> — no timing row for "
                + ", ".join(f"<code>{d}</code>" for d in dropped)
                + ": that arm either did not run or produced no timing record, so it is "
                  "absent from the plot rather than plotted at zero.")
    motif_note = (
        "<b>Motif arms are drawn open and set no line.</b> "
        + ", ".join(f"<code>{t}</code>" for t in motif_arms)
        + " reports the envelope of a discontinuous residue set rather than an alignment, "
          "so its residue overlap is not the same measurement. It is excluded from every "
          "IoU comparison elsewhere in this report, and it no longer gets a reference line "
          "here either: calibrating the frontier on an arm the report excludes was that "
          "comparison by the back door. The points stay so the number is not hidden."
    ) if motif_arms else ""
    # A trace from before the 2026-08-25 index/search split has kmerseek's index build
    # inside its search time, so the "search only" claim below would be false for it. Say
    # which it is rather than printing the claim unconditionally.
    fused = mt.fused_index_tools(trace)
    if fused:
        index_note = (
            "<b>Index cost</b> — this trace predates the index/search split, so for "
            + ", ".join(f"<code>{t}</code>" for t in fused)
            + " the CPU-hours INCLUDE building the target index; every other arm is search "
              "only. Those points are not comparable to the rest, and re-running against a "
              "post-split trace is what fixes it.")
    else:
        index_note = (
            "<b>Index cost</b> — the CPU-hours are the SEARCH only, for every arm, and "
            "they are one search of the whole query set against one target proteome, not "
            "the whole sweep and not the whole nine-proteome run. Each tool that needs a "
            "database builds it in its own process (<code>foldseekDb</code>, "
            "<code>prostt5Db</code>, <code>mmseqsDb</code>, <code>kmerseekIndex</code>) "
            "and only the search process is charged, so no tool pays for an index it "
            "builds once and reuses across every query. Index cost is reported separately "
            "under CPU time by process.")
    write_section(out, "qfo_frontier", {
        "id": "qfo_frontier",
        "section_name": "The frontier",
        "description": (
            "<p>Accuracy against cost. <b>Up and to the LEFT is better</b>: x is a cost, "
            "not a rate.</p>"
            + bullets(
                f"<b>y axis ({y_title})</b> — {y_note} Averaged over target proteomes.",
                "<b>x axis (CPU-hours)</b> — measured on this run: the CPU time of one "
                "search of the whole query set against one target proteome, log scale "
                "because the arms span decades. The previous version of this figure put "
                "accuracy on x and every point landed between 0.10 and 0.18 with the "
                "labels printed over each other.",
                "<b>Dashed vertical lines</b> are named CPU budgets rather than data, so "
                "an arm can be placed in the CPU-minute band or the CPU-day band without "
                "dividing anything.",
                "<b>Dashed horizontal line</b> is the best accuracy any existing tool "
                "reaches under the same interval semantics as everything else here, so "
                "the upper-left quadrant is the part of the space nothing occupies today.",
                motif_note,
                f"<b>kmerseek points</b> — the sweep contributes its top {top_kmerseek} "
                "combos rather than one point, so a lone spike is distinguishable from a "
                "plateau, and its points are drawn larger and at full opacity.",
                "<b>Point labels</b> are drawn only where they do not land on another "
                "label. Every point carries its full name, its accuracy and its cost on "
                "hover whether or not it is labelled.",
                "<b>Colour</b> marks the method class, which the legend names; "
                "<b>marker shape</b> marks the individual tool, since three tools share "
                "the colour of \"needs 3D structure\".",
                "<b>The gray-zone number is in the table below, not on an axis.</b> Fmax "
                "under 40% identity is the regime the hypothesis is about, and it is worth "
                "reading; it does not separate the arms well enough to be an axis.",
                index_note,
                note)),
        "plot_type": "scatter",
        "pconfig": pconfig,
        "data": points,
    })


    # The capability table that travels with the figure: sensitivity means little without
    # what a tool needs before it can produce it.
    board = attach_throughput(best_variants(ungrouped(cut), top_kmerseek),
                              trace, n_queries)
    # The gray-zone Fmax is off the figure's axes now, so this is where a reader reads it.
    # It comes from its own frame rather than from the plotted board, because the two rank
    # on different metrics and joining on the board's rows would silently blank the arms
    # that made the board on residue F1 but not on gray-zone Fmax.
    gray_of = {(r["tool"], r["variant"]): r["acc"] for r in gray_acc.to_dicts()}
    frontier_of = {(r["tool"], r["variant"]): r["acc"] for r in acc.to_dicts()}
    cap = {}
    for row in board.to_dicts():
        t = row["tool"]
        search_s = row.get("search_s")
        cap[row["label"]] = {
            "cls": CLASSES[tool_class(t)][0],
            "needs_3d": NEEDS_3D.get(t, "No"),
            "alignment_free": ALIGNMENT_FREE.get(t, "No"),
            "frontier_y": frontier_of.get((t, row["variant"])),
            "fmax": row.get("fmax"),
            "gray_fmax": gray_of.get((t, row["variant"])),
            # The figure's own y value, in minutes, so a reader can read off the exact
            # number a point sits at instead of estimating it off a log axis.
            "search_min": (search_s / 60) if search_s is not None else None,
            "queries_per_s": row.get("queries_per_s"),
            "cpu_hours": row.get("cpu_hours"),
        }
    write_section(out, "qfo_capability", {
        "id": "qfo_capability",
        "section_name": "What each tool needs",
        "description": ("<p>Inputs required, accuracy, and measured cost, side by side.</p>"
                        + bullets(
                            "<b>Class</b> is what a tool needs before it can answer at "
                            "all.",
                            "<b>Needs 3D?</b> is whether the tool requires a structure.",
                            "<b>Alignment-free?</b> names the alignment the tool does, "
                            "where it does one.",
                            "<b>Fmax (all)</b> is over every domain instance; "
                            "<b>Fmax (&lt;40% id)</b> is over the gray zone the hypothesis "
                            "is about.",
                            "<b>Minutes / proteome</b> is the wall clock of one search of "
                            "the whole query set against one target proteome, median over "
                            "target species. This is the frontier plot's y axis; lower is "
                            "better.",
                            "<b>CPU-hours</b> is summed over this combo's SEARCH tasks, or "
                            "over the whole arm where the trace cannot separate variants. "
                            "Database construction is excluded for every arm, "
                            "kmerseekIndex included; it is under CPU time by process.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_capability_table", "title": "Method capabilities and cost",
                    "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "cls": dict(title="Class", scale=False),
            "needs_3d": dict(title="Needs 3D?", scale=False),
            "alignment_free": dict(title="Alignment-free?", scale=False),
            "frontier_y": dict(title=f"{y_title} (frontier y)", min=0, max=1,
                               scale="RdYlGn", format="{:,.3f}",
                               description="The accuracy axis of the frontier plot above"),
            "fmax": dict(title="Fmax (all)", min=0, max=1, scale="RdYlGn", format="{:,.3f}"),
            "gray_fmax": dict(title="Fmax (<40% id)", min=0, max=1, scale="RdYlGn",
                              format="{:,.3f}"),
            "search_min": dict(title="Minutes / proteome", scale="RdYlGn-rev",
                               format="{:,.1f}",
                               description="Wall clock of one search of the whole query "
                                           "set against one target proteome, median over "
                                           "target species. This is the frontier plot's "
                                           "y axis; lower is better"),
            "queries_per_s": dict(title="Queries/s", scale="Blues", format="{:,.1f}"),
            "cpu_hours": dict(title="CPU-hours", scale="Reds", format="{:,.2f}",
                              description="Summed over this combo's SEARCH tasks, or "
                                          "over the whole arm where the trace cannot "
                                          "separate variants. Database construction is "
                                          "excluded for every arm, kmerseekIndex "
                                          "included; it is under CPU time by process"),
        },
        "data": cap,
    })


def section_search_cost(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                        n_queries: int, primary_truth: str,
                        max_tools: int) -> None:
    """What one search costs, ordered by cost, with the method class on every bar.

    The speed claim this project makes is specific and the report has to keep it specific.
    kmerseek is cheap against the methods that have to build or run something before they
    can answer -- a profile HMM, a structure alignment, a 3B-parameter language model --
    and it is NOT cheap against a plain sequence-sequence search. mmseqs2 in seq-seq mode
    is faster per query and cheaper per search than every kmerseek arm in this run.

    Ordering the bars by cost and colouring them by method class is what makes that
    readable without anybody having to write the caveat down: the alignment-grey bars sit
    at the cheap end with kmerseek, the profile and structure bars sit decades away. A
    figure that sorted by tool name, or that plotted only kmerseek against the expensive
    arms, would be making a blanket claim the numbers do not support.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    if cut.height == 0:
        return
    board = attach_throughput(best_variants(cut).head(max_tools), trace, n_queries)
    rows = [r for r in board.to_dicts()
            if r.get("cpu_hours") is not None and r["cpu_hours"] > 0]
    if not rows:
        return
    rows.sort(key=lambda r: r["cpu_hours"])
    data, table = {}, {}
    for r in rows:
        cls = tool_class(r["tool"])
        rate = r.get("queries_per_s")
        # "queries per s", never "q/s". MultiQC cleans a sample name as though it were a
        # file path, so a slash anywhere in it truncates the name to whatever follows the
        # last one -- every arm here collapsed to a single row labelled "s".
        key = (f"{r['label']} · {rate:,.1f} queries per s" if rate else r["label"])
        data[key] = {cls: r["cpu_hours"]}
        table[r["label"]] = {"cls": CLASSES[cls][0], "cpu_hours": r["cpu_hours"],
                             "queries_per_s": rate,
                             "fmax": r.get("fmax")}
    cheapest = rows[0]
    km = next((r for r in rows if r["tool"] == "kmerseek"), None)
    cheaper = [r for r in rows if r["tool"] != "kmerseek"
               and km is not None and r["cpu_hours"] < km["cpu_hours"]]
    # Cost and rate are different orderings and the honest caveat has to check both. A tool
    # that costs more CPU-hours per search can still answer more queries per second, which
    # is exactly the mmseqs2 case, and a panel that only ranked cost would let a blanket
    # speed claim through.
    faster = [r for r in rows
              if r["tool"] != "kmerseek" and km is not None
              and r.get("queries_per_s") and km.get("queries_per_s")
              and r["queries_per_s"] > km["queries_per_s"]]
    if km is None:
        caveat = ""
    elif cheaper or faster:
        parts = []
        if cheaper:
            parts.append("cheaper per search — "
                         + ", ".join(f"<code>{r['tool']}</code> at "
                                     f"{r['cpu_hours']:.3f} CPU-h" for r in cheaper))
        if faster:
            best_km = max((r["queries_per_s"] for r in rows if r["tool"] == "kmerseek"
                           and r.get("queries_per_s")), default=None)
            parts.append(
                "faster per query — "
                + ", ".join(f"<code>{r['tool']}</code> at "
                            f"{r['queries_per_s']:,.0f} queries per second" for r in faster)
                + (f", against {best_km:,.0f} for the quickest kmerseek arm"
                   if best_km else ""))
        caveat = (
            "<b>Baselines that beat kmerseek on this panel</b> — " + "; ".join(parts)
            + ". The speed argument this benchmark makes is against the methods that need "
              "a profile, a structure or a language model before they can answer. It is "
              "not a claim to beat a plain sequence-sequence search, and the ordering here "
              "is so that a reader does not have to take the distinction on trust.")
    else:
        caveat = (
            "<b>Every baseline in this run costs more per search and answers fewer queries "
            "per second than the cheapest kmerseek arm.</b> That is this run's ordering, "
            "not a general claim: a plain sequence-sequence search is the arm to watch, "
            "and where it lands under kmerseek the comparison is with the profile and "
            "structure methods only.")
    write_section(out, "qfo_search_cost", {
        "id": "qfo_search_cost",
        "section_name": "What one search costs",
        "description": (
            f"<p>CPU-hours for one search of the whole query set against one target "
            f"proteome, cheapest first ({primary_truth} truth, <code>{split}</code> "
            f"split, median over target proteomes).</p>"
            + bullets(
                caveat,
                "<b>Colour is the method class</b>, which is the axis the cost argument is "
                "actually about: what a tool needs before it can answer at all.",
                "<b>Each tick label carries that arm's measured query rate</b>, so cost per "
                "search and throughput are readable off one panel.",
                f"<b>The cheapest arm here is <code>{cheapest['label']}</code></b> at "
                f"{cheapest['cpu_hours']:.3f} CPU-hours.",
                "<b>Database construction is excluded for every arm</b>, kmerseekIndex "
                "included. It is reported separately under CPU time by process, so no tool "
                "is charged here for an index it builds once and reuses.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_search_cost_plot",
                    "title": "CPU-hours per search, cheapest first",
                    "ylab": "CPU-hours per search (log scale)",
                    # Grouped, not stacked. Every arm carries exactly one category -- its
                    # method class -- so a stacked bar normalises each row to itself and
                    # the whole panel collapses to one full-width bar. Grouping draws the
                    # one category each arm has, in its class colour.
                    "xlog": True, "cpswitch": False, "sort_samples": False,
                    "stacking": "group", "height": 620},
        "categories": {key: {"name": name, "color": color}
                       for key, (name, color) in CLASSES.items()},
        "data": data,
    })
    write_section(out, "qfo_search_cost_table", {
        "id": "qfo_search_cost_table",
        "section_name": "Search cost and rate",
        "description": ("<p>The panel above as numbers, cheapest search first.</p>"
                        + bullets(
                            "<b>CPU-hours</b> is one search of the query set against one "
                            "target proteome, median over proteomes.",
                            "<b>Queries/s</b> is the same measurement as a rate.",
                            "<b>Fmax</b> is beside them so a reader can see what the "
                            "extra cost buys, or does not.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_search_cost_table_table",
                    "title": "CPU-hours and query rate per arm",
                    "col1_header": "Arm", "sort_rows": False, "no_violin": True},
        "headers": {
            "cls": dict(title="Class", scale=False),
            "cpu_hours": dict(title="CPU-hours / search", format="{:,.3f}",
                              scale="Reds-rev"),
            "queries_per_s": dict(title="Queries/s", format="{:,.1f}", scale="Blues"),
            "fmax": dict(title="Fmax", min=0, max=1, format="{:,.3f}", scale="Greens"),
        },
        "data": table,
    })


def _search_tasks(trace: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """Search tasks with a usable run time, annotated with their query rate."""
    if trace.height == 0:
        return trace.head(0)
    searches = trace.filter(
        pl.col("is_search") & pl.col("realtime_s").is_not_null() & (pl.col("realtime_s") > 0)
    )
    if searches.height == 0:
        return searches
    return searches.with_columns(
        (n_queries / pl.col("realtime_s")).alias("qps"),
        pl.struct("process", "tag")
          .map_elements(lambda r: mt.variant_from_tag(r["process"], r["tag"]),
                        return_dtype=pl.String)
          .alias("trace_variant"),
    )


def throughput_per_tool(trace: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """Median query rate and median search wall clock per tool, plus that arm's CPU-hours.

    Median over target species rather than mean: one species dominating the sweep's wall
    time should not move a rate that is meant to describe the method.

    `search_s` is the same measurement as `queries_per_s` read the other way round, and it
    is aggregated here rather than derived from the rate afterwards so the two can never
    disagree about which tasks went into the median.
    """
    empty = pl.DataFrame(schema={"tool": pl.String, "queries_per_s": pl.Float64,
                                 "search_s": pl.Float64, "cpu_hours": pl.Float64})
    per_task = _search_tasks(trace, n_queries)
    if per_task.height == 0:
        return empty
    rate = per_task.group_by("tool").agg(
        pl.col("qps").median().alias("queries_per_s"),
        pl.col("realtime_s").median().alias("search_s"),
        pl.col("cpu_hours").sum().alias("cpu_hours"),
    )
    # mmseqs2 runs two variants under one process name, so the trace's process column
    # cannot separate them; the metrics table spells them apart. Emit both spellings from
    # the one rate rather than dropping mmseqs2 out of the join entirely.
    mm = rate.filter(pl.col("tool") == "mmseqs2")
    if mm.height:
        rate = pl.concat([
            rate.filter(pl.col("tool") != "mmseqs2"),
            mm.with_columns(pl.lit("mmseqs2_seqseq").alias("tool")),
            mm.with_columns(pl.lit("mmseqs2_iterative").alias("tool")),
        ])
    return rate


def throughput_per_variant(trace: pl.DataFrame, n_queries: int) -> pl.DataFrame:
    """Rate and cost per (tool, variant), for tools whose trace tags carry the variant.

    kmerseek tags read `<species>_<alphabet>_k<k>_lc<bool>`, so each of the sweep's combos
    has its own timings and must not be plotted at a shared per-arm rate -- k=18 on a
    2-letter alphabet and k=29 on the same alphabet are different jobs entirely. Every
    other arm has one variant per process and falls back to the tool-level figure.
    """
    empty = pl.DataFrame(schema={"tool": pl.String, "variant": pl.String,
                                 "queries_per_s": pl.Float64, "search_s": pl.Float64,
                                 "cpu_hours": pl.Float64})
    per_task = _search_tasks(trace, n_queries)
    if per_task.height == 0:
        return empty
    return (
        per_task.filter(pl.col("trace_variant") != "default")
        .group_by("tool", "trace_variant")
        .agg(pl.col("qps").median().alias("queries_per_s"),
             pl.col("realtime_s").median().alias("search_s"),
             pl.col("cpu_hours").sum().alias("cpu_hours"))
        .rename({"trace_variant": "variant"})
    )


def attach_throughput(sel: pl.DataFrame, trace: pl.DataFrame,
                      n_queries: int) -> pl.DataFrame:
    """Add queries_per_s, search_s and cpu_hours to a (tool, variant) selection.

    Per variant where the trace can tell them apart, per arm otherwise. Rows with neither
    are returned with nulls; the caller decides whether to drop them.
    """
    by_variant = throughput_per_variant(trace, n_queries)
    by_tool = throughput_per_tool(trace, n_queries)
    out = sel
    if by_variant.height:
        out = out.join(by_variant, on=["tool", "variant"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("queries_per_s"),
                               pl.lit(None, dtype=pl.Float64).alias("search_s"),
                               pl.lit(None, dtype=pl.Float64).alias("cpu_hours"))
    if by_tool.height:
        out = out.join(by_tool.rename({"queries_per_s": "_qps_tool",
                                       "search_s": "_search_tool",
                                       "cpu_hours": "_cpu_tool"}),
                       on="tool", how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("_qps_tool"),
                               pl.lit(None, dtype=pl.Float64).alias("_search_tool"),
                               pl.lit(None, dtype=pl.Float64).alias("_cpu_tool"))
    # coalesce, not min/max_horizontal: those skip nulls in a way that would silently
    # substitute the arm total wherever a variant genuinely measured zero.
    return out.with_columns(
        pl.coalesce("queries_per_s", "_qps_tool").alias("queries_per_s"),
        pl.coalesce("search_s", "_search_tool").alias("search_s"),
        pl.coalesce("cpu_hours", "_cpu_tool").alias("cpu_hours"),
    ).drop("_qps_tool", "_search_tool", "_cpu_tool")


# Positions down each species' curve that the pooled curve is evaluated at. 101 is one
# point per percent of the way down, which is already finer than a rendered pixel on a
# 500px plot.
POOLED_CURVE_GRID = 101

# Precision the recall panel reports at. 0.5 is the point where a call is as likely right
# as wrong, which is the weakest cutoff a reader would act on.
RECALL_AT_PRECISION = 0.5


def pool_curve_over_species(sub: pl.DataFrame, xcol: str, ycol: str,
                            n_points: int = POOLED_CURVE_GRID
                            ) -> list[tuple[float, float]]:
    """Mean x and mean y across species, matched by rank down each species' own curve.

    Both axes are averaged and the pairing is by RANK, not by x and not by score. Each
    species' curve is a list of operating points ordered from its most stringent score to
    its most permissive; position u in [0, 1] down that list is defined for every species
    at every u, so the average is over every proteome at every drawn point and never over a
    subset. Point u carries (mean recall at u, mean precision at u), which is the same kind
    of quantity every table in this report holds: a mean over target proteomes.

    The endpoint is the reason to prefer this to the obvious alternatives. At u = 1 the x
    value is the mean over species of each species' own maximum recall, which is exactly
    the denominator "Recall at precision >= 0.5" is built from, so the curve and that table
    can be read against each other.

    Rank rather than matched score threshold, which is the other way to make every species
    contribute at every point. A threshold is not the same stringency in two proteomes: an
    E-value depends on the size of the database it was computed against, and the nine
    target proteomes differ several-fold in size, so pairing on the number would pair
    different operating points and call it a match. Rank is scale-free and pairs each
    species at the same position down its own curve.

    What this replaces, and why. The previous version binned x to two decimals, averaged y
    inside each bin, and stopped at the first bin that did not hold every species. The
    intent was to stop where the shortest species curve ends -- past that point a mean
    inside an x bin silently becomes the mean of whichever species got that far, and those
    are the easiest ones, so precision turns back upward showing a subset. The effect was
    to stop at the first x value no species happened to land on, and a species curve is a
    few hundred scattered points rather than a dense grid, so a gap appears within the
    first bin or two. In the midi-plus report that truncated four kmerseek arms to a SINGLE
    point at recall 0.0 and hhblits to recall 0.02, while the same arms' recall at
    precision 0.5 was 0.087 and their recall_reachable 0.78. The curve panel and the recall
    table contradicted each other, and the curve was the one that was wrong.

    Averaging y inside a shared x bin is what forces the choice between truncating and
    silently changing the set of species being averaged. Averaging both coordinates at
    matched rank removes the choice: there is no bin to fall out of.

    Returns (x, y) pairs from the most stringent end to the most permissive, so the caller
    can both draw them and read a value off them.
    """
    if sub.height == 0 or "species" not in sub.columns:
        return []
    # Sort within a species, not globally. score_threshold is unique inside one species'
    # curve, so it is a total order there and the rank is reproducible; a global sort would
    # interleave species.
    order_col = "score_threshold" if "score_threshold" in sub.columns else xcol
    descending = order_col == "score_threshold"
    per_species = []
    for (species,), one in sub.group_by("species", maintain_order=True):
        one = one.sort(order_col, descending=descending, nulls_last=True)
        xs = one[xcol].to_numpy().astype(float)
        ys = one[ycol].to_numpy().astype(float)
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[keep], ys[keep]
        if xs.size == 0:
            continue
        per_species.append((species, xs, ys))
    if not per_species:
        return []

    us = np.linspace(0.0, 1.0, n_points)
    xg, yg = [], []
    for _, xs, ys in per_species:
        # A one-point species curve is flat at its own value rather than dropped: dropping
        # it would put that species back in the "averaged over a subset" failure this
        # function exists to remove.
        src = np.linspace(0.0, 1.0, xs.size) if xs.size > 1 else np.zeros(1)
        xg.append(np.interp(us, src, xs) if xs.size > 1 else np.full(n_points, xs[0]))
        yg.append(np.interp(us, src, ys) if ys.size > 1 else np.full(n_points, ys[0]))
    mean_x = np.mean(np.vstack(xg), axis=0)
    mean_y = np.mean(np.vstack(yg), axis=0)
    return [(float(x), float(y)) for x, y in zip(mean_x, mean_y)]



def curve_series(pairs: list[tuple[float, float]], decimals: int = 4) -> dict:
    """Pairs as the {x: y} map MultiQC's linegraph wants, one y per distinct x.

    JSON object keys are strings, which MultiQC converts back to numbers for a numeric
    axis, so the key is written at fixed precision rather than left to repr. Grid points
    that round onto the same x are averaged rather than overwriting each other, so the
    drawn line is the same function the caller reads values off.
    """
    grouped: dict[str, list[float]] = {}
    for x, y in pairs:
        grouped.setdefault(f"{x:.{decimals}f}", []).append(y)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def recall_at_precision(pairs: list[tuple[float, float]],
                        min_precision: float = RECALL_AT_PRECISION) -> float:
    """Highest recall on this curve at precision at or above `min_precision`.

    Read off the DRAWN series rather than off the raw pairs, so the panel and the plot
    cannot disagree even in the last decimal. curve_series rounds x to a fixed precision and
    averages the y of any grid points that land on the same rounded x, which is what keeps
    the drawn line single-valued; reading the raw pairs instead let the table report a
    recall the line does not pass through. The difference is small and it is exactly the
    kind of small that makes a reader distrust both numbers.

    An arm that never reaches the precision returns 0: recall 0 is achievable at any
    precision by calling nothing, so there is no such thing as "undefined" here.
    """
    reached = [float(x) for x, y in curve_series(pairs).items()
               if y is not None and y >= min_precision]
    return max(reached) if reached else 0.0


def curve_lines(board: pl.DataFrame, max_lines: int) -> list[tuple[str, str, str]]:
    """One kmerseek arm and up to five baselines, as (tool, variant, label).

    The pinned arm when there is one, otherwise the board's best kmerseek row. Six lines is
    a ceiling rather than a target: the panel exists to show one shape against a
    background, and every line past about six is drawn over the ones under it.
    """
    rows = board.to_dicts()
    km = [r for r in rows if r["tool"] == "kmerseek"]
    if CANONICAL is not None:
        pinned = [r for r in km if r["variant"] == CANONICAL[1]]
        km = pinned or km
    keep = km[:1] + [r for r in rows if r["tool"] != "kmerseek"]
    return [(r["tool"], r["variant"], r["label"]) for r in keep[:min(max_lines, 6)]]


def _motif_tools(cut: pl.DataFrame) -> set[str]:
    """Tools whose calls are the envelope of a discontinuous residue set, not an interval."""
    if "interval_semantics" not in cut.columns:
        return set()
    return set(cut.filter(pl.col("interval_semantics") == "motif")["tool"].to_list())


def section_recall_at_precision(out: Path, cached: dict, cut: pl.DataFrame,
                                board: pl.DataFrame, primary_truth: str, split: str,
                                n_species: int) -> None:
    """How far each arm reaches while still being right half the time. 3.4.

    Read off the same pooled curves the PR panel draws, through the same two functions, so
    the bar and the line cannot disagree. `cached` carries the pairs already computed for
    the arms the PR panel drew; every other arm on the board is pooled here.

    Motif arms are left out rather than ranked last. A tool that reports the envelope of a
    discontinuous residue set is answering a different question from one that reports an
    interval, and the boundary section already says those rows must not be ranked against
    the alignment rows. Ranking them here would be that comparison by the back door.
    """
    dropped = _motif_tools(cut)
    data = {}
    for row in board.to_dicts():
        tool, variant, label = row["tool"], row["variant"], row["label"]
        if tool in dropped:
            continue
        pairs = cached.get(label)
        if pairs is None:
            sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
            if sub.height == 0:
                continue
            pairs = pool_curve_over_species(sub, "recall_reachable", "precision")
        if not pairs:
            continue
        data[label] = {"recall_at_p": recall_at_precision(pairs),
                       "max_precision": max(y for _, y in pairs),
                       "max_recall": max(x for x, _ in pairs)}
    if not data:
        return
    ordered = dict(sorted(data.items(), key=lambda kv: -kv[1]["recall_at_p"]))
    excluded = ("<b>Excluded</b> — "
                + ", ".join(f"<code>{t}</code>" for t in sorted(dropped))
                + ": motif semantics. Those calls are the envelope of a discontinuous "
                  "residue set rather than an interval, so their precision is not the "
                  "same measurement." if dropped else "")
    write_section(out, "qfo_recall_at_precision", {
        "id": "qfo_recall_at_precision",
        "section_name": f"Recall at precision >= {RECALL_AT_PRECISION:g}",
        "description": (
            f"<p>The highest recall each arm reaches while at least "
            f"{RECALL_AT_PRECISION:g} of its calls are correct "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                f"<b>Both axes are means over the {n_species} target proteomes at a "
                "matched score threshold</b>, read off the same pooled curves the "
                "precision/recall panel draws. The bar and the line are the same numbers.",
                "<b>Threshold-free.</b> No tool's shipped default cutoff enters this; each "
                "arm is allowed its own best operating point.",
                "<b>An arm at 0</b> never reaches this precision anywhere on its curve, "
                "not at any threshold. That is a real result, not missing data.",
                "<b>Max precision</b> and <b>max recall</b> are the two ends of the same "
                "curve, so an arm with a high max precision and a zero bar is precise only "
                "where it has almost stopped calling.",
                excluded)),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_recall_at_precision_plot",
                    "title": f"Recall at precision >= {RECALL_AT_PRECISION:g} "
                             f"({primary_truth})",
                    "ylab": f"recall at precision >= {RECALL_AT_PRECISION:g} "
                            f"(mean over {n_species} species)",
                    "height": 450, "cpswitch": False, "use_legend": False,
                    "tt_decimals": 4},
        "categories": {"recall_at_p": {"name": f"recall at P >= {RECALL_AT_PRECISION:g}",
                                       "color": CLASSES["kmerseek"][1]}},
        "data": {k: {"recall_at_p": v["recall_at_p"]} for k, v in ordered.items()},
    })
    write_section(out, "qfo_recall_at_precision_table", {
        "id": "qfo_recall_at_precision_table",
        "section_name": f"Recall at precision >= {RECALL_AT_PRECISION:g}, with curve ends",
        "description": ("<p>The bar above as numbers, with the two ends of each arm's "
                        "pooled curve beside it.</p>"
                        + bullets(
                            "<b>Recall at P</b> is the bar.",
                            "<b>Max precision</b> is the highest precision anywhere on the "
                            "curve, which is usually at its strictest threshold.",
                            "<b>Max recall</b> is the reach at the most lenient threshold. "
                            "An arm whose max recall is far above its recall at P is "
                            "buying reach with errors.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_recall_at_precision_table_table",
                    "title": f"Recall at precision >= {RECALL_AT_PRECISION:g}",
                    "col1_header": "Arm", "sort_rows": False, "no_violin": True},
        "headers": {
            "recall_at_p": dict(title=f"Recall at P>={RECALL_AT_PRECISION:g}",
                                format="{:,.4f}", scale="Greens", min=0),
            "max_precision": dict(title="Max precision", format="{:,.3f}", scale="Blues",
                                  min=0, max=1),
            "max_recall": dict(title="Max recall", format="{:,.3f}", scale="Purples",
                               min=0, max=1),
        },
        "data": ordered,
    })


def section_curves(out: Path, curves: pl.DataFrame, metrics: pl.DataFrame,
                   primary_truth: str, max_lines: int) -> None:
    """Precision / recall and ROC, one line per tool at its best variant.

    Six lines by default rather than max_lines: the argument the PR panel makes is a shape
    difference -- kmerseek's curve is high, flat and short, phmmer's is long and decaying
    -- and that is not readable through fourteen overprinted lines. One kmerseek arm is
    drawn in colour and every baseline in grey with its own dash, so the comparison is
    between one thing and a background rather than between fourteen categorical colours.
    """
    if curves.height == 0:
        return
    cut, split = pick_split(curves.filter(pl.col("truth_set") == primary_truth))
    if cut.height == 0:
        return
    mcut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    board = best_variants(mcut)
    keep = curve_lines(board, max_lines)
    if not keep:
        return
    n_species = int(cut.filter(pl.col("species") != "all")["species"].n_unique())

    pr_pairs = {}
    for kind, xcol, ycol, xlab, ylab, title in (
        ("pr", "recall_reachable", "precision", "recall (reachable instances)",
         "precision", "Precision / recall"),
        ("roc", "fpr", "tpr", "false positive rate", "true positive rate", "ROC"),
    ):
        if xcol not in cut.columns or ycol not in cut.columns:
            continue
        data, colors, dashes = {}, {}, {}
        greys = cycle(["#8c8c8c", "#5c5c5c", "#b0b0b0", "#3f3f3f", "#9c9c9c"])
        grey_dashes = cycle(["solid", "dash", "dot", "dashdot", "longdash"])
        for tool, variant, label in keep:
            sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
            if sub.height == 0:
                continue
            pairs = pool_curve_over_species(sub, xcol, ycol)
            if not pairs:
                continue
            data[label] = curve_series(pairs)
            if kind == "pr":
                pr_pairs[label] = pairs
            if tool == "kmerseek":
                colors[label] = CLASSES["kmerseek"][1]
                dashes[label] = "solid"
            else:
                colors[label] = next(greys)
                dashes[label] = next(grey_dashes)
        if not data:
            continue
        # The axis fits the data rather than the metric's bound. Recall is bounded at 1 and
        # nowhere near it here: on a 0-to-1 axis every line in this panel collapses onto
        # the left edge, which is the same defect as a colour ramp pinned to 0-to-1 over a
        # grid that tops out at 0.28. Padded 10% and never past 1.
        xtop = max(float(x) for series in data.values() for x in series)
        xmax = min(1.0, max(xtop * 1.1, 0.02))
        write_section(out, f"qfo_{kind}_curve", {
            "id": f"qfo_{kind}_curve",
            "section_name": f"{title} curves",
            "description": (f"<p>{primary_truth} truth, <code>{split}</code> split. One "
                            f"kmerseek arm against the baselines, each at its best "
                            f"variant.</p>"
                            + bullets(
                                "<b>Recall</b> counts distinct true domain instances; "
                                "<b>precision</b> counts calls.",
                                f"<b>Both axes are means over the {n_species} target "
                                "proteomes at a matched score threshold.</b> Threshold is "
                                "the curve's own parameter and means the same thing in "
                                "every species of one arm, so every proteome is in the "
                                "mean at every point of the line. A proteome whose calls "
                                "have run out contributes recall 0 and stays in, rather "
                                "than dropping out and leaving the easiest proteomes "
                                "averaging with each other.",
                                "<b>Read the shape, not only the height.</b> An arm whose "
                                "curve is high and stops early is precise over a short "
                                "reach; one whose curve runs far to the right while "
                                "decaying is trading precision for reach. Those are "
                                "different failure modes and a single Fmax cannot tell "
                                "them apart.",
                                "<b>Colour</b> marks the kmerseek arm; every baseline is "
                                "grey with its own dash pattern. Six lines, not fourteen: "
                                "the panel is a comparison against a background.",
                                f"<b>The x axis stops at {xmax:.2f}</b>, which is where "
                                "the data stops, not at the 1 the metric is bounded by. On "
                                "a 0-to-1 axis every line here collapses onto the left "
                                "edge.",
                                "The per-species curves stay in "
                                "<code>all_domain_curves.parquet</code>.")),
            "plot_type": "linegraph",
            "pconfig": {"id": f"qfo_{kind}_curve_plot",
                        "title": f"{title} ({primary_truth})",
                        "xlab": xlab, "ylab": ylab, "xmin": 0, "xmax": xmax,
                        "ymin": 0, "ymax": 1, "height": 500,
                        "showlegend": True,
                        "colors": colors, "dash_styles": dashes},
            "data": data,
        })

    section_recall_at_precision(out, pr_pairs, cut, board.head(max_lines), primary_truth,
                                split, n_species)


def identity_single_bin_note(primary_truth: str, populated: list[str]) -> str:
    """Say what the identity axis has, instead of drawing a bar chart that is not it."""
    listed = ", ".join(f"<code>{b}</code>" for b in populated) or "no bin"
    note = (
        f"<p>Not plotted: only {listed} has data on the <code>{primary_truth}</code> "
        "truth set, so there is no identity gradient to read. A one-category grouped "
        "bargraph renders as a plain Fmax-per-tool chart carrying this section's title, "
        "which is a different figure than the title promises.</p>"
    )
    if populated == ["no_homolog"]:
        note += (
            "<p>Percent identity was measured; it cannot attach to this truth set. "
            "<code>attach_identity</code> joins the identity table on (accession, "
            "pfam_id, domain_start, domain_end), and on the Swiss-Prot truth set "
            "<code>pfam_id</code> holds a curated feature type (DOMAIN, TRANSMEM, "
            "ACT_SITE, ...) rather than a Pfam accession, so nothing matches and every "
            "instance falls into <code>no_homolog</code>. The Pfam truth set has all six "
            "bins populated in the same run &mdash; read the identity axis there, or "
            "pass <code>--primary-truth pfam</code>.</p>"
        )
    return note


def section_identity(out: Path, metrics: pl.DataFrame, primary_truth: str,
                     max_tools: int) -> None:
    """Fmax by percent-identity bin: the twilight-zone axis the claim is stated on."""
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    ident = cut.filter(pl.col("stratum_axis") == "identity")
    if ident.height == 0:
        return
    board = best_variants(ungrouped(cut)).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    order = [b for b in ["0-20%", "20-30%", "30-40%", "40-60%", "60-100%", "no_homolog"]
             if b in ident["stratum"].unique().to_list()]
    data = {}
    for tool, variant, label in keep:
        sub = ident.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        by_bin = sub.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()
        lookup = {r["stratum"]: r["fmax"] for r in by_bin}
        data[label] = {b: lookup.get(b) for b in order}
    if not data:
        return
    order = [b for b in order
             if any(series.get(b) is not None for series in data.values())]
    # Same guard, and the same reason, as MIN_COVARIATE_BINS in section_covariates: a
    # grouped bargraph with one category renders as a plain Fmax-per-tool bar chart
    # wearing this section's title, and identity is the axis the paper's claim is stated
    # on, so a figure that silently is not it is the worst one to leave standing.
    #
    # It fires on the Swiss-Prot truth set, which is the default primary. attach_identity
    # joins the identity table on (accession, pfam_id, domain_start, domain_end), and on
    # that truth set `pfam_id` holds a feature type rather than a Pfam accession, so the
    # join never matches and all 7_000 instances land in `no_homolog`. Identity WAS
    # computed -- the Pfam truth set has all six bins populated -- it just cannot attach
    # to a truth set keyed on something else.
    if len(order) < MIN_COVARIATE_BINS:
        write_section(out, "qfo_identity", {
            "id": "qfo_identity",
            "section_name": "Twilight zone",
            "description": (f"<p>{primary_truth} truth, <code>{split}</code> split.</p>"),
            "plot_type": "html",
            "data": identity_single_bin_note(primary_truth, order),
        })
        return
    data = {label: {b: series.get(b) for b in order} for label, series in data.items()}
    write_section(out, "qfo_identity", {
        "id": "qfo_identity",
        "section_name": "Twilight zone",
        "description": (
            f"<p>Fmax by percent identity between a domain instance and its closest "
            f"same-family domain in the target proteome ({primary_truth} truth, "
            f"<code>{split}</code> split).</p>"
            + bullets(
                "<b>Each bin</b> is a percent-identity band between the human domain "
                "instance and its closest same-family domain in the target proteome.",
                "<b><code>no_homolog</code></b> is instances with no same-family target "
                "domain at all: unreachable by transfer, kept as their own bin so they "
                "never contaminate the &lt;20% bin the hypothesis cares about.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_identity_plot", "title": "Fmax by percent identity",
                    "ylab": "Fmax", "cpswitch": False, "stacking": "group", "height": 500},
        "categories": {b: {"name": b, "color": IDENTITY_COLORS.get(b)} for b in order},
        "data": data,
    })


# The three covariate axes evaluate_domain_calls stratifies on that nothing rendered until
# now. Every metrics row already carries them -- build_query_covariates computes disorder
# from AlphaFold pLDDT and omega from the dN/dS pipeline, and STRATA bins all three -- so
# this was scored data being thrown away at report time.
#
# Disorder is the one to read first for this benchmark's claim. Every structure-based
# baseline needs a confident structure to work from, so their accuracy should fall as the
# disordered fraction rises. A sequence-only method has no such dependency, and whether
# that shows up as a flatter curve is exactly the kind of thing a reviewer asks for
# evidence of rather than assertion. It cuts both ways: notebook 202 found the HP-family
# alphabets have a real low-pLDDT dip where protein and dayhoff are flat, so this plot is
# equally capable of showing kmerseek looking worse.
COVARIATE_AXES = {
    "disorder": (
        "Disorder",
        bullets(
            "<b>x axis</b> is the fraction of the query's residues with pLDDT below 50, "
            "AlphaFold's standard disorder proxy.",
            "<b>Structure-based tools</b> need a confident structure, so their accuracy is "
            "expected to fall to the right. A sequence-only method has no such dependency.",
            "<b>MobiDB's curated annotation</b> is used instead when "
            "<code>--mobidb_cache</code> is set.",
        ),
    ),
    "disorder_seq": (
        "Disorder (sequence-based)",
        bullets(
            "<b>x axis</b> is the fraction of residues metapredict calls disordered from "
            "sequence alone.",
            "<b>Read this against the pLDDT disorder axis, not instead of it.</b> pLDDT "
            "below 50 also drops when AlphaFold merely modelled a protein badly, which "
            "usually means a shallow MSA, and a shallow MSA independently hurts the "
            "profile baselines.",
            "metapredict needs no structure and no alignment, so it shares neither "
            "confound. <b>The two axes disagreeing is itself a finding.</b>",
        ),
    ),
    "disorder_target": (
        "Disorder (target side)",
        bullets(
            "<b>x axis</b> is the disorder of the TARGET each human instance could best "
            "transfer from: the same closest same-family domain the identity axis uses, so "
            "the two describe the same target.",
            "<b>This is the half that bites a structure-based method.</b> foldseek and "
            "reseek align a structure to a structure, so a target with no confident "
            "structure defeats them however well-ordered the human query is.",
            "A sequence-only method has no such dependency on either side, and this is "
            "where that should show.",
        ),
    ),
    "plddt": (
        "Model confidence",
        bullets(
            "<b>x axis</b> is the query structure's mean pLDDT.",
            "<b>Distinct from the disorder axis</b>: a protein can be confidently "
            "modelled overall while carrying a disordered tail, and the two axes separate "
            "those cases.",
        ),
    ),
    "omega": (
        "Selective pressure",
        bullets(
            "<b>omega</b> is dN/dS for the human query gene, one value per gene, from the "
            "human-mouse-dnds-omega pipeline. Low omega is purifying selection, high omega "
            "is relaxed or positive selection.",
            "It is a property of the QUERY, not of the pair being searched: the same value "
            "labels a human protein's rows against every target species. Mouse is the only "
            "species dN/dS was computed against, because past roughly 300 MYA the "
            "synonymous sites are at the Jukes-Cantor ceiling and omega stops meaning "
            "anything.",
            "Coverage is the constraint. dN/dS exists for 1,335 human genes against ~19.4k "
            "query proteins, so this axis is a statement about that subset and nothing "
            "wider. A query set restricted to one chromosome leaves too few genes per bin "
            "for the axis to exist at all.",
        ),
    ),
}

# Why an axis can be present in the metrics and still say nothing, written per axis rather
# than inferred, because the reason is different each time and a reader deciding whether to
# re-run something needs the specific one.
COVARIATE_EMPTY_NOTE = {
    "omega": (
        "dN/dS covers 1,335 human genes. Only 71 of them are on chromosome 6, so the "
        "chr6 (midi) query set puts 17 / 31 / 23 / 0 proteins in the four omega bins and "
        "only one clears the per-stratum protein floor in "
        "<code>evaluate_domain_calls.py</code>. Whole-proteome query sets do not have "
        "this problem: genome-wide the same four bins hold 411 / 659 / 240 / 25 genes."
    ),
}


def _bin_order(values: list[str]) -> list[str]:
    """Sort bin labels by their lower edge.

    Derived from the labels themselves rather than from a copy of STRATA's edges, because
    a second copy of those numbers is a thing that drifts. attach_strata writes them as
    f"{lo}-{hi}", so the lower edge is parseable; anything unparseable sorts last rather
    than raising, since a missing bin should not take the whole report down.
    """
    def lo(label: str) -> float:
        try:
            return float(label.split("-")[0])
        except (ValueError, IndexError):
            return float("inf")
    return sorted(values, key=lo)


# A covariate axis is a claim about a gradient: Fmax is supposed to move as the covariate
# moves. One bin is not a gradient, and MultiQC will not refuse to draw it -- a grouped
# bargraph built from a single category renders as one plain bar per series, so what comes
# out is an Fmax-per-tool bar chart wearing the covariate's title with the covariate itself
# nowhere in the figure. That is exactly what the omega axis produced on the chr6 (midi)
# query set: 19 bars, no omega axis, and a caption confidently describing selective
# pressure. A section that silently turns into a different plot is worse than one that says
# it has no data, so below this many populated bins the section writes the note instead.
MIN_COVARIATE_BINS = 2


def empty_covariate_note(axis: str, populated: list[str],
                         sub_axis: pl.DataFrame) -> str:
    """Say what this axis has instead of drawing a chart that cannot show it."""
    if populated:
        n = sub_axis.filter(pl.col("stratum").is_in(populated))
        counts = (
            n.group_by("stratum").agg(pl.col("n_stratum_proteins").max())
            .to_dicts() if "n_stratum_proteins" in sub_axis.columns else []
        )
        by_bin = {r["stratum"]: r["n_stratum_proteins"] for r in counts}
        listed = ", ".join(
            f"<code>{b}</code>" + (f" ({by_bin[b]} proteins)" if by_bin.get(b) else "")
            for b in populated
        )
        head = (
            f"<p>Not plotted: only one <code>{axis}</code> bin has data in this run "
            f"({listed}), so there is no gradient to read. Plotting it would draw a plain "
            "Fmax-per-tool bar chart carrying this section's title, which is a different "
            "figure than the one the title promises.</p>"
        )
    else:
        head = (
            f"<p>Not plotted: this run scored the <code>{axis}</code> axis but no bin came "
            "back with a number for any tool.</p>"
        )
    why = COVARIATE_EMPTY_NOTE.get(axis)
    return head + (f"<p>{why}</p>" if why else "")


def section_covariates(out: Path, metrics: pl.DataFrame, primary_truth: str,
                       max_tools: int) -> None:
    """Fmax against disorder, model confidence and selective pressure."""
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    board = best_variants(ungrouped(cut)).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]

    for axis, (title, blurb) in COVARIATE_AXES.items():
        sub_axis = cut.filter(pl.col("stratum_axis") == axis)
        # The axis was never scored in this run at all -- no structures, no metapredict, no
        # dN/dS file. Nothing to report and nothing to explain, so the section is absent
        # rather than present-and-empty. Different case from the one below, where the axis
        # WAS scored and still cannot be read.
        if sub_axis.height == 0:
            continue
        order = _bin_order(sub_axis["stratum"].drop_nulls().unique().to_list())
        data = {}
        for tool, variant, label in keep:
            sub = sub_axis.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
            if sub.height == 0:
                continue
            by_bin = sub.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()
            lookup = {r["stratum"]: r["fmax"] for r in by_bin}
            data[label] = {b: lookup.get(b) for b in order}

        # Bins that no tool has a number in are not bins; carrying them into `categories`
        # would draw empty slots next to the real ones and invite the reader to read a
        # gap as a zero.
        order = [b for b in order
                 if any(series.get(b) is not None for series in data.values())]
        if len(order) < MIN_COVARIATE_BINS:
            write_section(out, f"qfo_{axis}", {
                "id": f"qfo_{axis}",
                "section_name": title,
                "description": (f"<p>{primary_truth} truth, <code>{split}</code> "
                                f"split.</p>" + blurb),
                "plot_type": "html",
                "data": empty_covariate_note(axis, order, sub_axis),
            })
            continue
        data = {label: {b: series[b] for b in order} for label, series in data.items()}
        write_section(out, f"qfo_{axis}", {
            "id": f"qfo_{axis}",
            "section_name": title,
            "description": (f"<p>{primary_truth} truth, <code>{split}</code> split.</p>"
                            + blurb),
            "plot_type": "bargraph",
            "pconfig": {"id": f"qfo_{axis}_plot", "title": f"Fmax by {axis}",
                        "ylab": "Fmax", "cpswitch": False, "stacking": "group",
                        "height": 500, "showlegend": True},
            "categories": {b: {"name": b} for b in order},
            "data": data,
        })


# --- the disorder axis, drawn as dots at measured disorder ---------------------------
#
# What the four-bar Disorder section above cannot do, and why this is not just that plot
# with more bars.
#
# THE Y AXIS IS NOT PER PROTEIN, AND CANNOT BE. Fmax is a threshold-optimised, protein-
# macro-averaged F: cafa_metrics.protein_centric_curve sweeps a score threshold, averages
# precision and recall over the proteins in the cut at each one, and takes the best F over
# the sweep. It is defined over a POPULATION of queries and has no value for a single
# protein. One dot per query protein labelled "Fmax" would be a fabricated statistic.
#
# The per-protein alternatives were measured rather than assumed, on the real midi-plus
# calls (foldseek 3di_aa against mouse, Pfam truth, at that arm's own Fmax threshold of
# 359.0): the protein's own F takes the value 0 on 472 of 998 queries and 1.0 on another
# 110, so 58% of the dots would sit on two horizontal lines and four distinct values would
# account for 65% of them. That is what a per-query scatter of this benchmark looks like,
# and the reason is in the answer key -- 492 of the 998 human queries carry exactly one
# annotated Pfam instance and the median is two, so per-query recall is a coin flip with
# no room in between. The dense figure has to come from more populations, not from
# smaller ones.
#
# So: more cuts, each still a population, each drawn where its proteins actually are.
# STRATA["disorder_fine"] in evaluate_domain_calls.py cuts the query set fourteen ways
# instead of four, and every metrics row carries `stratum_value_mean` -- the mean disorder
# of the proteins in that cell -- which is what this plots as x. Not the bin midpoint: on
# the widest cell those differ by 0.11.
#
# Falls back to the coarse four-bin `disorder` axis, and to the bin midpoint, on a results
# tree scored before either existed, and says in the description which it drew.

# How many kmerseek arms this section draws. The covariate bargraph takes --max-tools
# (19 on the midi run), and at that width the legend is taller than the plot and the
# fourteen kmerseek arms are fourteen lines of the same green: the figure stops being
# readable as the comparison it exists to make. Every baseline still contributes its one
# variant, so the trim only narrows the sweep block, and the pinned --canonical-variant
# arm still rides along through best_variants.
DISORDER_SCATTER_TOP_KMERSEEK = 2


def _fine_disorder_axis(cut: pl.DataFrame) -> tuple[str, pl.DataFrame] | None:
    """The finest disorder cut this results tree actually holds, with its rows."""
    for axis in ("disorder_fine", "disorder"):
        sub = cut.filter(pl.col("stratum_axis") == axis)
        if sub.height:
            return axis, sub
    return None


def _bin_x(rows: pl.DataFrame, label: str) -> float | None:
    """Where a bin goes on the disorder axis: measured mean if scored, else midpoint.

    The mean is a property of the query set, so every arm and every species in a cell
    reports the same number; taking the mean over those rows is only a way to ignore the
    nulls a partially-rescored tree would carry.
    """
    if "stratum_value_mean" in rows.columns:
        measured = rows["stratum_value_mean"].drop_nulls()
        if measured.len():
            return float(measured.mean())
    return band_midpoint(label)


# Two points are a slope. This section's whole claim is that the slope is readable at a
# resolution the four-bin plot does not have, so below this it has nothing the covariate
# bargraph does not already say and writes nothing.
MIN_DISORDER_POINTS = 4


def section_disorder_scatter(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """Fmax against measured disorder, one dot per cut, for a legible set of arms."""
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    found = _fine_disorder_axis(cut)
    if found is None:
        return
    axis, sub = found

    # best_variants takes the UNION of the top combos under every ranking metric, so
    # top_kmerseek=2 comes back with four kmerseek arms, not two -- and on the midi run
    # they arrive as two near-duplicate pairs (gbmr4_k21 with and without the
    # low-complexity filter differ by 0.0005 Fmax) drawn in the same green. Trimmed here
    # rather than by lowering top_kmerseek further, because the union is what keeps a
    # sensitivity winner in every other section and this is the only place it hurts.
    board = best_variants(ungrouped(cut), top_kmerseek=DISORDER_SCATTER_TOP_KMERSEEK)
    keep, combos = [], []
    for r in board.to_dicts():
        if r["tool"] == "kmerseek" and not is_canonical(r["tool"], r["variant"]):
            # Deduped on the alphabet x ksize combo, ignoring the low-complexity arm. The
            # two lc arms of one combo are the same point twice: on the midi run the top
            # two kmerseek arms by Fmax are gbmr4_k21_lcTrue and gbmr4_k21_lcFalse, 0.0005
            # apart and drawn in the same green, so spending both slots on them would trim
            # the legend without making the figure any more readable. The low-complexity
            # comparison has its own section.
            combo = r["variant"].rpartition("_lc")[0] or r["variant"]
            if combo in combos:
                continue
            if len(combos) >= DISORDER_SCATTER_TOP_KMERSEEK:
                continue
            combos.append(combo)
        keep.append((r["tool"], r["variant"], r["label"]))
    if not keep:
        return

    labels = _bin_order(sub["stratum"].drop_nulls().unique().to_list())
    bins = []
    for label in labels:
        rows = sub.filter(pl.col("stratum") == label)
        x = _bin_x(rows, label)
        if x is None:
            continue
        bins.append({
            "label": label, "x": x,
            "proteins": (int(rows["n_stratum_proteins"].max())
                         if "n_stratum_proteins" in rows.columns
                            and rows["n_stratum_proteins"].drop_nulls().len() else None),
            "instances": (int(rows["n_truth_instances"].max())
                          if "n_truth_instances" in rows.columns
                             and rows["n_truth_instances"].drop_nulls().len() else None),
            "measured": ("stratum_value_mean" in rows.columns
                         and rows["stratum_value_mean"].drop_nulls().len() > 0),
        })
    if len(bins) < MIN_DISORDER_POINTS:
        return

    # All measured or none. A tree where only some arms were rescored has a measured mean
    # on some cells and nothing on others, and mixing the two would draw part of the axis
    # at measured disorder and part at bin midpoints under a single label saying one of
    # them -- an axis whose meaning changes along its own length. The midpoint is the
    # reading that exists for every cell, so a partial tree gets midpoints throughout and
    # the description says so.
    measured_x = all(b["measured"] for b in bins)
    if not measured_x:
        for b in bins:
            b["x"] = band_midpoint(b["label"])
        bins = [b for b in bins if b["x"] is not None]
        if len(bins) < MIN_DISORDER_POINTS:
            return

    # A dict of series, not a list of points: custom content reads a top-level list as a
    # list of DATASETS and dies in its numeric-x coercion. See section_ceiling_cardinality.
    points: dict[str, list[dict]] = {}
    slopes = {}
    for tool, variant, label in keep:
        arm = sub.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if arm.height == 0:
            continue
        by_bin = {r["stratum"]: r["fmax"]
                  for r in arm.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()}
        series = []
        drawn = []
        for b in bins:
            y = by_bin.get(b["label"])
            if y is None:
                continue
            series.append(scatter_point(
                round(b["x"], 4), y, name=f"{label} @ disorder {b['x']:.3f}",
                group=label, color=tool_color(tool),
                n_proteins=b["proteins"], stratum=b["label"]))
            drawn.append((b["x"], y))
        if len(drawn) < MIN_DISORDER_POINTS:
            continue
        points[label] = series
        drawn.sort()
        # Ordered half against disordered half, not first cut against last. The endpoint
        # cells are the two smallest populations on the axis and the most-ordered cut runs
        # high for every arm in this benchmark, so a first-minus-last reading called every
        # arm falling, kmerseek included, when its middle cuts are its best ones. Halves
        # are still a description of these fourteen numbers rather than a fitted trend.
        half = len(drawn) // 2
        lo_half = sum(y for _, y in drawn[:half]) / half
        hi_half = sum(y for _, y in drawn[-half:]) / half
        slopes[label] = hi_half - lo_half
    if not points:
        return

    n_bins = len(bins)
    smallest = min((b["proteins"] for b in bins if b["proteins"] is not None), default=None)
    falling = sorted(lb for lb, d in slopes.items() if d < 0)
    rising = sorted(lb for lb, d in slopes.items() if d >= 0)

    def listed(labels_):
        return ", ".join(f"<code>{lb}</code>" for lb in labels_) or "none"

    x_bullet = (
        "<b>x is the measured mean disorder of the query proteins in that cut</b>, not "
        "the cut's midpoint. The two agree to within 0.006 on the narrow cells and part "
        "company at the wide end, where the top cell's midpoint (0.805) sits 0.092 above "
        "the mean disorder of the proteins in it (0.713)."
        if measured_x else
        "<b>x is the cut's midpoint</b>, because this results tree was scored before "
        "<code>stratum_value_mean</code> existed. Re-score to put each point at the "
        "measured mean of its proteins instead; on the coarse axis the widest cell's "
        "midpoint (0.805) is 0.092 above the mean disorder of what is in it (0.713).")
    axis_bullet = (
        f"<b>{n_bins} cuts of the query set</b>, from "
        f"<code>STRATA[\"disorder_fine\"]</code>."
        if axis == "disorder_fine" else
        f"<b>{n_bins} cuts of the query set</b>, from the coarse four-bin "
        "<code>disorder</code> axis — this results tree has no <code>disorder_fine</code> "
        "rows. Re-score to get the fourteen-cut version this section is built for.")

    write_section(out, "qfo_disorder_scatter", {
        "id": "qfo_disorder_scatter",
        "section_name": "Fmax against disorder",
        "description": (
            f"<p>Fmax against the fraction of the query's residues with pLDDT below 50 "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>One dot is one cut of the query set, not one protein.</b> Its y is "
                "the Fmax of the arm over the proteins in that cut: the score threshold "
                "is swept, precision and recall are averaged over those proteins at each "
                "threshold, and the best F over the sweep is the dot. Fmax is a "
                "population statistic and has no per-protein value, so there is no "
                "honest way to put one dot per query protein on this axis.",
                axis_bullet,
                x_bullet,
                (f"<b>The smallest cut holds {smallest} query proteins</b> in this run. "
                 "Every cut clears the per-stratum protein floor in "
                 "<code>evaluate_domain_calls.py</code>, which is what stops the "
                 "resolution being bought with cells too small to score."
                 if smallest is not None else ""),
                (f"<b>Lower on the disordered half of the axis</b>: {listed(falling)}. "
                 f"<b>Level or higher</b>: {listed(rising)}. Each arm's mean Fmax over "
                 "the more disordered half of the cuts against its mean over the more "
                 "ordered half — a description of these points, not a fitted trend and "
                 "not a test. Halves rather than the two end cuts, because those are the "
                 "smallest populations on the axis."),
                f"<b>Arms are trimmed to every baseline plus kmerseek's top "
                f"{DISORDER_SCATTER_TOP_KMERSEEK} by Fmax</b>, rather than the "
                "<code>--max-tools</code> set the four-bin Disorder section draws. That "
                "set is mostly sweep combos, they are all drawn in the same green, and "
                "several are near-duplicates of each other — the legend ends up taller "
                "than the figure and says less. <b>The whole sweep is in the Disorder "
                "section</b>; the table under this one carries only what is drawn here.",
                "<b>Colour is the method class</b>, as everywhere else in this report.",
                "<b>No error bars, here or anywhere in this report.</b> Each dot is a "
                "mean over target proteomes with no resampling behind it, and the cuts "
                "hold a few dozen proteins each, so a gap narrower than the scatter "
                "between neighbouring cuts of the same arm is not a result.")),
        "plot_type": "scatter",
        "pconfig": {"id": "qfo_disorder_scatter_plot",
                    "title": "Fmax vs disorder",
                    "xlab": ("mean disorder of the proteins in the cut "
                             "(fraction of residues with pLDDT < 50)" if measured_x
                             else "disorder (cut midpoint)"),
                    "ylab": "Fmax over the proteins in the cut",
                    "xmin": 0, "xmax": 1, "ymin": 0, "height": 560,
                    "showlegend": True},
        "data": points,
    })

    table = {}
    headers = {"x": {"title": "Disorder", "format": "{:,.3f}",
                     "description": ("Measured mean disorder of the proteins in the cut"
                                     if measured_x else "Cut midpoint")},
               "proteins": {"title": "Proteins", "format": "{:,d}",
                            "description": "Query proteins in the cut"},
               "instances": {"title": "Instances", "format": "{:,d}",
                             "description": "Annotated domain instances in the cut"}}
    for b in bins:
        row = {"x": b["x"], "proteins": b["proteins"], "instances": b["instances"]}
        for label, series in points.items():
            hit = next((p for p in series if p["stratum"] == b["label"]), None)
            row[label] = hit["y"] if hit else None
            headers.setdefault(label, {"title": label, "format": "{:,.3f}",
                                       "scale": "RdYlGn", "min": 0, "max": 1,
                                       "description": f"Fmax for {label} in this cut"})
        table[b["label"]] = row
    write_section(out, "qfo_disorder_scatter_table", {
        "id": "qfo_disorder_scatter_table",
        "section_name": "Fmax against disorder, per cut",
        "description": (
            f"<p>The numbers behind the scatter above: every cut's disorder, its "
            f"denominators, and each drawn arm's Fmax over it ({primary_truth} truth, "
            f"<code>{split}</code> split).</p>"
            + bullets(
                "<b>Proteins and instances are the denominators</b> the cut's Fmax was "
                "computed over. They are the only thing a reader has to judge a gap by, "
                "since nothing in this report carries a sampling error.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_disorder_scatter_table_plot",
                    "title": "Fmax by disorder cut", "col1_header": "Disorder cut",
                    "sort_rows": False, "no_violin": True},
        "headers": headers,
        "data": table,
    })


# Families ranked by the gap between kmerseek and the best baseline, not by size. "The
# twenty biggest families" answers a question nobody asked; "where does this method win and
# where does it lose" is the one a reader has.
#
# This is NOT a confound check, which is the usual reason to look at family identity.
# Notebook 206 excluded C2H2 zinc fingers because tandem arrays inflate PROTEIN-level
# k-mer sharing through repeat content, which is real when the scored object is a protein
# pair. Here it is a domain instance -- a twelve-finger protein contains twelve domains and
# the right answer is twelve correctly-bounded regions -- so that exclusion belonged to a
# different unit of analysis and attach_strata keeps them in. This section answers the
# other question: which families each method is actually good and bad at.
# The curated cuts evaluate_domain_calls already scores and the report has never read.
# `stratum_mhc` is the 7 MHC classes from bin/gene_sets.py, `stratum_geneset` the curated
# sets on one shared axis. Both are in UNFLOORED_AXES, so their protein floor is 1 and a
# 6-gene class survives scoring -- and then went nowhere, because nothing here asked for
# them. The numbers have been in all_domain_metrics.parquet the whole time.
CURATED_AXES = {
    "mhc": ("MHC classes",
            "The 25 MHC genes notebooks 210-216 score, grouped into the 7 classes those "
            "notebooks report by. The class split is not cosmetic: notebook 211 found "
            "class I and class II answer the k-size question in opposite directions, so a "
            "single pooled MHC number hides the result."),
    "geneset": ("Curated gene sets",
                "Each curated set from bin/gene_sets.py as its own stratum. Non-members "
                "are null rather than an \"everything else\" cell, so a set is compared "
                "against the report's other cuts and not against its own complement."),
}


def section_curated_sets(out: Path, metrics: pl.DataFrame, max_tools: int) -> None:
    """MHC classes and the curated gene sets, per truth set.

    These axes carry the vignette the report otherwise has no trace of. A reader of the
    report alone cannot currently see the MHC result at all, and a reader of the MHC
    notebooks cannot see it in the same units as every baseline. One section fixes both,
    and it needs no new computation -- only the columns the scoring stage already writes.

    Per truth set and never pooled, like everything else here. Small n is the whole point
    of these cuts rather than a defect, so the section states the protein count beside
    each bar instead of filtering on it.
    """
    for axis, (title, why) in CURATED_AXES.items():
        sub_all = metrics.filter(pl.col("stratum_axis") == axis)
        if sub_all.height == 0:
            continue
        for ts in sorted(sub_all["truth_set"].unique().to_list()):
            cut, split = pick_split(
                metrics.filter(pl.col("truth_set") == ts))
            sub = cut.filter(pl.col("stratum_axis") == axis)
            if sub.height == 0:
                continue
            board = best_variants(ungrouped(cut)).head(max_tools)
            keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
            if not keep:
                continue
            sub = sub.with_columns(label_column())

            order = sorted(sub["stratum"].drop_nulls().unique().to_list())
            n_by = {}
            if "n_stratum_proteins" in sub.columns:
                n_by = {r["stratum"]: r["n_stratum_proteins"] for r in
                        sub.group_by("stratum")
                           .agg(pl.col("n_stratum_proteins").max()).to_dicts()}
            data = {}
            for _, _, label in keep:
                one = sub.filter(pl.col("label") == label)
                if one.height == 0:
                    continue
                by_bin = {r["stratum"]: r["fmax"] for r in
                          one.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()}
                data[label] = {b: by_bin.get(b) for b in order}
            order = [b for b in order
                     if any(series.get(b) is not None for series in data.values())]
            if not order or not data:
                continue
            data = {k: {b: v[b] for b in order} for k, v in data.items()}

            write_section(out, f"qfo_curated_{axis}_{ts}", {
                "id": f"qfo_curated_{axis}_{ts}",
                "section_name": f"{title} — {ts} truth",
                "description": (
                    f"<p>Fmax per {axis} stratum, each tool at its best variant, averaged "
                    f"over target species ({ts} truth, <code>{split}</code> split).</p>"
                    + bullets(
                        why,
                        "<b>n per stratum</b> — " + ", ".join(
                            f"<code>{b}</code> {n_by[b]}" for b in order if b in n_by)
                        + " proteins." if n_by else "",
                        "<b>These cuts are small on purpose</b>, so the per-stratum "
                        "protein count is printed above rather than used to filter. A bar "
                        "over 6 proteins is a vignette and must not carry a headline "
                        "number; notebook 215 is explicit that the six class I heavy "
                        "chains are three independent lineages, so read that cut as n=3.",
                        "<b>Nothing here is new computation.</b> evaluate_domain_calls "
                        "has always written these strata with the protein floor waived "
                        "(UNFLOORED_AXES); the report simply never read them.")),
                "plot_type": "bargraph",
                "pconfig": {"id": f"qfo_curated_{axis}_{ts}_plot",
                            "title": f"Fmax by {axis} ({ts})", "ylab": "Fmax",
                            "cpswitch": False, "stacking": "group", "height": 500,
                            "showlegend": True},
                "categories": {b: {"name": f"{b} (n={n_by[b]})" if b in n_by else b}
                               for b in order},
                "data": data,
            }, supplementary=not is_primary_truth(ts))


def section_search_space(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """Accuracy against how much each arm said, which is the confound nothing else here
    separates.

    The Disk I/O section states that HP alphabets at low k produce enormous match volume
    by design, because the p-value filter is deliberately lenient so Bonferroni correction
    can happen downstream. That makes "HP detects more" and "HP reports more" the same
    observation in every recall-shaped panel. Fmax partly controls for it -- precision is
    in there -- but nothing in the report lets a reader see the call volume an arm needed
    to reach its Fmax, which is the first thing an unfriendly reviewer will ask for.

    n_calls and n_gray_calls are already on every metrics row, so this is a plot of
    columns that exist rather than a new measurement.
    """
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    cut = ungrouped(cut)
    if cut.height == 0 or "n_calls" not in cut.columns:
        return
    board = best_variants(cut)
    if board.height == 0:
        return
    agg = (cut.with_columns(label_column())
              .group_by("tool", "variant", "label")
              .agg(pl.col("n_calls").sum().alias("calls"),
                   pl.col("n_gray_calls").sum().alias("gray"),
                   pl.col("fmax").mean().alias("fmax"),
                   pl.col("precision").mean().alias("precision")))
    wanted = {(r["tool"], r["variant"]) for r in board.to_dicts()}
    # Keyed by label, not a list. MultiQC's custom content reads a top-level LIST as a list
    # of DATASETS, then raises TypeError coercing a numeric x inside custom_content.py --
    # and its except clause catches only ValueError, so the whole custom_content module
    # aborts and NO section in the report renders. Every other scatter here is a dict for
    # this reason; this one was not, and it took the entire report down with it.
    points = {}
    for r in agg.to_dicts():
        if (r["tool"], r["variant"]) not in wanted:
            continue
        if not r["calls"] or r["fmax"] is None:
            continue
        gray_frac = (r["gray"] / r["calls"]) if r["gray"] is not None else None
        point = scatter_point(
            r["calls"], r["fmax"], name=r["label"],
            group=CLASSES[tool_class(r["tool"])][0],
            color=tool_color(r["tool"]),
            marker_symbol=TOOL_SYMBOL.get(r["tool"], "circle"),
            annotation=r["label"] if agg.height < ANNOTATE_EVERY_POINT_BELOW else None)
        if gray_frac is not None:
            point["gray_fraction"] = round(gray_frac, 4)
        points[r["label"]] = point
    if not points:
        return
    write_section(out, "qfo_search_space", {
        "id": "qfo_search_space",
        "section_name": "Accuracy against how much was said",
        "description": (
            f"<p>Each arm's Fmax against the total number of calls it reported "
            f"({primary_truth} truth, <code>{split}</code> split, summed over target "
            f"species).</p>"
            + bullets(
                "<b>x is call volume, not a score.</b> An arm far to the right reached its "
                "Fmax by reporting more, and one to the upper LEFT reached the same Fmax "
                "by reporting less. Two arms level on y at opposite ends of x are not the "
                "same result.",
                "<b>This is the confound the rest of the report cannot separate.</b> The "
                "p-value filter is left lenient on purpose so Bonferroni correction can "
                "happen downstream, and HP alphabets at low k generate enormous match "
                "volume by design — the Disk I/O section is where that cost shows up. "
                "Every recall-shaped panel therefore reads \"detects more\" and "
                "\"reports more\" identically.",
                "<b>Fmax already carries precision</b>, so this is not a correction to it. "
                "It is the axis that says whether an alphabet's advantage survives being "
                "charged for its search space, which a reviewer will ask before accepting "
                "the headline alphabet result.",
                "<b>Colour</b> marks the method class and <b>shape</b> the individual "
                "tool, as in the frontier plot. Hover carries each point's gray-call "
                "fraction: a high one means much of the extra volume landed where the "
                "annotation says nothing, which is neither a hit nor a charged error.")),
        "plot_type": "scatter",
        "pconfig": {"id": "qfo_search_space_plot",
                    "title": f"Fmax against calls reported ({primary_truth})",
                    "xlab": "calls reported (log scale)", "ylab": "Fmax",
                    "xlog": True, "height": 560, "ymin": 0,
                    "xsuffix": "", "ysuffix": "", "showlegend": True},
        "data": points,
    })


def section_hgnc(out: Path, metrics: pl.DataFrame, primary_truth: str,
                 min_instances: int, top_n: int) -> None:
    """Where kmerseek beats the best baseline by family, and where it loses."""
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    fam = cut.filter(pl.col("stratum_axis") == "hgnc")
    if fam.height == 0:
        return
    # The row key this section ranks on. Added here rather than assumed: `cut` is the raw
    # metrics table, and only the leaderboard path passes through best_variants.
    fam = fam.with_columns(label_column())

    # Small families rank on noise: one instance found or missed swings Fmax a long way,
    # and the tail of HGNC groups is mostly singletons. Filtered on the ANSWER KEY's size
    # rather than on how many any tool found, so the threshold cannot depend on the tools
    # being compared.
    if "n_truth_instances" in fam.columns:
        sizes = (fam.group_by("stratum")
                    .agg(pl.col("n_truth_instances").max().alias("n_inst"))
                    .filter(pl.col("n_inst") >= min_instances))
        fam = fam.join(sizes, on="stratum", how="inner")
    else:
        fam = fam.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_inst"))

    is_ks = pl.col("tool") == "kmerseek"
    ks = fam.filter(is_ks).group_by("stratum").agg(pl.col("fmax").max().alias("ks_fmax"))
    bl = (fam.filter(~is_ks).sort("fmax", descending=True, nulls_last=True)
             .group_by("stratum")
             .agg(pl.col("fmax").first().alias("bl_fmax"),
                  pl.col("label").first().alias("bl_label")))
    joined = (ks.join(bl, on="stratum", how="inner")
                .join(fam.group_by("stratum").agg(pl.col("n_inst").max()),
                      on="stratum", how="left")
                .with_columns((pl.col("ks_fmax") - pl.col("bl_fmax")).alias("gap"))
                .sort("gap", descending=True, nulls_last=True))
    if joined.height == 0:
        return

    # Both ends, not one. A section that showed only the wins would be marketing.
    head = joined.head(top_n)
    tail = joined.tail(top_n).filter(~pl.col("stratum").is_in(head["stratum"].to_list()))
    rows = {}
    for r in head.to_dicts() + tail.sort("gap").to_dicts():
        rows[r["stratum"]] = {"instances": r["n_inst"], "kmerseek": r["ks_fmax"],
                              "baseline": r["bl_fmax"], "best_baseline": r["bl_label"],
                              "gap": r["gap"]}
    write_section(out, "qfo_hgnc", {
        "id": "qfo_hgnc",
        "section_name": "Families won and lost",
        "description": (
            f"<p>Best kmerseek variant against the best non-kmerseek tool, per HGNC gene "
            f"group ({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>Gap</b> is kmerseek minus the best baseline, so positive is kmerseek "
                "ahead.",
                f"<b>Rows shown</b> are the {top_n} largest gaps in each direction.",
                f"<b>Families included</b> carry at least {min_instances} instances in the "
                "answer key. Below that, one instance found or missed is enough to rank a "
                "family on noise.",
                "<b>C2H2 zinc fingers are included.</b> The repeat-content confound that "
                "excludes them elsewhere is about protein-level k-mer sharing, and the "
                "scored object here is the domain instance.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_hgnc_table", "title": "kmerseek minus best baseline, by family",
                    "col1_header": "HGNC gene group", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "instances": dict(title="Instances", format="{:,.0f}", scale="Blues"),
            "kmerseek": dict(title="kmerseek Fmax", format="{:.3f}", scale="Greens", min=0, max=1),
            "baseline": dict(title="Best baseline Fmax", format="{:.3f}", scale="Purples", min=0, max=1),
            "best_baseline": dict(title="Which baseline", scale=False),
            "gap": dict(title="Gap", format="{:+.3f}", scale="RdYlGn"),
        },
        "data": rows,
    })


# The alphabet the internal control is drawn from. protein20 is the identity encoding --
# no reduction at all -- so an arm built on it isolates what the reduction is doing, on one
# engine with nothing else varying.
CONTROL_ALPHABET = "protein20"

# Greys the de-emphasised series cycle through, paired with dash patterns so a static
# export can still tell two of them apart. Colour is never the only channel.
EMPHASIS_GREYS = ["#8c8c8c", "#5c5c5c", "#b0b0b0", "#3f3f3f", "#9c9c9c", "#6e6e6e"]
EMPHASIS_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def emphasis_colors(labels: list[str], *, lead: str | None,
                    control: str | None = None) -> tuple[dict, dict]:
    """One series in colour, one control in a second colour, everything else grey.

    Nine categorical colours is a legend the reader has to learn before they can read the
    figure, and on this report's palette two of the nine were nearly the same hue. What a
    divergence panel is read for is one line against a background, so that is what it is
    drawn as.
    """
    colors, dashes = {}, {}
    greys, dash_cycle = cycle(EMPHASIS_GREYS), cycle(EMPHASIS_DASHES)
    for label in labels:
        if label == lead:
            colors[label], dashes[label] = CLASSES["kmerseek"][1], "solid"
        elif label == control:
            colors[label], dashes[label] = "#d55e00", "dash"
        else:
            colors[label], dashes[label] = next(greys), next(dash_cycle)
    return colors, dashes


def _divergence_control(cut: pl.DataFrame,
                        already: set[tuple[str, str]]) -> tuple[str, str, str, bool]:
    """The best-Fmax protein20 kmerseek arm, and whether it still has to be added.

    An arm already on the board is the control; picking a DIFFERENT protein20 arm because
    the obvious one was taken would silently draw a k nobody would run, which is exactly
    the failure this control exists to avoid. So the same row is reused and only its colour
    changes.
    """
    km = cut.filter(pl.col("tool") == "kmerseek")
    if km.height == 0:
        return ()
    parsed = parse_kmerseek_variants(km)
    ctrl = parsed.filter(pl.col("alphabet") == CONTROL_ALPHABET)
    if ctrl.height == 0:
        return ()
    ranked = (ctrl.group_by("variant").agg(pl.col("fmax").mean())
                  .sort("fmax", descending=True, nulls_last=True))
    if ranked.height == 0:
        return ()
    variant = ranked.to_dicts()[0]["variant"]
    return ("kmerseek", variant, label_of("kmerseek", variant),
            ("kmerseek", variant) not in already)


def _panel_ymax(panel: dict) -> float:
    """Top of a divergence panel's axis, from the panel's own values.

    A fixed ymax of 1 was squashing Fmax into the bottom fifth of the plot, which is where
    the per-species spread lives. The panels are on different scales -- Fmax tops out near
    0.2 while reachable recall reaches 0.6 -- so each gets its own limit rather than one
    shared limit sized by whichever panel happens to be largest. Padded 15%, and never
    below a small floor so an all-zero panel still draws an axis.
    """
    values = [v for series in panel.values() for v in series.values() if v is not None]
    return max(max(values, default=0.0) * 1.15, 0.05)


def section_divergence(out: Path, metrics: pl.DataFrame, primary_truth: str,
                       max_tools: int) -> None:
    """Both headline metrics against divergence time. The species IS the divergence axis.

    This is the per-species view the leaderboard's means are taken over: nine points per
    line, one per target proteome. The leaderboard carries the spread as SD and range;
    this carries its shape, ordered on the axis that explains it. Sensitivity to the first
    false positive gets a panel here too, because it disagrees with Fmax on which variants
    are best and a reader comparing them needs both against the same axis.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cut = cut.filter(pl.col("species") != "all")
    if cut.height == 0 or "species_mya" not in cut.columns:
        return
    board = best_variants(cut).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    # The internal control, forced in whether or not it ranks. protein20 is the same engine,
    # the same index and the same query set as every other kmerseek arm with nothing varying
    # but how many amino-acid classes the alphabet keeps, so the gap between it and the
    # coarse arms at the far proteome has no tool choice in it to attribute to. It does not
    # win any board -- that is the point of it -- so best_variants never keeps it.
    control = _divergence_control(cut, {(t, v) for t, v, _ in keep})
    if control and control[3]:
        keep.append(control[:3])
    has_sens = "sens_first_fp_mean" in cut.columns
    data, recall, sens = {}, {}, {}
    for tool, variant, label in keep:
        aggs = [pl.col("fmax").mean(), pl.col("recall").mean(),
                pl.col("recall_reachable").mean()]
        if has_sens:
            aggs.append(pl.col("sens_first_fp_mean").mean())
        sub = (cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
                  .group_by("species_mya").agg(aggs).sort("species_mya"))
        data[label] = {str(r["species_mya"]): r["fmax"] for r in sub.to_dicts()}
        recall[label] = {str(r["species_mya"]): r["recall_reachable"]
                         for r in sub.to_dicts()}
        if has_sens:
            sens[label] = {str(r["species_mya"]): r["sens_first_fp_mean"]
                           for r in sub.to_dicts()}
    panels = [("Fmax", "Fmax", data), ("Recall (reachable)", "recall_reachable", recall)]
    if has_sens:
        panels.append(("Sens. to 1st FP", "sens_first_fp_mean", sens))
    lead = next((lbl for t, _, lbl in keep if t == "kmerseek"), None)
    colors, dashes = emphasis_colors(
        [lbl for _, _, lbl in keep], lead=lead,
        control=control[2] if control else None)
    emphasis = (
        f"<b>Colour is emphasis, not category.</b> <code>{lead}</code> is drawn in colour, "
        + (f"<code>{control[2]}</code> in a second colour as the internal control, "
           if control else "")
        + "and every baseline in grey with its own dash pattern. Nine categorical colours "
          "would make the reader look the line up in a legend before they could read the "
          "figure."
    ) if lead else ""
    control_note = (
        f"<b>The internal control is <code>{control[2]}</code>.</b> It is the same engine, "
        "the same index and the same query set as the coarse arms, differing only in how "
        "many amino-acid classes its alphabet keeps, so the distance between it and them "
        "at the far proteome has no tool choice in it to attribute to. It is forced into "
        "this panel rather than selected: it does not win any board, which is what makes "
        "it a control."
    ) if control else ""
    write_section(out, "qfo_divergence", {
        "id": "qfo_divergence",
        "section_name": "Divergence",
        "description": (
            f"<p>Both headline metrics and reachable recall against divergence time from "
            f"human, in millions of years ({primary_truth} truth, <code>{split}</code> "
            f"split).</p>"
            + bullets(
                "<b>Each line</b> is nine points, one per target proteome, and these are "
                "the values the leaderboard's means are taken over.",
                emphasis,
                control_note,
                "<b>The shape is the result.</b> A tool that keeps its score from mouse "
                "out to E. coli is making a claim about remote homology; one that falls "
                "away is reporting close matches. The leaderboard's Fmax SD column is that "
                "same spread as a single number.",
                "<b>Sens. to 1st FP</b> is threshold-free and ranks variants differently "
                "from Fmax, so the lines are not in the same order between panels. That is "
                "the disagreement between the two metrics, not an error.",
                "<b>Raw recall is deliberately absent.</b> A human family that does not "
                "exist in the target proteome cannot be transferred by any search, and "
                "E. coli holds 971 of human's 8,909 families against mouse's 8,805. "
                "Comparing tools on raw recall would mostly compare proteomes.",
                "<b>What separates these lines is not their level, it is what survives.</b>"
                " Two kmerseek arms on the same engine, the same index and the same query "
                "set can differ by more than an order of magnitude at the furthest "
                "proteome purely in how coarse their alphabet is. That comparison is one "
                "curve rather than nineteen lines in the Divergence retention section, "
                "and it is an internal control: there is no tool choice in it to attribute "
                "the difference to.",
                "<b>Read the Fmax panel first.</b> Reachable recall and sensitivity to the "
                "first false positive are both under an open correctness check at the time "
                "of writing — the reachability denominator and the continuity of the "
                "precision-recall curves — so a conclusion drawn from those two panels is "
                "provisional in a way the Fmax panel is not.",
                "<b>A proteome where every line drops at once is a property of the run, "
                "not of the tools.</b> Where one divergence point sits far below both its "
                "neighbours for every arm including the baselines, treat it as unresolved "
                "rather than as a result about that lineage.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_divergence_plot", "title": "Accuracy vs divergence time",
                    "xlab": "divergence from human (Mya)", "ylab": "score",
                    "ymin": 0, "height": 500,
                    "style": "lines+markers",
                    "colors": colors, "dash_styles": dashes,
                    "data_labels": [{"name": name, "ylab": ylab,
                                     "ymax": _panel_ymax(panel)}
                                    for name, ylab, panel in panels]},
        "data": [panel for _, _, panel in panels],
    })


# Nine proteomes on an ordered axis, so they get a ramp rather than nine categorical
# colours. Divergence is the order, near at the pale end. The count is not fixed to nine:
# the ramp is sampled to however many proteomes a run has.
SPECIES_RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
                "#08306b", "#54278f", "#3f007d"]


def species_colors(order: list[str]) -> dict[str, str]:
    """A colour per species, sampled along the divergence ramp in the order given."""
    if not order:
        return {}
    if len(order) == 1:
        return {order[0]: SPECIES_RAMP[0]}
    step = (len(SPECIES_RAMP) - 1) / (len(order) - 1)
    return {sp: SPECIES_RAMP[round(i * step)] for i, sp in enumerate(order)}


def section_tool_by_species(out: Path, metrics: pl.DataFrame) -> None:
    """Every tool's score against every target proteome, as nine points rather than a mean.

    This was a line per tool across divergence time, and the shape it drew was the mean.
    What the mean hides is the ceiling: the kmerseek arms peak higher than any baseline's
    best proteome while sitting lower on average, which is the whole difference between "as
    good as" and "good at a different set of cases". A mean cannot carry that and a line
    through nine species asserts a continuum between proteomes that are nine separate
    measurements.

    So: one column per tool, nine points in it, coloured along the divergence ramp, with
    the mean drawn as a bar across the column. Reading a column tells you the spread; the
    mean is one mark inside it rather than the whole figure.
    """
    base = ungrouped(metrics)
    if base.height == 0 or "species_mya" not in base.columns:
        return

    for ts in sorted(base["truth_set"].unique().to_list()):
        cut, split = pick_split(base.filter(pl.col("truth_set") == ts))
        cut = cut.filter(pl.col("species") != "all")
        if cut.height == 0:
            continue
        # top_kmerseek=1, which is what gives one row per tool. best_variants takes each
        # non-kmerseek tool's best variant and kmerseek's top N separately, so 0 drops
        # kmerseek from the section entirely rather than reducing it to its best -- and a
        # larger N puts five sweep combos in beside the baselines this section exists to
        # compare them against. The leaderboard and the alphabet matrix are where the rest
        # of the sweep belongs.
        board = best_variants(cut, 1)
        if board.height == 0:
            continue
        keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
        cols = [c for c in HEADLINE if c in cut.columns]
        if not cols:
            continue

        order = (cut.group_by("species").agg(pl.col("species_mya").min())
                    .sort("species_mya")["species"].to_list())
        palette = species_colors(order)
        labels = [label for _, _, label in keep]

        panels, panel_names = [], []
        for metric in cols:
            points = {}
            for i, (tool, variant, label) in enumerate(keep):
                sub = (cut.filter((pl.col("tool") == tool)
                                  & (pl.col("variant") == variant))
                          .group_by("species").agg(pl.col(metric).mean()))
                values = {r["species"]: r[metric] for r in sub.to_dicts()
                          if r[metric] is not None}
                if not values:
                    continue
                for sp in order:
                    if sp not in values:
                        continue
                    # group is what the legend calls the colour, and the colour here is
                    # the proteome, so the legend entry has to lead with the proteome's
                    # name. MultiQC builds it as "{group}: {the names in it}" cropped at
                    # 60 characters, so a constant group put the same fourteen words in
                    # front of every entry and cropped away the part that identified it.
                    points[f"{label} · {sp} · {metric}"] = scatter_point(
                        i, values[sp],
                        name=f"{label} {values[sp]:.3f}",
                        group=species_tick(sp, ts),
                        color=palette[sp],
                        marker_size=9, marker_line_width=0,
                        marker_symbol="circle")
                mean_all = sum(values.values()) / len(values)
                points[f"{label} · mean · {metric}"] = scatter_point(
                    i, mean_all,
                    name=f"{label} {mean_all:.3f}",
                    group=f"mean over {len(values)} proteomes",
                    color="#000000", marker_size=20, marker_symbol="line-ew",
                    marker_line_width=3)
            if points:
                spec = fmt_metric_headers([metric]).get(metric, {})
                panels.append(suppress_auto_labels(points))
                panel_names.append(spec.get("title", metric))
        if not panels:
            continue

        write_section(out, f"qfo_tool_by_species_{ts}", {
            "id": f"qfo_tool_by_species_{ts}",
            "section_name": f"Tool by proteome — {ts} truth",
            "description": (
                f"<p>Each tool's best variant scored against every target proteome "
                f"separately, on the <code>{split}</code> split ({ts} truth).</p>"
                + bullets(
                    f"<b>One point per target proteome</b>, {len(order)} of them per tool, "
                    "no aggregation. Colour runs along the divergence ramp, pale for the "
                    "closest proteome and dark for the furthest.",
                    "<b>The black bar is the mean over proteomes.</b> It is one mark "
                    "inside the column rather than the column itself.",
                    "<b>Read the top of a column, not only its bar.</b> An arm whose best "
                    "proteome sits above every baseline's best while its mean sits below "
                    "them is not a worse version of the same tool, it is a tool that works "
                    "on a different set of cases. Averaging over proteomes is what made "
                    "that unreadable in the line plot this replaces.",
                    "<b>Smin is the panel where lower is better</b>, so its columns read "
                    "the other way. Every other metric here is higher-is-better.",
                    "<b>Raw recall is deliberately absent</b>, for the reason the "
                    "Divergence section gives: a human family with no instance in the "
                    "target proteome cannot be transferred by any search, so comparing "
                    "tools on it would mostly compare proteomes. Reachable recall is the "
                    "corrected form.",
                    curation_caveat_note(ts))),
            "plot_type": "scatter",
            "pconfig": {"id": f"qfo_tool_by_species_{ts}_plot",
                        "title": f"Tool accuracy by target proteome ({ts})",
                        "xlab": "tool", "ylab": "score",
                        "categories": labels,
                        "xmin": -0.5, "xmax": len(labels) - 0.5,
                        "ymin": 0, "height": 560, "showlegend": True,
                        "xsuffix": "", "ysuffix": "", "y_decimals": 3,
                        "data_labels": [{"name": name, "ylab": name}
                                        for name in panel_names]},
            "data": panels,
        }, supplementary=not is_primary_truth(ts))

        # The same values as a table, for the metric the report leads on. A dot plot is
        # read for spread; the number a sentence quotes has to be legible somewhere.
        headline = cols[0]
        by_species = {}
        for tool, variant, label in keep:
            sub = (cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
                      .group_by("species", "species_mya").agg(pl.col(headline).mean()))
            for r in sub.to_dicts():
                key = species_tick(r["species"], ts, Mya=r["species_mya"])
                row = by_species.setdefault(key, {"Mya": r["species_mya"],
                                                  "_species": r["species"]})
                row[label] = r[headline]
        if not by_species:
            continue
        ordered = dict(sorted(by_species.items(), key=lambda kv: kv[1]["Mya"]))
        # Two summary rows rather than one. The pair is the point: where they differ, the
        # mean is being moved by a proteome whose annotation is the limit rather than its
        # divergence.
        measured = list(ordered.values())
        summaries = [(f"mean over {len(measured)} proteomes", measured)]
        for name, rows in summaries:
            summary = {"Mya": None}
            for label in (lbl for _, _, lbl in keep):
                vals = [r[label] for r in rows if r.get(label) is not None]
                summary[label] = sum(vals) / len(vals) if vals else None
            ordered[name] = summary
        for row in ordered.values():
            row.pop("_species", None)
        headers = {"Mya": {"title": "Mya", "description": "Divergence from human",
                           "format": "{:,.0f}"}}
        headers.update({label: {"title": label, "format": "{:,.3f}", "scale": "RdYlGn"}
                        for _, _, label in keep})
        write_section(out, f"qfo_tool_by_species_table_{ts}", {
            "id": f"qfo_tool_by_species_table_{ts}",
            "section_name": f"Tool by proteome, {panel_names[0]} — {ts} truth",
            "description": (
                f"<p>The <b>{panel_names[0]}</b> panel above as numbers ({split} split).</p>"
                + bullets(
                    "<b>One row per target proteome</b>, ordered by divergence from human, "
                    "then the two summary rows.",
                    "<b>One column per tool</b>, each at its best variant.",
                    "Read <b>down a column</b> for how one tool holds up, and <b>across a "
                    "row</b> for which tool won that proteome.",
                    "<b>The last row is the mean over proteomes</b>, the aggregation "
                    "every leaderboard in this report ranks on.",
                    curation_caveat_note(ts))),
            "plot_type": "table",
            "pconfig": {"id": f"qfo_tool_by_species_table_{ts}_table",
                        "title": f"{panel_names[0]} by species and tool ({ts})",
                        "col1_header": "Species", "sort_rows": False, "no_violin": True},
            "headers": headers,
            "data": ordered,
        }, supplementary=not is_primary_truth(ts))



def parse_kmerseek_variants(df: pl.DataFrame) -> pl.DataFrame:
    """Split kmerseek's variant string into alphabet, ksize and low-complexity arm.

    One parser for every section that needs it. The pattern is the variant NAME the
    pipeline writes, so a second copy would keep matching an old naming scheme in one
    section while the other had moved on -- and the failure mode is an empty plot, not an
    error. Rows whose variant does not parse (every non-kmerseek tool) are dropped.
    """
    return df.with_columns(
        pl.col("variant").str.extract(r"^(.*)_k\d+_lc(?:True|False)$", 1).alias("alphabet"),
        pl.col("variant").str.extract(r"_k(\d+)_lc", 1).cast(pl.Int64).alias("ksize"),
        pl.col("variant").str.extract(r"_lc(True|False)$", 1).alias("lowcomp"),
    ).filter(pl.col("alphabet").is_not_null())


def alphabet_classes(alphabet: str) -> int:
    """Class count off the end of the alphabet name -- protein20 -> 20, hp_lehninger2 -> 2.

    Every alphabet was renamed in kmerseek PR #43 to state its class count, so the name is
    the authority. Anything unparsable sorts last rather than raising.
    """
    m = re.search(r"(\d+)$", alphabet)
    return int(m.group(1)) if m else 10_000


# Metrics where a SMALLER number is the better one. Everything else in HEADLINE is
# higher-is-better. Getting this wrong would silently crown the worst combo, so it is a
# named set rather than a sign buried in a sort call.
LOWER_IS_BETTER = {"smin"}


def pick_selection_split(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """The SELECTION half if the run produced one, otherwise whatever pick_split finds.

    Every other section reads the heldout half, because that is the honest one to report.
    This table is the exception, and deliberately: it exists to CHOOSE an alphabet, and
    choosing on the heldout half is the thing the split exists to prevent. Reporting the
    winner's score on the data that picked it is optimistically biased -- see
    build_domain_truth.assign_split, which sets the split up for exactly this reason.

    Only Pfam is partitioned. Swiss-Prot and Pfam-N are scored whole, so they fall back to
    `all`, and the section says which it used per truth set rather than implying one.
    """
    if "split" in df.columns:
        sel = df.filter(pl.col("split") == "selection")
        if sel.height:
            return sel, "selection"
    return pick_split(df)


def _combo_label(row: dict) -> str:
    lc = "lcT" if row["lowcomp"] == "True" else "lcF"
    return f"{row['alphabet']} k{row['ksize']} {lc}"


def section_species_winners(out: Path, metrics: pl.DataFrame, top_n: int = 3) -> None:
    """How identifiable each proteome's best encoding is, rather than which one it was.

    This was a table of argmaxes: for each proteome and each headline metric, the three
    best combos with their scores, in cells reading "<alphabet> k<n> lcT (0.137)". Ten
    metric columns times three rows per proteome is 270 strings, and every one of them was
    a winner drawn from a sweep where the winner routinely beats the runner-up in the
    fourth decimal.

    The columns here are the two numbers that say whether any of that was a result:

      margin              the winner's score minus the next DISTINCT encoding's. On a
                          sweep this size a margin below one SE is a coin flip.
      encodings within    how many encodings score inside one standard error of the best.
      1 SE                Where that count runs into the dozens the winner is not
                          identifiable, and the caption says so instead of leaving the
                          table to imply otherwise.

    Ranked over distinct (alphabet, ksize) encodings, each at its better low-complexity
    arm, for the reason best_encodings_per_species gives: raw combos put the same encoding
    with the filter flipped in second place and the margin would be measuring the filter.
    """
    base = ungrouped(metrics.filter(pl.col("tool") == "kmerseek"))
    if base.height == 0 or "species" not in base.columns:
        return
    parsed = parse_kmerseek_variants(base)
    if parsed.height == 0:
        return

    for ts in sorted(parsed["truth_set"].unique().to_list()):
        cut, split = pick_selection_split(parsed.filter(pl.col("truth_set") == ts))
        # `all` is the hmmscan ceiling's species, which reads no target proteome.
        cut = cut.filter(pl.col("species") != "all")
        if cut.height == 0 or "species_mya" not in cut.columns:
            continue
        cut = encoding_axes(cut.filter(pl.col("species_mya").is_not_null()))
        if cut.height == 0:
            continue
        metric = "fmax" if "fmax" in cut.columns else None
        if metric is None:
            continue
        surface = encoding_surface(cut, metric)
        if surface.height == 0:
            continue

        n_encodings = surface.select(pl.struct("alphabet", "ksize").n_unique()).item()
        data, unidentifiable = {}, []
        for row in (surface.group_by("species", "species_mya")
                           .agg(pl.col("n_reachable").max())
                           .sort("species_mya").to_dicts()):
            sub = (surface.filter(pl.col("species") == row["species"])
                          .drop_nulls("value").sort("value", descending=True))
            if sub.height == 0:
                continue
            ranked = sub.to_dicts()
            best = ranked[0]
            se = one_se(best["value"], row["n_reachable"])
            margin = (best["value"] - ranked[1]["value"]) if len(ranked) > 1 else None
            within = (sum(1 for r in ranked if r["value"] >= best["value"] - se)
                      if se is not None else None)
            if within is not None and within > 1:
                unidentifiable.append((row["species"], within))
            data[species_tick(row["species"], ts, Mya=row["species_mya"])] = {
                "encoding": encoding_label(best["alphabet"], best["ksize"]),
                "bits": best["bits_per_kmer"],
                "best": best["value"],
                "margin": margin,
                "one_se": se,
                "within_1se": within,
                "n_ranked": len(ranked),
                "n_reachable": row["n_reachable"],
            }
        if not data:
            continue

        worst = max(unidentifiable, key=lambda p: p[1], default=None)
        verdict = (
            f"<b>The winner is not identifiable on {len(unidentifiable)} of "
            f"{len(data)} proteomes.</b> At <code>{worst[0]}</code> "
            f"{worst[1]:,} of {n_encodings:,} encodings score inside one standard error "
            f"of the best, so naming one of them the winner is naming one member of a tie. "
            f"Read the bits column of the panel above instead: what the data supports is a "
            f"band, not a choice."
            if worst else
            "<b>Every proteome's best encoding clears one standard error of the field.</b> "
            "The margin column is the size of that lead.")

        write_section(out, f"qfo_species_winners_{ts}", {
            "id": f"qfo_species_winners_{ts}",
            "section_name": f"Is the best encoding identifiable — {ts} truth",
            "description": (
                f"<p>For each target proteome, the best of {n_encodings:,} distinct "
                f"kmerseek encodings on "
                f"{fmt_metric_headers([metric]).get(metric, {}).get('title', metric)}, "
                f"and how far clear of the field it is ({ts} truth, "
                f"<code>{split}</code> split).</p>"
                + bullets(
                    verdict,
                    "<b>Margin</b> is the winner's score minus the next distinct "
                    "encoding's. <b>1 SE</b> is <code>sqrt(v(1-v)/n)</code> at the "
                    "winner's score over that proteome's reachable instances: the standard "
                    "error of a proportion at the same value and denominator, used as a "
                    "noise scale, not a confidence interval. A margin under 1 SE is not a "
                    "difference between encodings.",
                    "<b>Encodings within 1 SE</b> counts how many of the field score at "
                    "or above <code>best - 1 SE</code>, the winner included. One means the "
                    "winner stands alone; forty means the top of the board is one number.",
                    "<b>Reachable</b> is the denominator the SE is computed on, and it "
                    "differs several-fold between proteomes, so the same margin is "
                    "significant against mouse and noise against E. coli.",
                    "<b>Ranked over distinct alphabet x ksize encodings</b>, each at its "
                    "better low-complexity arm. Ranking raw combos would put the same "
                    "encoding with the filter flipped in second place and the margin "
                    "would be measuring the filter.",
                    "<b>Split</b> — the <code>selection</code> half wherever the truth set "
                    "has one. This section is for choosing an encoding, and choosing on "
                    "the heldout half is what the split exists to prevent.",
                    curation_caveat_note(ts))),
            "plot_type": "table",
            "pconfig": {"id": f"qfo_species_winners_{ts}_table",
                        "title": f"Best encoding and its margin, per proteome ({ts})",
                        "col1_header": "Species · Mya", "sort_rows": False, "no_violin": True},
            "headers": {
                "encoding": dict(title="Best encoding", scale=False,
                                 description="Alphabet and k with the highest score here"),
                "bits": dict(title="Bits/k-mer", format="{:,.1f}", scale="Purples",
                             description="k · log2(classes) for that encoding"),
                "best": dict(title="Best score", format="{:,.4f}", scale="Greens"),
                "margin": dict(title="Margin over 2nd", format="{:,.4f}", scale="Blues",
                               description="Winner minus the next distinct encoding. "
                                           "Compare it with the 1 SE column"),
                "one_se": dict(title="1 SE", format="{:,.4f}", scale=False,
                               description="sqrt(v(1-v)/n) at the winner's score over the "
                                           "reachable instances. A noise scale, not a CI"),
                "within_1se": dict(title="Encodings within 1 SE", format="{:,.0f}",
                                   scale="Reds",
                                   description="How many encodings score at or above "
                                               "best - 1 SE. Large means the winner is "
                                               "one member of a tie"),
                "n_ranked": dict(title="Encodings ranked", format="{:,.0f}", scale=False),
                "n_reachable": dict(title="Reachable instances", format="{:,.0f}",
                                    scale="Blues"),
            },
            "data": data,
        }, supplementary=not is_primary_truth(ts))



# --- the encoding score surface, and the bits budget as a number ----------------------
#
# This block used to plot an ARGMAX. For each target proteome it took the encoding with the
# best Fmax and drew that encoding's alphabet size, its k and its bits per k-mer against
# divergence time, three metrics times three ranks, nine crossing lines behind a three-way
# truth-set switcher.
#
# Three things were wrong with it and they compound. The winner's margin over the runner-up
# is routinely in the third or fourth decimal across a sweep of this size, so the quantity
# being plotted is the least stable summary of the score surface rather than a property of
# it. Alphabet size and k do not interpolate between species, so the connecting lines
# asserted intermediate encodings that do not exist. And the whole score distribution --
# the thing that says whether the winner means anything -- was thrown away before drawing.
#
# What replaces it is two panels, both visible, no switcher:
#
#   A  the surface. Every encoding against every proteome, coloured by Fmax. The argmax is
#      one cell of it rather than the whole figure.
#   B  the budget as a number. Per species, the encodings that are within one standard
#      error of that species' best, strip-plotted on the bits axis with their median. If
#      there is a bits budget the strips sit at the same height across species and the
#      caption can quote it with an interval. If they do not, the figure says that too.

# Metrics the panels rank on. More than one deliberately: this benchmark's metrics disagree
# about which combo wins, so a single "best encoding" line would be a choice made silently.
# Fmax is precision-dominated, reachable recall is not, and sensitivity to the first false
# positive is threshold-free and per-query. They pick different alphabets, and the
# disagreement is itself a result rather than a fault.
ENCODING_RANK_METRICS = ["fmax", "recall_reachable", "sens_first_fp_mean"]

# What alphabet_classes returns for a name that carries no class count. Sorting last is the
# right behaviour for a sort key and the wrong one here: log2(10_000) would draw a
# fabricated 13-bits-per-residue alphabet onto the bits axis, which reads as a result. Rows
# with it are dropped and the alphabet is NAMED, in the build log and in the section text.
UNKNOWN_ALPHABET_CLASSES = 10_000

# How many distinct encodings the margin table ranks per species. Two is the minimum that
# can report a margin at all; three keeps the third in view for a reader checking whether
# the gap closes immediately behind the winner.
ENCODING_TOP_N = 3


def encoding_axes(df: pl.DataFrame) -> pl.DataFrame:
    """Attach class count and bits-per-k-mer to a frame already through the variant parser.

    `alphabet_classes` is the single source for the class count -- every alphabet was
    renamed in kmerseek PR #43 to state it, so the name is the authority. Anything it
    cannot read is dropped here rather than defaulted, because a default would still plot.
    """
    counts = {a: alphabet_classes(a) for a in df["alphabet"].unique().to_list()}
    unknown = sorted(a for a, c in counts.items() if c == UNKNOWN_ALPHABET_CLASSES)
    if unknown:
        print("encoding-vs-divergence: these alphabet names carry no class count, so "
              "bits per k-mer is not computable for them and they are left out of the "
              "encoding panels: " + ", ".join(unknown))
    return (df.with_columns(pl.col("alphabet")
                              .replace_strict(counts, return_dtype=pl.Int64)
                              .alias("n_classes"))
              .filter(pl.col("n_classes") != UNKNOWN_ALPHABET_CLASSES)
              .with_columns((pl.col("ksize") * pl.col("n_classes").log(2))
                            .alias("bits_per_kmer")))


def best_encodings_per_species(cut: pl.DataFrame, metric: str,
                               top_n: int = ENCODING_TOP_N) -> pl.DataFrame:
    """Top `top_n` DISTINCT (alphabet, ksize) encodings per species, on one metric.

    Collapsed over the low-complexity arm first. That arm is gone from the report for the
    reason the supplementary panel measures, and it has to be gone from the RANKING too:
    ranking raw combos makes the 2nd and 3rd best usually the same alphabet and k with the
    filter flipped, scoring within 0.001, so a margin over the runner-up would be reporting
    the filter's noise rather than the distance to the next real encoding.
    """
    lower_better = metric in LOWER_IS_BETTER
    agg = pl.col(metric).min() if lower_better else pl.col(metric).max()
    per_encoding = (cut.drop_nulls(metric)
                       .group_by("species", "species_mya", "alphabet", "ksize",
                                 "n_classes", "bits_per_kmer")
                       .agg(agg))
    if per_encoding.height == 0:
        return per_encoding
    return (per_encoding
            .sort(metric, descending=not lower_better)
            .group_by("species", maintain_order=True)
            .head(top_n)
            .with_columns(pl.int_range(pl.len()).over("species").alias("rank")))


def one_se(value: float | None, n: int | None) -> float | None:
    """A noise scale for a rate measured over `n` instances: sqrt(v(1-v)/n).

    This is the standard error of a PROPORTION at the same value over the same denominator.
    Fmax is not a proportion -- it is the best F1 over a threshold sweep -- so this is a
    scale rather than a confidence interval, and the sections that use it say so. What it
    is for is the only question anyone asks of a 203-encoding leaderboard: is the winner
    distinguishable from the field, or is the whole top of the board one number. No number
    in this benchmark carries a resampling error, so the alternative to an explicit scale
    is not a better interval, it is the reader assuming the third decimal means something.
    """
    if value is None or n is None or n <= 0:
        return None
    v = min(max(float(value), 0.0), 1.0)
    return math.sqrt(v * (1.0 - v) / n)


def encoding_surface(cut: pl.DataFrame, metric: str) -> pl.DataFrame:
    """One row per (encoding, species): the metric, at that encoding's better lc arm."""
    return (cut.drop_nulls(metric)
               .group_by("species", "species_mya", "alphabet", "ksize", "n_classes",
                         "bits_per_kmer")
               .agg(pl.col(metric).max().alias("value"),
                    pl.col("n_reachable_instances").max().alias("n_reachable")
                    if "n_reachable_instances" in cut.columns
                    else pl.lit(None, dtype=pl.Int64).alias("n_reachable")))


def encoding_label(alphabet: str, ksize: int) -> str:
    return f"{alphabet} k{ksize}"


def section_encoding_vs_divergence(out: Path, metrics: pl.DataFrame,
                                   top_n: int = ENCODING_TOP_N) -> None:
    """The encoding score surface, and the bits budget, against divergence time.

    Per truth set, never pooled, and on the selection split for the reason
    pick_selection_split gives -- these panels are about choosing an encoding, and the
    heldout half exists so the chosen one can be reported honestly elsewhere.
    """
    base = ungrouped(metrics.filter(pl.col("tool") == "kmerseek"))
    if base.height == 0 or "species" not in base.columns:
        return
    if "species_mya" not in base.columns:
        return
    parsed = parse_kmerseek_variants(base)
    if parsed.height == 0:
        return

    for ts in sorted(parsed["truth_set"].unique().to_list()):
        cut, split = pick_selection_split(parsed.filter(pl.col("truth_set") == ts))
        # `all` is the hmmscan ceiling's species, which reads no target proteome and so
        # sits at no divergence time.
        cut = cut.filter((pl.col("species") != "all")
                         & pl.col("species_mya").is_not_null())
        if cut.height == 0:
            continue
        cut = encoding_axes(cut)
        if cut.height == 0:
            continue
        metric = "fmax" if "fmax" in cut.columns else None
        if metric is None:
            continue
        surface = encoding_surface(cut, metric)
        if surface.height == 0:
            continue
        section_encoding_score_surface(out, surface, ts, split, metric)
        section_bits_budget(out, surface, ts, split, metric)


def section_encoding_score_surface(out: Path, surface: pl.DataFrame, ts: str, split: str,
                                   metric: str) -> None:
    """Panel A: every encoding against every proteome, coloured by Fmax.

    Rows are sorted by bits per k-mer, which is the axis Panel B quotes a number on, so the
    two panels read against each other: a horizontal band of colour partway up this grid is
    the same claim as a flat strip in the next one.
    """
    order = (surface.group_by("species", "species_mya")
                    .agg(pl.col("n_reachable").max())
                    .sort("species_mya").to_dicts())
    encodings = (surface.select("alphabet", "ksize", "bits_per_kmer").unique()
                        .sort("bits_per_kmer").to_dicts())
    cells = {(r["alphabet"], r["ksize"], r["species"]): r["value"]
             for r in surface.to_dicts()}
    rows = [[cells.get((e["alphabet"], e["ksize"], s["species"])) for s in order]
            for e in encodings]
    vmax = heat_max(rows)
    if vmax is None:
        return
    title = fmt_metric_headers([metric]).get(metric, {}).get("title", metric)

    # The argmax per species, named in the text rather than drawn as its own series. A
    # MultiQC heatmap has no annotation layer to put a marker on a cell with, so the
    # selection is marked in Panel B, where the same encodings are drawn as points.
    winners = []
    for s in order:
        col = [(cells.get((e["alphabet"], e["ksize"], s["species"])), e)
               for e in encodings]
        got = [(v, e) for v, e in col if v is not None]
        if not got:
            continue
        best_v, best_e = max(got, key=lambda p: p[0])
        se = one_se(best_v, s["n_reachable"])
        within = sum(1 for v, _ in got if se is not None and v >= best_v - se)
        winners.append(
            f"<code>{s['species']}</code> "
            f"{encoding_label(best_e['alphabet'], best_e['ksize'])} at {best_v:.3f}"
            + (f", with {within:,} of {len(got):,} encodings inside one SE of it"
               if se is not None else ""))

    reach = {s["species"]: s["n_reachable"] for s in order if s["n_reachable"]}
    write_section(out, f"qfo_encoding_divergence_{ts}", {
        "id": f"qfo_encoding_divergence_{ts}",
        "section_name": f"Encoding score surface — {ts} truth",
        "description": (
            f"<p>{title} for every kmerseek encoding against every target proteome "
            f"({ts} truth, <code>{split}</code> split, {len(encodings):,} distinct "
            f"alphabet x ksize encodings).</p>"
            + bullets(
                f"<b>Colour is {title}</b>, one cell per encoding and proteome, at that "
                "encoding's better low-complexity arm.",
                "<b>Columns</b> are target proteomes in divergence order, left to right, "
                "with the reachable instances each is scored over in the tick label. "
                "<b>Rows</b> are encodings sorted by bits per k-mer, low at the bottom.",
                "<b>This panel replaces a plot of the argmax.</b> The winner's margin over "
                "the next encoding is in the third decimal for most proteomes here, so a "
                "figure of which encoding won was drawing the least stable summary of this "
                "grid. Read a band, not a cell.",
                "<b>Per-proteome best</b> — " + "; ".join(winners) + "."
                if winners else "",
                heat_range_note(vmax),
                curation_caveat_note(ts))),
        "plot_type": "heatmap",
        "pconfig": {"id": f"qfo_encoding_divergence_{ts}_plot",
                    "title": f"{title} by encoding and target proteome ({ts})",
                    "xlab": "target proteome (divergence order)",
                    "ylab": "encoding, sorted by bits per k-mer",
                    "min": 0, "max": vmax, "colstops": SEQUENTIAL_COLSTOPS,
                    "square": False,
                    "height": min(1800, 120 + 12 * len(encodings)),
                    "tt_decimals": 3},
        "xcats": [species_tick(s["species"], ts, Mya=s["species_mya"],
                               n=reach.get(s["species"])) for s in order],
        "ycats": [encoding_label(e["alphabet"], e["ksize"]) for e in encodings],
        "data": rows,
    }, supplementary=not is_primary_truth(ts))


def section_bits_budget(out: Path, surface: pl.DataFrame, ts: str, split: str,
                        metric: str) -> None:
    """Panel B: the bits per k-mer of every encoding that is as good as that proteome's best.

    One point per encoding inside one standard error of the proteome's best score, so the
    strip is the set of encodings the data cannot tell apart from the winner. If the method
    needs a fixed amount of information in one k-mer, those strips sit at the same height
    all the way out and the caption can quote the height with an interval. The median of
    each strip is drawn on top, and the pooled median as a rule across the panel.
    """
    order = (surface.group_by("species", "species_mya")
                    .agg(pl.col("n_reachable").max())
                    .sort("species_mya").to_dicts())
    if not order:
        return
    span = (order[-1]["species_mya"] - order[0]["species_mya"]) or 1.0
    points, medians, per_species, pooled = {}, {}, {}, []
    for i, s in enumerate(order):
        sub = surface.filter(pl.col("species") == s["species"]).drop_nulls("value")
        if sub.height == 0:
            continue
        best = float(sub["value"].max())
        se = one_se(best, s["n_reachable"])
        if se is None:
            continue
        band = sub.filter(pl.col("value") >= best - se).to_dicts()
        if not band:
            continue
        bits = sorted(r["bits_per_kmer"] for r in band)
        mid = bits[len(bits) // 2] if len(bits) % 2 else (
            (bits[len(bits) // 2 - 1] + bits[len(bits) // 2]) / 2)
        per_species[s["species"]] = mid
        pooled.extend(bits)
        # A jitter of 1.5% of the axis span, alternating direction, so a strip of 40
        # encodings at one proteome is a cloud rather than one overprinted dot. The x
        # values are real divergence times; the offset is drawing, and it is small enough
        # that two proteomes never overlap.
        for j, r in enumerate(sorted(band, key=lambda r: r["bits_per_kmer"])):
            jitter = ((j % 7) - 3) / 3 * 0.015 * span
            is_best = r["value"] >= best - 1e-12
            points[f"{s['species']} {encoding_label(r['alphabet'], r['ksize'])}"] = (
                scatter_point(
                    s["species_mya"] + jitter, r["bits_per_kmer"],
                    name=f"{s['species']} {encoding_label(r['alphabet'], r['ksize'])} "
                         f"({metric} {r['value']:.3f})",
                    # Constant group strings. MultiQC builds one legend entry per unique
                    # (colour, size, line width, group), so putting the strip's n in the
                    # group name produced a separate legend row per proteome.
                    group=("best encoding" if is_best else "within 1 SE of best"),
                    color=("#d55e00" if is_best else "#a8c8e0"),
                    marker_size=13 if is_best else 6,
                    marker_line_width=2 if is_best else 0,
                    marker_symbol="circle",
                    annotation=(encoding_label(r["alphabet"], r["ksize"])
                                if is_best else None)))
        medians[f"median {s['species']}"] = scatter_point(
            s["species_mya"], mid,
            name=f"median bits, {s['species']} (n={len(band)})",
            group="median of the strip", color="#000000",
            marker_size=12, marker_symbol="line-ew", marker_line_width=3)
    if not points:
        return
    points.update(medians)
    suppress_auto_labels(points)
    pooled.sort()
    lo, hi = min(per_species.values()), max(per_species.values())
    grand = (pooled[len(pooled) // 2] if len(pooled) % 2
             else (pooled[len(pooled) // 2 - 1] + pooled[len(pooled) // 2]) / 2)
    q1 = pooled[len(pooled) // 4]
    q3 = pooled[(3 * len(pooled)) // 4]
    flat = (hi - lo) <= 0.15 * grand
    verdict = (
        f"<b>The strips sit at the same height across the run.</b> Every proteome's median "
        f"is between {lo:.1f} and {hi:.1f} bits, a spread of {hi - lo:.1f} bits on a "
        f"median of {grand:.1f}, so a fixed information budget per k-mer is consistent "
        f"with this surface."
        if flat else
        f"<b>The strips do not sit at one height.</b> The per-proteome medians run from "
        f"{lo:.1f} to {hi:.1f} bits, a spread of {hi - lo:.1f} bits against a pooled "
        f"median of {grand:.1f}, so there is no single bits budget to quote.")
    print(f"bits budget ({ts}, {split}): per-species medians {lo:.2f}-{hi:.2f} bits, "
          f"pooled median {grand:.2f}, IQR {q1:.2f}-{q3:.2f}, "
          f"{len(pooled):,} encodings inside one SE across {len(per_species)} proteomes")

    write_section(out, f"qfo_bits_budget_{ts}", {
        "id": f"qfo_bits_budget_{ts}",
        "section_name": f"Bits budget per k-mer — {ts} truth",
        "description": (
            f"<p>Bits per k-mer of every encoding whose score is within one standard error "
            f"of its proteome's best ({ts} truth, <code>{split}</code> split, ranked on "
            f"{metric}).</p>"
            + bullets(
                "<b>One point per encoding</b>, not per winner. The strip at each proteome "
                "is the set of encodings this data cannot separate from the best one, so "
                "its width is how identifiable the winner is.",
                "<b>One SE</b> is <code>sqrt(v(1-v)/n)</code> at the proteome's best score "
                "over its reachable instances. That is the standard error of a proportion "
                "at the same value and denominator, used as a noise scale rather than a "
                "confidence interval: no number in this benchmark carries a resampling "
                "error.",
                "<b>The orange point</b> is the argmax and carries its encoding's name. "
                "<b>The black bar</b> is the median of that proteome's strip; the dashed "
                "rule is the median over every point in the panel.",
                verdict,
                f"<b>Pooled median {grand:.1f} bits, interquartile range {q1:.1f} to "
                f"{q3:.1f} bits</b>, over {len(pooled):,} encoding-proteome points inside "
                f"one SE.",
                "<b>x is divergence time</b>, a continuous axis, so points are offset "
                "sideways by up to 1.5% of the span to keep a strip from overprinting "
                "itself. The offsets are drawing, not measurement.",
                curation_caveat_note(ts))),
        "plot_type": "scatter",
        "pconfig": {"id": f"qfo_bits_budget_{ts}_plot",
                    "title": f"Bits per k-mer among encodings within 1 SE of best ({ts})",
                    "xlab": "divergence from human (Mya)",
                    "ylab": "bits per k-mer (k · log2 classes)",
                    # The scatter DROPS points outside the range rather than clipping the
                    # axis, and a point at the limit loses half its label, so the range is
                    # padded past the data rather than left to autorange.
                    "ymin": min(pooled) - 0.06 * (max(pooled) - min(pooled) or 1),
                    "ymax": max(pooled) + 0.06 * (max(pooled) - min(pooled) or 1),
                    "xmin": order[0]["species_mya"] - 0.06 * span,
                    "xmax": order[-1]["species_mya"] + 0.10 * span,
                    "height": 520, "showlegend": True,
                    "xsuffix": "", "ysuffix": "",
                    "x_decimals": 0, "y_decimals": 1,
                    "y_lines": [{"value": grand, "color": "#555555", "dash": "dash",
                                 "width": 1,
                                 "label": f"pooled median {grand:.1f} bits"}]},
        "data": points,
    }, supplementary=not is_primary_truth(ts))


# --- Fmax against model confidence, as a shape rather than a ranking -----------------
#
# The covariate section already draws the pLDDT axis, as a grouped bar chart with one
# group per band. That form answers "who is best inside this band" and hides the thing the
# bands were computed to show: whether a tool's accuracy RISES with structure confidence.
# That is a shape ACROSS bands, and a reader cannot see a peak in a grouped bar chart.
#
# The shape is what separates the two readings of this benchmark. Read only at the
# disordered end, kmerseek's coarse-alphabet arms look like pure degradation. Read across
# every band, they can instead be non-monotone -- worst where structure is confident, best
# in the middle -- which is a claim about a REGIME rather than about the dark proteome, and
# it is the claim the bands can actually support.
#
# Bands come from STRATA["plddt"], edges 0/50/70/90/100. The lowest band is defined and
# came back empty in the runs this was written against: no protein under mean pLDDT 50
# cleared the per-stratum floor in evaluate_domain_calls.py. So a fall-off at the low end
# is a fall-off at 50-70, not at the disordered tail, and the section says which bands it
# actually has rather than letting the axis imply the missing one.

# Two points make a slope; a peak needs three. Below this the section would draw the
# covariate bar chart again with lines instead of bars, and claim a shape it cannot see.
MIN_REGIME_BANDS = 3


def band_midpoint(label: str) -> float | None:
    """Numeric x for a "70-90" band label, so the axis is pLDDT and not three equal slots.

    Equal-width categories would put 90-100 as far from 70-90 as 70-90 is from 50-70,
    which is twice the pLDDT it spans -- and the crossover this plot exists to show
    happens between exactly those two bands, so the spacing is load-bearing.
    """
    parts = label.split("-")
    if len(parts) != 2:
        return None
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (lo + hi) / 2


def _peak_band(series: dict[float, float | None]) -> float | None:
    """Which band an arm scores highest in. None when it has no number anywhere."""
    scored = {x: y for x, y in series.items() if y is not None}
    return max(scored, key=scored.get) if scored else None


def section_plddt_regime(out: Path, metrics: pl.DataFrame, primary_truth: str,
                         max_tools: int) -> None:
    """Fmax against pLDDT band as lines, so a peak in the middle is visible.

    Same rows and the same board as the pLDDT covariate section, drawn as a shape instead
    of as a per-band ranking, plus the per-band denominators that section does not carry.
    """
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    sub = cut.filter(pl.col("stratum_axis") == "plddt")
    if sub.height == 0:
        return
    board = best_variants(ungrouped(cut)).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    if not keep:
        return

    labels = _bin_order(sub["stratum"].drop_nulls().unique().to_list())
    bands = [(b, band_midpoint(b)) for b in labels]
    bands = [(b, x) for b, x in bands if x is not None]

    series, colors = {}, {}
    for tool, variant, label in keep:
        arm = sub.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if arm.height == 0:
            continue
        by_band = {r["stratum"]: r["fmax"]
                   for r in arm.group_by("stratum").agg(pl.col("fmax").mean()).to_dicts()}
        drawn = {x: by_band.get(b) for b, x in bands}
        if all(v is None for v in drawn.values()):
            continue
        series[label] = drawn
        colors[label] = tool_color(tool)

    bands = [(b, x) for b, x in bands
             if any(s.get(x) is not None for s in series.values())]
    if not series or len(bands) < MIN_REGIME_BANDS:
        write_section(out, "qfo_plddt_regime", {
            "id": "qfo_plddt_regime",
            "section_name": "Model-confidence regime",
            "description": f"<p>{primary_truth} truth, <code>{split}</code> split.</p>",
            "plot_type": "html",
            "data": (
                f"<p>Not plotted: this run has {len(bands)} populated pLDDT band(s) and a "
                f"peak needs at least {MIN_REGIME_BANDS}. Two points are a slope, and a "
                "slope drawn here would be read as the shape this section is named for. "
                "The per-band numbers are in the Model confidence section above.</p>"),
        })
        return
    series = {label: {x: s.get(x) for _, x in bands} for label, s in series.items()}

    # Denominators, per band, shared by every arm: how many query proteins the band holds
    # and how many annotated instances sit on them. Nothing in this report carries a
    # sampling error, so the count is the only thing a reader has to judge a gap by.
    sizes = {}
    for b, x in bands:
        rows = sub.filter(pl.col("stratum") == b)
        sizes[x] = {
            "band": b,
            "proteins": (rows["n_stratum_proteins"].max()
                         if "n_stratum_proteins" in rows.columns else None),
            "instances": (rows["n_truth_instances"].max()
                          if "n_truth_instances" in rows.columns else None),
        }

    top_x = bands[-1][1]
    peaks = {label: _peak_band(s) for label, s in series.items()}
    mid = [lb for lb, pk in peaks.items() if pk is not None and pk != top_x]
    n_mid, n_arms = len(mid), len([p for p in peaks.values() if p is not None])
    if mid:
        shape = (f"<b>In this run {n_mid} of {n_arms} arms peak below the top band</b> — "
                 + ", ".join(f"<code>{lb}</code>" for lb in sorted(mid))
                 + ". Every other arm scores highest where the structure is most "
                 "confident. An arm that peaks in the middle is describing a regime, not "
                 "a monotone dependence on structure quality.")
    else:
        shape = ("<b>In this run every arm peaks in the top band</b>, so nothing here is "
                 "non-monotone and there is no regime to claim.")

    lo_label = sizes[bands[0][1]]["band"]
    write_section(out, "qfo_plddt_regime", {
        "id": "qfo_plddt_regime",
        "section_name": "Model-confidence regime",
        "description": (
            f"<p>Fmax against the query structure's mean pLDDT, as lines, so the shape "
            f"across bands is readable rather than the ranking inside one "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>Same numbers as the Model confidence section</b>, same arms and same "
                "split. Only the form differs: that one is a grouped bar chart, which "
                "answers who wins a band and hides whether a line rises, falls or peaks.",
                "<b>x is the band's midpoint</b>, not a category slot, so 90-100 sits the "
                "distance from 70-90 that it actually spans. The crossover happens "
                "between those two bands, so equal spacing would move it.",
                shape,
                f"<b>The lowest band drawn is {lo_label}.</b> The strata define a 0-50 "
                "band as well; where it is absent, no query protein under mean pLDDT 50 "
                "cleared the per-stratum protein floor. A fall-off at the left edge of "
                "this plot is therefore a fall-off in low-confidence structure, not in "
                "the disordered tail — the disorder axes are where that is measured.",
                "<b>Colour is the method class</b>, as everywhere else in this report, so "
                "a kmerseek line crossing above or below the structure methods is legible "
                "without reading the legend.",
                "<b>No error bars, here or anywhere in this report.</b> Every point is a "
                "mean over target proteomes with no resampling behind it, so a gap "
                "narrower than the spread between species is not a result. The table "
                "below carries each band's protein and instance counts, which is the only "
                "denominator a reader has to judge a gap by.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_plddt_regime_plot",
                    "title": "Fmax vs model confidence",
                    "xlab": "mean pLDDT (band midpoint)", "ylab": "Fmax",
                    "ymin": 0, "height": 520, "style": "lines+markers",
                    "colors": colors, "showlegend": True},
        "data": series,
    })

    table, headers = {}, {"peak": {"title": "Peaks in",
                                   "description": "Band this arm scores highest in"}}
    for b, x in bands:
        n = sizes[x]
        headers[b] = {"title": b, "format": "{:,.3f}", "scale": "RdYlGn",
                      "min": 0, "max": 1,
                      "description": (f"Fmax in the {b} pLDDT band — "
                                      f"{n['proteins'] or 0:,} query proteins, "
                                      f"{n['instances'] or 0:,} annotated instances")}
    for label, s in series.items():
        row = {b: s.get(x) for b, x in bands}
        pk = peaks.get(label)
        row["peak"] = sizes[pk]["band"] if pk is not None else None
        table[label] = row
    counts = "; ".join(
        f"<code>{sizes[x]['band']}</code>: {sizes[x]['proteins'] or 0:,} proteins, "
        f"{sizes[x]['instances'] or 0:,} instances" for _, x in bands)
    write_section(out, "qfo_plddt_regime_table", {
        "id": "qfo_plddt_regime_table",
        "section_name": "Model-confidence regime by band",
        "description": (
            f"<p>The lines above as numbers, with each band's size "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                f"<b>Band sizes</b> — {counts}. These are shared by every arm: the bands "
                "partition the query proteins, not a tool's calls.",
                "<b>Peaks in</b> names the band each arm scores highest in. It is the "
                "column that separates a monotone dependence on structure quality from a "
                "regime, and reading it down the table is faster than reading the lines.",
                "<b>A band with few proteins carries few instances</b>, and no number "
                "here has a sampling error attached. Treat a gap smaller than the "
                "species-to-species spread in the leaderboard as unresolved.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_plddt_regime_table_table",
                    "title": "Fmax by pLDDT band", "col1_header": "Tool",
                    "sort_rows": False, "no_violin": True},
        "headers": headers,
        "data": table,
    })


# --- how much of its own accuracy an alphabet keeps out to the far proteome -----------
#
# The Divergence section draws one line per ARM, which on a full sweep is nineteen lines
# whose only visible difference is where they land. The quantity that separates them is
# not the level, it is how much of the level survives: on this benchmark two kmerseek arms
# on the SAME engine, the same k-mer machinery and the same index, differ by a factor of
# sixty at the far proteome purely in how coarse their alphabet is. Plotted as retention
# against alphabet size that is one curve, and it is an internal control -- no reviewer can
# attribute it to tool choice, because there is no tool choice in it.
#
# Retention alone is a trap, which is why the table beside the plot is not optional: a
# method that starts low and stays low retains 100%. The absolute Fmax at both ends is on
# every row so the ratio is never read without the level it is a ratio of.

# Two target proteomes is the minimum this section can exist on: one to start from and one
# to retain out to. A run with a single proteome has no divergence axis at all.
RETENTION_MIN_SPECIES = 2

# Fraction of the drawn y range two reference lines have to be apart before their labels
# are left separate. Deliberately generous: a pooled label is always readable and an
# overlapped pair never is, so the cost of merging one pair too many is a longer string
# and the cost of merging one too few is a label nobody can read.
RETENTION_LABEL_GAP = 0.08


def _hp_family(alphabet: str) -> str:
    """Which band of the alphabet axis a name belongs to, for the plot's three groups."""
    if alphabet.startswith("hp_"):
        return "HP (hydrophobic / polar)"
    if alphabet_classes(alphabet) >= 20:
        return "unreduced (20 classes)"
    return "other reduced alphabet"


RETENTION_GROUP_COLORS = {
    "HP (hydrophobic / polar)": "#0f9d76",
    "other reduced alphabet": "#0072b2",
    "unreduced (20 classes)": "#7f7f7f",
}


def _endpoint_fmax(cut: pl.DataFrame, near: float, far: float,
                   keys: list[str]) -> pl.DataFrame:
    """Fmax at the nearest and the most diverged proteome, one row per key.

    The low-complexity arm is collapsed first, by taking the better arm, which is the same
    convention best_encodings_per_species uses -- an encoding is the alphabet and the k,
    and the filter is a setting on it rather than a different encoding.
    """
    ends = cut.filter(pl.col("species_mya").is_in([near, far]))
    if ends.height == 0:
        return ends
    per = (ends.drop_nulls("fmax")
               .group_by(keys + ["species_mya"])
               .agg(pl.col("fmax").max()))
    return (per.group_by(keys).agg(
        pl.col("fmax").filter(pl.col("species_mya") == near).max().alias("fmax_near"),
        pl.col("fmax").filter(pl.col("species_mya") == far).max().alias("fmax_far")))


def _retention_rows(picked: pl.DataFrame, ends: pl.DataFrame) -> pl.DataFrame:
    """The plotted rows for one pair of endpoints: one per alphabet, with far/near.

    Split out of section_alphabet_retention so the endpoint search and the section body
    build the rows the same way rather than two copies drifting apart.
    """
    if ends.height == 0:
        return picked.head(0)
    return (picked.join(ends, on=["alphabet", "ksize"], how="inner")
                  .drop_nulls(["fmax_near", "fmax_far"])
                  .filter(pl.col("fmax_near") > 0)
                  .with_columns((pl.col("fmax_far") / pl.col("fmax_near"))
                                .alias("retention"))
                  .sort("n_classes"))


def _pick_far_endpoint(km_report: pl.DataFrame, picked: pl.DataFrame,
                       mya: list[float], near: float):
    """The most diverged proteome the plotted encodings can still be measured at.

    The furthest proteome in the run is the one this section wants, and on Swiss-Prot it
    is the one it gets. On the Pfam and Pfam-N truth sets of the midi-plus run it is not
    usable: every kmerseek arm scores exactly Fmax 0 at E. coli -- 0 of 812 Pfam-N rows
    have a single true-positive call there -- so `fmax_far` is 0.0 rather than null, the
    ratio is 0.0 for all nineteen alphabets at once, and the plot is a flat row of zeroes
    on the alphabet axis. The baselines are unaffected, because they do score at E. coli,
    so the section reads as "kmerseek retains nothing, structure retains a quarter" when
    what actually happened is that the numerator is below every arm's detection floor.

    The section already refuses to divide by a zero denominator. This is the same refusal
    on the numerator: step inward to the furthest proteome where at least one plotted
    encoding scores above zero, and hand back the ones stepped over so the caption can
    name them. An arm that is zero at a proteome where others are not keeps its zero --
    that is a real difference between alphabets, which is what the section is for.

    Returns (far, rows, skipped, saw_any_near). `far` is None when no proteome past
    `near` is usable; `saw_any_near` says whether that was the denominator's fault.
    """
    skipped: list[float] = []
    saw_any_near = False
    for cand in reversed(mya[1:]):
        ends = _endpoint_fmax(km_report, near, cand, ["alphabet", "ksize"])
        rows = _retention_rows(picked, ends)
        if rows.height:
            saw_any_near = True
            if rows["fmax_far"].max() > 0:
                return cand, rows, skipped, saw_any_near
        skipped.append(cand)
    return None, picked.head(0), skipped, saw_any_near


def _space_line_labels(lines: list[dict], gap: float) -> list[dict]:
    """Pool the labels of reference lines that sit too close to be read separately.

    MultiQC 1.35 renders a `y_lines` label as a plotly shape label centred on the line and
    keeps nothing but the text -- `FlatLine.parse_label` in multiqc/plots/plot.py drops
    every other field of a dict label as deprecated -- so there is no offset or anchor to
    nudge with. Two baselines a percentage point apart print their labels on top of each
    other, which is what "reseek (26%)" and "prostt5 (25%)" did here.

    The lines stay exactly where their values put them. Only the text moves: each cluster
    of lines within `gap` of each other gets one label on its topmost line carrying every
    name in the cluster, and the rest are drawn unlabelled.
    """
    ordered = sorted(lines, key=lambda ln: ln["value"])
    out: list[dict] = []
    cluster: list[dict] = []

    def flush():
        if not cluster:
            return
        text = " · ".join(ln["label"] for ln in cluster if ln.get("label"))
        for ln in cluster:
            out.append({**ln, "label": None})
        if text:
            out[-1]["label"] = text

    for ln in ordered:
        if cluster and ln["value"] - cluster[-1]["value"] < gap:
            cluster.append(ln)
            continue
        flush()
        cluster = [ln]
    flush()
    return out


def section_alphabet_retention(out: Path, metrics: pl.DataFrame) -> None:
    """Fraction of Fmax an alphabet keeps from the nearest proteome out to the furthest.

    The companion to section_encoding_vs_divergence and deliberately the other way round:
    that section asks which encoding won each proteome, this one asks what each alphabet
    keeps across them. Same variant parser, same class counts from `alphabet_classes`, and
    the same collapse of the low-complexity arm, so the two cannot disagree about what an
    encoding is. Per truth set, never pooled.

    Two proteomes only. Everything between them is the Divergence section's job, and a
    ratio of two endpoints is not a claim about the shape in between -- including any
    single proteome where an arm collapses for its own reasons.
    """
    if "species_mya" not in metrics.columns or "species" not in metrics.columns:
        return
    base = ungrouped(metrics)
    if base.height == 0:
        return

    for ts in sorted(base["truth_set"].unique().to_list()):
        whole = base.filter((pl.col("truth_set") == ts) & (pl.col("species") != "all")
                            & pl.col("species_mya").is_not_null())
        if whole.height == 0:
            continue
        mya = sorted(whole["species_mya"].unique().to_list())
        if len(mya) < RETENTION_MIN_SPECIES:
            continue
        near = mya[0]
        name_of = {r["species_mya"]: r["species"]
                   for r in whole.select("species_mya", "species").unique().to_dicts()}

        report, report_split = pick_split(whole)
        choose, choose_split = pick_selection_split(whole)
        km_report = parse_kmerseek_variants(report.filter(pl.col("tool") == "kmerseek"))
        km_choose = parse_kmerseek_variants(choose.filter(pl.col("tool") == "kmerseek"))
        if km_report.height == 0 or km_choose.height == 0:
            continue
        km_report, km_choose = encoding_axes(km_report), encoding_axes(km_choose)
        if km_report.height == 0 or km_choose.height == 0:
            continue

        # One k per alphabet, chosen on the selection half by mean Fmax over every
        # proteome -- not by Fmax at either endpoint, which would be choosing the answer.
        picked = (km_choose.drop_nulls("fmax")
                           .group_by("alphabet", "ksize", "n_classes")
                           .agg(pl.col("fmax").mean().alias("mean_fmax"))
                           .sort("mean_fmax", descending=True, nulls_last=True)
                           .group_by("alphabet", maintain_order=True).head(1))
        if picked.height == 0:
            continue
        # Retention is far/near, so a zero at EITHER end is a missing measurement rather
        # than a retention of zero: a zero denominator has no ratio at all, and a zero
        # numerator shared by every arm is a proteome past the whole method's detection
        # floor, which flattens the alphabet axis to a row of zeroes while the baselines
        # -- which do score there -- keep real values beside it.
        far, rows, skipped, saw_any_near = _pick_far_endpoint(km_report, picked, mya, near)
        if far is None:
            reason = (
                f"no kmerseek alphabet has a non-zero Fmax at "
                f"{name_of.get(near, near)} ({near:,.0f} Mya) on this truth set, and "
                "retention is that number's divisor. A ratio out of zero is not a "
                "retention of zero"
                if not saw_any_near else
                "no kmerseek alphabet has a non-zero Fmax at any target proteome past "
                f"{name_of.get(near, near)} ({near:,.0f} Mya) on this truth set, so the "
                "ratio has no numerator to report. A numerator of zero at every "
                "alphabet at once is a detection floor, not a retention of zero")
            write_section(out, f"qfo_retention_{ts}", {
                "id": f"qfo_retention_{ts}",
                "section_name": f"Divergence retention vs alphabet size — {ts} truth",
                "description": f"<p>{ts} truth, <code>{report_split}</code> split.</p>",
                "plot_type": "html",
                "data": (f"<p>Not plotted: {reason}, so nothing is drawn rather than a "
                         "row of zeroes.</p>"),
            }, supplementary=not is_primary_truth(ts))
            continue
        floored = sorted(skipped)
        floor_note = ""
        if floored:
            named = ", ".join(f"{name_of.get(m, m)} ({m:,.0f} Mya)" for m in floored)
            floor_note = (
                "<b>The furthest proteome in this run is not the far end of this "
                f"plot.</b> Every kmerseek alphabet scores Fmax 0 at {named}, so the "
                "ratio there is 0 for all of them and says nothing about the alphabet. "
                f"The far end shown is {name_of.get(far, far)} ({far:,.0f} Mya), the "
                "most diverged proteome "
                f"where any alphabet still scores, and the baseline lines are recomputed "
                f"on that same pair. Retention out to "
                f"{name_of.get(floored[-1], floored[-1])} is not measurable here for "
                f"kmerseek at all, which is a stronger statement than a low retention.")

        # Baselines: one variant each, the same two endpoints, on the reporting half.
        base_rows = report.filter(pl.col("tool") != "kmerseek")
        others = _endpoint_fmax(base_rows, near, far, ["tool"])
        if others.height:
            others = (others.drop_nulls(["fmax_near", "fmax_far"])
                            .filter(pl.col("fmax_near") > 0)
                            .with_columns((pl.col("fmax_far") / pl.col("fmax_near"))
                                          .alias("retention")))

        points = {}
        for r in rows.to_dicts():
            group = _hp_family(r["alphabet"])
            points[r["alphabet"]] = scatter_point(
                r["n_classes"], r["retention"],
                name=f"{r['alphabet']} k{r['ksize']}",
                group=group, color=RETENTION_GROUP_COLORS[group],
                annotation=f"{r['alphabet']} k{r['ksize']}",
                marker_size=13, marker_line_width=1)

        # One reference line per method class, the best retainer in it, so the plot
        # carries the comparison the claim has to survive without turning into a grid.
        lines = []
        if others.height:
            by_class = {}
            for r in others.to_dicts():
                cls = tool_class(r["tool"])
                if r["retention"] > by_class.get(cls, (None, -1.0))[1]:
                    by_class[cls] = (r["tool"], r["retention"])
            for cls, (tool, ret) in sorted(by_class.items()):
                lines.append({"value": ret, "color": CLASSES[cls][1], "dash": "dash",
                              "width": 2, "label": f"{tool} ({ret:.0%})"})
        if lines:
            span = max([ln["value"] for ln in lines]
                       + rows["retention"].to_list() + [0.0])
            lines = _space_line_labels(lines, span * RETENTION_LABEL_GAP)

        best = rows.sort("retention", descending=True).row(0, named=True)
        worst = rows.sort("retention").row(0, named=True)
        spread = (f"<b>In this run the spread across alphabets is "
                  f"{best['retention']:.0%} down to {worst['retention']:.1%}</b> — "
                  f"<code>{best['alphabet']}</code> ({best['n_classes']} classes, "
                  f"k{best['ksize']}) against <code>{worst['alphabet']}</code> "
                  f"({worst['n_classes']} classes, k{worst['ksize']}), on the same "
                  "engine, the same index and the same query set. Nothing but the "
                  "alphabet differs between those points, an internal control "
                  "rather than a comparison between tools.")
        same_split = choose_split == report_split
        split_note = (
            f"<b>Split</b> — k is chosen on the <code>{choose_split}</code> half and "
            f"retention is reported on the <code>{report_split}</code> half"
            + (". Those are the same rows for this truth set, which has no partition, so "
               "the reported retention is optimistically biased by however much the "
               "choice of k mattered."
               if same_split else
               ", so the alphabet's k is not chosen on the numbers being reported."))

        pconfig = {"id": f"qfo_retention_{ts}_plot",
                   "title": f"Divergence retention vs alphabet size ({ts})",
                   "xlab": "amino-acid classes in the alphabet",
                   "ylab": f"Fmax at {far:,.0f} Mya / Fmax at {near:,.0f} Mya",
                   "ymin": 0, "height": 520, "showlegend": True,
                   "xsuffix": "", "ysuffix": "",
                   "x_decimals": 0, "y_decimals": 3}
        if lines:
            pconfig["y_lines"] = lines
        write_section(out, f"qfo_retention_{ts}", {
            "id": f"qfo_retention_{ts}",
            "section_name": f"Divergence retention vs alphabet size — {ts} truth",
            "description": (
                f"<p>The fraction of its Fmax each kmerseek alphabet keeps from "
                f"{name_of.get(near, 'the nearest proteome')} ({near:,.0f} Mya) out to "
                f"{name_of.get(far, 'the furthest')} ({far:,.0f} Mya), against how many "
                f"amino-acid classes that alphabet collapses the 20 residues into "
                f"({ts} truth).</p>"
                + bullets(
                    floor_note,
                    "<b>y is a ratio and must never be read on its own.</b> A method that "
                    "starts low and stays low retains 100%. The table below carries the "
                    "absolute Fmax at both ends on every row, and that is the number that "
                    "says whether a high retention is worth anything.",
                    spread,
                    "<b>Dashed lines</b> are the best-retaining baseline in each method "
                    "class, on the same two proteomes. Structure-based methods retain "
                    "more in relative terms than any alphabet here, and that is the "
                    "honest frame: a coarse alphabet buys divergence robustness "
                    "approaching what structure buys, without needing a structure. It "
                    "does not beat structure at holding its own level.",
                    "<b>Green points are the HP family</b>, which reduce to hydrophobic "
                    "and polar; grey is the unreduced 20-class alphabet; blue is every "
                    "other reduced alphabet.",
                    "<b>One k per alphabet</b>, the k with the best mean Fmax over every "
                    "target proteome. Taking the best k separately at each endpoint would "
                    "manufacture retention out of two different encodings.",
                    "<b>Two proteomes, not nine.</b> A ratio of endpoints says nothing "
                    "about the shape between them, including any single proteome where an "
                    "arm collapses for its own reasons. The Divergence section carries "
                    "every point.",
                    split_note,
                    "<b>Class counts come from the alphabet name</b>, which has stated "
                    "them since kmerseek PR #43. A name that carries no count is left out "
                    "rather than defaulted, and the build log names it.")),
            "plot_type": "scatter",
            "pconfig": pconfig,
            "data": points,
        }, supplementary=not is_primary_truth(ts))

        near_col = f"{name_of.get(near, near)} ({near:,.0f} Mya)"
        far_col = f"{name_of.get(far, far)} ({far:,.0f} Mya)"
        headers = {
            "classes": {"title": "Classes", "format": "{:,.0f}", "scale": "Blues",
                        "description": "Amino-acid classes, or blank for a baseline"},
            "k": {"title": "k", "format": "{:,.0f}", "scale": "Greens",
                  "description": "K-mer length chosen for this alphabet"},
            "near": {"title": near_col, "format": "{:,.3f}", "scale": "RdYlGn",
                     "min": 0, "max": 1,
                     "description": "Fmax at the least diverged target proteome"},
            "far": {"title": far_col, "format": "{:,.3f}", "scale": "RdYlGn",
                    "min": 0, "max": 1,
                    "description": "Fmax at the far endpoint of this plot, the most "
                                   "diverged target proteome any alphabet still scores "
                                   "above zero at"},
            "retention": {"title": "Retention", "format": "{:,.1%}", "scale": "Purples",
                          "description": "Far divided by near. Meaningless without the "
                                         "two columns it is a ratio of"},
        }
        tbl = {}
        for r in rows.to_dicts():
            tbl[f"kmerseek {r['alphabet']} k{r['ksize']}"] = {
                "classes": r["n_classes"], "k": r["ksize"], "near": r["fmax_near"],
                "far": r["fmax_far"], "retention": r["retention"]}
        for r in (others.sort("retention", descending=True).to_dicts()
                  if others.height else []):
            tbl[r["tool"]] = {"classes": None, "k": None, "near": r["fmax_near"],
                              "far": r["fmax_far"], "retention": r["retention"]}
        write_section(out, f"qfo_retention_table_{ts}", {
            "id": f"qfo_retention_table_{ts}",
            "section_name": f"Divergence retention by alphabet — {ts} truth",
            "description": (
                f"<p>The plot above as numbers, with every baseline on the same two "
                f"proteomes ({ts} truth, <code>{report_split}</code> split).</p>"
                + bullets(
                    floor_note,
                    f"<b>{near_col} and {far_col}</b> are the absolute Fmax the retention "
                    "column is a ratio of. Read them first: a row retaining 90% of a "
                    "score of 0.05 has nothing to report.",
                    "<b>Retention</b> is far divided by near, per row, never pooled and "
                    "never averaged across rows.",
                    "<b>Baselines carry no class count or k</b>, because they have no "
                    "alphabet. They are here so the kmerseek rows are read against "
                    "something rather than against each other.")),
            "plot_type": "table",
            "pconfig": {"id": f"qfo_retention_table_{ts}_table",
                        "title": f"Divergence retention ({ts})",
                        "col1_header": "Arm", "sort_rows": False, "no_violin": True},
            "headers": headers,
            "data": tbl,
        }, supplementary=not is_primary_truth(ts))


def contiguous_k_axis(ks: list[int]) -> list[int]:
    """Every integer k between the smallest and largest run, so the axis is metric.

    A heatmap's x is a category axis: a cell is one column wide whether the next k is one
    step away or six. The sweep's k grid has holes, and drawing only the k that ran puts
    k=9 next to k=15 at the same spacing as k=9 next to k=10, which reads as a smooth
    gradient across a jump nothing was measured in. Filling the range makes an unrun k a
    blank column, which is what it is. The range is capped so a single outlying k cannot
    turn the panel into a field of blanks.
    """
    if not ks:
        return []
    lo, hi = min(ks), max(ks)
    if hi - lo + 1 > MAX_CONTIGUOUS_K:
        return sorted(ks)
    return list(range(lo, hi + 1))


# Past this many columns a filled k axis costs more in unreadable width than it buys in
# honesty, and the panel falls back to plotting only the k that ran.
MAX_CONTIGUOUS_K = 40


def section_alphabet_matrix(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """The sweep itself: Fmax over alphabet x ksize, one panel.

    The low-complexity filter was a second dimension here, two heatmaps of the same grid
    with one switch flipped. It is gone from the main report: at each alphabet's own best k
    the entire effect of the filter is under 0.001 Fmax across the whole sweep, which is
    two orders of magnitude below the differences between alphabets this panel is read for.
    Each (alphabet, k) cell is now that combo's better arm, and the filter gets one
    supplementary panel below. Both arms stay in the metrics; only the report's second axis
    goes.
    """
    cut, split = pick_split(ungrouped(metrics.filter(
        (pl.col("truth_set") == primary_truth) & (pl.col("tool") == "kmerseek"))))
    if cut.height == 0:
        return
    parsed = parse_kmerseek_variants(cut)
    if parsed.height == 0:
        return

    # Mean over species first, then the better low-complexity arm. The other order would
    # let a cell take its filtered arm on one proteome and its unfiltered arm on another.
    per_arm = parsed.group_by("alphabet", "ksize", "lowcomp").agg(pl.col("fmax").mean())
    grid = per_arm.group_by("alphabet", "ksize").agg(pl.col("fmax").max())
    ks = contiguous_k_axis(sorted(grid["ksize"].unique().to_list()))
    # Ranked by each alphabet's best cell, so the top rows are the ones worth reading.
    alphas = (grid.group_by("alphabet").agg(pl.col("fmax").max())
                  .sort("fmax", descending=True)["alphabet"].to_list())
    lookup = {(r["alphabet"], r["ksize"]): r["fmax"] for r in grid.to_dicts()}
    rows = [[lookup.get((a, k)) for k in ks] for a in alphas]
    vmax = heat_max(rows)
    unrun = [k for k in ks if all(lookup.get((a, k)) is None for a in alphas)]
    unrun_note = (
        "<b>Blank columns are k the sweep did not run</b> at any alphabet: "
        + ", ".join(str(k) for k in unrun)
        + ". The axis carries every integer k between its ends so that neighbouring "
          "columns are one step apart in k wherever they touch."
    ) if unrun else ""

    write_section(out, "qfo_alphabet", {
        "id": "qfo_alphabet",
        "section_name": "Alphabet x ksize",
        "description": (
            f"<p>Fmax for every alphabet and k-mer size in the sweep ({primary_truth} "
            f"truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>Fmax (mean over target species)</b>, then the better of the two "
                "low-complexity arms for that alphabet and k.",
                "<b>Rows</b> are alphabets ordered by their own best cell, <b>columns</b> "
                "are k-mer sizes.",
                "<b>Blank cells inside an alphabet's row</b> are k outside that alphabet's "
                "range. The floor is set from measured bits per symbol, not from class "
                "count, so a 2-letter alphabet starts at k=18 while protein20 starts "
                "at k=4.",
                unrun_note,
                "<b>The low-complexity filter is collapsed here.</b> At each alphabet's "
                "own best k the whole effect of turning it on is under 0.001 Fmax "
                "anywhere in the sweep. Carrying it as a second axis doubled this panel "
                "and every panel downstream of it to show a difference smaller than the "
                "third decimal. The supplementary panel below is the measurement.",
                heat_range_note(vmax))),
        "plot_type": "heatmap",
        "pconfig": {"id": "qfo_alphabet_plot",
                    "title": "Fmax by alphabet and ksize",
                    "xlab": "k", "ylab": "alphabet",
                    "min": 0, "max": vmax, "colstops": SEQUENTIAL_COLSTOPS,
                    "square": False, "height": 500},
        "xcats": [str(k) for k in ks],
        "ycats": alphas,
        "data": rows,
    })

    section_lowcomplexity_delta(out, parsed, primary_truth, split)


def section_lowcomplexity_delta(out: Path, parsed: pl.DataFrame, primary_truth: str,
                                split: str) -> None:
    """One panel for the whole low-complexity story, because the result is negative.

    This replaces two figures: a 19-dataset switcher of paired bars, one panel per
    alphabet, and a per-alphabet delta bar chart. Both were readable and neither was worth
    opening. Recomputing the toggle at each alphabet's own best k -- the k anyone would
    actually run -- the largest effect anywhere in the sweep is under 0.001 Fmax, which is
    two orders of magnitude below the differences between alphabets the rest of the report
    turns on.

    The k grid is kept rather than collapsed to one number per alphabet, because the two
    facts a reader needs are different: the toggle does nothing at the k each alphabet
    operates at, AND it does a great deal at k values far below that, where Fmax is near
    zero anyway. A per-alphabet bar can only carry the first, and it invites reading the
    second off the sweep's low-k tail as though it were a result. One grid carries both.
    """
    grid = parsed.group_by("alphabet", "ksize", "lowcomp").agg(pl.col("fmax").mean())
    wide = grid.pivot(on="lowcomp", index=["alphabet", "ksize"], values="fmax")
    if not {"True", "False"}.issubset(set(wide.columns)):
        return
    wide = wide.with_columns((pl.col("True") - pl.col("False")).alias("delta"))
    cells = {(r["alphabet"], r["ksize"]): r["delta"] for r in wide.to_dicts()}
    ks = contiguous_k_axis(sorted({k for _, k in cells}))
    # Best-k first, so the rows a reader would ever run are at the top of the grid.
    # "At the k you would choose" has to mean a PAIRED comparison at one k, not the change
    # in an alphabet's best achievable Fmax. The two differ, and the difference is the
    # trap: an alphabet whose filtered arm peaks at a different k from its unfiltered arm
    # gets credited with a jump that is a k choice rather than a filter effect. gbmr7 is
    # that case. So the k is fixed to where the UNFILTERED arm is best -- the k anyone
    # sweeping without the filter would land on -- and the number reported is what turning
    # the filter on does there.
    best_k, at_best = {}, {}
    for r in wide.to_dicts():
        off = r.get("False")
        if off is None or r["delta"] is None:
            continue
        prev = best_k.get(r["alphabet"])
        if prev is None or off > prev:
            best_k[r["alphabet"]] = off
            at_best[r["alphabet"]] = r["delta"]
    if not at_best:
        return
    alphas = sorted(at_best, key=lambda a: -abs(at_best[a]))
    rows = [[cells.get((a, k)) for k in ks] for a in alphas]
    if not any(v is not None for row in rows for v in row):
        return

    worst = max(at_best, key=lambda a: abs(at_best[a]))
    hp_worst = max((a for a in at_best if a.startswith("hp_")),
                   key=lambda a: abs(at_best[a]), default=None)
    hp_note = (f"Across the HP alphabets the largest is <code>{hp_worst}</code> at "
               f"{at_best[hp_worst]:+.4f}." if hp_worst else "")
    span = max(abs(v) for row in rows for v in row if v is not None)
    write_section(out, "qfo_lowcomplexity", {
        "id": "qfo_lowcomplexity",
        "section_name": "Supplementary: low-complexity filter",
        "description": (
            f"<p>Fmax with low-complexity k-mers removed minus Fmax with them kept, for "
            f"every alphabet and k in the sweep ({primary_truth} truth, "
            f"<code>{split}</code> split, mean over target species).</p>"
            f"<p><b>Methods sentence.</b> Every kmerseek arm was swept with and without "
            f"the low-complexity k-mer filter; at each alphabet's own best k the filter "
            f"changes Fmax by at most {max(abs(v) for v in at_best.values()):.4f}, so the "
            f"main report collapses the two arms to the better of the pair and this panel "
            f"is the whole record of the comparison.</p>"
            + bullets(
                "<b>This is the report's only low-complexity panel.</b> It was a second "
                "axis on the alphabet grid, on the k-mer spectra, and on nineteen figures "
                "downstream of those. The measurement below is why they are gone, not a "
                "reason to put them back.",
                "<b>The result is negative and that is the finding.</b> Fixing each "
                "alphabet at the k where its UNFILTERED arm scores best -- the k anyone "
                "sweeping would land on -- and turning the filter on there, the largest "
                f"effect anywhere in the run is <code>{worst}</code> at "
                f"{at_best[worst]:+.4f}. " + hp_note + " That is two orders of magnitude "
                "below the differences between alphabets the rest of this report rests "
                "on. "
                "At the k you would choose, the toggle does nothing.",
                f"<b>The strong cells are at other k</b>, and <code>{span:.3f}</code> is "
                "the largest single cell in this grid. Where removing homopolymer-like "
                "k-mers does move Fmax, it moves a combo whose unfiltered score is near "
                "zero to another number that is still near zero, and it does not reach "
                "the k that alphabet is actually run at. Comparing an alphabet's best "
                "filtered combo with its best unfiltered one credits the filter with that "
                "k change, "
                "which is why the number above is a paired comparison at one k instead.",
                "<b>Rows are ordered by the size of the best-k effect</b>, largest first, "
                "so the top row is the strongest case the toggle has anywhere.",
                "<b>Positive</b> means removal helped. The scale is diverging and "
                "symmetric because the sign is the reading.",
                "This was two figures -- paired bars per alphabet behind a 19-button "
                "switcher, and a per-alphabet delta bar. Neither said anything this one "
                "does not, and a legible figure of a negligible effect is still a figure "
                "of a negligible effect.")),
        "plot_type": "heatmap",
        "pconfig": {"id": "qfo_lowcomplexity_plot",
                    "title": "Fmax change from low-complexity removal",
                    "xlab": "k", "ylab": "alphabet",
                    "min": -span, "max": span, "colstops": DIVERGING_COLSTOPS,
                    # The default two decimals renders every cell of a negative result as
                    # "0.00", which reads as missing data rather than as the finding.
                    "tt_decimals": 4,
                    "square": False, "height": 500},
        "xcats": [str(k) for k in ks],
        "ycats": alphas,
        "data": rows,
    })


# ---------------------------------------------------------------------------
# Reduced-alphabet information ceiling
# ---------------------------------------------------------------------------
#
# The thesis these three panels exist to measure, rather than assert: HP-alphabet
# performance is a function of the TARGET FEATURE's length and type, not of the alphabet
# alone.
#
# Rannon & Burstein (bioRxiv 2026.02.08.701987v2, doi 10.64898/2026.02.08.701987) trained
# protein language models on reduced alphabets and found their 2-letter model worst on
# signal peptides (ROC-AUC 0.75, PR-AUC 0.47) while nearly lossless on solubility (relative
# F1 ~0.97) and strong on enzyme detection (~0.90). Signal peptides are ~20 residues;
# solubility and enzyme class are whole-protein properties. Their BPE tokens are short and
# our HP k floor is 18, so if that is one gradient rather than three unrelated task
# results, their negative result is the low-k arm of this sweep measured independently by
# another lab. These panels put both gradients in domain units so the comparison is a
# measurement instead of an analogy.
#
# No expected ordering is encoded anywhere below. The numbers are emitted and fall where
# they fall.
CEILING_PARENT = {
    "parent_id": "qfo_ceiling",
    "parent_name": "Reduced-alphabet information ceiling",
    "parent_description": (
        "<p>Whether a coarse alphabet works is a question about the feature being found, "
        "not about the alphabet on its own.</p>"
        + bullets(
            "A 2-letter alphabet at k=19 spans 19 residues, so a 21-residue TRANSMEM helix "
            "admits three k-mers and a 400-residue kinase domain admits 380.",
            "These panels cut the sweep on <b>feature length</b> and on <b>feature "
            "type</b>, to see whether that is where the reduced alphabets lose.")
    ),
}

# A cell is "containment-scored" once point features are the majority of its instances.
# evaluate_domain_calls scores a point instance by whether the call covers the annotated
# residue, because IoU against a 1-residue interval is arithmetically unsatisfiable -- so
# those numbers answer a different question from the interval cells and must never share an
# axis or a colour scale with them. On the mini run ACT_SITE comes out ABOVE DOMAIN, which
# is a criterion difference and not a result, and putting the two in one heatmap row is how
# that becomes a claim nobody meant to make.
POINT_MAJORITY = 0.5


def is_containment_scored(df: pl.DataFrame) -> pl.Expr:
    """point_fraction past the majority mark, false when the column predates this."""
    if "point_fraction" not in df.columns:
        return pl.lit(False)
    return pl.col("point_fraction").fill_null(0.0) > POINT_MAJORITY


def _ratio_bin(col: pl.Expr) -> pl.Expr:
    """Snap a feature_length/ksize ratio to a log2 grid, returning the bin's centre ratio.

    Log2, not linear. The ratio spans roughly 0.03 (a one-residue point feature against
    k=30) to 60 (titin's longest domain against k=18), and a linear grid would put every
    short feature -- the whole left-hand end the claim is about -- into one column.
    """
    return (2.0 ** col.log(2).round(0)).round(4)


def _ratio_label(ratio: float) -> str:
    """A log2 bin's name, as a fraction below 1 and a plain integer at or above it.

    The x axis used to be a numeric log axis, and Plotly drew log10 MINOR ticks on it: the
    labels read 2, 3, ... 9, 1, 2, ... 9, 10, which is a decade's worth of minor ticks
    repeated across a grid whose steps are powers of two. Nothing about that is readable.
    The bins are already snapped to powers of two, so they are equally spaced by
    construction and belong on a category axis with one label per bin.
    """
    if ratio >= 1:
        return f"{ratio:g}"
    return f"1/{round(1 / ratio):g}"


def _ratio_categories(ratios) -> list[str]:
    """Bin labels in ascending numeric order, which is the order the axis has to be in."""
    return [_ratio_label(r) for r in sorted(set(ratios))]


def on_categories(series: dict, cats: list[str]) -> dict:
    """A line's points over the FULL category list, with null where it has no value.

    A MultiQC line plot with `categories: True` becomes a Plotly category axis, and Plotly
    takes the category ORDER from the traces: the first trace's x values in order, then any
    value a later trace introduces, appended at the end. A series that skips a bin the next
    series has therefore reorders the axis -- which is how a log2 grid came out reading
    1/2, 1, 2, 4, 8, 32, 1/4, 16. Emitting every series over the same ordered list fixes
    the order at the first trace and keeps it. Nulls survive into the pairs and Plotly
    draws them as gaps, so a line still stops where its data stops.
    """
    return {c: series.get(c) for c in cats}


def section_ceiling_length(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """best_f1 against feature_length / ksize, one line per HP alphabet.

    The RATIO, not the raw length. Raw length would show every alphabet declining together
    toward short features, which is true and uninformative -- every method finds short
    things less reliably. The claim under test is specifically that a coarse alphabet needs
    a long window, so the quantity is how many k-mers the feature can hold, and that is
    feature_length / ksize. A ratio of 1 is a feature exactly one k-mer long.

    Coverage rides along as a switchable second dataset rather than as a footnote. A
    best_f1 computed over 12% of the calls in a cell is a different claim from the same
    number over 90%, and the short-feature cells are exactly where coverage drops.
    """
    cut, split = pick_split(metrics.filter(
        (pl.col("truth_set") == primary_truth)
        & (pl.col("stratum_axis") == "feature_length_bin")
        & (pl.col("tool") == "kmerseek")
    ))
    if cut.height == 0 or "median_feature_length" not in cut.columns:
        return
    # The point-feature bin is dropped from the ratio curve, not plotted at ratio ~0.05.
    # It is scored by containment while every other bin is scored by placement, so it is
    # not the left-hand end of this curve -- it is a different measurement that happens to
    # sit at a small feature length. The feature-type panels report it on its own scale.
    parsed = parse_kmerseek_variants(cut).filter(
        pl.col("alphabet").str.starts_with("hp_")
        & pl.col("median_feature_length").is_not_null()
        & (pl.col("median_feature_length") > 0)
        & ~is_containment_scored(cut)
    )
    if parsed.height == 0:
        return

    parsed = parsed.with_columns(
        _ratio_bin(pl.col("median_feature_length") / pl.col("ksize")).alias("ratio")
    )
    alphas = sorted(parsed["alphabet"].unique().to_list(), key=alphabet_classes)

    cats = _ratio_categories(parsed["ratio"].to_list())
    datasets, labels = [], []
    for metric, ylab in (("best_f1", "best F1"), ("coverage", "coverage")):
        if metric not in parsed.columns:
            continue
        data = {}
        for alpha in alphas:
            sub = parsed.filter(pl.col("alphabet") == alpha)
            # Averaged over ksize, low-complexity arm and target species within a ratio
            # bin, because the ratio is the axis: two combos landing on the same ratio by
            # different routes are two measurements of the same quantity.
            by_ratio = (sub.group_by("ratio").agg(pl.col(metric).mean())
                           .sort("ratio").to_dicts())
            series = {_ratio_label(r["ratio"]): r[metric] for r in by_ratio
                      if r[metric] is not None}
            if series:
                data[alpha] = on_categories(series, cats)
        if data:
            datasets.append(data)
            labels.append({"name": ylab, "ylab": ylab})
    if not datasets:
        return

    n_cells = parsed.height
    cov = parsed["coverage"].median() if "coverage" in parsed.columns else None
    cov_note = f", median coverage {cov:.2f}" if cov is not None else ""
    write_section(out, "qfo_ceiling_length", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_length",
        "section_name": "Feature length against k",
        "description": (
            f"<p>Best achievable F1 by how many k-mers the annotated feature can hold "
            f"({primary_truth} truth, <code>{split}</code> split, {n_cells} scored "
            f"cells{cov_note}).</p>"
            + bullets(
                "<b>x axis</b> is the median feature length in the cell divided by the "
                "variant's k, snapped to a log2 grid: <code>"
                + "</code>, <code>".join(cats) + "</code>. <code>1</code> is a feature "
                "exactly one k-mer long, and the bins are equally spaced because each is "
                "twice the one before it.",
                "<b>One line per HP alphabet.</b>",
                "<b>Buttons</b> switch between best F1 and the coverage each number was "
                "computed over.",
                "<b>Truth sets are never pooled.</b> This panel is one truth set only, and "
                "the feature-type panel below is Swiss-Prot only because Pfam carries no "
                "type variation to cut on.",
                "<b>The point-feature bin is not the left-hand end of this curve</b> and "
                "is not drawn on it. Point instances are scored by containment rather than "
                "placement, so they answer a different question and belong on their own "
                "scale.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_ceiling_length_plot",
                    "title": "best F1 by feature length / k",
                    "xlab": "feature length / k (log2 bins)", "ylab": "best F1",
                    "categories": True, "ymin": 0, "ymax": 1, "height": 500,
                    "data_labels": labels},
        "data": datasets if len(datasets) > 1 else datasets[0],
    })


def section_ceiling_length_by_k(out: Path, metrics: pl.DataFrame,
                               primary_truth: str) -> None:
    """The same axis, one line per ksize instead of averaged over them.

    This is the panel that decides what the averaged one means, and it has to exist beside
    it rather than instead of it.

    If the curves for every k COLLAPSE onto each other against feature_length / ksize, then
    the ratio is the sufficient statistic: k trades against feature length one for one, and
    there is no k-floor to read here -- only a statement about how many k-mers a feature has
    to hold. If they SEPARATE, there is an absolute-k effect on top of the ratio, and the k
    at which the curves stop improving is a k-floor measured on annotated domains rather
    than derived from keyspace arithmetic. Those are different claims and the averaged
    panel cannot tell them apart, because averaging over k inside a ratio bin is precisely
    the operation that hides the separation.

    One dataset per alphabet behind a switcher, for the same reason the low-complexity
    section uses one: 7 HP alphabets x 12 ksizes is 84 lines in a single plot.
    """
    cut, split = pick_split(metrics.filter(
        (pl.col("truth_set") == primary_truth)
        & (pl.col("stratum_axis") == "feature_length_bin")
        & (pl.col("tool") == "kmerseek")
    ))
    if cut.height == 0 or "median_feature_length" not in cut.columns:
        return
    parsed = parse_kmerseek_variants(cut).filter(
        pl.col("alphabet").str.starts_with("hp_")
        & pl.col("median_feature_length").is_not_null()
        & (pl.col("median_feature_length") > 0)
        # Same exclusion as the averaged panel, for the same reason.
        & ~is_containment_scored(cut)
    )
    if parsed.height == 0:
        return
    parsed = parsed.with_columns(
        _ratio_bin(pl.col("median_feature_length") / pl.col("ksize")).alias("ratio")
    )

    cats = _ratio_categories(parsed["ratio"].to_list())

    def lines_by_k(frame: pl.DataFrame) -> dict:
        """One series per k over the ratio axis, dropping k values with a single point."""
        out_lines = {}
        for k in sorted(frame["ksize"].unique().to_list()):
            by_ratio = (frame.filter(pl.col("ksize") == k)
                             .group_by("ratio").agg(pl.col("best_f1").mean())
                             .sort("ratio").to_dicts())
            series = {_ratio_label(r["ratio"]): r["best_f1"] for r in by_ratio
                      if r["best_f1"] is not None}
            # A single point cannot show a plateau or a collapse, and a legend entry for it
            # costs more than it carries.
            if len(series) > 1:
                out_lines[f"k={k}"] = on_categories(series, cats)
        return out_lines

    # The pooled panel FIRST, so the default view answers the section's question once
    # instead of asking a reader to open seven near-identical per-alphabet panels and hold
    # them in their head. The per-alphabet datasets stay behind the buttons, because the
    # pooled view cannot rule out one alphabet behaving differently from the rest.
    datasets, labels = [], []
    pooled = lines_by_k(parsed)
    if pooled:
        datasets.append(pooled)
        labels.append({"name": "all HP alphabets", "ylab": "best F1"})
    for alpha in sorted(parsed["alphabet"].unique().to_list(), key=alphabet_classes):
        data = lines_by_k(parsed.filter(pl.col("alphabet") == alpha))
        if data:
            datasets.append(data)
            labels.append({"name": alpha, "ylab": "best F1"})
    if not datasets:
        return

    write_section(out, "qfo_ceiling_length_by_k", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_length_by_k",
        "section_name": "Feature length against k, per k",
        "description": (
            f"<p>The panel above, split by k instead of averaged over it "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>The first panel pools every HP alphabet</b> and is the one to read. "
                "The per-alphabet panels behind the buttons exist to check that the pooled "
                "answer is not one alphabet's behaviour averaged over six that disagree.",
                "<b>Buttons</b> switch alphabet; <b>one line per k</b>, named in the "
                "legend.",
                "<b>x axis</b> is on the same log2 bins as the panel above: <code>"
                + "</code>, <code>".join(cats) + "</code>.",
                "<b>Read it for one thing:</b> do the lines lie on top of each other or "
                "not?",
                "<b>If they collapse</b>, feature_length / k is the whole story and k "
                "trades against feature length one for one.",
                "<b>If they separate</b>, there is an absolute-k effect on top of the "
                "ratio, and the k at which the curves stop improving is a k floor measured "
                "on annotated domains rather than derived from keyspace arithmetic.",
                "The averaged panel cannot distinguish those, because averaging over k "
                "inside a ratio bin is exactly what hides the separation.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_ceiling_length_by_k_plot",
                    "title": "best F1 by feature length / k, per k",
                    "xlab": "feature length / k (log2 bins)", "ylab": "best F1",
                    "categories": True, "ymin": 0, "ymax": 1, "height": 500,
                    "showlegend": True, "data_labels": labels},
        "data": datasets if len(datasets) > 1 else datasets[0],
    })


def section_ceiling_feature_type(out: Path, metrics: pl.DataFrame) -> None:
    """Alphabet x Swiss-Prot feature type, and the coverage the same grid was scored over.

    Swiss-Prot only, and not because it is the default primary truth set: `pfam_id` holds
    the FT type there, while for Pfam and Pfam-N it holds a family accession with no type
    variation to cut on and for M-CSA an entry id. evaluate_domain_calls leaves the axis
    null on those sets, so there is nothing to plot even when one of them is primary.

    Rows are ordered by class count, coarsest at the top, so any narrowing of the gap
    between long structural features and short functional ones as the alphabet grows reads
    down the figure.
    """
    sp = metrics.filter(
        (pl.col("truth_set") == "swissprot")
        & (pl.col("stratum_axis") == "feature_type")
        & (pl.col("tool") == "kmerseek")
    )
    if sp.height == 0:
        return
    cut, split = pick_split(sp)
    parsed = parse_kmerseek_variants(cut)
    if parsed.height == 0:
        return

    # Types ordered by median feature length, shortest first, so the x axis is itself the
    # length gradient rather than an alphabetical list. Ties and missing lengths sort last.
    if "median_feature_length" in parsed.columns:
        lengths = (parsed.group_by("stratum")
                         .agg(pl.col("median_feature_length").median().alias("len"))
                         .sort("len", nulls_last=True))
        types = lengths["stratum"].to_list()
    else:
        types = sorted(parsed["stratum"].unique().to_list())
    alphas = sorted(parsed["alphabet"].unique().to_list(), key=alphabet_classes)

    # Split the columns by scoring criterion before anything is drawn. ACT_SITE and BINDING
    # are scored by containment and DOMAIN by placement; on the mini run that puts ACT_SITE
    # above DOMAIN, which is a criterion difference and not a result. Sliding an eye across
    # one colour scale from one to the other is exactly the misreading this benchmark's
    # "truth sets are never pooled" rule exists to prevent, so the two go in separate
    # figures rather than in one with a footnote.
    point_types = set(
        parsed.filter(is_containment_scored(parsed))["stratum"].unique().to_list()
    )
    groups = [
        ("", "placement (IoU)", [t for t in types if t not in point_types]),
        ("_point", "containment", [t for t in types if t in point_types]),
    ]

    for group_suffix, criterion, group_types in groups:
        if not group_types:
            continue
        # Coverage only for the placement grid: the containment grid is small, and a second
        # figure per criterion is more navigation than the point rows can repay.
        metrics_here = ((("best_f1", "best F1"), ("coverage", "coverage"))
                        if group_suffix == "" else (("best_f1", "best F1"),))
        _feature_type_heatmaps(out, parsed, alphas, group_types, group_suffix,
                               criterion, metrics_here, split)


def _feature_type_heatmaps(out, parsed, alphas, types, group_suffix, criterion,
                           metrics_here, split) -> None:
    """One heatmap per metric, for a single scoring criterion's columns.

    `criterion` is not decoration. Placement columns and containment columns are separate
    figures precisely so that nothing invites reading across them, and the only thing left
    to stop a reader assuming otherwise is the figure saying which one it is.
    """
    sub = parsed.filter(pl.col("stratum").is_in(types))
    n_by_type = (
        sub.group_by("stratum").agg(pl.col("n_truth_instances").max().alias("n")).to_dicts()
        if "n_truth_instances" in sub.columns else []
    )
    counts = ", ".join(f"{r['stratum']} n={r['n']}"
                       for r in sorted(n_by_type, key=lambda r: -(r["n"] or 0)))
    scored_by = (
        ["<b>Scored by containment</b> — a point feature asserts a residue, and the "
         "question is whether the call covered it. IoU against a 1-residue interval is "
         "1/call_length and therefore unsatisfiable at any sane cutoff, which is why these "
         "columns are a separate figure rather than the left-hand end of the placement "
         "grid.",
         "<b>Do not compare these numbers with the placement heatmap.</b> Containment is "
         "the easier criterion, and a higher number here is not a better result there."]
        if group_suffix else
        ["<b>Scored by placement</b> — the call has to coincide with the annotated "
         "interval, not merely overlap it.",
         "Point features are in their own figure below, on the containment criterion."]
    )

    # These two used to share one 0..1 colour scale, on the reasoning that both metrics are
    # bounded there and could therefore be read against each other. Bounded is not the same
    # as occupied. The placement best-F1 grid tops out at about 0.28 while coverage on the
    # same grid reaches 0.84, so a shared range spends three quarters of the ramp on values
    # best F1 never takes and every cell of it lands in the first two stops -- a uniform
    # slab with no readable ordering, which was the single loudest complaint about this
    # report. Sharing the range with coverage does not survive that: the two are different
    # quantities and a colour match between "F1 0.28" and "coverage 0.28" was never a
    # comparison worth protecting. Each grid now gets its own range, both panels print
    # their numbers in the cells, and each description names both ranges so nobody reads a
    # shade across the pair.
    grids, vmaxes = {}, {}
    for metric, _ in metrics_here:
        if metric not in sub.columns:
            continue
        grid = sub.group_by("alphabet", "stratum").agg(pl.col(metric).mean())
        lookup = {(r["alphabet"], r["stratum"]): r[metric] for r in grid.to_dicts()}
        grids[metric] = [[lookup.get((a, t)) for t in types] for a in alphas]
        vmaxes[metric] = heat_max(grids[metric])
    if not any(v is not None for v in vmaxes.values()):
        return
    other = "; ".join(
        f"the {'best-F1' if m == 'best_f1' else m} panel runs 0 to {v:.2f}"
        for m, v in vmaxes.items() if v is not None)

    for metric, title in metrics_here:
        rows = grids.get(metric)
        vmax = vmaxes.get(metric)
        if rows is None or vmax is None:
            continue
        suffix = group_suffix + ("" if metric == "best_f1" else "_coverage")
        if metric == "best_f1":
            body = (
                f"<p>{title} per alphabet and Swiss-Prot feature type, "
                f"<code>{split}</code> split.</p>"
                + bullets(
                    *scored_by,
                    "<b>Rows</b> run coarsest alphabet at the top to finest at the bottom.",
                    "<b>Columns</b> run shortest median feature on the left to longest on "
                    "the right.",
                    "<b>The MIN_STRATUM_PROTEINS floor is waived on this axis.</b> "
                    "ACT_SITE and DNA_BIND are small in every proteome, and dropping them "
                    "would delete the short-feature end of the gradient.",
                    heat_range_note(vmax),
                    f"<b>Each panel of this criterion is on its own range</b> — {other}. "
                    "Read the numbers in the cells across the pair, never the shades.",
                    f"<b>Instances per type</b> — {counts}.")
            )
        else:
            body = (
                "<p>Share of calls that could be judged at all, on the same grid as the "
                "best-F1 heatmap above.</p>"
                + bullets(
                    "A high F1 over a low coverage is a different claim from the same F1 "
                    "over a high one.",
                    heat_range_note(vmax),
                    f"<b>Each panel of this criterion is on its own range</b> — {other}. "
                    "Read the numbers in the cells across the pair, never the shades.")
            )
        write_section(out, f"qfo_ceiling_feature_type{suffix}", {
            **CEILING_PARENT,
            "id": f"qfo_ceiling_feature_type{suffix}",
            "section_name": f"Feature type ({criterion}) — {title}",
            "description": body,
            "plot_type": "heatmap",
            "pconfig": {"id": f"qfo_ceiling_feature_type{suffix}_plot",
                        "title": f"{title} by alphabet and feature type ({criterion})",
                        "xlab": "feature type", "ylab": "alphabet",
                        "min": 0, "max": vmax, "colstops": SEQUENTIAL_COLSTOPS,
                        "square": False, "height": 520},
            "xcats": types,
            "ycats": alphas,
            "data": rows,
        })


def section_feature_type_crossover(out: Path, metrics: pl.DataFrame) -> None:
    """The same numbers as the feature-type heatmaps, read as a crossover.

    The heatmaps carry the mechanism and bury it. Read along a row, a coarse alphabet is
    best on the extended patterned features and worst on the ones defined by which residue
    is where; read down a column, the ordering by alphabet reverses between those two kinds
    of feature. That reversal is the whole explanation of both the win and the loss -- a
    2-letter alphabet keeps a hydrophobic/polar PATTERN and throws away the residue
    IDENTITY that defines a catalytic site -- and a reader has to reconstruct it from a
    grid of 180 numbers to see it.

    Nothing here is asserted. Every feature type in the criterion is drawn, none picked;
    the direction each one runs is the Spearman correlation between the alphabet's class
    count and its best F1, computed on the spot, and the description reports whichever
    types come out running down and whichever come out running up.

    Placement and containment stay in separate datasets behind the switcher, for the reason
    the heatmaps state: ACT_SITE is scored by containment and TRANSMEM by placement, and a
    shared axis would turn a criterion difference into a claim nobody meant to make.
    """
    sp = metrics.filter(
        (pl.col("truth_set") == "swissprot")
        & (pl.col("stratum_axis") == "feature_type")
        & (pl.col("tool") == "kmerseek")
    )
    if sp.height == 0:
        return
    cut, split = pick_split(sp)
    parsed = parse_kmerseek_variants(cut)
    if parsed.height == 0 or "best_f1" not in parsed.columns:
        return

    alphas = sorted(parsed["alphabet"].unique().to_list(), key=alphabet_classes)
    classes = {a: alphabet_classes(a) for a in alphas}
    point_types = set(
        parsed.filter(is_containment_scored(parsed))["stratum"].unique().to_list()
    )
    grid = parsed.group_by("alphabet", "stratum").agg(pl.col("best_f1").mean())
    cell = {(r["alphabet"], r["stratum"]): r["best_f1"] for r in grid.to_dicts()}

    datasets, labels, notes = [], [], []
    for suffix, criterion in (("", "placement (IoU)"), ("_point", "containment")):
        types = sorted({t for _, t in cell
                        if (t in point_types) == bool(suffix)})
        if not types:
            continue
        data, down, up = {}, [], []
        for t in types:
            series = {a: cell[(a, t)] for a in alphas
                      if cell.get((a, t)) is not None}
            if len(series) < 3:
                continue
            data[t] = on_categories(series, alphas)
            rho = spearman_rho([classes[a] for a in series],
                               list(series.values()))
            if rho is None:
                continue
            lo = min(series.values())
            hi = max(series.values())
            entry = (f"<code>{t}</code> {lo:.2f}&ndash;{hi:.2f}, "
                     f"rho&nbsp;{rho:+.2f}")
            (down if rho < 0 else up).append(entry)
        if not data:
            continue
        datasets.append(data)
        labels.append({"name": criterion, "ylab": "best F1"})
        notes.append((criterion, down, up))
    if not datasets:
        return

    # The direction each type runs is read off the data; nothing about WHICH types land in
    # which group is written here. The mechanism is stated once, as the reading a split
    # supports, so that a run where the split comes out differently does not carry a
    # sentence asserting a story its own numbers contradict.
    bullet_lines = []
    for criterion, down, up in notes:
        if down:
            bullet_lines.append(
                f"<b>{criterion}, coarser alphabet scores higher</b> (rho below zero) — "
                + "; ".join(down) + ".")
        if up:
            bullet_lines.append(
                f"<b>{criterion}, finer alphabet scores higher</b> (rho above zero) — "
                + "; ".join(up) + ".")

    write_section(out, "qfo_ceiling_crossover", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_crossover",
        "section_name": "Where the coarse alphabet wins and where it loses",
        "description": (
            f"<p>Best achievable F1 against alphabet size, one line per Swiss-Prot feature "
            f"type (swissprot truth, <code>{split}</code> split, averaged over ksize, "
            f"low-complexity arm and target species).</p>"
            + bullets(
                "<b>x axis</b> runs coarsest alphabet on the left to finest on the right. "
                "<b>Lines that fall</b> to the right are features a coarse alphabet is "
                "better at; <b>lines that rise</b> are features it is worse at. The two "
                "directions in one figure are the mechanism.",
                "<b>rho</b> beside each type is the Spearman correlation between the "
                "alphabet's class count and its best F1 over the alphabets in the sweep. "
                "It is computed from these numbers, not asserted.",
                *bullet_lines,
                "<b>What a split in those two directions would mean.</b> A 2-letter "
                "encoding keeps the hydrophobic/polar PATTERN along a sequence and "
                "discards which residue is at each position. Feature types defined by an "
                "extended run whose alternation is the signal would then survive the "
                "encoding, and types defined by the identity of one or a few residues "
                "would not. That is one mechanism accounting for a win and a loss at once, "
                "and this figure is where it can be checked rather than assumed: read the "
                "two lists above and see whether they sort that way.",
                "<b>Buttons</b> switch scoring criterion. Placement requires the call to "
                "coincide with the annotated interval; containment asks only whether the "
                "call covered the asserted residue, because IoU against a 1-residue "
                "interval is unsatisfiable. <b>Never read a number from one against a "
                "number from the other.</b>",
                "<b>Every feature type in the criterion is drawn</b>, none selected, so "
                "the pattern is not an artefact of which lines were chosen.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_ceiling_crossover_plot",
                    "title": "best F1 by feature type against alphabet size",
                    "xlab": "alphabet (coarsest to finest)", "ylab": "best F1",
                    "categories": True, "ymin": 0, "height": 520,
                    "showlegend": True, "data_labels": labels},
        "data": datasets if len(datasets) > 1 else datasets[0],
    })


def section_ceiling_recognition(out: Path, metrics: pl.DataFrame,
                                primary_truth: str) -> None:
    """Recognition against delineation per alphabet: family Fmax, Fmax, and the gap.

    `fmax` gates on interval placement, so it scores "never recognised this family" and
    "recognised it, drew the boundary wrong" identically at zero. `family_fmax` ignores
    placement entirely. The difference between them is therefore what boundary placement
    costs a given alphabet, and it is the quantity this parent exists to measure: an
    alphabet that recognises families as well as protein20 but cannot delineate them has a
    large gap, while one that has genuinely lost the family signal has a small gap and a
    low family Fmax. Those are different failures and one number cannot tell them apart.

    Every alphabet is drawn, not just the HP ones. protein20 is the reference the gap is
    read against; dropping it would leave the HP numbers with nothing to be large or small
    compared to.
    """
    cut, split = pick_split(ungrouped(metrics.filter(
        (pl.col("truth_set") == primary_truth) & (pl.col("tool") == "kmerseek"))))
    if cut.height == 0 or "family_fmax" not in cut.columns:
        return
    parsed = parse_kmerseek_variants(cut).with_columns(
        (pl.col("family_fmax") - pl.col("fmax")).alias("family_gap")
    )
    if parsed.height == 0:
        return
    alphas = sorted(parsed["alphabet"].unique().to_list(), key=alphabet_classes)

    # Averaged over ksize, low-complexity arm and target species. Each alphabet's row in
    # the heatmap below keeps the ksize axis, so the averaging here is not the only view.
    per_alpha = parsed.group_by("alphabet").agg(
        pl.col("fmax").mean(), pl.col("family_fmax").mean(), pl.col("family_gap").mean(),
        pl.col("coverage").mean() if "coverage" in parsed.columns else pl.lit(None).alias("coverage"),
        pl.col("n_family_truth").median().alias("n_family_truth"),
        pl.col("n_family_calls").median().alias("n_family_calls"),
        pl.len().alias("n_cells"),
    )
    lookup = {r["alphabet"]: r for r in per_alpha.to_dicts()}

    # Ordered by the gap, widest at the top, not by class count and not alphabetically.
    # MultiQC sorts bar plot samples by name unless told otherwise, which is what put an
    # alphabetical axis on the one figure whose whole content is an ordering: the bars run
    # from the alphabets that recognise families and cannot place them down to the ones
    # that lose nothing to placement, and that ranking is unreadable when dayhoff6 comes
    # first because d sorts before g. `sort_samples: False` in the pconfig is the half of
    # this that makes the order survive into the plot.
    ordered_by_gap = [name for name in
                      sorted((a for a in alphas if a in lookup),
                             key=lambda a: (lookup[a]["family_gap"] is None,
                                            -(lookup[a]["family_gap"] or 0.0)))]
    levels = {name: lookup[name] for name in ordered_by_gap}
    if not levels:
        return
    datasets = [
        {a: {"fmax": r["fmax"], "family_fmax": r["family_fmax"]} for a, r in levels.items()},
        {a: {"family_gap": r["family_gap"]} for a, r in levels.items()},
        {a: {"coverage": r["coverage"]} for a, r in levels.items()},
    ]
    categories = [
        {"fmax": {"name": "Fmax (family named AND placed)", "color": "#0f9d76"},
         "family_fmax": {"name": "family Fmax (named only)", "color": "#c9528f"}},
        {"family_gap": {"name": "family Fmax - Fmax", "color": "#c99a00"}},
        {"coverage": {"name": "share of calls that could be judged", "color": "#7f7f7f"}},
    ]
    labels = [{"name": "Fmax vs family Fmax", "ylab": "Fmax"},
              {"name": "gap", "ylab": "family Fmax - Fmax"},
              {"name": "coverage", "ylab": "coverage"}]

    n_cells = parsed.height

    # median() returns None when every value is null, and int(None) raises. That is not a
    # hypothetical: a run whose family columns exist but were only filled for some arms
    # leaves these all-null for kmerseek, and the whole section died on it. The counts are
    # context for the description, not the result, so a missing one drops out of the text.
    def _median(col: str) -> int | None:
        if col not in parsed.columns:
            return None
        med = parsed[col].median()
        return None if med is None else int(med)

    med_truth = _median("n_family_truth")
    med_calls = _median("n_family_calls")
    counts_note = (
        f"; median {med_truth} distinct (protein, family) pairs in the answer key per cell "
        f"and {med_calls} predicted"
        if med_truth is not None and med_calls is not None else "")
    write_section(out, "qfo_ceiling_recognition", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_recognition",
        "section_name": "Recognition against delineation",
        "description": (
            f"<p>For each alphabet, the interval-aware Fmax beside the family Fmax that "
            f"ignores where the call landed, averaged over ksize, low-complexity arm and "
            f"target species ({primary_truth} truth, <code>{split}</code> split, "
            f"{n_cells} scored cells{counts_note}).</p>"
            + bullets(
                "<b>Fmax</b> scores a tool that names the right family in the wrong place "
                "at zero, identically to one that never recognised the family.",
                "<b>Family Fmax</b> scores only the naming.",
                "<b>The distance between the two bars</b> is what boundary placement costs "
                "that alphabet, and the second dataset plots it directly.",
                "A coarse alphabet that has lost the family signal shows a low family "
                "Fmax; one that recognises families but cannot delineate them shows a high "
                "family Fmax and a wide gap.",
                "<b>The third dataset</b> is the share of calls that could be judged at "
                "all, on the same bars, because neither Fmax means the same thing over 12% "
                "of calls as over 90%.",
                "<b>Rows are ordered by the gap</b>, widest at the top, on all three "
                "datasets. The ordering is the result, so it is the axis rather than "
                "something a reader has to reconstruct from an alphabetical list.",
                "<b>Truth sets are never pooled.</b> This is one truth set only.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_ceiling_recognition_plot",
                    "title": "Fmax and family Fmax by alphabet",
                    "ylab": "Fmax", "cpswitch": False, "stacking": "group",
                    "sort_samples": False,
                    "height": 500, "data_labels": labels},
        "categories": categories,
        "data": datasets,
    })

    # The same gap with the ksize axis kept. Averaging over k is what hides whether a wide
    # gap is a property of the alphabet or of the window length it was run at.
    grid = parsed.group_by("alphabet", "ksize").agg(pl.col("family_gap").mean()).sort("ksize")
    cells = {(r["alphabet"], r["ksize"]): r["family_gap"] for r in grid.to_dicts()}
    # Every integer k between the ends, so an unrun k is a blank column rather than a
    # column boundary the eye reads as one step. See contiguous_k_axis.
    ks = contiguous_k_axis(sorted(grid["ksize"].unique().to_list()))
    rows = [[cells.get((a, k)) for k in ks] for a in alphas]
    if not any(v is not None for row in rows for v in row):
        return
    span = max(abs(v) for row in rows for v in row if v is not None)
    unrun = [k for k in ks if all(cells.get((a, k)) is None for a in alphas)]
    gap_note = (
        "<b>Blank columns are k the sweep did not run</b> at any alphabet: "
        + ", ".join(str(k) for k in unrun)
        + ". The axis carries every integer k between its ends so that neighbouring "
          "columns are one step apart in k wherever they touch."
    ) if unrun else ""
    write_section(out, "qfo_ceiling_recognition_k", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_recognition_k",
        "section_name": "Recognition against delineation, by k",
        "description": (
            f"<p>Family Fmax minus Fmax for every alphabet and k-mer size "
            f"({primary_truth} truth, <code>{split}</code> split, averaged over the "
            f"low-complexity arm and target species).</p>"
            + bullets(
                "<b>Larger</b> means more of what the alphabet recognised was thrown away "
                "by landing in the wrong place.",
                "<b>Blank cells</b> are combos outside that alphabet's k range.",
                "<b>The scale is symmetric around zero</b> because the gap is not "
                "guaranteed positive: the family reading also swaps the recall denominator "
                "from domain instances to families, so a cut dominated by a tandem array "
                "of one family can lose more from that swap than it gains from ignoring "
                "placement.",
                "<b>This is the one heatmap in the report on a diverging colour ramp</b>, "
                "and it is the one quantity that has a meaningful midpoint. Every other "
                "grid here is sequential, because Fmax and coverage have no neutral value "
                "for a diverging scale to sit on.",
                gap_note)),
        "plot_type": "heatmap",
        "pconfig": {"id": "qfo_ceiling_recognition_k_plot",
                    "title": "family Fmax - Fmax by alphabet and ksize",
                    "xlab": "k", "ylab": "alphabet",
                    "min": -span, "max": span, "colstops": DIVERGING_COLSTOPS,
                    "square": False, "height": 500},
        "xcats": [str(k) for k in ks],
        "ycats": alphas,
        "data": rows,
    })


# The two quantities section_ceiling_recognition draws as paired bars, re-read against
# alphabet cardinality. Same columns, same averaging; only the x axis is different.
# Okabe-Ito blue and orange, the first two SERIES_COLORS, so recognition and placement
# keep one colour each wherever they appear together.
CARDINALITY_SERIES = [
    ("recognition", "family_fmax", "Recognition (family Fmax, named only)",
     SERIES_COLORS[0]),
    ("placement", "fmax", "Placement (Fmax, named AND placed)", SERIES_COLORS[1]),
]


def rank_avg(values: list[float]) -> list[float]:
    """Ranks with ties averaged, which is what Spearman needs.

    Six alphabets in this sweep have two classes and two have twelve, so ties are the
    common case on the x axis rather than an edge case. Ranking them by position instead
    would make the correlation depend on the order the alphabets happened to be listed in.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, Pearson on tie-averaged ranks. None when either side is constant.

    Written out rather than pulled from scipy because scipy is not a dependency of this
    container and one correlation is not worth adding it for.
    """
    if len(xs) < 3:
        return None
    rx, ry = rank_avg(xs), rank_avg(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def descending_pairs(xs: list[float], ys: list[float]) -> tuple[int, int]:
    """Pairs of DIFFERENT x, counted as running down in y or running up.

    "Almost monotone" is the claim this section makes, and a reader cannot check it off a
    correlation alone. This is the plain version of it: of every pair of alphabets whose
    class counts differ, how many have the coarser one scoring higher, and how many the
    other way round. Pairs that tie on x or on y are counted in neither, because neither
    supports nor contradicts a direction.
    """
    down = up = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] == xs[j] or ys[i] == ys[j]:
                continue
            if (xs[j] > xs[i]) != (ys[j] > ys[i]):
                down += 1
            else:
                up += 1
    return down, up


def section_ceiling_cardinality(out: Path, metrics: pl.DataFrame) -> None:
    """Recognition and placement Fmax against alphabet CARDINALITY, on a log x axis.

    section_ceiling_recognition draws these same two columns, `family_fmax` and `fmax`, as
    paired bars with the alphabet as a bare category. That view answers "which alphabet"
    and cannot answer "does it depend on how coarse the alphabet is", because the category
    axis carries no cardinality and a reader would have to know each name's class count to
    see a trend. This plots the same two means against that class count instead. The
    averaging is deliberately identical -- mean over ksize, low-complexity arm and target
    species -- so the two sections cannot disagree; a point here is exactly the height of
    the matching bar there.

    The vertical distance between a point pair is still the placement cost the parent
    section's second dataset plots, so nothing is lost by reading them on one y.

    No uncertainty is computed anywhere in this report, and a monotone trend asserted over
    nineteen points with no interval is the first thing a reviewer will push on. Every
    point therefore carries the min and max of the cells it averaged, drawn as caps and
    given as columns, and the description says plainly that this is dispersion across the
    sweep rather than sampling error.
    """
    base = ungrouped(metrics.filter(pl.col("tool") == "kmerseek"))
    if base.height == 0 or "family_fmax" not in base.columns:
        return
    parsed = parse_kmerseek_variants(base)
    if parsed.height == 0:
        return

    for ts in sorted(parsed["truth_set"].unique().to_list()):
        cut, split = pick_split(parsed.filter(pl.col("truth_set") == ts))
        # Both metrics dropped together, so mean, min and max are over the same cells and
        # the gap between the two series is a difference of like-for-like averages.
        cut = cut.drop_nulls(["fmax", "family_fmax"])
        if cut.height == 0:
            continue
        cut = encoding_axes(cut)
        if cut.height == 0:
            continue
        per = cut.group_by("alphabet", "n_classes").agg(
            pl.len().alias("n_cells"),
            *[expr
              for key, col, _, _ in CARDINALITY_SERIES
              for expr in (pl.col(col).mean().alias(f"{key}_mean"),
                           pl.col(col).min().alias(f"{key}_min"),
                           pl.col(col).max().alias(f"{key}_max"),
                           pl.col(col).std().alias(f"{key}_sd"))],
        ).sort("n_classes", "alphabet")
        if per.height == 0:
            continue
        rows = per.to_dicts()

        # A DICT of series, not a list of points. Custom content reads a top-level list as
        # a list of DATASETS, so a bare list of points crashes the report in
        # custom_content's numeric-x coercion rather than drawing anything. Keying by
        # alphabet and hanging that alphabet's points off it is what scatter accepts, and
        # it makes the keys unique for free.
        points: dict[str, list[dict]] = {}
        for key, _, label, color in CARDINALITY_SERIES:
            for r in rows:
                series = points.setdefault(r["alphabet"], [])
                # Annotated unconditionally rather than through
                # ANNOTATE_EVERY_POINT_BELOW: six alphabets share x=2 and two share x=12,
                # so an unlabelled point at a shared cardinality cannot be identified at
                # all, which is the one thing this axis makes worse than the bar chart.
                series.append(scatter_point(
                    r["n_classes"], r[f"{key}_mean"],
                    name=f"{r['alphabet']} ({r['n_classes']})", group=label, color=color,
                    annotation=r["alphabet"], n_cells=r["n_cells"]))
                lo, hi = r[f"{key}_min"], r[f"{key}_max"]
                if lo is None or hi is None or hi <= lo:
                    continue
                # MultiQC 1.35's scatter builds each go.Scatter from a fixed key list and
                # has no error_y, so the range is drawn as two capped markers at the same
                # x. Same colour as the mean, smaller and faded, and its own group so the
                # legend says what the caps are.
                for edge in (lo, hi):
                    series.append(scatter_point(
                        r["n_classes"], edge,
                        name=f"{r['alphabet']} min/max", color=color,
                        group=f"{label} — min and max over cells",
                        marker_size=5, marker_symbol="line-ew-open", opacity=0.55))

        xs = [r["n_classes"] for r in rows]
        class_rules = [{"value": n, "color": "#cccccc", "dash": "dot", "width": 1,
                        "label": str(n)} for n in sorted(set(xs))]
        stats = {}
        for key, _, _, _ in CARDINALITY_SERIES:
            ys = [r[f"{key}_mean"] for r in rows]
            stats[key] = (spearman_rho(xs, ys), descending_pairs(xs, ys),
                          min(ys), max(ys))

        def trend(key: str) -> str:
            rho, (down, up), lo, hi = stats[key]
            rho_txt = "n/a" if rho is None else f"{rho:+.2f}"
            total = down + up
            share = f"{up}/{total}" if total else "0/0"
            return (f"Spearman rho {rho_txt}, range {lo:.3f}–{hi:.3f}, {share} of the "
                    f"pairs of alphabets with different class counts run the other way")

        n_alpha = len(rows)
        n_cells = int(per["n_cells"].sum())
        kmin, kmax = int(cut["ksize"].min()), int(cut["ksize"].max())
        n_species = int(cut["species"].n_unique()) if "species" in cut.columns else 0
        species_txt = f", {n_species} target species" if n_species else ""
        write_section(out, f"qfo_ceiling_cardinality_{ts}", {
            **CEILING_PARENT,
            "id": f"qfo_ceiling_cardinality_{ts}",
            "section_name": f"Recognition and placement vs alphabet size — {ts} truth",
            "description": (
                f"<p>Family Fmax and interval-aware Fmax against how many amino-acid "
                f"classes the alphabet collapses the 20 residues into, log x "
                f"({ts} truth, <code>{split}</code> split, {n_alpha} alphabets over "
                f"{n_cells:,} scored cells).</p>"
                + bullets(
                    "<b>Recognition (family Fmax, named only)</b> scores whether the "
                    "right family was named on the right protein and ignores where the "
                    "call landed.",
                    "<b>Placement (Fmax, named AND placed)</b> is the same machinery read "
                    "at the interval level: naming the family in the wrong place scores "
                    "zero, identically to never naming it. The vertical distance between "
                    "a point and the one under it is therefore what boundary placement "
                    "costs that alphabet, the same quantity the Recognition against "
                    "delineation section plots as a gap.",
                    f"<b>One point per alphabet</b>, the mean over ksize (k {kmin}–{kmax} "
                    f"across the sweep, though each alphabet was swept over its own "
                    f"range), low-complexity arm{species_txt} — the same averaging as "
                    "Recognition against delineation, so a point here is exactly the "
                    "height of the bar there. Several alphabets share a class count, so "
                    "points stack at one x; each is labelled with its alphabet name.",
                    "<b>The caps above and below each point</b> are the min and max of "
                    "the cells it averaged. They are the DISPERSION of the sweep, not a "
                    "confidence interval: nothing in this report estimates sampling "
                    "error, so no trend stated here is tested against one. A cap taller "
                    "than the gap to the neighbouring alphabet means the choice of k or "
                    "arm moves that point further than the alphabet does.",
                    f"<b>Recognition against class count</b> — {trend('recognition')}.",
                    f"<b>Placement against class count</b> — {trend('placement')}.",
                    "<b>Log x</b> because the class counts are 2, 3, 4 … 20 and a linear "
                    "axis would pile two thirds of the alphabets into the left tenth of "
                    "the plot.",
                    "<b>Truth sets are never pooled.</b> This is one truth set only; each "
                    "gets its own copy of this section.")),
            "plot_type": "scatter",
            "pconfig": {"id": f"qfo_ceiling_cardinality_{ts}_plot",
                        "title": f"Recognition and placement by alphabet size ({ts})",
                        "xlab": "amino-acid classes in the alphabet (letters, log scale)",
                        "ylab": "Fmax", "xlog": True, "ymin": 0, "height": 560,
                        # Padded past the data on both sides so the alphabet names next to
                        # the 2-class column are not cut off by the axis. Widening only:
                        # the scatter DROPS points outside xmin/xmax rather than clipping
                        # the axis, so a bound inside the data would silently lose an
                        # alphabet.
                        "xmin": 1.5, "xmax": 30,
                        "xsuffix": "", "ysuffix": "", "showlegend": True,
                        # One labelled rule per cardinality actually present. Plotly's log
                        # axis draws minor ticks and labels 20 as a bare "2", which on an
                        # axis whose left end really is 2 is worse than no label at all.
                        # The rules also give the stacked points a column to sit in.
                        "x_lines": class_rules},
            "data": points,
        }, supplementary=not is_primary_truth(ts))

        headers = {
            "n_classes": {"title": "Classes", "format": "{:,.0f}", "scale": "Blues",
                          "description": "Amino-acid classes the alphabet collapses the "
                                         "20 residues into, read off its name"},
            "recognition_mean": {"title": "Recognition", "format": "{:,.3f}", "min": 0,
                                 "scale": "PuBu",
                                 "description": "family Fmax, mean over ksize x "
                                                "low-complexity arm x target species"},
            "recognition_sd": {"title": "Recog. SD", "format": "{:,.3f}", "scale": "Greys",
                               "description": "SD across those cells. Dispersion of the "
                                              "sweep, not a standard error"},
            "recognition_min": {"title": "Recog. min", "format": "{:,.3f}",
                                "scale": "PuBu", "description": "Worst cell behind it"},
            "recognition_max": {"title": "Recog. max", "format": "{:,.3f}",
                                "scale": "PuBu", "description": "Best cell behind it"},
            "placement_mean": {"title": "Placement", "format": "{:,.3f}", "min": 0,
                               "scale": "Oranges",
                               "description": "Interval-aware Fmax, same cells"},
            "placement_sd": {"title": "Place. SD", "format": "{:,.3f}", "scale": "Greys",
                             "description": "SD across those cells, same caveat"},
            "placement_min": {"title": "Place. min", "format": "{:,.3f}",
                              "scale": "Oranges", "description": "Worst cell behind it"},
            "placement_max": {"title": "Place. max", "format": "{:,.3f}",
                              "scale": "Oranges", "description": "Best cell behind it"},
            "gap": {"title": "Gap", "format": "{:,.3f}", "scale": "RdPu",
                    "description": "Recognition minus placement: what boundary placement "
                                   "cost this alphabet"},
            "n_cells": {"title": "Cells", "format": "{:,.0f}", "scale": "Greens",
                        "description": "Scored cells averaged into this row"},
        }
        table = {}
        for r in rows:
            table[r["alphabet"]] = {
                "n_classes": r["n_classes"], "n_cells": r["n_cells"],
                "gap": r["recognition_mean"] - r["placement_mean"],
                **{k: r[k] for k in headers if k not in ("n_classes", "n_cells", "gap")},
            }
        write_section(out, f"qfo_ceiling_cardinality_table_{ts}", {
            **CEILING_PARENT,
            "id": f"qfo_ceiling_cardinality_table_{ts}",
            "section_name": f"Recognition and placement by alphabet size — {ts} truth",
            "description": (
                f"<p>The plot above as numbers, one row per alphabet, coarsest first "
                f"({ts} truth, <code>{split}</code> split).</p>"
                + bullets(
                    "<b>Recognition / Placement</b> are the two series, family Fmax and "
                    "interval-aware Fmax, averaged over the same cells.",
                    "<b>SD, min and max</b> are the spread across those cells and are "
                    "the column to read before believing a difference between two "
                    "neighbouring rows. They describe how much the sweep varies, not how "
                    "precisely the mean is known — this report estimates no sampling "
                    "error anywhere, and that is a gap rather than a claim of precision.",
                    "<b>Gap</b> is recognition minus placement, what boundary placement "
                    "cost that alphabet.",
                    "<b>Cells</b> is how many ksize x arm x species results the row "
                    "averaged. Rows built from one cell have no spread to report and "
                    "their SD is blank.")),
            "plot_type": "table",
            "pconfig": {"id": f"qfo_ceiling_cardinality_table_{ts}_table",
                        "title": f"Recognition and placement by alphabet size ({ts})",
                        "col1_header": "Alphabet", "sort_rows": False, "no_violin": True},
            "headers": headers,
            "data": table,
        }, supplementary=not is_primary_truth(ts))


def section_ceiling_bpe(out: Path, bpe: dict | None) -> None:
    """ProtBERTa_2's learned token boundaries against Pfam domain boundaries.

    Written from bin/hp_bpe_boundary_diagnostic.py's JSON, which is a standalone
    measurement rather than anything this pipeline searched -- which is why the panel is
    absent rather than empty when the diagnostic has not been run.
    """
    if not bpe or not bpe.get("alphabets"):
        return
    rows = bpe["alphabets"]
    order = sorted(rows, key=lambda k: rows[k].get("enrichment") or 0.0, reverse=True)
    data = {}
    for name in order:
        r = rows[name]
        if r.get("enrichment") is None:
            continue
        label = name + (" (= ProtBERTa_2)" if r.get("identical_to_protberta_2")
                                              and name != "protberta_2" else "")
        data[label] = {"enrichment": r["enrichment"]}
    if not data:
        return
    ctrl = rows.get("hp_random_control2", {}).get("enrichment")
    control_note = (
        f"<b>The random 10/10 control</b> sits at {ctrl:.2f}x. A bar that is not clearly "
        "above it is measuring the autocorrelation of any two-letter string, not "
        "hydrophobicity."
        if ctrl else "")
    write_section(out, "qfo_ceiling_bpe", {
        **CEILING_PARENT,
        "id": "qfo_ceiling_bpe",
        "section_name": "BPE token boundaries vs domain boundaries",
        "description": (
            f"<p>How often a ProtBERTa_2 BPE token boundary falls exactly on a Pfam domain "
            f"boundary, divided by the same rate on length- and composition-matched "
            f"shuffled sequences.</p>"
            + bullets(
                "<b>1.0 is the null</b>: no more agreement than shuffled sequence gives.",
                f"<b>Measured on</b> {bpe.get('n_proteins', '?')} human proteins carrying "
                f"{bpe.get('n_domain_instances', '?')} domain instances, with the "
                f"tokenizer released at doi "
                f"<code>{bpe.get('tokenizer_doi', '')}</code> applied to each alphabet's "
                f"own h/p encoding.",
                control_note,
                "<b>This is segmentation agreement, not end-to-end performance.</b> A "
                "tokenizer whose boundaries never coincide with domain boundaries can "
                "still support a model that finds domains, and one whose boundaries agree "
                "perfectly can still be beaten by a k-mer method.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_ceiling_bpe_plot",
                    "title": "Domain-boundary enrichment at BPE token boundaries",
                    "ylab": "observed / shuffled-null hit rate", "cpswitch": False,
                    "height": 420},
        "categories": {"enrichment": {"name": "enrichment over shuffled null",
                                      "color": "#4c72b0"}},
        "data": data,
    })


def section_boundary(out: Path, metrics: pl.DataFrame, primary_truth: str,
                     max_tools: int) -> None:
    """Right family in the wrong place is the failure mode this benchmark exists to catch."""
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    if cut.height == 0:
        return
    board = best_variants(cut).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    # No ndo: it is a literal alias of residue_recall, so the two columns were the same
    # number drawn twice under different names. See the note above HEADLINE.
    cols = ["residue_precision", "residue_recall", "residue_f1", "median_iou_tp",
            "precision_iou80", "recall_iou80", "n_tp_strict",
            "dbd_median", "dbd_mean",
            "nterm_offset_median", "nterm_offset_mean", "nterm_offset_iqr",
            "cterm_offset_median", "cterm_offset_mean", "cterm_offset_iqr",
            "domain_count_mcc", "domain_count_accuracy"]
    cols = [c for c in cols if c in cut.columns]
    if not cols:
        return
    data = {}
    for tool, variant, label in keep:
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        agg = sub.select([pl.col(c).mean() for c in cols]).to_dicts()[0]
        agg["semantics"] = (sub["interval_semantics"].first()
                            if "interval_semantics" in sub.columns else "alignment")
        data[label] = agg
    write_section(out, "qfo_boundary", {
        "id": "qfo_boundary",
        "section_name": "Boundary accuracy",
        "description": (
            f"<p>Where a call lands, not just whether the family is right "
            f"({primary_truth} truth, <code>{split}</code> split).</p>"
            + bullets(
                "<b>Offsets</b> are in residues and signed. Negative N-terminal means the "
                "call starts before the true domain, so a systematic bias shows as a "
                "median away from zero rather than as a wider IQR.",
                "<b>Rows marked <code>motif</code></b> report the envelope of a "
                "discontinuous residue set rather than an alignment, so their boundary "
                "numbers measure a different thing and should not be ranked against the "
                "alignment rows.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_boundary_table", "title": "Residue-level and boundary metrics",
                    "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "semantics": dict(title="Semantics", scale=False),
            "residue_precision": dict(title="Residue prec.", min=0, max=1, scale="BuGn",
                                      format="{:,.3f}",
                                      description="Measured from the prediction side"),
            "residue_recall": dict(title="Residue rec.", min=0, max=1, scale="BuGn",
                                   format="{:,.3f}",
                                   description="Measured from the truth side"),
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


# Alphabets the placement half of the report has to name, whatever their Fmax rank. Every
# HP arm is here by definition -- they are the hypothesis -- plus protein20 as the
# reference an HP number is large or small against.
def _boundary_must_include(parsed: pl.DataFrame) -> list[str]:
    """The alphabets qfo_boundary_dots shows regardless of where they rank on Fmax."""
    alphas = set(parsed["alphabet"].unique().to_list())
    keep = sorted(a for a in alphas if a.startswith("hp_"))
    return keep + [a for a in ("protein20",) if a in alphas]


def section_boundary_dots(out: Path, metrics: pl.DataFrame, primary_truth: str,
                          max_tools: int) -> None:
    """Median IoU per arm as a labelled dot plot, with the 2-letter HP arms in it.

    Two problems with the boundary table, and one figure fixes both.

    First, the arms. `section_boundary` selects its rows with `best_variants`, which keeps
    each tool's best variant plus kmerseek's top few under Fmax and sensitivity, ranked
    over the whole sweep. Those ranks are recognition ranks, and the HP alphabets do not
    win them -- polarity4 and wwmj5 do. So the table that reports where a call LANDS
    contained no 2-letter arm at all, and the report could not connect its recognition half
    to its placement half: the recognition winners and the placement winners were different
    alphabets and neither figure showed both. The boundary columns were computed for every
    arm in the sweep; nothing was missing from the measurement, only from the row
    selection. Every HP alphabet is pinned in here for that reason, at its own best-Fmax
    k, beside protein20 as the reference.

    Second, the drawing. MultiQC renders a 19-row table's plot view as a violin, and a
    kernel density over 19 points invents a shape that is not in the data, with unlabelled
    dots underneath it that a reader cannot attribute to a tool. This is the same numbers
    as one labelled point per arm.

    n_TP travels with every point, and it is the column to read first. Two arms can share
    a median IoU while one earned it over hundreds of correct calls and the other over
    twenty, which is what a 2-letter alphabet at high k produces: few calls, placed well
    when they land at all. The medians themselves sit in a narrow band -- the section text
    quotes this run's own range rather than a remembered one, because the range moved when
    point features stopped being scored as intervals.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    if cut.height == 0 or "median_iou_tp" not in cut.columns:
        return

    keep = [(r["tool"], r["variant"], r["label"])
            for r in best_variants(cut).head(max_tools).to_dicts()]
    chosen = {(t, v) for t, v, _ in keep}

    # The pinned arms, one k each: the k that alphabet reaches its own best Fmax at, so the
    # point is the arm as anyone would run it rather than its worst k.
    ks = parse_kmerseek_variants(cut.filter(pl.col("tool") == "kmerseek"))
    if ks.height:
        per_variant = ks.group_by("alphabet", "variant").agg(
            pl.col("fmax").mean().alias("fmax"))
        for alpha in _boundary_must_include(ks):
            best = (per_variant.filter(pl.col("alphabet") == alpha)
                               .sort("fmax", descending=True, nulls_last=True))
            if best.height == 0:
                continue
            variant = best["variant"][0]
            if ("kmerseek", variant) not in chosen:
                keep.append(("kmerseek", variant, label_of("kmerseek", variant)))
                chosen.add(("kmerseek", variant))

    from_table = {(r["tool"], r["variant"])
                  for r in best_variants(cut).head(max_tools).to_dicts()}
    rows = []
    for tool, variant, label in keep:
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        iou = sub["median_iou_tp"].mean()
        n_tp = sub["n_tp_strict"].mean() if "n_tp_strict" in sub.columns else None
        if iou is None:
            continue
        rows.append((label, tool, variant, iou, n_tp))
    if not rows:
        return
    rows.sort(key=lambda r: -r[3])

    # One horizontal bar per arm rather than one scatter point. The scatter drew every
    # label as text beside its marker, and at 20-odd arms the labels overprinted each other
    # and ran off the left edge -- the same unreadability as the violin, differently
    # shaped. A bar's row label is on the axis, so it cannot collide with anything, and n
    # goes into the label so the denominator arrives with the number instead of beside it.
    #
    # One series per method class, each row filled in exactly one of them. That is what
    # keeps the class colours and the legend on a bar plot, where colour is a property of
    # the series rather than of the row.
    classes_present, iou_data, n_data = [], {}, {}
    for label, tool, variant, iou, n_tp in rows:
        cls = CLASSES[tool_class(tool)][0]
        if cls not in classes_present:
            classes_present.append(cls)
        # Plain digit grouping. A `&thinsp;` here renders in the browser and then breaks
        # the static export: kaleido fails the PNG and PDF of any plot whose sample names
        # carry an HTML entity, with "plotly.js error 525", while the SVG still comes out.
        # The export is what goes in a figure, so the tick label has to survive it.
        n_txt = f"n={n_tp:,.0f}" if n_tp is not None else "n=?"
        row_label = f"{short_label(tool, variant)}  ({n_txt})"
        iou_data[row_label] = {cls: iou}
        n_data[row_label] = {cls: n_tp}
    categories = {CLASSES[c][0]: {"name": CLASSES[c][0], "color": CLASSES[c][1]}
                  for c in CLASSES if CLASSES[c][0] in classes_present}

    pinned = [short_label(tool, variant) for _, tool, variant, _, _ in rows
              if tool == "kmerseek" and (tool, variant) not in from_table]
    write_section(out, "qfo_boundary_dots", {
        "id": "qfo_boundary_dots",
        "section_name": "Boundary accuracy, one point per arm",
        "description": (
            f"<p>Median IoU of a correct call against the domain it names, one labelled "
            f"point per arm ({primary_truth} truth, <code>{split}</code> split, mean over "
            f"target species).</p>"
            + bullets(
                "<b>One row per arm, sorted by median IoU</b>, so the ordering is the axis "
                "and every row carries its own name. This is the same content as the "
                "violin view of the table above, which draws 19 unlabelled dots under a "
                "kernel density estimated from 19 points.",
                f"<b>The spread is narrow: {rows[-1][3]:.3f} to {rows[0][3]:.3f} across "
                f"{len(rows)} arms.</b> Read the ordering against that range rather than "
                "against the axis, and read n beside it. Point features are excluded from "
                "this column, so an arm is no longer penalised for the intervals it was "
                "never asked to place.",
                "<b>n in each row label</b> is that arm's strict true positives, calls at "
                "IoU &ge; 0.8, averaged over target species. A median IoU over 20 calls "
                "and one over 20_000 are not the same measurement, and the bar is the same "
                "length either way. <b>Switch to the second dataset</b> to see n on its "
                "own axis.",
                "<b>Every HP alphabet is drawn</b>, at its own best-Fmax k, whether or not "
                "it ranks high enough on Fmax to reach the table above. Fmax ranks "
                "recognition; this figure is about placement, and selecting placement rows "
                "on a recognition rank is what left the boundary table with no 2-letter "
                "arm in it.",
                ("<b>Pinned arms</b> — " + ", ".join(f"<code>{p}</code>" for p in pinned)
                 + ": present here and absent from the table above, for that reason.")
                if pinned else "",
                "<b>Rows marked <code>motif</code> in the table above</b> report the "
                "envelope of a discontinuous residue set rather than an alignment. Their "
                "IoU measures a different thing and should not be ranked against the "
                "alignment arms.",
                "<b>Colour</b> is the method class, which the legend names.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_boundary_dots_plot",
                    "title": f"Median IoU per arm ({primary_truth}, {split} split)",
                    "ylab": "median IoU over correct calls",
                    "cpswitch": False, "stacking": "group", "sort_samples": False,
                    "height": 120 + 26 * len(rows),
                    "data_labels": [
                        {"name": "median IoU", "ylab": "median IoU over correct calls"},
                        {"name": "strict TP (n)", "ylab": "calls at IoU >= 0.8"}]},
        "categories": [categories, categories],
        "data": [iou_data, n_data],
    })


def _per_tool_table(cut: pl.DataFrame, cols: list[str], max_tools: int) -> dict:
    """Mean of each column over species, one row per selected (tool, variant)."""
    board = best_variants(cut).head(max_tools)
    cols = [c for c in cols if c in cut.columns]
    data = {}
    for row in board.to_dicts():
        sub = cut.filter((pl.col("tool") == row["tool"])
                         & (pl.col("variant") == row["variant"]))
        if sub.height == 0 or not cols:
            continue
        data[row["label"]] = sub.select([pl.col(c).mean() for c in cols]).to_dicts()[0]
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
    cols = ["fmax", "family_fmax", "family_gap", "fmax_threshold", "fmax_precision",
            "fmax_recall", "family_fmax_precision", "family_fmax_recall", "wfmax",
            "family_wfmax", "n_family_truth", "n_family_calls", "n_family_found",
            "smin", "smin_threshold", "smin_ru", "smin_mi"]
    if {"family_fmax", "fmax"}.issubset(cut.columns):
        cut = cut.with_columns((pl.col("family_fmax") - pl.col("fmax")).alias("family_gap"))
    data = _per_tool_table(cut, cols, max_tools)
    if not data:
        return
    write_section(out, "qfo_cafa", {
        "id": "qfo_cafa",
        "section_name": "CAFA-style metrics",
        "description": (
            f"<p>{primary_truth} truth, <code>{split}</code> split, averaged over target "
            f"species.</p>"
            + bullets(
                "<b>Fmax</b> is the maximum F-score over score thresholds. The precision "
                "and recall columns are the operating point where it is reached.",
                "<b>Family Fmax</b> is the same curve read on the SET of Pfam families "
                "called per query protein against the set truly present, with interval "
                "placement ignored: the CAFA-classic reading. Fmax scores a tool that "
                "names the right family in the wrong place at zero, exactly as it scores a "
                "tool that never recognised the family, and the pair separates those.",
                "<b>Gap</b> is family Fmax minus Fmax, so it is what boundary placement "
                "costs. It is almost always positive but is not guaranteed to be: the "
                "family reading also swaps the recall denominator from instances to "
                "families, and on a protein carrying a tandem array of one family that "
                "swap can cost more than ignoring placement gains.",
                "<b>The three family counts</b> are the denominators: distinct "
                "(protein, family) pairs in the answer key, predicted, and correct.",
                "<b>wFmax</b> weights each family by its information content, "
                "IC = -log<sub>2</sub> P(family), so recovering a rare family counts for "
                "more than recovering a common one.",
                "<b>Smin</b> is the minimum of sqrt(remaining uncertainty<sup>2</sup> + "
                "misinformation<sup>2</sup>) in bits, and lower is better.",
                "<b><code>smin_ru</code></b> is information still missing (false "
                "negatives) and <b><code>smin_mi</code></b> is information invented (false "
                "positives) at that threshold, so the two say which way a tool is failing.",
                "<b>The weighting here is not CAFA's information accretion</b>, which is "
                "defined over an ontology's parent-child structure. Pfam is flat, so plain "
                "IC is used and the metric is reported under that narrower definition.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_cafa_table", "title": f"CAFA-style metrics ({primary_truth})",
                    "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "fmax": dict(title="Fmax", min=0, max=1, scale="RdYlGn", format="{:,.3f}",
                         description="Interval-aware: the call must land on the "
                                     "annotated interval"),
            "family_fmax": dict(title="Family Fmax", min=0, max=1, scale="RdPu",
                                format="{:,.3f}",
                                description="Set of families called per protein, "
                                            "placement ignored"),
            "family_gap": dict(title="Gap", scale="PuOr", format="{:,.3f}",
                               description="Family Fmax minus Fmax: what placement costs"),
            "fmax_threshold": dict(title="@ threshold", scale=False, format="{:,.2f}",
                                   description="Score cutoff where Fmax is reached"),
            "fmax_precision": dict(title="Prec. @ Fmax", min=0, max=1, scale="Oranges",
                                   format="{:,.3f}"),
            "fmax_recall": dict(title="Rec. @ Fmax", min=0, max=1, scale="Greens",
                                format="{:,.3f}"),
            "family_fmax_precision": dict(title="Fam. prec.", min=0, max=1, scale="Oranges",
                                          format="{:,.3f}", hidden=True),
            "family_fmax_recall": dict(title="Fam. rec.", min=0, max=1, scale="Greens",
                                       format="{:,.3f}", hidden=True),
            "n_family_truth": dict(title="Families (truth)", format="{:,.0f}", scale=False,
                                   description="Distinct (protein, family) pairs in the "
                                               "answer key for this cell"),
            "n_family_calls": dict(title="Families (called)", format="{:,.0f}", scale=False,
                                   description="Distinct (protein, family) pairs predicted, "
                                               "after collapsing redundant copies"),
            "n_family_found": dict(title="Families (correct)", format="{:,.0f}",
                                   scale="Greens"),
            "wfmax": dict(title="wFmax", min=0, max=1, scale="PuBuGn", format="{:,.3f}",
                          description="Fmax weighted by family information content"),
            "family_wfmax": dict(title="Family wFmax", min=0, max=1, scale="PuBuGn",
                                 format="{:,.3f}", hidden=True),
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
            f"<p>{primary_truth} truth, <code>{split}</code> split, averaged over target "
            f"species.</p>"
            + bullets(
                "<b>The left block</b> is what each tool reported at its own default "
                "cutoff, which differs between tools and is not a property of the method.",
                "<b>The right block</b> is threshold-free, and is the comparable one.",
                "<b>ROC AUC</b> is the probability a correct call outranks an incorrect "
                "one.",
                "<b>AUPRC</b> is average precision over score-ranked calls.",
                "<b>Best F1</b> is the optimum at any threshold, with the operating point "
                "it sits at.",
                "<b>Recall</b> is against <i>reachable</i> instances throughout: those "
                "whose family exists in the target proteome and could be transferred at "
                "all.",
                "<b><code>precision_strict</code></b> is the same precision with gray-zone "
                "calls charged as errors, kept visible so the gray-zone convention can "
                "never be mistaken for a free improvement.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_threshold_table",
                    "title": f"Threshold metrics ({primary_truth})",
                    "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
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
            "<p>What the metrics in this report are scored against.</p>"
            + bullets(
                "<b>Instance counts</b> are measured off this run: the largest answer key "
                "any tool was scored against, which is the size of the key itself rather "
                "than a total over rows.",
                "<b>For reference</b>, the full sets are 50,185 human Pfam domain "
                "instances, 142,857 human Swiss-Prot features, Pfam-N streamed from EBI, "
                "and 106 human proteins in M-CSA. A run scoped to fewer species or a mini "
                "test set will show less than that.",
                "<b>No number in this report is ever averaged across truth sets.</b> Pfam "
                "is circular with the profile baselines and Swiss-Prot is not, so a mean "
                "over the two has no interpretation.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_truth_provenance_table", "title": "Truth set provenance",
                    "col1_header": "Truth set", "sort_rows": False, "scale": False, "no_violin": True},
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
    # Resolved against Crossref and Europe PMC rather than transcribed from a summary: the
    # article is Protein Science 35(1) e70397, online 2025-12-23, print issue 2026-01, so
    # both years appear in the wild. Its significance warning is about STRUCTURE search --
    # the Gumbel tail fitting the high-scoring end badly and overestimating significance --
    # and a general call for new methods to match BLAST's E-value rigor. It does NOT
    # discuss k-mer or word-statistic independence, so cite it for the general standard and
    # not as authority on the overlapping-k-mer problem.
    ("Significance statistics — the standard this report's threshold-free numbers are "
     "measured against, and the reason a null-model panel belongs here",
     "Sahakyan, H., Mutz, P., Tobiasson, V., &amp; Koonin, E. V. (2026). Exploring the "
     "protein universe with distant similarity detection methods. <i>Protein Science</i>, "
     "35(1), e70397. Published online 23 December 2025.",
     "https://doi.org/10.1002/pro.70397"),
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
    """How much of each tool's output could be judged at all.

    Grouped bars on a log axis rather than one stacked bar per arm, and no percentage view.
    The three counts span five orders of magnitude across the arms in this run -- roughly
    two thousand calls at one end and tens of millions at the other -- and a stack cannot
    be drawn on a log axis, so the linear stack rendered every small arm as a hairline and
    the percentage switch turned the whole panel into a ranking on denominators that are
    not comparable. An arm at a 50% true-positive share over 40 calls was drawn the same
    width as one over 40 million.

    So: each count gets its own bar, the axis is log, and every arm's tick label carries
    its own call total and true-positive share. The share is still there for a reader who
    wants it, attached to the denominator that makes it mean something.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    need = {"n_tp_calls", "n_fp_calls", "n_gray_calls"}
    if cut.height == 0 or not need.issubset(set(cut.columns)):
        return
    board = best_variants(cut).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    raw = {}
    for tool, variant, label in keep:
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        raw[label] = sub.select(
            [pl.col(c).sum() for c in ("n_tp_calls", "n_fp_calls", "n_gray_calls")]
        ).to_dicts()[0]
    if not raw:
        return
    # Sorted by total so the axis runs from the arms that barely call to the arms that call
    # tens of millions of times, which is the shape the section is about.
    ordered = sorted(raw.items(), key=lambda kv: -sum(v or 0 for v in kv[1].values()))
    data = {}
    for label, row in ordered:
        total = sum(v or 0 for v in row.values())
        tp = row.get("n_tp_calls") or 0
        share = f"{100 * tp / total:.1f}% TP" if total else "no calls"
        # Plain digit grouping, not an HTML entity. A `&thinsp;` in a tick label renders in
        # the browser and breaks the static export: kaleido fails the PNG and PDF of any
        # plot whose sample names carry an entity, with "plotly.js error 525", while the
        # SVG still comes out. The tick label has to survive the export, because the export
        # is what goes in a figure.
        data[f"{label} · {total:,} calls · {share}"] = row
    smallest = ordered[-1]
    largest = ordered[0]
    write_section(out, "qfo_grayzone", {
        "id": "qfo_grayzone",
        "section_name": "Gray-zone accounting",
        "description": (
            f"<p>Every call this run produced, split three ways ({primary_truth} truth, "
            f"<code>{split}</code> split, summed over species).</p>"
            + bullets(
                "<b>Gray calls</b> land in territory the annotation never covered.",
                "They are <b>excluded from the precision denominator</b> rather than "
                "charged as errors, because a region Pfam never annotated is not evidence "
                "the tool was wrong.",
                "<b>The size of the gray slice</b> is how much that convention is worth to "
                "each tool, so it is shown rather than folded away.",
                "<b>The axis is log and the bars are grouped, not stacked.</b> The counts "
                f"run from {sum(v or 0 for v in smallest[1].values()):,} calls for "
                f"<code>{smallest[0]}</code> to "
                f"{sum(v or 0 for v in largest[1].values()):,} for "
                f"<code>{largest[0]}</code>. A linear stack draws the first as a hairline; "
                "a percentage view draws both the same width.",
                "<b>Each arm's tick label carries its own total and its true-positive "
                "share.</b> The share is not a leaderboard: a high one over a few thousand "
                "calls is a different claim from the same share over tens of millions, and "
                "the arm with the highest share here has the smallest denominator in the "
                "run.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_grayzone_plot", "title": "Calls by outcome",
                    "ylab": "calls (log scale)",
                    "xlog": True, "stacking": "group", "cpswitch": False,
                    "sort_samples": False, "height": 520},
        "categories": {"n_tp_calls": {"name": "true positive", "color": "#0f9d76"},
                       "n_fp_calls": {"name": "false positive", "color": "#c9528f"},
                       "n_gray_calls": {"name": "gray (unscoreable)", "color": "#c8c8c8"}},
        "data": data,
    })


# --- what the side-by-side plot can and cannot show --------------------------------
#
# Pfam is built from the same HMMs the profile baselines run, so this section used to say
# that a tool's drop from its Pfam bar to its Swiss-Prot bar IS the size of that
# circularity. That reading only works if the drop is specific to the profile methods. If
# every method class falls by roughly the same factor then the ratio is measuring how much
# harder Swiss-Prot is as a task, and it says nothing about who wrote the answer key --
# in which case the plot does not detect circularity at all and the caption was claiming a
# measurement the figure cannot make.
#
# Which of the two it is, is a property of the run, so it is computed from the bars rather
# than asserted. A ratio between two truth sets is not a number pooled across them:
# nothing is averaged and each half is that set's own Fmax, which is the one comparison
# this section exists to make.
CIRCULARITY_SETS = ("pfam", "swissprot")


def _by_class(data: dict, tools: dict, key: str) -> dict[str, list[float]]:
    """Every bar of one truth set, grouped by the method class that drew it."""
    out = {}
    for label, cell in data.items():
        v = cell.get(key)
        if v is None:
            continue
        out.setdefault(CLASSES[tool_class(tools[label])][0], []).append(v)
    return out


def circularity_bullet(data: dict, tools: dict) -> str:
    """Whether the Pfam-to-Swiss-Prot drop separates the method classes, or does not."""
    spans = {}
    for label, cell in data.items():
        a, b = cell.get(CIRCULARITY_SETS[0]), cell.get(CIRCULARITY_SETS[1])
        if a is None or not b:
            continue
        spans.setdefault(CLASSES[tool_class(tools[label])][0], []).append(a / b)
    spans = {c: (min(v), max(v)) for c, v in spans.items()}
    aligned = CLASSES["alignment"][0]
    if len(spans) < 2 or aligned not in spans:
        return ("<b>Pfam and Swiss-Prot cannot be compared in this run</b>: the "
                f"<i>{aligned}</i> class and at least one other both need a bar on each "
                "set before there is a ratio to read.")
    listed = "; ".join(f"{c} {lo:.2f}-{hi:.2f}" for c, (lo, hi) in sorted(spans.items()))
    # The directional test, because the hypothesis is directional. Circularity in the
    # answer key means the class that WROTE the answer key loses more by leaving it than
    # anyone else does. A symmetric "do the ranges overlap" test answers a different and
    # weaker question, and a class with a single tool in it is a point rather than a range,
    # which such a test reads as separation on its own.
    ours_lo = min(spans[aligned])
    others_hi = max(hi for c, (_, hi) in spans.items() if c != aligned)
    if ours_lo <= others_hi:
        return (f"<b>The Pfam-to-Swiss-Prot ratio does not single out the methods Pfam "
                f"was built from</b> — {listed}. The <i>{aligned}</i> class does not drop "
                "further than every other class, so the drop is measuring how much harder "
                "Swiss-Prot is rather than how much Pfam flatters the methods that "
                "defined it. This figure therefore does not detect circularity, and the "
                "Pfam bars have to be argued about on what they are.")
    return (f"<b>The Pfam-to-Swiss-Prot ratio does single out the methods Pfam was built "
            f"from</b> — {listed}. Every <i>{aligned}</i> method drops further than every "
            "method outside that class, which is the shape circularity in the answer key "
            "would take: the class that defined Pfam gains most from being scored on it.")


def pfam_lead_bullet(data: dict, tools: dict) -> str:
    """Where kmerseek's Pfam bar sits against the methods Pfam itself was built from.

    Compared against the sequence-alignment class specifically, not against the best bar
    on the plot. Pfam is circular with the profile and alignment methods and with nothing
    else, so they are the comparison the circularity argument is about; a structure method
    that happens to score higher is beating kmerseek on a truth set it has no privileged
    relationship with, which is a different claim and gets its own sentence.

    The annotation ceiling is left out of both. hmmscan runs the Pfam HMMs against the very
    proteins Pfam annotated, so it is the top of the scale rather than a baseline.
    """
    per_class = _by_class(data, tools, CIRCULARITY_SETS[0])
    ours, aligned = CLASSES["kmerseek"][0], CLASSES["alignment"][0]
    if ours not in per_class or aligned not in per_class:
        return ""
    lo, hi = min(per_class[ours]), max(per_class[ours])
    rival = max(per_class[aligned])
    others = {c: max(v) for c, v in per_class.items()
              if c not in (ours, aligned, CLASSES["ceiling"][0])}
    over = ""
    if others:
        top_class, top = max(others.items(), key=lambda kv: kv[1])
        if top > hi:
            over = (f" <i>{top_class}</i> is higher still at {top:.3f}, on a truth set it "
                    "has no privileged relationship with.")
    if lo <= rival:
        return (f"<b>On Pfam, kmerseek's arms span {lo:.3f}-{hi:.3f}</b> against "
                f"{rival:.3f} for the best <i>{aligned}</i> method, so they straddle the "
                f"methods Pfam was built from rather than clearing them.{over}")
    return (f"<b>On Pfam, every kmerseek arm shown beats every <i>{aligned}</i> "
            f"method</b> — {lo:.3f}-{hi:.3f} against {rival:.3f}, a lead of "
            f"{lo / rival - 1:.0%} to {hi / rival - 1:.0%}. Pfam is the truth set those "
            "methods' own HMMs define, which makes it the least flattering place for that "
            f"lead to appear, not the most.{over}")


def pfamn_bullet(data: dict, tools: dict) -> str:
    """Where each method class lands on Pfam-N, stated rather than left to be noticed.

    Pfam-N is Pfam's neural extension: family members a deep model matched beyond what the
    HMM did. It is the one set in this section where the Pfam ordering does not carry
    over, so it goes in the caption instead of being left for a reader to spot.
    """
    per_class = _by_class(data, tools, "pfamn")
    if len(per_class) < 2:
        return ""
    best = {c: max(v) for c, v in per_class.items()}
    listed = ", ".join(f"{c} {v:.3f}"
                       for c, v in sorted(best.items(), key=lambda kv: -kv[1]))
    ours = CLASSES["kmerseek"][0]
    if ours not in best:
        return f"<b>Pfam-N</b> — best Fmax per method class: {listed}."
    rank = sorted(best.values(), reverse=True).index(best[ours]) + 1
    return (f"<b>Pfam-N reverses the Pfam ordering</b>: kmerseek ranks {rank} of "
            f"{len(best)} method classes there. Best Fmax per class — {listed}. Pfam-N is "
            "Pfam's neural extension, the family members a deep model matched beyond the "
            "HMM, and it is the truth set this benchmark's headline does not survive.")


def section_canonical(out: Path, metrics: pl.DataFrame) -> None:
    """Name the pinned arm and say what pinning does, when one is pinned.

    Written only under --canonical-variant. With no pin the report behaves exactly as it
    did before the flag existed, and a section explaining a mechanism nobody switched on
    would be noise.
    """
    if CANONICAL is None:
        return
    tool, variant = CANONICAL
    # The bare row key, without the mark label_of appends -- this sentence names the mark
    # separately, and printing both reads as the arm being called "... ★ ★".
    label = f"kmerseek {variant}" if tool == "kmerseek" else tool
    rows = metrics.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
    if rows.height == 0:
        body = (f"<p><b>Pinned arm not found.</b> <code>--canonical-variant</code> asked "
                f"for <code>{tool}:{variant}</code> and no row in the metrics carries that "
                "tool and variant, so nothing is marked anywhere in this report. Check the "
                "variant spelling against the Alphabet sweep section, which lists every "
                "arm that was scored.</p>")
    else:
        per_set = (ungrouped(rows).group_by("truth_set")
                   .agg(pl.col("fmax").mean().alias("fmax"),
                        pl.col("species").n_unique().alias("n"))
                   .sort("truth_set").to_dicts())
        listed = ", ".join(f"<b>{r['truth_set']}</b> {r['fmax']:.3f} over {r['n']} target "
                           f"proteome(s)" for r in per_set)
        body = (f"<p>Every per-tool figure in this report marks <code>{label}</code> with "
                f"{CANONICAL_MARK.strip()}, and forces it into the row set even where it "
                f"did not rank into one. Mean Fmax: {listed}.</p>")
    write_section(out, "qfo_canonical", {
        "id": "qfo_canonical",
        "section_name": "Pinned arm",
        "description": (
            "<p>One kmerseek configuration followed across every section, instead of "
            "whichever arm topped that section's own ranking.</p>"
            + bullets(
                "<b>Why this exists</b> — every per-tool section runs its own selection, "
                "so the kmerseek arm being drawn changes between figures without the "
                "report saying so. The covariate and divergence sections rank on the "
                "primary truth set's heldout half; the frontier and the PR/ROC curves "
                "rank on a gray-zone-weighted Fmax; each leaderboard ranks on its own "
                "truth set; the side-by-side plot ranks across all three at once.",
                "<b>Off by default.</b> Without <code>--canonical-variant</code> nothing "
                "is pinned and no number in this report differs from before the flag "
                "existed. No arm is hard-coded as the canonical one.",
                "<b>What pinning does not do</b> — it does not change any section's "
                "ranking, only adds a row and marks it. A section that trims to "
                "<code>--max-tools</code> can still drop the pinned arm if it ranks below "
                "the cut.")),
        "plot_type": "html",
        "data": body,
    })


def section_truthsets(out: Path, metrics: pl.DataFrame, max_tools: int) -> None:
    """The circularity check, in one plot."""
    cut, picked = split_per_truth_set(ungrouped(metrics))
    sets = sorted(picked)
    if len(sets) < 2:
        return
    split = ", ".join(f"{ts}: {s}" for ts, s in sorted(picked.items()))
    # The one place the truth-set guard is opted out of, and only for row selection: which
    # tools are worth a bar is decided across all three sets, but every number drawn is a
    # single set's own Fmax. Nothing here is a cross-set mean.
    ranked = best_variants(cut, across_truth_sets=True).head(max_tools)
    data, tools = {}, {}
    for row in ranked.to_dicts():
        sub = cut.filter((pl.col("tool") == row["tool"])
                         & (pl.col("variant") == row["variant"]))
        cell = {}
        for ts in sets:
            hit = sub.filter(pl.col("truth_set") == ts)
            cell[ts] = hit["fmax"].max() if hit.height else None
        data[row["label"]] = cell
        tools[row["label"]] = row["tool"]
    write_section(out, "qfo_truthsets", {
        "id": "qfo_truthsets",
        "section_name": "Truth sets side by side",
        "description": (
            f"<p>Best Fmax per tool against each truth set, each on its own split "
            f"(<code>{split}</code>).</p>"
            + bullets(
                "<b>Only Pfam is swept</b>, so only Pfam has a heldout half. The others "
                "are scored whole.",
                "<b>Pfam is defined by the HMMs the profile baselines run</b>, so it is "
                "the truth set those methods should be most at home on.",
                pfam_lead_bullet(data, tools),
                circularity_bullet(data, tools),
                pfamn_bullet(data, tools),
                "<b>What this figure cannot do on its own is measure circularity.</b> "
                "That needs a drop specific to the class that wrote the answer key, and "
                "the per-class ratios above are the test of whether there is one. Read "
                "them before reading the bar heights as evidence about Pfam.",
                "<b>Nothing here is averaged across truth sets.</b> Each bar is one set's "
                "own Fmax on its own split, and the ratios compare two of them rather "
                "than pooling them.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_truthsets_plot", "title": "Fmax by truth set",
                    "ylab": "Fmax", "cpswitch": False, "stacking": "group", "height": 450},
        "categories": {ts: {"name": ts} for ts in sets},
        "data": data,
    })


def section_dedup_transfers(out: Path, metrics_all: pl.DataFrame, primary_truth: str,
                            max_tools: int) -> None:
    """Detection against redundancy: the same arm scored with and without collapsing
    calls that are one prediction reached through many targets.

    Homology transfer emits one call per target domain covered, so a human kinase hitting
    300 mouse kinases produces 300 calls of the same family over the same residues. The
    matcher makes one a true positive and the rest false positives. That is real behaviour
    worth charging a tool for, but it is also a property of how redundant the target
    proteome is rather than of whether the domain was found -- which is why precision falls
    as target proteomes get larger even while recall rises. Both numbers are reported
    because neither one alone answers the question.
    """
    if "dedup_transfers" not in metrics_all.columns:
        return
    base = ungrouped(metrics_all.filter(pl.col("truth_set") == primary_truth))
    if base.height == 0 or base["dedup_transfers"].n_unique() < 2:
        return

    per_mode = {}
    for mode in (False, True):
        cut, split = pick_split(base.filter(pl.col("dedup_transfers") == mode))
        if cut.height == 0:
            return
        agg = (cut.group_by("tool", "variant")
                  .agg(pl.col("fmax").mean().alias("fmax"),
                       pl.col("precision").mean().alias("precision"),
                       pl.col("recall_reachable").mean().alias("recall"),
                       pl.col("n_calls").mean().alias("n_calls")))
        per_mode[mode] = agg
        last_split = split

    # The arm is held fixed at whichever variant wins WITHOUT dedup, so the two bars are
    # the same tool configuration measured twice. Letting each mode pick its own best
    # variant would fold a variant change into what is meant to be a one-difference
    # comparison.
    best = (per_mode[False].sort("fmax", descending=True, nulls_last=True)
            .group_by("tool", maintain_order=True).first())
    joined = (best.select("tool", "variant", "fmax", "precision", "n_calls")
              .join(per_mode[True].select(
                  "tool", "variant",
                  pl.col("fmax").alias("fmax_dedup"),
                  pl.col("precision").alias("precision_dedup"),
                  pl.col("n_calls").alias("n_calls_dedup")),
                  on=["tool", "variant"], how="inner")
              .with_columns(
                  (pl.col("fmax_dedup") - pl.col("fmax")).alias("fmax_delta"),
                  pl.when(pl.col("n_calls") > 0)
                    .then(1 - pl.col("n_calls_dedup") / pl.col("n_calls"))
                    .otherwise(None).alias("redundant_fraction"))
              .sort("fmax_dedup", descending=True, nulls_last=True)
              .head(max_tools))
    if joined.height == 0:
        return

    label = lambda r: r["tool"] if r["variant"] in (None, "", "-") else f"{r['tool']} ({r['variant']})"
    rows = joined.to_dicts()

    write_section(out, "qfo_dedup_bar", {
        "id": "qfo_dedup_bar",
        "section_name": "Detection vs redundant transfer",
        "description": (
            f"<p>{primary_truth} truth, <code>{last_split}</code> split, averaged over "
            f"target species.</p>"
            + bullets(
                "<b>Each tool is shown at the variant that wins <i>without</i> dedup</b>, "
                "so both bars are the same configuration scored twice.",
                "<b>As reported</b> charges every redundant copy of a call as a false "
                "positive.",
                "<b>One call per region</b> collapses calls of the same family that "
                "overlap each other on the same query protein, keeping the best-scoring "
                "one.",
                "<b>Tandem repeats survive the collapse.</b> Adjacent domains of one "
                "family barely overlap, so they are separate calls either way.",
                "<b>The gap</b> is how much of a tool's score is redundancy in the target "
                "proteome rather than detection.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_dedup_bar_plot",
                    "title": f"Fmax with and without collapsing redundant transfers "
                             f"({primary_truth})",
                    "ylab": "Fmax", "height": 450, "stacking": "group"},
        "categories": {"fmax": {"name": "as reported", "color": "#b0b0b0"},
                       "fmax_dedup": {"name": "one call per region", "color": "#0f9d76"}},
        "data": {label(r): {"fmax": r["fmax"], "fmax_dedup": r["fmax_dedup"]} for r in rows},
    })

    write_section(out, "qfo_dedup_table", {
        "id": "qfo_dedup_table",
        "section_name": "Redundant transfer, per tool",
        "description": (
            "<p>The numbers behind the plot above.</p>"
            + bullets(
                "<b>Redundant calls</b> is the fraction of reported calls that were "
                "another call of the same family over the same residues of the same query "
                "protein.",
                "It rises with how many homologues the target proteome contains, which is "
                "why it is largest for the profile methods on vertebrate targets.",
                "<b>A large Fmax gap</b> means the tool found the domain and was charged "
                "for saying so repeatedly.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_dedup_table_plot",
                    "title": f"Detection vs redundancy ({primary_truth})",
                    "col1_header": "Tool", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "fmax": dict(title="Fmax", min=0, max=1, scale="Blues", format="{:,.3f}",
                         description="As reported, redundant copies charged as errors"),
            "fmax_dedup": dict(title="Fmax (deduped)", min=0, max=1, scale="Greens",
                               format="{:,.3f}", description="One call per query region"),
            "fmax_delta": dict(title="Gap", scale="RdYlGn", format="{:,.3f}",
                               description="How much of the score was redundancy"),
            "redundant_fraction": dict(title="Redundant calls", min=0, max=1,
                                       scale="Reds", format="{:,.1%}"),
            "precision": dict(title="Precision", min=0, max=1, scale="Oranges",
                              format="{:,.3f}", hidden=True),
            "precision_dedup": dict(title="Prec. (deduped)", min=0, max=1, scale="Oranges",
                                    format="{:,.3f}"),
            "n_calls": dict(title="Calls", format="{:,.0f}", hidden=True),
        },
        "data": {label(r): {k: r[k] for k in
                            ("fmax", "fmax_dedup", "fmax_delta", "redundant_fraction",
                             "precision", "precision_dedup", "n_calls")} for r in rows},
    })


# Below this many distinct labels in the answer key, `pfam_id` is a category vocabulary
# rather than a family vocabulary: every proteome carries nearly all of it, the reachability
# join matches everything, and the bar stops being a ceiling. Kept in step with
# MIN_REACHABILITY_VOCAB in aggregate_domain_metrics.py.
MIN_REACHABILITY_VOCAB = 50

# Ratio to the median at which a target species has so little annotation that its recall is
# capped by curation coverage rather than by divergence. In step with THIN_TARGET_RATIO in
# aggregate_domain_metrics.py.
THIN_TARGET_RATIO = 0.05


def reachability_caveat(per: pl.DataFrame, primary_truth: str) -> str:
    """Withdraw the ceiling claim above when the numbers on this run do not support it.

    Two ways this bar lies, both of which it did in the midi run:

    Degenerate vocabulary. On the Swiss-Prot truth set `pfam_id` holds a curated feature
    type from a ~15-value vocabulary, not a Pfam accession. Every proteome has nearly all
    of them, so reachable / truth is ~1.0 for every species and `recall_reachable` is
    plain recall wearing a reachability label. Eight of nine species returned exactly
    7_000 / 7_000.

    Thin target annotation. The bar counts LABELS present in the target, not annotated
    PROTEINS, so a species with almost no annotation still scores as fully reachable.
    Ciona intestinalis has 28 Swiss-Prot entries against 2_309-20_417 for every other
    target species; it read 6_991 / 7_000 reachable while every arm's transferable calls
    collapsed 30-130x. The bar was the thing that was supposed to catch that.
    """
    notes = []
    if "n_truth_families" in per.columns:
        vocab = int(per["n_truth_families"].max() or 0)
        if vocab < MIN_REACHABILITY_VOCAB:
            notes.append(
                f"<p><b>This bar is not a ceiling on the <code>{primary_truth}</code> "
                f"truth set.</b> Its <code>pfam_id</code> column holds one of {vocab} "
                "curated feature types rather than a protein family, and every proteome "
                "carries nearly all of them, so the reachability join matches almost "
                "everything and <code>recall_reachable</code> here is plain recall. Read "
                "the ceiling on the Pfam truth set, where the label is a family.</p>"
            )
    if "n_target_map_proteins" in per.columns:
        sub = per.filter(pl.col("n_target_map_proteins").is_not_null())
        if sub.height >= 3:
            median = float(sub["n_target_map_proteins"].median())
            thin = sub.filter(pl.col("n_target_map_proteins") < median
                              * THIN_TARGET_RATIO)
            if median > 0 and thin.height:
                listed = ", ".join(
                    f"<b>{r['species']}</b> ({r['n_target_map_proteins']} annotated "
                    f"target proteins against a median of {median:.0f})"
                    for r in thin.sort("n_target_map_proteins").to_dicts()
                )
                notes.append(
                    "<p><b>Full-height bars do not mean full annotation.</b> This bar "
                    "counts labels present in the target, not annotated proteins, and "
                    f"{listed} has so little target annotation that every arm's recall "
                    "there is capped by curation coverage rather than by divergence. Do "
                    "not read that species as an evolutionary result.</p>"
                )
    return "".join(notes)


def _reachability_per_species(metrics: pl.DataFrame, truth_set: str) -> pl.DataFrame:
    cut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == truth_set)))
    cut = cut.filter(pl.col("species") != "all")
    need = {"n_truth_instances", "n_reachable_instances"}
    if cut.height == 0 or not need.issubset(set(cut.columns)):
        return cut.head(0)
    agg = [pl.col("n_truth_instances").max(), pl.col("n_reachable_instances").max()]
    if "n_target_map_proteins" in cut.columns:
        agg.append(pl.col("n_target_map_proteins").max())
    if "n_truth_families" in cut.columns:
        agg.append(pl.col("n_truth_families").max())
    return (cut.group_by("species", "species_mya").agg(agg)
               .with_columns((pl.col("n_reachable_instances")
                              / pl.col("n_truth_instances").clip(lower_bound=1))
                             .alias("reachable_frac")))


# The answer key whose label is a Pfam family and whose reachability is therefore shared
# family content. Preferred over any other family-labelled set for the lead panel so that
# the report's framing figure is drawn on the same key the domain definitions come from.
REACHABILITY_LEAD_TRUTH = "pfam"


def pick_reachability_truth_set(metrics: pl.DataFrame,
                                primary_truth: str) -> tuple[str, pl.DataFrame]:
    """The truth set whose reachability is shared family content rather than curation depth.

    A curated answer key ranks proteomes by how much of each has been curated -- Ciona has
    23 curated Swiss-Prot proteins against E. coli's 2_999 -- and its label is a feature
    type from a vocabulary every proteome carries nearly all of, so the bar is also nearly
    flat. Neither of those is a recall ceiling. The lead panel is drawn on a family-labelled
    key: Pfam when the run has it, otherwise whichever uncurated key varies most between
    proteomes. Every other key still gets its own panel, in the supplementary group, saying
    what it ranks on.
    """
    candidates = []
    for ts in sorted(metrics["truth_set"].unique().to_list()):
        per = _reachability_per_species(metrics, ts)
        if per.height == 0:
            continue
        spread = float(per["reachable_frac"].max() - per["reachable_frac"].min())
        candidates.append((ts, per, spread))
    if not candidates:
        return primary_truth, _reachability_per_species(metrics, primary_truth)
    for ts, per, _ in candidates:
        if ts == REACHABILITY_LEAD_TRUTH:
            return ts, per
    uncurated = [c for c in candidates if c[0] not in CURATED_TRUTH_SETS] or candidates
    ts, per, _ = max(uncurated, key=lambda c: c[2])
    return ts, per


def section_reachability(out: Path, metrics: pl.DataFrame, primary_truth: str) -> None:
    """The framing figure: how much of the answer key any search could reach at all.

    Every Fmax, every recall and every leaderboard row in this report sits on this
    denominator. A human domain whose family has no instance anywhere in the target
    proteome cannot be transferred by any method, so a proteome where a large share of the
    key is unreachable caps every arm equally and a tool comparison drawn there is a
    comparison of what is left.

    ONE PANEL PER TRUTH SET, and that is not a formality here. The two answer keys measure
    different things about a proteome and they disagree about the ordering. On Pfam the
    label is a protein family, so the bar is what it says it is: shared family content,
    with Ciona above fly and chicken and E. coli last. On Swiss-Prot the label is a
    manually curated feature type, so the bar tracks how much of the proteome has been
    curated -- 23 curated Ciona proteins against 2_999 for E. coli -- and it ranks a
    tunicate below a bacterium. Drawing one of those and calling it the framing figure
    would publish a curation artefact as a biological result. The panel drawn in the main
    report is the one whose reachability varies with family content; the rest are
    supplementary and say what they rank on.

    Descending by reachable share rather than by divergence time. Divergence order is the
    right axis for the accuracy sections, and the wrong one here: the question this panel
    answers is how uneven the ceiling is, and sorting on the quantity is what makes that
    readable.
    """
    lead, _ = pick_reachability_truth_set(metrics, primary_truth)
    for ts in sorted(metrics["truth_set"].unique().to_list()):
        per = _reachability_per_species(metrics, ts)
        if per.height == 0:
            continue
        _reachability_panel(out, per, ts, is_lead=(ts == lead), lead=lead)


def _reachability_panel(out: Path, per: pl.DataFrame, truth_set: str, *,
                        is_lead: bool, lead: str) -> None:
    rows = per.sort("reachable_frac", descending=True).to_dicts()
    total = int(max(r["n_truth_instances"] for r in rows))
    data = {}
    for r in rows:
        key = species_tick(r["species"], truth_set, Mya=r["species_mya"],
                           reachable=r["n_reachable_instances"])
        data[key] = {"reachable_pct": 100 * r["reachable_frac"],
                     "unreachable_pct": 100 * (1 - r["reachable_frac"])}
    lowest = rows[-1]
    mean_frac = sum(r["reachable_frac"] for r in rows) / len(rows)
    reach = {r["species"]: r["reachable_frac"] for r in rows}
    flat = (rows[0]["reachable_frac"] - lowest["reachable_frac"]) < 0.05
    curated = truth_set in CURATED_TRUTH_SETS

    if curated:
        what = (f"<p><b>This bar ranks proteomes by curation depth, not by shared family "
                f"content.</b> The <code>{truth_set}</code> answer key is built from "
                f"manually curated entries, so a species' position here is partly how much "
                f"of it somebody has curated. Read <code>{lead}</code>'s panel for the "
                f"biological ordering.</p>")
    else:
        what = (f"<p>The label on this answer key is a protein family, so the bar is "
                f"shared family content: what any search could transfer at all.</p>")

    if flat:
        shape = (f"<b>The bar is nearly flat, and that is the artefact.</b> Every proteome "
                 f"reads between {100 * lowest['reachable_frac']:.1f}% and "
                 f"{100 * rows[0]['reachable_frac']:.1f}%, so the reachability join is "
                 "matching almost everything and <code>recall_reachable</code> on this "
                 "truth set is plain recall wearing a reachability label. The constant "
                 "denominator is not a shared ceiling.")
    else:
        shape = (f"<b>The ceiling is uneven.</b> It runs from "
                 f"{100 * rows[0]['reachable_frac']:.1f}% at "
                 f"<code>{rows[0]['species']}</code> down to "
                 f"{100 * lowest['reachable_frac']:.1f}% at "
                 f"<code>{lowest['species']}</code>, a mean of {100 * mean_frac:.1f}% over "
                 f"{len(rows)} proteomes. An arm's mean Fmax is a mean over "
                 f"{len(rows)} different ceilings.")

    write_section(out, "qfo_reachability" if is_lead else f"qfo_reachability_{truth_set}", {
        "id": "qfo_reachability" if is_lead else f"qfo_reachability_{truth_set}",
        "section_name": f"Recall ceiling per species — {truth_set} truth",
        "description": (
            f"<p>The share of the {total:,}-instance answer key whose family exists "
            f"somewhere in each target proteome, ordered from most reachable to least "
            f"({truth_set} truth).</p>" + what
            + bullets(
                ("<b>Read this before any leaderboard.</b> " if is_lead else "")
                + "No search of any kind can transfer a family the target proteome does "
                  "not have, so the pale part of each bar is out of reach for every arm in "
                  "this report and every recall_reachable divides by the green part only.",
                shape,
                "<b>Each bar's tick label carries its own reachable count</b>, since the "
                "denominators are what the shares are of.",
                curation_caveat_note(truth_set, reach))
            + reachability_caveat(per, truth_set)),
        "plot_type": "bargraph",
        "pconfig": {"id": ("qfo_reachability_plot" if is_lead
                           else f"qfo_reachability_{truth_set}_plot"),
                    "title": f"Reachable share of the answer key ({truth_set} truth)",
                    "ylab": "% of answer-key instances",
                    "cpswitch": False, "sort_samples": False,
                    "ymax": 100, "height": 420, "tt_decimals": 1},
        "categories": {
            "reachable_pct": {"name": "family present in target", "color": "#0f9d76"},
            "unreachable_pct": {"name": "no family in target", "color": "#d9d9d9"}},
        "data": data,
    }, supplementary=not is_lead)


def grouped(n: int) -> str:
    """14_873, the way this project writes integers, without a comma anywhere near prose."""
    return f"{int(n):,}".replace(",", "_")


def identity_bin_midpoint(label: str) -> float | None:
    """Numeric x for a "20-30%" identity band, so the axis is identity and not five slots.

    Same argument as band_midpoint, which this defers to once the percent sign is off: the
    bins are not equal width (0-20 spans twenty points, 60-100 spans forty), and the claim
    this panel tests is about where on the identity axis a tool stops working, so the
    spacing carries the result.
    """
    return band_midpoint(label.rstrip("%"))


IDENTITY_BIN_ORDER = ["0-20%", "20-30%", "30-40%", "40-60%", "60-100%"]


def _identity_truth_set(metrics: pl.DataFrame, primary_truth: str) -> tuple[str, bool]:
    """The truth set whose identity bins are actually populated, and whether it was forced.

    attach_identity joins on (accession, pfam_id, domain_start, domain_end). On Swiss-Prot
    `pfam_id` holds a curated feature type rather than a Pfam accession, so nothing matches
    and every instance lands in no_homolog -- one bin, no axis. Falling back to Pfam is the
    only way to draw this panel at all, and Pfam is the truth set the sweep also selected
    on, so the fallback is reported in the caption rather than made quietly.
    """
    for ts in [primary_truth] + [t for t in ("pfam", "pfamn") if t != primary_truth]:
        bins = (metrics.filter((pl.col("truth_set") == ts)
                               & (pl.col("stratum_axis") == "identity"))["stratum"]
                .unique().to_list())
        if len([b for b in bins if b in IDENTITY_BIN_ORDER]) >= MIN_COVARIATE_BINS:
            return ts, ts != primary_truth
    return "", False


# The comparison set for the annotated divergence panel: one arm per thing a reader might
# reach for instead of kmerseek, rather than every arm that scored. Ordered profile ->
# structure so the legend reads down the cost axis.
DIVERGENCE_PEERS = ["mmseqs2_iterative", "hmmer3_phmmer", "hhblits", "foldseek"]


def peer_annotation(tool: str, cost: dict[str, float]) -> str:
    """What a reader has to spend, or install, to buy that line.

    A tool's Fmax is not comparable to another's without this: hhblits sits above every
    other line here and costs three orders of magnitude more per search, and foldseek needs
    a structure for every query. Both facts belong on the line, not in a table elsewhere in
    the report that the reader has to hold in their head.
    """
    bits = []
    if cost.get(tool):
        bits.append(f"{cost[tool]:.3g} CPU-h/search")
    if NEEDS_3D.get(tool) == "Yes":
        bits.append("needs structure")
    return f" ({', '.join(bits)})" if bits else ""


# The two axes that are now measured over the domain rather than over its protein, with
# the label each one carries on a figure.
REGION_AXES = {
    "plddt_region": ("mean pLDDT of the domain", "pLDDT"),
    "disorder_region": ("mean disorder of the domain", "disorder"),
}


def region_axis_series(cut: pl.DataFrame, axis: str, keep: list) -> tuple[dict, dict, dict]:
    """Fmax and reachable recall per arm along one region axis, plus the n behind each point.

    x is `stratum_value_mean`, the measured mean of the covariate over the instances in that
    bin, not the bin's midpoint. The bins are deliberately uneven -- modelled domains bunch
    above pLDDT 60 -- so a midpoint would put a point where no domain sits.
    """
    sub = cut.filter(pl.col("stratum_axis") == axis)
    if sub.height == 0 or "stratum_value_mean" not in sub.columns:
        return {}, {}, {}
    fmax, recall, counts = {}, {}, {}
    for tool, variant, label in keep:
        arm = sub.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if arm.height == 0:
            continue
        agg = (arm.group_by("stratum")
               .agg(pl.col("stratum_value_mean").mean().alias("x"),
                    pl.col("fmax").mean().alias("fmax"),
                    pl.col("recall_reachable").mean().alias("recall"),
                    pl.col("n_truth_instances").max().alias("n"))
               .filter(pl.col("x").is_not_null())
               .sort("x"))
        if agg.height == 0:
            continue
        fmax[label] = {str(r["x"]): r["fmax"] for r in agg.to_dicts()}
        recall[label] = {str(r["x"]): r["recall"] for r in agg.to_dicts()}
        for r in agg.to_dicts():
            counts[str(r["x"])] = max(counts.get(str(r["x"]), 0), r["n"] or 0)
    return fmax, recall, counts


def section_region_axes(out: Path, metrics: pl.DataFrame, primary_truth: str,
                        max_tools: int) -> None:
    """Recovery against the domain's own confidence and disorder, two readings per axis.

    Fmax is computed over a SET of instances at a threshold, so there is no Fmax for a
    single domain and no honest way to put one on a per-region scatter. What this does
    instead is compute it within each narrow bin of the region covariate and plot it against
    the measured mean of that bin, which is a continuous axis in everything but name.

    Two panels per axis because the two readings can disagree and the disagreement is
    informative: Fmax moves with the threshold an arm happens to pick, reachable recall does
    not. An arm that holds Fmax across the axis while its recall falls is being carried by
    precision on a shrinking set of calls.
    """
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == primary_truth))
    if cut.height == 0 or "stratum_axis" not in cut.columns:
        return
    board = best_variants(ungrouped(cut)).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    lead = next((lbl for t, _, lbl in keep if t == "kmerseek"), None)

    for axis, (xlab, short) in REGION_AXES.items():
        fmax, recall, counts = region_axis_series(cut, axis, keep)
        if len(fmax) < 2:
            continue
        colors, dashes = emphasis_colors(list(fmax), lead=lead)
        n_row = ", ".join(
            f"{float(x):.3g}: n={grouped(n)}"
            for x, n in sorted(counts.items(), key=lambda kv: float(kv[0])))
        protein_twin = "plddt" if axis == "plddt_region" else "disorder_seq"
        write_section(out, f"qfo_{axis}", {
            "id": f"qfo_{axis}",
            "section_name": f"Region {short}",
            "description": (
                f"<p>Fmax and reachable recall against the {xlab}, mean over target "
                f"proteomes ({primary_truth} truth, <code>{split}</code> split).</p>"
                + bullets(
                    f"<b>This is the domain's own {short}</b>, averaged over the residues "
                    f"of that domain instance. The <code>{protein_twin}</code> axis "
                    f"elsewhere in this report is the same quantity taken over the whole "
                    f"query protein, which gives every domain of a protein the same value "
                    f"-- an ordered domain inside a disordered protein reads as disordered "
                    f"there and does not here.",
                    "<b>x is the measured mean of each bin</b>, not the bin's midpoint. The "
                    "bins are uneven because the data is, and a midpoint would put a point "
                    "where no domain sits.",
                    f"<b>Instances per bin:</b> {n_row}. A bin holding tens of domains "
                    f"carries no result whatever height it draws at.",
                    "<b>Read both panels.</b> Fmax moves with the threshold an arm picks "
                    "and reachable recall does not, so an arm that holds Fmax while its "
                    "recall falls is being carried by precision over a shrinking set of "
                    "calls rather than still finding the domains.")),
            "plot_type": "linegraph",
            "pconfig": {
                "id": f"qfo_{axis}_plot",
                "title": f"Recovery vs {xlab}",
                "xlab": xlab, "ylab": "score (mean over target proteomes)",
                "ymin": 0, "height": 520, "style": "lines+markers",
                "colors": colors, "dash_styles": dashes,
                "data_labels": [
                    {"name": "Fmax", "ylab": "Fmax (mean over target proteomes)"},
                    {"name": "Recall (reachable)",
                     "ylab": "recall_reachable (mean over target proteomes)"},
                ],
            },
            "data": [fmax, recall],
        })


def section_divergence_annotated(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                                 n_queries: int, primary_truth: str) -> None:
    """Fmax against divergence time for one kmerseek arm and four peers, cost on the label.

    The full Divergence section draws every arm that ranked and answers "which is best".
    This one answers the question a reader actually arrives with: for a target this far from
    human, what do the realistic alternatives score, and what does each cost. Five lines,
    because that is how many a reader can hold; the cost and the structure requirement ride
    in the line's own label so the comparison cannot be read without them.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    cut = cut.filter(pl.col("species") != "all")
    if cut.height == 0 or "species_mya" not in cut.columns:
        return

    # Ciona is dropped from THIS panel rather than marked, because a five-line figure has no
    # room to explain it and a point every line dips on is read as biology. It stays in the
    # full Divergence section, where the caveat has somewhere to live.
    dropped = CURATION_CAVEAT_SPECIES if (
        cut["species"] == CURATION_CAVEAT_SPECIES).any() else None
    if dropped:
        cut = cut.filter(pl.col("species") != CURATION_CAVEAT_SPECIES)

    board = best_variants(cut)
    km = next((r for r in board.to_dicts() if r["tool"] == "kmerseek"), None)
    keep = ([(km["tool"], km["variant"], km["label"])] if km else [])
    for tool in DIVERGENCE_PEERS:
        row = next((r for r in board.to_dicts() if r["tool"] == tool), None)
        if row:
            keep.append((row["tool"], row["variant"], row["label"]))
    if len(keep) < 2:
        return

    cost = {}
    if trace is not None and trace.height:
        per_tool = throughput_per_tool(trace, n_queries)
        if per_tool.height:
            n_species = max(int(cut["species"].n_unique()), 1)
            cost = {r["tool"]: r["cpu_hours"] / n_species
                    for r in per_tool.to_dicts() if r.get("cpu_hours")}

    data, labels = {}, {}
    for tool, variant, label in keep:
        sub = (cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
               .group_by("species_mya").agg(pl.col("fmax").mean()).sort("species_mya"))
        if sub.height == 0:
            continue
        shown = label + peer_annotation(tool, cost)
        labels[label] = shown
        data[shown] = {str(r["species_mya"]): r["fmax"] for r in sub.to_dicts()}
    if not data:
        return

    lead = labels.get(km["label"]) if km else None
    colors, dashes = emphasis_colors(list(data), lead=lead)
    ticks = (cut.select("species", "species_mya").unique()
             .sort("species_mya").to_dicts())
    tick_note = ", ".join(f"{r['species']} {r['species_mya']:.0f}" for r in ticks)
    omitted = (
        f"<b><code>{dropped}</code> is omitted from this panel.</b> On the "
        f"<code>{primary_truth}</code> answer key it carries 23 curated proteins against a "
        f"median of 3,174, so its point measures curation depth rather than divergence. It "
        f"is kept, and explained, in the full Divergence section and in the Pfam "
        f"reachability panel where there is room to say why."
    ) if dropped else ""

    write_section(out, "qfo_divergence_annotated", {
        "id": "qfo_divergence_annotated",
        "section_name": "Divergence, against the alternatives",
        "description": (
            f"<p>Fmax against divergence time from human, for one kmerseek arm and the "
            f"peers a reader would otherwise reach for ({primary_truth} truth, "
            f"<code>{split}</code> split). Proteomes on the axis: {tick_note} Mya.</p>"
            + bullets(
                "<b>Cost and requirements are on the line's own label</b>, because Fmax "
                "alone does not say what an arm costs to run or what it needs installed, "
                "and the highest line here is also the most expensive by three orders of "
                "magnitude.",
                "<b>CPU-hours are per search, per proteome</b> -- the tool's total divided "
                "by the number of target proteomes, so it is what one more species costs.",
                omitted,
                "<b>Five lines, not nineteen.</b> The full sweep is in the Divergence "
                "section; this one is the comparison a reader arrives with.")),
        "plot_type": "linegraph",
        "pconfig": {"id": "qfo_divergence_annotated_plot",
                    "title": f"Fmax vs target proteome ({primary_truth} truth)",
                    "xlab": "divergence from human (Mya)",
                    "ylab": "Fmax (per target proteome)",
                    "ymin": 0, "height": 520, "style": "lines+markers",
                    "colors": colors, "dash_styles": dashes},
        "data": data,
    })


def section_identity_vs_divergence(out: Path, metrics: pl.DataFrame, primary_truth: str,
                                   max_tools: int) -> None:
    """The same arms on two axes for the same question, drawn on one shared y range.

    Divergence time is a property of the target proteome; percent identity is a property of
    the individual domain pair. A tool that holds its score out to 2000 Mya and a tool that
    holds it down to 20% identity are not making the same claim, and where the two axes
    disagree for the SAME arm is where the hypothesis is actually tested rather than
    restated. They are two sections rather than one two-panel figure because MultiQC has no
    side-by-side layout; the y range is pinned across both so the heights are comparable,
    which is the only thing the shared axis was for.
    """
    ts, forced = _identity_truth_set(metrics, primary_truth)
    if not ts:
        return
    cut, split = pick_split(metrics.filter(pl.col("truth_set") == ts))
    per_species = ungrouped(cut).filter(pl.col("species") != "all")
    ident = cut.filter(pl.col("stratum_axis") == "identity")
    if per_species.height == 0 or "species_mya" not in per_species.columns:
        return

    board = best_variants(per_species).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    control = _divergence_control(per_species, {(t, v) for t, v, _ in keep})
    if control and control[3]:
        keep.append(control[:3])

    bins = [b for b in IDENTITY_BIN_ORDER if b in ident["stratum"].unique().to_list()]
    mya, pident = {}, {}
    for tool, variant, label in keep:
        sub = (per_species.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
               .group_by("species_mya").agg(pl.col("fmax").mean()).sort("species_mya"))
        mya[label] = {str(r["species_mya"]): r["fmax"] for r in sub.to_dicts()}
        isub = (ident.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
                .group_by("stratum").agg(pl.col("fmax").mean()))
        lookup = {r["stratum"]: r["fmax"] for r in isub.to_dicts()}
        pident[label] = {str(identity_bin_midpoint(b)): lookup[b]
                         for b in bins if lookup.get(b) is not None}
    mya = {k: v for k, v in mya.items() if v}
    pident = {k: v for k, v in pident.items() if v}
    if not mya or not pident:
        return

    # One y range over both panels. Read off the drawn values, not off the metric's 0-1
    # bound: Fmax tops out near 0.3 here, and a 0-1 axis would flatten both panels into the
    # bottom third and hide the very difference they are drawn to show.
    ceiling = max([v for s in list(mya.values()) + list(pident.values())
                   for v in s.values() if v is not None] or [1.0])
    shared_ymax = round(ceiling * 1.08, 3)

    # n per bin, so a bin holding forty instances is not read against one holding thousands.
    # The count is a property of the answer key, not of the arm, so any arm's rows carry it;
    # max guards against an arm that happens to be missing a species.
    counts = {r["stratum"]: r["n"] for r in
              ident.group_by("stratum")
              .agg(pl.col("n_truth_instances").max().alias("n")).to_dicts()}
    n_row = ", ".join(f"{b} n={grouped(counts.get(b, 0))}" for b in bins)
    unreachable = counts.get("no_homolog", 0)
    binned_total = sum(counts.get(b, 0) for b in bins)
    top_share = (100 * counts.get("60-100%", 0) / binned_total) if binned_total else 0.0

    lead = next((lbl for t, _, lbl in keep if t == "kmerseek"), None)
    colors, dashes = emphasis_colors([lbl for _, _, lbl in keep], lead=lead,
                                     control=control[2] if control else None)
    forced_note = (
        f"<b>Drawn on the <code>{ts}</code> truth set, not <code>{primary_truth}</code>.</b> "
        f"On <code>{primary_truth}</code> the identity join has nothing to match: it keys on "
        f"(accession, pfam_id, domain_start, domain_end) and that truth set's "
        f"<code>pfam_id</code> holds a curated feature type rather than a Pfam accession, so "
        f"every instance lands in <code>no_homolog</code> and there is no axis. "
        f"<code>{ts}</code> is also the truth set the sweep selected its alphabet and k on, "
        f"so read this pair as the weaker, circular evidence it is."
    ) if forced else ""
    shared = (
        f"<b>Both panels share one y axis, 0 to {shared_ymax}.</b> They are the same arms "
        f"and the same metric; only the x axis differs, so a height in one is a height in "
        f"the other. The range is taken from the data rather than from Fmax's 0-1 bound."
    )
    skew = (
        f"<b>{top_share:.0f}% of the instances that have any same-family target at all sit "
        f"in the 60-100% band.</b> That is what the divergence panel is mostly built from, "
        f"so a flat line across Mya is compatible with detecting only well-conserved "
        f"domains. This is the stratification that decides between ancient-core detection "
        f"and twilight-zone detection, and on this run it points at the former."
    ) if binned_total else ""
    disagree = (
        "<b>Where the two panels disagree is the finding.</b> An arm that holds its level "
        "across divergence time while falling away across identity is detecting an "
        "ancient conserved core, not working in the twilight zone -- the distant proteomes "
        "keep the domains that stayed similar, so a flat Mya line can be drawn entirely by "
        "high-identity pairs. Read the identity panel before claiming remote homology."
    )

    for sid, name, panel, xlab, title, extra in (
        ("qfo_identity_pair_mya", "Divergence axis", mya,
         "divergence from human (Mya)", "Fmax vs divergence time",
         "<b>Each point is one target proteome</b>, and these are the values the "
         "leaderboard's means are taken over."),
        ("qfo_identity_pair_pident", "Identity axis", pident,
         "identity to closest same-family target domain (%)", "Fmax vs percent identity",
         f"<b>Each point is an identity band, plotted at its midpoint</b> because the bands "
         f"are not equal width, and the bands are nowhere near equally populated: {n_row}. "
         f"A band holding tens of instances carries no result, whatever height it draws at. "
         f"<b><code>no_homolog</code> is not on this axis</b> -- {grouped(unreachable)} "
         f"instances with no same-family target domain at all have no identity to plot, and "
         f"putting them at zero would read as an identity of zero rather than as absence. "
         + skew),
    ):
        write_section(out, sid, {
            "id": sid,
            "section_name": name,
            "description": (
                f"<p>Fmax per arm, mean over target proteomes ({ts} truth, "
                f"<code>{split}</code> split).</p>"
                + bullets(extra, shared, forced_note, disagree)),
            "plot_type": "linegraph",
            "pconfig": {"id": f"{sid}_plot", "title": title, "xlab": xlab,
                        "ylab": "Fmax (mean over target proteomes)",
                        "ymin": 0, "ymax": shared_ymax, "height": 460,
                        "style": "lines+markers",
                        "colors": colors, "dash_styles": dashes},
            "data": panel,
        })


ALPHABET_DOSE_METRICS = [
    ("family_fmax", "recognition — right family, anywhere in the protein", "#0072b2"),
    ("fmax", "placement — right family in the right interval", "#d55e00"),
]


def section_alphabet_dose_response(out: Path, metrics: pl.DataFrame,
                                   primary_truth: str) -> None:
    """What alphabet reduction buys, recognition and placement on one axis.

    These two numbers are the same CAFA machinery read at two levels. family_fmax asks
    whether the right family was named anywhere in the query protein; fmax asks whether it
    was named over the right interval. Reduction is supposed to help the first -- a coarser
    alphabet survives substitutions -- and there is no reason it should help the second,
    because collapsing residue classes throws away exactly the resolution a boundary needs.

    Drawn together on one x axis, that is a two-line figure and the separation between the
    lines IS the result. The report had these in different sections on different axes,
    where a reader had to hold one plot in their head to read the other.
    """
    cut, split = pick_split(ungrouped(metrics.filter(
        (pl.col("truth_set") == primary_truth) & (pl.col("tool") == "kmerseek"))))
    if cut.height == 0:
        return
    parsed = parse_kmerseek_variants(cut)
    if parsed.height == 0:
        return
    cols = [(m, name, color) for m, name, color in ALPHABET_DOSE_METRICS
            if m in parsed.columns]
    if len(cols) < 2:
        return
    parsed = parsed.with_columns(
        pl.col("alphabet").map_elements(alphabet_classes, return_dtype=pl.Int64)
          .alias("n_classes")).filter(pl.col("n_classes") != UNKNOWN_ALPHABET_CLASSES)
    if parsed.height == 0:
        return

    # Each alphabet at its own best k, on each metric separately. Fixing one k across
    # alphabets would compare a 2-letter alphabet at a k it is not run at; taking the best
    # k per metric is the arm anyone sweeping for that metric would land on.
    per = (parsed.group_by("alphabet", "n_classes", "ksize")
                 .agg([pl.col(m).mean() for m, _, _ in cols]))
    points, spans, table = {}, {}, {}
    for metric, name, color in cols:
        best = (per.drop_nulls(metric).sort(metric, descending=True)
                   .group_by("alphabet", maintain_order=True).head(1)
                   .sort("n_classes"))
        rows = best.to_dicts()
        if not rows:
            continue
        top = max(rows, key=lambda r: r[metric])
        bottom = min(rows, key=lambda r: r[metric])
        spans[metric] = (bottom, top, name)
        for r in rows:
            # x is log2 of the class count on a LINEAR axis rather than the class count on
            # a log10 one. The quantity is a halving of the alphabet, so log2 is its own
            # scale, and Plotly's log10 axis over 2 to 20 classes draws mantissa minor
            # ticks that read "...8, 9, 10, 2", where the trailing 2 means 20. On this axis
            # a tick of 1 is 2 classes, 2 is 4, 3 is 8, 4 is 16, and one step is one
            # halving.
            points[f"{r['alphabet']} · {metric}"] = scatter_point(
                math.log2(r["n_classes"]), r[metric],
                name=f"{r['alphabet']} k{r['ksize']} — {name} ({r[metric]:.3f})",
                group=name, color=color,
                marker_size=10,
                marker_symbol="circle" if metric == "fmax" else "diamond",
                annotation=(r["alphabet"] if r is top or r is bottom else None))
            row = table.setdefault(r["alphabet"], {"n_classes": r["n_classes"]})
            row[metric] = r[metric]
            row[f"{metric}__k"] = r["ksize"]
    if not points:
        return
    suppress_auto_labels(points)
    for label, row in table.items():
        f, p = row.get("family_fmax"), row.get("fmax")
        row["gap"] = (f - p) if f is not None and p is not None else None

    def fold(metric: str) -> str:
        lo, hi, name = spans[metric]
        ratio = hi[metric] / lo[metric] if lo[metric] else float("inf")
        return (f"<b>{name}</b> runs {hi[metric]:.3f} at "
                f"<code>{hi['alphabet']}</code> ({hi['n_classes']} classes) down to "
                f"{lo[metric]:.3f} at <code>{lo['alphabet']}</code> "
                f"({lo['n_classes']} classes), a {ratio:.1f}x span.")

    write_section(out, "qfo_alphabet_dose_response", {
        "id": "qfo_alphabet_dose_response",
        "section_name": "Alphabet dose-response: recognition and placement",
        "description": (
            f"<p>What alphabet reduction buys, on both halves of the task at once "
            f"({primary_truth} truth, <code>{split}</code> split, mean over target "
            f"species, each alphabet at its own best k).</p>"
            + bullets(
                "<b>x is log2 of the number of amino-acid classes the alphabet collapses "
                "to</b>, so one step is one halving of the alphabet: a tick of 1 is 2 "
                "classes, 2 is 4, 3 is 8, 4 is 16. The left of the panel is heavy "
                "reduction and the right is none. It is drawn as log2 on a linear axis "
                "rather than as the class count on a log10 one, because a log10 axis over "
                "2 to 20 classes labels its minor ticks by mantissa and the last one reads "
                "\"2\" where it means 20.",
                *[fold(m) for m, _, _ in cols if m in spans],
                "<b>The separation between the two series is the result.</b> Reduction "
                "buys recognition much more than it buys placement, which is what the "
                "boundary section measures from the other direction: naming the family and "
                "putting it in the right interval are different problems, and a coarse "
                "alphabet is only good at one of them.",
                "<b>Each alphabet is taken at its own best k for each metric</b>, so the "
                "two series can sit at different k for the same alphabet. Fixing one k "
                "across alphabets would score a 2-letter alphabet at a k it is never run "
                "at.",
                "<b>Only the two ends of each series are labelled</b>; every point carries "
                "its alphabet, its k and its value on hover.")),
        "plot_type": "scatter",
        "pconfig": {"id": "qfo_alphabet_dose_response_plot",
                    "title": "Recognition and placement against alphabet size",
                    "xlab": "log2(amino-acid classes): 1 = 2 classes, 2 = 4, 3 = 8, "
                            "4 = 16",
                    "ylab": "Fmax (mean over target species)", "height": 520,
                    "x_decimals": 2,
                    "ymin": 0, "showlegend": True,
                    "xsuffix": "", "ysuffix": "", "y_decimals": 3},
        "data": points,
    })

    headers = {"n_classes": dict(title="Classes", format="{:,.0f}", scale="Blues")}
    for metric, name, _ in cols:
        title = "Family Fmax" if metric == "family_fmax" else "Fmax"
        headers[metric] = dict(title=title, min=0, max=1, format="{:,.3f}",
                               scale="Greens", description=name)
        headers[f"{metric}__k"] = dict(title=f"{title}: best k", format="{:,.0f}",
                                       scale=False)
    headers["gap"] = dict(title="Recognition − placement", format="{:,.3f}", scale="Reds",
                          description="How much of the family the alphabet finds but "
                                      "cannot place")
    write_section(out, "qfo_alphabet_dose_response_table", {
        "id": "qfo_alphabet_dose_response_table",
        "section_name": "Alphabet dose-response, as numbers",
        "description": ("<p>The panel above as numbers, one row per alphabet, ordered by "
                        "class count.</p>"
                        + bullets(
                            "<b>Recognition − placement</b> is the gap between the two "
                            "series. It widens as the alphabet coarsens, which is the "
                            "figure's argument in one column.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_alphabet_dose_response_table_table",
                    "title": "Recognition and placement by alphabet",
                    "col1_header": "Alphabet", "sort_rows": False, "no_violin": True},
        "headers": headers,
        "data": dict(sorted(table.items(), key=lambda kv: kv[1]["n_classes"])),
    })


def section_call_volume(out: Path, metrics: pl.DataFrame, primary_truth: str,
                        max_tools: int) -> None:
    """How many calls each arm makes at its shipped default, and what tuning would buy.

    The report's main interpretive risk is here. Precision, and every metric built on it,
    is compared between arms at whatever cutoff each tool ships, and those cutoffs are not
    a common operating point: the arms in this run differ by three orders of magnitude in
    how many calls they emit per proteome before anything is scored. A precision comparison
    at defaults is partly a comparison of vendor defaults.

    The threshold gain column is the other half. Fmax is the best F1 over a threshold sweep
    and f1 is the F1 at the default, so their difference is what tuning the cutoff would
    buy that arm. An arm with a large gain is being under-reported at its default; one with
    a small gain is already near its own best.
    """
    cut, split = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    if cut.height == 0 or "n_calls" not in cut.columns:
        return
    board = best_variants(cut).head(max_tools)
    keep = [(r["tool"], r["variant"], r["label"]) for r in board.to_dicts()]
    has_gain = {"fmax", "f1"}.issubset(set(cut.columns))
    rows = []
    for tool, variant, label in keep:
        sub = cut.filter((pl.col("tool") == tool) & (pl.col("variant") == variant))
        if sub.height == 0:
            continue
        calls = sub["n_calls"].mean()
        if calls is None:
            continue
        gain = None
        if has_gain:
            fmax, f1 = sub["fmax"].mean(), sub["f1"].mean()
            gain = (fmax - f1) if fmax is not None and f1 is not None else None
        rows.append((label, float(calls), gain, tool))
    if not rows:
        return
    rows.sort(key=lambda r: -r[1])
    data = {f"{label} · {gain:+.3f} from tuning" if gain is not None else label:
            {"n_calls": calls}
            for label, calls, gain, _ in rows}
    table = {label: {"n_calls": calls, "threshold_gain": gain,
                     "cls": CLASSES[tool_class(tool)][0]}
             for label, calls, gain, tool in rows}
    lo, hi = rows[-1], rows[0]
    write_section(out, "qfo_call_volume", {
        "id": "qfo_call_volume",
        "section_name": "Call volume at each tool's default threshold",
        "description": (
            f"<p>Calls per target proteome at the cutoff each tool ships with "
            f"({primary_truth} truth, <code>{split}</code> split, mean over target "
            f"proteomes).</p>"
            + bullets(
                f"<b>The axis is log because the arms span three orders of magnitude</b>: "
                f"{hi[1]:,.0f} calls per proteome for <code>{hi[0]}</code> against "
                f"{lo[1]:,.0f} for <code>{lo[0]}</code>.",
                "<b>This is why a precision comparison at defaults is not a fair fight.</b>"
                " An arm that emits a thousand calls and an arm that emits a million are "
                "not being asked the same question, and the difference between their "
                "precisions is partly the difference between two vendors' ideas of a "
                "sensible cutoff. Every threshold-free metric in this report — Fmax, "
                "AUPRC, sensitivity to the first false positive — exists to route around "
                "this, and the threshold-dependent ones should be read with it in view.",
                "<b>Each tick label carries that arm's threshold gain</b>: Fmax minus F1 "
                "at the default, which is what tuning the cutoff would buy it. A large "
                "gain means the default is costing that arm; a small one means it is "
                "already near its own best."
                if has_gain else "")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_call_volume_plot",
                    "title": "Calls per target proteome at the default threshold",
                    "ylab": "calls per proteome (log scale)",
                    "xlog": True, "cpswitch": False, "sort_samples": False,
                    "use_legend": False, "height": 480},
        "categories": {"n_calls": {"name": "calls per proteome",
                                   "color": CLASSES["kmerseek"][1]}},
        "data": data,
    })
    write_section(out, "qfo_call_volume_table", {
        "id": "qfo_call_volume_table",
        "section_name": "Call volume and threshold gain",
        "description": ("<p>The panel above as numbers, with what tuning the cutoff would "
                        "buy each arm.</p>"
                        + bullets(
                            "<b>Calls per proteome</b> is at the tool's shipped default.",
                            "<b>Threshold gain</b> is Fmax minus F1 at that default. It is "
                            "the headroom a fair comparison would give the arm."
                            if has_gain else "")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_call_volume_table_table",
                    "title": "Calls per proteome and threshold gain",
                    "col1_header": "Arm", "sort_rows": False, "no_violin": True},
        "headers": {
            "cls": dict(title="Class", scale=False),
            "n_calls": dict(title="Calls / proteome", format="{:,.0f}", scale="Reds"),
            "threshold_gain": dict(title="Threshold gain (Fmax − F1)", format="{:,.3f}",
                                   scale="Blues",
                                   description="What tuning the cutoff would buy this arm"),
        },
        "data": table,
    })


# ---------------------------------------------------------------------------
# sections: resources
# ---------------------------------------------------------------------------

def section_resources(out: Path, trace: pl.DataFrame, n_queries: int,
                      source: str = "this run") -> None:
    """Resource plots. `source` names the run these numbers describe.

    Naming it matters because the same code renders a nine-proteome Sherlock sweep and a
    two-species smoke test, and the absolute memory and time figures differ by orders of
    magnitude between them. A reader who cannot tell which one they are looking at will
    size a SLURM request off a 200-protein run.
    """
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
    # Work this run actually PERFORMED, as opposed to the cold-equivalent totals beside it.
    # Every other figure in this table counts cached and stored tasks at the cost of the
    # execution that filled them, which is the right number for "what does this pipeline
    # cost" and the wrong one for "what did this run do".
    #
    # Measured on the 2026-08-31 midi run, the two differ by 5.6x in CPU-hours and 11x in
    # task-hours: 6_499 tasks / 1_609 CPU-h / 329 task-h cold, against 550 tasks /
    # 289 CPU-h / 29 task-h executed. 92% of the tasks in that report were cache or store
    # hits. Without this column a reader cannot tell which of the two they are quoting,
    # and both numbers have been quoted for the same run.
    executed = done.filter(pl.col("status") == "COMPLETED")
    summary = {
        "run": {
            "tasks": trace.height,
            "completed": int((trace["status"] == "COMPLETED").sum()),
            "cached": int((trace["status"] == "CACHED").sum()),
            "stored": int((trace["status"] == mt.STORED).sum()),
            "failed": int((trace["status"] == "FAILED").sum()),
            "retried": int((trace["attempt"] > 1).sum()) if "attempt" in trace.columns else 0,
            "cpu_hours": total_cpu_h,
            "cpu_hours_executed": float(executed["cpu_hours"].sum() or 0),
            "wall_hours": float((done["realtime_s"].sum() or 0) / 3600),
            "wall_hours_executed": float((executed["realtime_s"].sum() or 0) / 3600),
            "peak_rss_gb": float(peak / 1024**3) if peak else None,
            "read_gb": float((done["read_b"].sum() or 0) / 1024**3),
            "write_gb": float((done["write_b"].sum() or 0) / 1024**3),
        }
    }
    write_section(out, "qfo_run_summary", {
        "id": "qfo_run_summary",
        "section_name": "Run totals",
        "description": (
            f"<p>Every task in {source}. <b>Two different totals are shown and they "
            f"answer different questions</b> &mdash; quote the one you mean.</p>"
            + bullets(
                "<b>CPU-hours</b> and <b>Task-hours</b> are the COLD-RUN cost: what this "
                "pipeline would take from an empty cache and an empty store. Cached and "
                "stored tasks are counted at the cost of the execution that filled them.",
                "<b>CPU-h executed</b> and <b>Task-h executed</b> are what THIS run "
                "performed: completed tasks only, cache and store hits excluded. This is "
                "the number to quote for how long a resumed run took.",
                "<b>Cached tasks</b> keep the resource figures from the execution that "
                "filled the cache, so a <code>-resume</code> run still reports honest "
                "cold-run totals for work it did not repeat.",
                "<b>Stored</b> counts kmerseek tasks served from <code>storeDir</code>. "
                "Nextflow never scheduled them, so they appear in no trace, and their "
                "times come from the timing record each task wrote beside its own result.",
                "<b>Tasks</b> is the count for the DAG this run actually built. It is not "
                "comparable across runs whose process structure differs: batching the "
                "scoring arms and splitting kmerseek index from search between them "
                "changed this figure by more than caching ever does.",
                "<b>Task-hours</b> is the sum of per-task run times, not elapsed clock "
                "time. The two differ by however much ran in parallel, and neither "
                "includes queue wait.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_run_summary_table", "title": "Run totals",
                    "col1_header": "", "sort_rows": False, "scale": False, "no_violin": True},
        "headers": {
            "tasks": dict(title="Tasks", format="{:,.0f}", scale=False),
            "completed": dict(title="Completed", format="{:,.0f}", scale=False),
            "cached": dict(title="Cached", format="{:,.0f}", scale=False),
            "stored": dict(title="Stored", format="{:,.0f}", scale=False,
                           description="Served from storeDir; timed by the task itself"),
            "failed": dict(title="Failed", format="{:,.0f}", scale=False),
            "retried": dict(title="Retried", format="{:,.0f}", scale=False),
            "cpu_hours": dict(title="CPU-hours", format="{:,.1f}", scale="Reds",
                              description="Cold-run cost: cached and stored tasks "
                                          "included at the cost that filled them"),
            "cpu_hours_executed": dict(
                title="CPU-h executed", format="{:,.1f}", scale="Reds",
                description="Work THIS run performed: completed tasks only"),
            "wall_hours": dict(title="Task-hours", format="{:,.1f}", scale="Blues",
                               description="Cold-run cost, summed per task, no queue wait"),
            "wall_hours_executed": dict(
                title="Task-h executed", format="{:,.1f}", scale="Blues",
                description="Work THIS run performed: completed tasks only"),
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
        "description": (f"<p>Every process in {source}.</p>"
                        + bullets(
                            "<b>CPU-hours</b> is run time times allotted cores, so a task "
                            "that requested 16 cores and used one still bills 16.",
                            "<b>The gap between the two bars</b> is exactly that waste.",
                            "<b><code>kmerseekIndex</code> and "
                            "<code>kmerseekSearch</code></b> are separate bars on purpose: "
                            "the index is built once per target proteome and reused across "
                            "runs, so charging it to the search would misprice both.")),
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
            "description": ("<p>Minutes per task.</p>"
                            + bullets(
                                "<b>The spread within a process</b> is what decides the "
                                "SLURM <code>time</code> request: sizing for the median "
                                "means the tail gets killed and requeued.",
                                "<b>For the search processes</b> this is also the spread "
                                "behind the frontier plot, which takes one median per arm "
                                "out of these boxes and turns it into a rate.")),
            "plot_type": "box",
            "pconfig": {"id": "qfo_res_walltime_plot", "title": "Run time per task",
                        "xlab": "minutes", "height": 500},
            "data": times,
        })

    # --- memory: what was asked for against what was used ---
    mem = done.filter(pl.col("peak_rss_b").is_not_null()
                      & pl.col("requested_mem_b").is_not_null()
                      & (pl.col("requested_mem_b") > 0) & (pl.col("peak_rss_b") > 0))
    if mem.height:
        # y is the FRACTION of the request that was touched, not the absolute peak, and
        # the change is deliberate. Plotting peak GB against requested GB needs a y=x
        # reference, and MultiQC's scatter has no line primitive, so the previous version
        # drew that reference as an 81-point `extra_series`. Those points are appended to
        # the real ones: they were counted in the plot's own "198 points" subtitle, took a
        # legend entry, and visually outnumbered the 117 tasks the plot exists to show. A
        # ratio puts the reference on a horizontal line at 1.0, which `y_lines` draws as a
        # layout shape rather than as data, so the count is honest and the tasks are what
        # the reader sees.
        #
        # What that reveals is over-allocation, and it is real rather than a plotting
        # artefact: the 2026-08-20 Sherlock trace has ecoli_hp_k20 reserving 128 GB and
        # peaking at 1.02 GB. Log y keeps a task at 0.01 and one at 0.9 on the same plot.
        annotate_all = mem.height <= ANNOTATE_EVERY_POINT_BELOW
        points = {}
        for i, r in enumerate(mem.to_dicts()):
            tag = r["tag"] if r["tag"] and r["tag"] != "-" else None
            label = f"{r['process']} [{tag}]" if tag else r["process"]
            peak_gb = r["peak_rss_b"] / 1024**3
            req_gb = r["requested_mem_b"] / 1024**3
            points[f"{label} #{i}"] = scatter_point(
                # mem_used_frac rather than a second division here, so this plot and the
                # efficiency bars can never disagree about what "used" means.
                req_gb, r["mem_used_frac"],
                # Both absolute figures live in the point name, so the hover still answers
                # "how many GB" even though the axis is now a ratio.
                name=f"{label} — {peak_gb:.2f} of {req_gb:.0f} GB",
                group=r["process"], color=tool_color(r["tool"]),
                annotation=label if annotate_all else None,
            )
        # The 1.0 guide is a layout shape, and Plotly's autorange ignores shapes, so on a
        # run where nothing came close to its request the line would sit off the top of
        # the plot and the reader would lose the reference the axis is measured against.
        # Derived from the data rather than fixed at 1.05, because MultiQC's scatter DROPS
        # points above ymax instead of clipping the axis -- a hard-coded ceiling would
        # silently delete exactly the over-request tasks this plot exists to catch.
        y_top = max(1.05, float(mem["mem_used_frac"].max() or 0) * 1.1)
        write_section(out, "qfo_res_memory", {
            "id": "qfo_res_memory",
            "section_name": "Memory: requested against used",
            "description": (
                f"<p>One point per task in {source}: the fraction of its memory request "
                f"the task actually touched, against the size of that request.</p>"
                + bullets(
                    "<b>The dashed line at 1.0</b> is peak = requested.",
                    "<b>Points far below it</b> are queue time paid for nothing. On "
                    "<code>hns</code> a 16-core, 64 GB ask waits 14.6 min median against "
                    "3.0 min for a 2-4 core one.",
                    "<b>Points at or above it</b> are the shape that gets OOM-killed on "
                    "the next combo.",
                    "<b>Memory is sized from keyspace bits</b> (ksize x log2 of the "
                    "alphabet's class count) in <code>kmerseekSearchMemory</code>: the "
                    "measured envelope halves every 6.8 bits. This plot is how that rule "
                    "gets checked against reality.",
                    "<b>Point labels</b> carry the process and its alphabet, k and "
                    "species; hover or read them.",
                    "<b>A smoke run</b> over a few hundred sequences will sit near the "
                    "bottom on every arm. Only a full-proteome run says anything about the "
                    "sizing rule.",
                    "<b>kmerseek tasks appear here only when they genuinely executed.</b> "
                    "A <code>storeDir</code> hit has no peak RSS to report, because peak "
                    "RSS is kernel accounting for a task Nextflow supervised and there was "
                    "no task.")),
            "plot_type": "scatter",
            "pconfig": {"id": "qfo_res_memory_plot",
                        "title": "Memory used as a fraction of the request",
                        "xlab": "requested (GB)",
                        "ylab": "peak RSS / requested",
                        "xlog": True, "ylog": True, "height": 520, "marker_size": 7,
                        "showlegend": True, "ymax": y_top,
                        "y_lines": [{"value": 1.0, "color": "#999999", "dash": "dash",
                                     "width": 2, "label": "peak = requested"}]},
            "data": points,
        })

    # --- kmerseek memory against ksize: the sizing rule, measured ---
    # peak_rss_b must be present, not merely coerced to zero. Stored tasks carry a wall
    # time but no memory figure -- there was no supervised process to read it from -- and
    # `or 0` would draw every cached combo along the x axis as if it had used no memory at
    # all, which a reader would take as evidence the sizing rule is over-provisioning.
    ks = done.filter(pl.col("process").is_in(sorted(mt.KMERSEEK_PROCESSES))
                     & pl.col("peak_rss_b").is_not_null())
    if ks.height:
        ks = ks.with_columns(
            pl.col("tag").str.extract(r"_k(\d+)_lc", 1).cast(pl.Int64).alias("ksize"),
            pl.col("tag").str.extract(r"^[^_]+_(.+)_k\d+_lc", 1).alias("alphabet"),
        ).filter(pl.col("ksize").is_not_null())
        if ks.height:
            alphas = sorted(set(ks["alphabet"].drop_nulls().to_list()))
            color_of = dict(zip(alphas, cycle(SERIES_COLORS)))
            annotate_all = ks.height <= ANNOTATE_EVERY_POINT_BELOW
            points = {}
            for i, r in enumerate(ks.to_dicts()):
                alpha = r["alphabet"] or "unknown"
                points[f"{r['tag']} #{i}"] = scatter_point(
                    r["ksize"], r["peak_rss_b"] / 1024**3,
                    # The tag is `<species>_<alphabet>_k<k>_lc<bool>`, which is the whole
                    # identity of the task; the alphabet leads the legend because it is
                    # what the colour encodes.
                    name=r["tag"], group=alpha, color=color_of.get(alpha, "#888888"),
                    annotation=r["tag"] if annotate_all else None,
                )
            write_section(out, "qfo_res_kmerseek_memory", {
                "id": "qfo_res_kmerseek_memory",
                "section_name": "kmerseek memory against k",
                "description": (
                    f"<p>Peak RSS of each kmerseek task in {source} against its k-mer "
                    f"size.</p>"
                    + bullets(
                        "<b>Colour</b> is the alphabet.",
                        "<b>Each point</b> is named for the species, alphabet, k and "
                        "low-complexity arm it measures.",
                        "<b>The expensive corner is a 2-letter alphabet at low k.</b> The "
                        "inverted index scales with the most degenerate k-mer's occurrence "
                        "count.",
                        "<b>Anything flat here</b> means the per-combo memory rule is "
                        "over-provisioning.",
                        "<b>Only tasks that actually executed appear.</b> A "
                        "<code>storeDir</code> hit has a timing record but no peak RSS, "
                        "since nothing supervised it.")),
                "plot_type": "scatter",
                "pconfig": {"id": "qfo_res_kmerseek_memory_plot",
                            "title": "kmerseek peak RSS by k and alphabet",
                            "xlab": "k", "ylab": "peak RSS (GB)", "height": 500,
                            "marker_size": 7, "showlegend": True},
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
                "<p>Mean fraction of the requested CPU and memory a process actually "
                "used.</p>"
                + bullets(
                    "<b>Low CPU efficiency</b> means cores are reserved and idle.",
                    "<b>High memory efficiency</b> means the next slightly larger input "
                    "gets OOM-killed.",
                    "<b>On macOS without containers Nextflow records neither</b>, so a "
                    "local run shows this section empty. The cluster run is the one with "
                    "real numbers.")),
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
                "<p>Bytes read and written per process.</p>"
                + bullets(
                    "<b>The search arms dominate.</b> HP alphabets at low k produce "
                    "enormous match volume by design, since the p-value filter is left "
                    "lenient so Bonferroni correction can happen downstream.",
                    "This plot is where that cost shows up.")),
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
            "<p>Tasks per process by final status.</p>"
            + bullets(
                "<b>A FAILED kmerseek task matters more than it looks.</b> The process "
                "uses <code>errorStrategy 'finish'</code> rather than "
                "<code>'ignore'</code>, because a combo that dies and gets skipped leaves "
                "an empty result that reads downstream as \"this alphabet found nothing\", "
                "which is indistinguishable from a real negative.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "qfo_res_status_plot", "title": "Task status by process",
                    "ylab": "tasks", "height": 450},
        "categories": {c: {"name": c.title(), "color": palette.get(c, "#888888")}
                       for c in cats},
        "data": {r["process"]: {c: r[c] for c in cats} for r in status.to_dicts()},
    })

    # There was a "Throughput per search" scatter here, one point per search task, plotting
    # queries per second against that task's CPU-hours. Removed 2026-08-27, when the
    # frontier plot gained the named duration lines and became the one place the report
    # argues speed. Two reasons, and the weaker one first: its axes were the same
    # measurement twice, since queries/s is 1/wall-time scaled by a constant and CPU-hours
    # is wall-time times cores, so what the plot showed was mostly how many cores each arm
    # was given. The real reason is that the same trade-off was then drawn twice, on the
    # same y quantity, several sections apart, and a reader who has to reconcile two
    # pictures of one trade-off takes away less than one who reads a single figure. What
    # it uniquely carried is still here: per-task spread is the "Task run times" box plot
    # above, billed cost is the CPU-hours bar in "CPU time by process", and the accuracy
    # side of the trade is on the frontier.


# ---------------------------------------------------------------------------
# general statistics
# ---------------------------------------------------------------------------

def section_general_stats(out: Path, metrics: pl.DataFrame, trace: pl.DataFrame,
                          n_queries: int, primary_truth: str) -> None:
    cut, _ = pick_split(ungrouped(metrics.filter(pl.col("truth_set") == primary_truth)))
    board = best_variants(cut)
    if board.height == 0:
        return
    board = attach_throughput(board, trace, n_queries)
    data = {}
    for row in board.to_dicts():
        data[row["label"]] = {
            "fmax": row.get("fmax"),
            "auprc": row.get("auprc"),
            "recall_reachable": row.get("recall_reachable"),
            "precision": row.get("precision"),
            "queries_per_s": row.get("queries_per_s"),
            "cpu_hours": row.get("cpu_hours"),
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

# spectrum.<species>.<alphabet>.k<ksize>.lc<true|false>.csv.gz -- the filename carries every
# coordinate, because the CSV body only carries moltype and ksize.
SPECTRUM_NAME = re.compile(
    r"^spectrum\.(?P<species>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)"
    r"\.lc(?P<lowcomp>true|false)\.csv\.gz$")


def load_spectra(path: Path | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the k-mer frequency spectra kmerseek writes beside each index.

    kmerseekIndex has always emitted one of these per (species, alphabet, ksize,
    low-complexity) combo via `--kmer-stats-out`, and the pipeline publishes them to
    `${outdir}/spectra` under a comment in main.nf that says they are "published for
    plotting". Nothing ever plotted them.

    Each file is a header comment carrying the totals, then a CSV of
    `moltype,ksize,occurrences,n_kmers` -- how many distinct k-mers occurred exactly N
    times. Returns (spectrum rows, one summary row per file).

    Unreadable files are skipped rather than fatal, like the BPE sidecar: this is a side
    measurement and a malformed one must not cost a finished sweep its report.
    """
    empty = pl.DataFrame(), pl.DataFrame()
    if path is None or not path.exists():
        return empty
    files = sorted(path.glob("spectrum.*.csv.gz")) if path.is_dir() else [path]
    rows, summaries, skipped = [], [], []
    for f in files:
        m = SPECTRUM_NAME.match(f.name)
        if m is None:
            skipped.append(f.name)
            continue
        meta = m.groupdict()
        try:
            with gzip.open(f, "rt") as fh:
                head = fh.readline()
                body = fh.read()
            if not head.startswith("#"):
                body, head = head + body, ""
            sub = pl.read_csv(io.StringIO(body))
        except (OSError, EOFError, pl.exceptions.PolarsError) as exc:
            print(f"could not read {f.name}: {exc}; skipping it")
            continue
        if sub.height == 0 or "occurrences" not in sub.columns:
            continue
        stats = dict(kv.split("=", 1) for kv in head.lstrip("# ").split()
                     if "=" in kv) if head else {}
        rows.append(sub.with_columns(
            pl.lit(meta["species"]).alias("species"),
            pl.lit(meta["alphabet"]).alias("alphabet"),
            pl.lit(int(meta["ksize"])).alias("ksize"),
            pl.lit(meta["lowcomp"] == "true").alias("lowcomp")))
        summaries.append({
            "species": meta["species"], "alphabet": meta["alphabet"],
            "ksize": int(meta["ksize"]), "lowcomp": meta["lowcomp"] == "true",
            # The totals come from the header rather than being re-summed, so a truncated
            # body shows up as a disagreement instead of being silently reconstructed.
            "total_kmers": int(stats.get("total_kmers", 0) or 0),
            "unique_kmers": int(stats.get("unique_kmers", 0) or 0),
            "mean_seqs_per_kmer": float(stats.get("mean_seqs_per_kmer", "nan")),
            "median_seqs_per_kmer": float(stats.get("median_seqs_per_kmer", "nan")),
            "max_occurrences": int(sub["occurrences"].max() or 0),
        })
    if skipped:
        print("spectra: filenames that do not carry alphabet/ksize/lc, so they are not "
              "plottable: " + ", ".join(skipped[:5]))
    if not rows:
        return empty
    return pl.concat(rows, how="diagonal"), pl.DataFrame(summaries)


def section_kmer_spectra(out: Path, spectra: pl.DataFrame,
                         summary: pl.DataFrame) -> None:
    """The k-mer frequency spectrum per alphabet and ksize.

    This is the shape behind three separate results the report already reports without
    ever showing the cause: the memory scatter (the inverted index scales with the most
    degenerate k-mer's occurrence count), the low-complexity filter (which removes exactly
    the right-hand tail), and the bits-per-k-mer floor (which is a claim about how quickly
    a coarse alphabet saturates its keyspace). One curve per combo, so the tail is visible
    rather than summarised.
    """
    if spectra.height == 0:
        return
    # The unfiltered arm only. The filter's whole effect on Fmax at each alphabet's best k
    # is under 0.001, and drawing both arms doubled this section to show a tail difference
    # that changes no result. The supplementary low-complexity panel is the measurement.
    cut = spectra.filter(~pl.col("lowcomp"))
    if cut.height == 0:
        cut = spectra
    # One dataset per species: the spectrum is a property of the indexed proteome, so
    # overlaying two proteomes on one panel would compare two different key sets.
    species = sorted(cut["species"].unique().to_list())
    panels, labels = [], []
    for sp in species:
        one = cut.filter(pl.col("species") == sp)
        series = {}
        for row in (one.select("alphabet", "ksize").unique()
                      .sort("alphabet", "ksize").to_dicts()):
            sel = one.filter((pl.col("alphabet") == row["alphabet"])
                             & (pl.col("ksize") == row["ksize"]))
            pts = {str(r["occurrences"]): r["n_kmers"]
                   for r in sel.sort("occurrences").to_dicts()}
            if pts:
                series[f"{row['alphabet']} k{row['ksize']}"] = pts
        if series:
            panels.append(series)
            labels.append({"name": sp, "ylab": "distinct k-mers"})
    if panels:
        write_section(out, "qfo_kmer_spectra", {
            "id": "qfo_kmer_spectra",
            "section_name": "K-mer spectra",
            "description": (
                "<p>The k-mer frequency spectrum of each indexed proteome, one curve per "
                "alphabet and k-mer size, with the low-complexity filter off.</p>"
                + bullets(
                    "<b>x</b> is how many sequences a k-mer occurs in; <b>y</b> is how "
                    "many distinct k-mers occur that many times. Both are log scale, "
                    "because the distribution is heavy-tailed by construction.",
                    "<b>The left-hand end is specificity.</b> A k-mer seen once is a "
                    "k-mer that can place a domain. An alphabet whose mass sits at "
                    "occurrences=1 is discriminative at that k.",
                    "<b>The right-hand tail is cost.</b> The inverted index scales with "
                    "the most degenerate k-mer's occurrence count, which is why a "
                    "2-letter alphabet at low k is the expensive corner in the memory "
                    "scatter. This is that scatter's cause rather than its symptom.",
                    "<b>The filtered arm is not drawn.</b> It was a second copy of this "
                    "section differing in the tail, and the filter moves Fmax by under "
                    "0.001 at every alphabet's best k. The supplementary low-complexity "
                    "panel carries that measurement.",
                    "<b>One dataset per target proteome</b>, switchable. The spectrum is a "
                    "property of the indexed proteome, so two proteomes are two different "
                    "key sets and are never drawn on one panel.",
                    "<b>Produced by the run, not recomputed here.</b> kmerseekIndex writes "
                    "these with <code>--kmer-stats-out</code> and the pipeline publishes "
                    "them to <code>spectra/</code>.")),
            "plot_type": "linegraph",
            "pconfig": {"id": "qfo_kmer_spectra_plot",
                        "title": "K-mer frequency spectrum",
                        "xlab": "sequences a k-mer occurs in", "ylab": "distinct k-mers",
                        "xlog": True, "ylog": True, "height": 560,
                        "xsuffix": "", "ysuffix": "", "showlegend": True,
                        "data_labels": labels},
            "data": panels,
        })

    if summary.height == 0:
        return
    summary = summary.filter(~pl.col("lowcomp")) if (~summary["lowcomp"]).any() else summary
    table = {}
    for r in summary.sort("alphabet", "ksize", "species").to_dicts():
        key = f"{r['alphabet']} k{r['ksize']} · {r['species']}"
        dup = (r["total_kmers"] / r["unique_kmers"]) if r["unique_kmers"] else None
        table[key] = {
            "total_kmers": r["total_kmers"], "unique_kmers": r["unique_kmers"],
            "duplication": dup, "mean_seqs_per_kmer": r["mean_seqs_per_kmer"],
            "max_occurrences": r["max_occurrences"],
        }
    write_section(out, "qfo_kmer_spectra_table", {
        "id": "qfo_kmer_spectra_table",
        "section_name": "K-mer spectra summary",
        "description": (
            "<p>The totals behind the spectra above, one row per combo and proteome.</p>"
            + bullets(
                "<b>total_kmers</b> is every k-mer occurrence; <b>unique_kmers</b> is the "
                "distinct ones.",
                "<b>duplication</b> is total / unique: how many times an average k-mer is "
                "seen. It rises as the alphabet coarsens and as k falls, and it is the "
                "one-number version of the tail.",
                "<b>max occurrences</b> is the single most degenerate k-mer in that "
                "proteome, which is what sizes the inverted index. Read it beside the "
                "kmerseek memory-against-k scatter.",
                "<b>Totals come from each file's header</b> rather than being re-summed "
                "from its body, so a truncated file shows up as a disagreement instead of "
                "being quietly reconstructed.")),
        "plot_type": "table",
        "pconfig": {"id": "qfo_kmer_spectra_table_table",
                    "title": "K-mer spectra summary", "col1_header": "Combo",
                    "sort_rows": False, "no_violin": True},
        "headers": {
            "total_kmers": dict(title="Total k-mers", format="{:,.0f}", scale="Blues"),
            "unique_kmers": dict(title="Unique k-mers", format="{:,.0f}", scale="Greens"),
            "duplication": dict(title="Duplication", format="{:,.3f}", scale="Reds",
                                description="total / unique"),
            "mean_seqs_per_kmer": dict(title="Mean seqs/k-mer", format="{:,.3f}",
                                       scale="Purples"),
            "max_occurrences": dict(title="Max occurrences", format="{:,.0f}",
                                    scale="Oranges",
                                    description="The most degenerate k-mer; this is what "
                                                "sizes the inverted index"),
        },
        "data": table,
    })


def load_bpe_boundary(path: Path | None) -> dict | None:
    """Read the BPE boundary diagnostic's JSON, or None.

    Unreadable is treated the same as absent, on purpose. This panel is a side measurement
    that no search produced; taking the whole report down over a malformed sidecar would
    cost a finished sweep its figures.
    """
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"could not read {path}: {exc}; skipping the BPE panel")
        return None


def load_curves(path: Path | None) -> pl.DataFrame:
    """Curves are optional. The pipeline passes an assets/NO_CURVES sentinel when a run has
    none, so a file that exists but is not a parquet is a supported input, not an error --
    the PR and ROC sections drop out and everything else still builds."""
    if path is None or not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        print(f"no usable curves at {path} ({exc}); skipping the PR and ROC sections")
        return pl.DataFrame()


def main():
    # Sections that do not take the setting as an argument read the module default, so it
    # is rebound once from the CLI rather than threaded through a dozen signatures. The
    # declaration has to precede every mention of the name in this function, argparse's
    # default included.
    global TOP_KMERSEEK, CANONICAL, PRIMARY_TRUTH

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--curves", type=Path)
    p.add_argument("--trace", type=Path,
                   help="Nextflow trace file. Missing or unparsable is not an error; the "
                        "resource sections say so instead.")
    p.add_argument("--kmerseek-timings", type=Path,
                   help="Directory of *.timings.jsonl written by the kmerseek processes. "
                        "That arm is on storeDir, so a store hit leaves no trace row and "
                        "these records are the only measurement of it. Missing is not an "
                        "error.")
    p.add_argument("--n-queries", type=int, required=True,
                   help="Human query proteins searched, for throughput")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--primary-truth", default=None,
                   help="Truth set the frontier and curve sections use. Defaults to "
                        "swissprot when present, since Pfam is circular with the profile "
                        "baselines, else the first set found.")
    p.add_argument("--hgnc-min-instances", type=int, default=20,
                   help="Skip HGNC families with fewer instances in the answer key than "
                        "this; below it Fmax ranks on noise")
    p.add_argument("--hgnc-top-n", type=int, default=15,
                   help="Families shown at each end of the kmerseek-minus-baseline gap")
    p.add_argument("--max-tools", type=int, default=20,
                   help="Rows per grouped plot, ranked by Fmax")
    p.add_argument("--top-kmerseek", type=int, default=TOP_KMERSEEK,
                   help="Alphabet x ksize x low-complexity combos to carry into the "
                        "comparison plots. Every baseline contributes one variant; only "
                        "the sweep contributes several.")
    p.add_argument("--canonical-variant", default=None,
                   help="One arm to follow across the whole report, as 'tool:variant' or "
                        "a bare variant, which means kmerseek. It is forced into every "
                        "board and its row key is marked, so a reader can follow the same "
                        "configuration between sections that each rank on their own "
                        "metric. Off by default: with no pin, nothing in the report "
                        "changes, and no arm is hard-coded as the canonical one.")
    p.add_argument("--dedup-mode", choices=["off", "on"], default="off",
                   help="which dedup-transfer scoring the report's sections use. 'off' is "
                        "the tool's output as reported, redundant copies of a call charged "
                        "as errors; 'on' collapses calls of one family over one query "
                        "region. The comparison section always shows both regardless.")
    p.add_argument("--max-lines", type=int, default=12,
                   help="Curves per PR/ROC plot")
    p.add_argument("--spectra", type=Path,
                   help="directory of spectrum.<species>.<alphabet>.k<k>.lc<t|f>.csv.gz "
                        "written by kmerseekIndex --kmer-stats-out and published to "
                        "${outdir}/spectra. Absent means the panel is skipped.")
    p.add_argument("--bpe-boundary", type=Path,
                   help="JSON from bin/hp_bpe_boundary_diagnostic.py. That diagnostic is "
                        "run by hand against a downloaded tokenizer, not by a search arm, "
                        "so a missing file is normal and drops the panel rather than "
                        "failing the report.")
    args = p.parse_args()

    TOP_KMERSEEK = args.top_kmerseek
    CANONICAL = parse_canonical(args.canonical_variant)

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics = pl.read_parquet(args.metrics)
    # Every arm is scored under both dedup-transfer settings. The whole frame is kept for
    # the one section that compares them; every other section sees a single mode, because
    # a frame with both has two rows per arm and any mean over it is halfway between two
    # different measurements. Default 'off' so every existing number in this report keeps
    # meaning exactly what it did before the second mode existed.
    metrics_all = metrics
    if "dedup_transfers" in metrics.columns:
        want = args.dedup_mode == "on"
        picked = metrics.filter(pl.col("dedup_transfers") == want)
        if picked.height == 0 and metrics.height:
            raise SystemExit(
                f"--dedup-mode {args.dedup_mode} selected no rows; the metrics carry "
                f"{sorted(set(metrics['dedup_transfers'].to_list()))}")
        metrics = picked
    curves = load_curves(args.curves)
    trace = mt.load_trace(args.trace) if args.trace else mt.load_trace(None)
    trace = mt.merge_timings(trace, mt.load_timing_sidecars(args.kmerseek_timings))
    # After the merge, not before: a run where kmerseek genuinely executed has both a
    # trace row and a fresh record for the same task, and only the trace row is kept.
    n_stored = int((trace["status"] == mt.STORED).sum()) if trace.height else 0

    sets = metrics["truth_set"].unique().to_list() if "truth_set" in metrics.columns else []
    primary = args.primary_truth or ("swissprot" if "swissprot" in sets
                                     else (sets[0] if sets else "pfam"))
    # Bound once here so the sections that build one panel per truth set can tell which
    # copy is the main report's without every one of them taking the argument.
    PRIMARY_TRUTH = primary

    section_frontier(args.outdir, metrics, trace, args.n_queries, primary,
                     args.top_kmerseek)
    section_search_cost(args.outdir, metrics, trace, args.n_queries, primary,
                        args.max_tools)
    section_leaderboards(args.outdir, metrics, args.top_kmerseek)
    section_cafa(args.outdir, metrics, primary, args.max_tools)
    section_threshold_metrics(args.outdir, metrics, primary, args.max_tools)
    section_truth_provenance(args.outdir, metrics)
    section_curves(args.outdir, curves, metrics, primary, args.max_lines)
    section_identity(args.outdir, metrics, primary, args.max_tools)
    section_covariates(args.outdir, metrics, primary, args.max_tools)
    section_disorder_scatter(args.outdir, metrics, primary)
    section_plddt_regime(args.outdir, metrics, primary, args.max_tools)
    section_curated_sets(args.outdir, metrics, args.max_tools)
    section_search_space(args.outdir, metrics, primary)
    section_hgnc(args.outdir, metrics, primary,
                 args.hgnc_min_instances, args.hgnc_top_n)
    section_divergence(args.outdir, metrics, primary, args.max_tools)
    # The five-line version, drawn before the full sweep: it is the comparison a reader
    # arrives with, and the nineteen-line panel is the follow-up.
    section_divergence_annotated(args.outdir, metrics, trace, args.n_queries, primary)
    # The region-level twins of the pLDDT and disorder sections, drawn on the domain's
    # own value rather than its protein's.
    section_region_axes(args.outdir, metrics, primary, args.max_tools)
    # Drawn beside the divergence panel at 148/149, just ahead of it: the identity half
    # is only meaningful read against the Mya half, so the two must stay adjacent.
    section_identity_vs_divergence(args.outdir, metrics, primary, args.max_tools)
    section_canonical(args.outdir, metrics)
    section_truthsets(args.outdir, metrics, args.max_tools)
    section_tool_by_species(args.outdir, metrics)
    section_species_winners(args.outdir, metrics)
    section_encoding_vs_divergence(args.outdir, metrics)
    section_alphabet_retention(args.outdir, metrics)
    section_alphabet_matrix(args.outdir, metrics, primary)
    section_ceiling_length(args.outdir, metrics, primary)
    section_ceiling_length_by_k(args.outdir, metrics, primary)
    section_ceiling_feature_type(args.outdir, metrics)
    section_feature_type_crossover(args.outdir, metrics)
    section_ceiling_recognition(args.outdir, metrics, primary)
    section_ceiling_cardinality(args.outdir, metrics)
    section_ceiling_bpe(args.outdir, load_bpe_boundary(args.bpe_boundary))
    section_kmer_spectra(args.outdir, *load_spectra(args.spectra))
    section_boundary(args.outdir, metrics, primary, args.max_tools)
    section_boundary_dots(args.outdir, metrics, primary, args.max_tools)
    section_grayzone(args.outdir, metrics, primary, args.max_tools)
    section_reachability(args.outdir, metrics, primary)
    section_alphabet_dose_response(args.outdir, metrics, primary)
    section_call_volume(args.outdir, metrics, primary, args.max_tools)
    section_dedup_transfers(args.outdir, metrics_all, primary, args.max_tools)
    # The resource sections say which run they describe. A `-entry report` rebuild points
    # at some earlier run's trace, so "this run" is not always true and a reader sizing a
    # SLURM request needs to know whether they are looking at the mini smoke set or the
    # nine-proteome sweep.
    source = f"<code>{args.trace.name}</code>" if args.trace else "this run"
    if n_stored:
        source += f" plus {n_stored:,} stored kmerseek timing records"
    section_resources(args.outdir, trace, args.n_queries, source)
    section_general_stats(args.outdir, metrics, trace, args.n_queries, primary)
    section_citations(args.outdir)

    written = sorted(f.name for f in args.outdir.glob("*_mqc.json"))
    print(f"primary truth set: {primary}")
    print(f"{metrics.height} metric rows, {curves.height} curve points, "
          f"{trace.height} resource records "
          f"({n_stored} of them stored kmerseek timings)")
    print(f"wrote {len(written)} sections to {args.outdir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
