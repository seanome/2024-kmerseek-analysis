"""Loaders and figures for notebook 254 (does seed extension keep notebook 244's cases?).

The searches are run by ``scripts/run_254_extension_searches.py`` and scored by
``scripts/score_254_extension.py``, which calls the landing reduction notebook 244 used
(``scripts/reduce_swissprot_instance_landing.reduce_one``) on the new region tables. This
module only reads those outputs and draws them.

Coordinates. Swiss-Prot features are 1-based and inclusive. kmerseek calls are 0-based and
end-exclusive; the IoU values come from the pipeline's own arithmetic, which compares the
two directly, as notebook 244 did. Residues printed and bars drawn convert kmerseek
coordinates to 1-based (start + 1 .. end).
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

import hero_example_utils as he

RUN_DIR = Path("/Users/olga/data/hero-244-extension")
REPO = Path(__file__).resolve().parents[1]
CASES_CSV = REPO / "tables" / "244_hero_candidates.csv"
SCORED_CSV = REPO / "tables" / "254_extension_cases.csv"
ARMS_TSV = REPO / "tables" / "254_extension_arms.tsv"
N_CASES = 87
KEY = ["query", "feature_type", "feature_start", "feature_end", "species", "target"]
TOOL_LABELS = list(he.COMPARISON_ARMS.values())

#: The tolerance and counts of the decision rule, fixed before any search was run.
KEEP_TOL = 0.05
KEEP_MIN = 70
BUILD_CHECK_MIN = 80


def load_cases() -> pl.DataFrame:
    cases = pl.read_csv(CASES_CSV).filter(pl.col("admitted_under") == "all criteria")
    if cases.height != N_CASES:
        raise ValueError(f"expected {N_CASES} cases, found {cases.height}")
    pattern = r"^kmerseek\.(.+)_k(\d+)_lc(True|False)$"
    arm = pl.col("kmerseek_chosen_arm")
    return cases.with_columns(
        alphabet=arm.str.extract(pattern, 1),
        ksize=arm.str.extract(pattern, 2).cast(pl.Int64),
        mask_on=arm.str.extract(pattern, 3) == "True",
        best_tool_iou=pl.max_horizontal([f"{t}_iou" for t in TOOL_LABELS]),
        best_tool=pl.concat_list([f"{t}_iou" for t in TOOL_LABELS])
        .list.arg_max()
        .replace_strict(dict(enumerate(TOOL_LABELS))),
    )


def load_scored() -> pl.DataFrame:
    """One row per case, with the new-build exact and extended results side by side."""
    s = pl.read_csv(SCORED_CSV, infer_schema_length=None)
    keep = [c for c in s.columns if c not in KEY and c != "condition"]
    wide = None
    for cond in ("exact", "extended"):
        part = s.filter(pl.col("condition") == cond).select(
            *KEY, *[pl.col(c).alias(f"{cond}_{c}") for c in keep]
        )
        wide = (
            part if wide is None else wide.join(part, on=KEY, how="full", coalesce=True)
        )
    return wide


def index_fits() -> pl.DataFrame:
    """Whether each extended index carries a Karlin-Altschul fit, read off the search log."""
    rows = []
    for log in sorted((RUN_DIR / "extended").glob("*.log")):
        m = re.match(r"human_vs_([a-z]+)\.(.+)\.k(\d+)\.lctrue\.log", log.name)
        text = log.read_text()
        fit = re.search(r"Karlin-Altschul: K ([0-9.e-]+), r_database ([0-9.]+)", text)
        rows.append(
            dict(
                species=m[1],
                alphabet=m[2],
                ksize=int(m[3]),
                has_ka_fit=fit is not None,
                K=float(fit[1]) if fit else None,
                r_database=float(fit[2]) if fit else None,
            )
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
#: One colour per call, the same in every figure of the notebook.
CALL_STYLE = {
    "old": dict(
        facecolor="#9E9E9E",
        edgecolor="#555555",
        hatch="",
        label="old build, exact seeds (notebook 244)",
    ),
    "exact": dict(
        facecolor="#2B6C9E",
        edgecolor="#1B4466",
        hatch="",
        label="new build, exact seeds",
    ),
    "extended": dict(
        facecolor="#D98C1F",
        edgecolor="#8A5510",
        hatch="////",
        label="new build, extended seeds",
    ),
}


def draw_calls(ax, name, length, lo, hi, features, instance, calls):
    """A protein as a line with its Swiss-Prot features as boxes, the true feature
    outlined in black, and one bar per call underneath.

    ``calls``: list of (style key, (start, end) 1-based inclusive, or None).
    """
    from matplotlib.patches import Rectangle

    feats = he._feature_rows(features, lo, hi)
    n_tiers = max([t for *_, t in feats], default=-1) + 1
    ax.plot([max(1, lo), min(length, hi)], [0, 0], color="#333333", lw=1.2, zorder=1)
    for s, e, ftype, tier in feats:
        y = 0.25 + tier * 0.75
        is_inst = instance is not None and (s, e) == instance
        ax.add_patch(
            Rectangle(
                (s - 0.5, y - 0.25),
                e - s + 1,
                0.5,
                facecolor=he.TARGET_FEATURE_COLOR if is_inst else he.FEATURE_COLOR,
                edgecolor="#000000" if is_inst else he.FEATURE_EDGE,
                lw=1.8 if is_inst else 0.8,
                zorder=2,
            )
        )
        ax.text(
            max(s, lo) + 0.5,
            y,
            ftype,
            va="center",
            ha="left",
            fontsize=7,
            zorder=3,
            clip_on=True,
        )
    labels = []
    for i, (key, iv) in enumerate(calls):
        y = -0.9 - i * 0.8
        st = CALL_STYLE[key]
        labels.append((y, st["label"]))
        if iv is None:
            ax.plot(
                [(lo + hi) / 2], [y], marker="x", color="#777777", ls="", ms=6, mew=1.5
            )
            ax.text(
                (lo + hi) / 2 + (hi - lo) * 0.015,
                y,
                "no call",
                va="center",
                fontsize=7,
                color="#555555",
            )
            continue
        s, e = iv
        ax.add_patch(
            Rectangle(
                (s - 0.5, y - 0.28),
                e - s + 1,
                0.56,
                lw=1.2,
                zorder=3,
                **{k: v for k, v in st.items() if k != "label"},
            )
        )
        pad = (hi - lo) * 0.01
        if e + pad * 8 < hi:
            ax.text(e + 0.5 + pad, y, f"{s}-{e}", va="center", ha="left", fontsize=7)
        else:
            ax.text(
                s + pad,
                y,
                f"{s}-{e}",
                va="center",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                zorder=4,
            )
    ax.set_yticks(
        [y for y, _ in labels] + [0], [lab for _, lab in labels] + [name], fontsize=8
    )
    ax.set_ylim(-0.9 - len(calls) * 0.8 + 0.2, 0.25 + max(n_tiers, 1) * 0.75 + 0.2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        f"residue position on {name} (aa; protein is {length} aa)", fontsize=8.5
    )
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def extended_regions_over_feature(r: dict) -> pl.DataFrame:
    """Every extended region on the case's query and target protein that overlaps the human
    feature, before any filter, with the Bonferroni-corrected p the landing reduction tests
    (load_regions: tail probability x region_search_space x db_n_targets, capped at 1).
    """
    path = (
        RUN_DIR
        / "extended"
        / (
            f"human_vs_{r['species']}.{r['alphabet']}.k{r['ksize']}.lctrue.regions.parquet"
        )
    )
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("query_name").str.contains(f"|{r['query']}|", literal=True)
            & pl.col("target_name").str.contains(f"|{r['target']}|", literal=True)
        )
        .with_columns(
            overlap=pl.min_horizontal("region_end", pl.lit(r["feature_end"]))
            - pl.max_horizontal("region_start", pl.lit(r["feature_start"])),
            bonferroni_p=pl.min_horizontal(
                pl.col("region_tail_probability")
                * pl.col("region_search_space")
                * pl.col("db_n_targets"),
                pl.lit(1.0),
            ),
        )
        .filter(pl.col("overlap") > 0)
        .select(
            "region_start",
            "region_end",
            "region_length",
            "region_n_shared_kmers",
            "region_expected_shared_kmers",
            "bonferroni_p",
            "region_evalue",
            "overlap",
        )
        .collect()
        .sort("overlap", "region_length", descending=[True, False])
    )
