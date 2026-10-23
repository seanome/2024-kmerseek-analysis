"""Shared code for notebook 357 (clade-grouped SPLASH on reduced-alphabet anchors).

Both the notebook and scripts/make_nb357.py import this, so the numbers in the notebook's
summary paragraph and the numbers its cells print come from the same functions.

The tables are what `make pull-results` in nextflow-runs/clade-splash copies to
tables/357/. Set NB357_DIR to read another copy (the pilot is tables/357_pilot/).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_BIN = ROOT / "nextflow-runs" / "clade-splash" / "bin"
DATA = Path(os.environ.get("NB357_DIR", ROOT / "tables" / "357"))

PRIMARY = ("dayhoff:13", 0)
SETTINGS = {"dayhoff:13": (0, 5, 10), "hp:29": (0,)}
SETTING_LABEL = {
    "dayhoff:13": "Dayhoff (6 letters), k = 13",
    "hp:29": "HP (2 letters), k = 29",
}
LABELLING_LABEL = {
    "real": "real class labels",
    "shuffle_random": "labels shuffled across all species",
    "shuffle_tree": "labels shuffled within each phylum",
}
ALPHA = 0.05
# The pre-registered thresholds (notebooks/357_preregistration.md).
RULE = {
    "n_significant": ("at least", 100),
    "n_tree_shuffle": ("fewer than", 5),
    "n_outside_pfam": ("at least", 30),
    "n_outside_not_composition": ("at least", 20),
}
LOW_COMPLEXITY_BITS = 0.75  # notebook 086's HP threshold, reported only
CHANCE_TARGET = 0.2  # the expected chance count the k values were picked for

sys.path.insert(0, str(PIPELINE_BIN))
import clade_splash as cs  # noqa: E402

PFAM_SCHEMA = {
    "setting": pl.String,
    "gap": pl.Int64,
    "anchor": pl.UInt64,
    "n_occurrences": pl.UInt32,
    "n_occ_in_pfam": pl.UInt32,
    "main_pfam": pl.String,
    "outside_pfam": pl.Boolean,
}
ALIGN_SCHEMA = {
    "setting": pl.String,
    "gap": pl.Int64,
    "anchor": pl.UInt64,
    "pfam_acc": pl.String,
    "match_column": pl.Int64,
    "n_species": pl.Int64,
    "p_column": pl.Float64,
    "q_column": pl.Float64,
}


def _read_or_empty(path: Path, schema: dict) -> pl.DataFrame:
    """A table the run has not written yet comes back empty, and the notebook says so."""
    if path.exists():
        return pl.read_parquet(path)
    print(
        f"MISSING: {path.name} (the run has not written it); its sections show nothing"
    )
    return pl.DataFrame(schema=schema)


def load(data: Path = DATA) -> dict:
    """Every table the notebook reads, by name."""
    d = {
        "species": pl.read_csv(data / "species" / "species.tsv", separator="\t"),
        "selection": json.loads(
            (data / "species" / "selection_summary.json").read_text()
        ),
        "inputs": json.loads((data / "species" / "inputs" / "inputs.json").read_text()),
        "totals": json.loads((data / "residue_totals.json").read_text()),
        "fetch": pl.read_csv(data / "fetch_stats.tsv", separator="\t"),
        "counts": pl.read_csv(data / "counts.tsv", separator="\t"),
        "sig": pl.read_parquet(data / "significant.parquet"),
        "pfam": _read_or_empty(data / "pfam_by_anchor.parquet", PFAM_SCHEMA),
        "top": _read_or_empty(data / "top_occurrences.parquet", {}),
        "align": _read_or_empty(data / "alignment_control.parquet", ALIGN_SCHEMA),
    }
    hist = []
    for setting in SETTINGS:
        folder = data / "candidates" / setting.replace(":", "_")
        files = sorted(folder.glob("*.hist.parquet"))
        if files:
            h = (
                pl.concat([pl.read_parquet(f) for f in files])
                .group_by("n_species")
                .agg(pl.col("len").sum())
            )
            hist.append(h.with_columns(pl.lit(setting).alias("setting")))
    d["hist"] = pl.concat(hist) if hist else pl.DataFrame()
    return d


def oasis_module(data: Path = DATA):
    """refresh-bio/SPLASH's OASIS.py, the copy the run used (results/provenance/)."""
    return cs.load_oasis(str(data / "provenance" / "OASIS.py"))


# ---------------------------------------------------------------------------------------
# k arithmetic
# ---------------------------------------------------------------------------------------


def chance_table(totals: dict, ks=range(8, 45)) -> pl.DataFrame:
    """Expected chance occurrences of one anchor, lambda = L / |Sigma|^k (equal letter
    frequencies), and two versions with the measured letter frequencies p_i: an anchor of
    typical probability, L * 2^(k * sum p_i log2 p_i), and the most likely anchor (made only
    of the commonest letter, which the mask would drop, so its runner-up made of the two
    commonest letters alternating is used instead), L * (p_1 p_2)^(k/2)."""
    L = totals["n_residues"]
    rows = []
    for setting, counts in totals["letter_counts"].items():
        alphabet, k0 = cs.parse_setting(setting)
        p = np.array(counts, dtype=float)
        p = p / p.sum()
        size = len(p)
        h = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        top2 = np.sort(p)[::-1][:2]
        for k in ks:
            rows.append(
                {
                    "setting": setting,
                    "alphabet": alphabet,
                    "k": k,
                    "chosen": k == k0,
                    "uniform": L / size**k,
                    "typical": L * 2.0 ** (-k * h),
                    "commonest_pair": L * float(np.prod(top2)) ** (k / 2),
                }
            )
    return pl.DataFrame(rows)


def smallest_k(table: pl.DataFrame, column: str, target: float = CHANCE_TARGET) -> dict:
    out = {}
    for setting, sub in table.group_by("setting"):
        below = sub.filter(pl.col(column) < target).sort("k")
        out[setting[0]] = int(below["k"][0]) if below.height else None
    return out


def best_p_table(
    oasis, n_clades=(3, 4, 6), per_clade=(5, 6, 8, 10, 15, 20, 30, 40, 60)
) -> pl.DataFrame:
    """OASIS p-value of a perfectly separated table: one class has only target A, the
    others only target B, n species in each class. This is one illustrative pattern,
    not a lower bound over all possible target partitions."""
    rows = []
    for j in n_clades:
        for n in per_clade:
            x = np.zeros((2, j), dtype=int)
            x[0, 0] = n
            x[1, 1:] = n
            rows.append(
                {
                    "n_clades": j,
                    "species_per_clade": n,
                    "n_species": j * n,
                    "best_p": float(oasis.OASIS_pvalue(x)),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------------------


def annotated_significant(d: dict) -> pl.DataFrame:
    """Significant anchors (every setting, gap, labelling) with Pfam position and the
    composition call."""
    sig = d["sig"].join(
        d["pfam"].select(
            "setting",
            "gap",
            "anchor",
            "n_occurrences",
            "n_occ_in_pfam",
            "main_pfam",
            "outside_pfam",
        ),
        on=["setting", "gap", "anchor"],
        how="left",
    )
    return sig.with_columns(
        (pl.col("q_comp") < ALPHA).alias("explained_by_composition"),
        (pl.col("anchor_entropy_bits") < LOW_COMPLEXITY_BITS).alias(
            "low_complexity_flag"
        ),
    )


def verdict(d: dict, setting: str = PRIMARY[0], gap: int = PRIMARY[1]) -> dict:
    """The four pre-registered numbers and the verdict."""
    sig = annotated_significant(d)
    s = sig.filter((pl.col("setting") == setting) & (pl.col("gap") == gap))
    real = s.filter(pl.col("labelling") == "real")
    tree = s.filter(pl.col("labelling") == "shuffle_tree")
    rand = s.filter(pl.col("labelling") == "shuffle_random")
    outside = real.filter(pl.col("outside_pfam") == True)  # noqa: E712
    n_missing_pfam = real.filter(pl.col("outside_pfam").is_null()).height
    nums = {
        "n_significant": real.height,
        "n_tree_shuffle": tree.height,
        "n_outside_pfam": outside.height,
        "n_outside_not_composition": outside.filter(
            ~pl.col("explained_by_composition")
        ).height,
    }
    passed = {}
    for key, (word, thr) in RULE.items():
        passed[key] = nums[key] >= thr if word == "at least" else nums[key] < thr
    tested = d["counts"].filter((pl.col("setting") == setting) & (pl.col("gap") == gap))
    n_tested = {
        r["labelling"]: r["n_anchors_tested"] for r in tested.iter_rows(named=True)
    }
    incomplete = (
        any(
            label not in n_tested
            for label in ("real", "shuffle_tree", "shuffle_random")
        )
        or n_missing_pfam > 0
        or outside.filter(pl.col("explained_by_composition").is_null()).height > 0
    )
    return {
        **nums,
        "passed": passed,
        "verdict": (
            "INSUFFICIENT_EVIDENCE"
            if incomplete
            else "PASS" if all(passed.values()) else "FAIL"
        ),
        "missed": [k for k, v in passed.items() if not v],
        "n_random_shuffle": rand.height,
        "n_tested": n_tested,
        "n_real_without_pfam_row": n_missing_pfam,
    }


def tree_shuffle_groups(species: pl.DataFrame) -> pl.DataFrame:
    """The groups the within-phylum shuffle permutes inside (phylum, else kingdom, else
    domain, as clade_splash.labellings does), with the number of kept classes in each. A
    species in a group with one class keeps its real label under that shuffle."""
    g = species.with_columns(
        pl.col("phylum")
        .fill_null(pl.col("kingdom"))
        .fill_null(pl.col("domain"))
        .fill_null("none")
        .alias("group")
    )
    return (
        g.group_by("group")
        .agg(
            pl.col("class").n_unique().alias("n_classes"),
            pl.len().alias("n_species"),
            pl.col("class").unique().sort().str.join(", ").alias("classes"),
        )
        .with_columns((pl.col("n_classes") == 1).alias("labels_cannot_move"))
        .sort("n_species", descending=True)
    )


# ---------------------------------------------------------------------------------------
# per-anchor displays
# ---------------------------------------------------------------------------------------


def clade_ends(row: dict) -> tuple[str, str]:
    """The classes with the largest and smallest OASIS column weight: the two classes the
    targets split most."""
    clades = row["clades"].split(";")
    w = [float(x) for x in row["clade_weights"].split(";")]
    return clades[int(np.argmax(w))], clades[int(np.argmin(w))]


def count_table(top: pl.DataFrame, species: pl.DataFrame, row: dict) -> pl.DataFrame:
    """Targets x classes, species counts, for one anchor, from its occurrences (real labels)."""
    occ = top.filter(
        (pl.col("setting") == row["setting"])
        & (pl.col("gap") == row["gap"])
        & (pl.col("anchor") == row["anchor"])
    )
    occ = occ.join(species.select("species_idx", "class"), on="species_idx")
    occ = occ.filter(pl.col("class").is_in(row["clades"].split(";")))
    t = (
        occ.select("target_residues", "class", "species_idx")
        .unique()
        .group_by("target_residues", "class")
        .len("n")
    )
    wide = t.pivot(on="class", index="target_residues", values="n").fill_null(0)
    order = sorted(
        row["clades"].split(";"),
        key=lambda c: -float(
            row["clade_weights"].split(";")[row["clades"].split(";").index(c)]
        ),
    )
    wide = wide.select(["target_residues", *[c for c in order if c in wide.columns]])
    wide = wide.with_columns(
        pl.sum_horizontal(pl.exclude("target_residues")).alias("_total")
    )
    return wide.sort("_total", "target_residues", descending=[True, False]).drop(
        "_total"
    )


def identity(a: str, b: str) -> int:
    return sum(x == y for x, y in zip(a, b))


def residue_examples(
    top: pl.DataFrame, species: pl.DataFrame, row: dict, per_clade: int = 2
) -> pl.DataFrame:
    """Two species from each end class, with accession, coordinates and residues.

    In each class: species whose target is that class's commonest target, lowest species
    index first; one occurrence per species (the lowest protein index, then position).
    Coordinates are 1-based, inclusive, anchor start to target end.
    """
    a_clade, b_clade = clade_ends(row)
    k = cs.parse_setting(row["setting"])[1]
    occ = top.filter(
        (pl.col("setting") == row["setting"])
        & (pl.col("gap") == row["gap"])
        & (pl.col("anchor") == row["anchor"])
    )
    occ = occ.join(species.select("species_idx", "class", "organism"), on="species_idx")
    out = []
    for clade in (a_clade, b_clade):
        sub = occ.filter(pl.col("class") == clade)
        if not sub.height:
            continue
        common = (
            sub.select("species_idx", "target_residues")
            .unique()
            .group_by("target_residues")
            .len()
            .sort("len", "target_residues", descending=[True, False])[
                "target_residues"
            ][0]
        )
        pick = (
            sub.filter(pl.col("target_residues") == common)
            .sort("species_idx", "protein_idx", "pos0")
            .unique("species_idx", keep="first", maintain_order=True)
            .head(per_clade)
        )
        out.append(pick)
    ex = pl.concat(out)
    ref = ex["target_residues"][0]
    return ex.with_columns(
        (pl.col("pos0") + 1).alias("start"),
        (pl.col("pos0") + k + pl.col("gap") + cs.TARGET_LENGTH).alias("end"),
        pl.col("target_residues")
        .map_elements(
            lambda t: f"{identity(t, ref)} of {cs.TARGET_LENGTH} identical",
            return_dtype=pl.String,
        )
        .alias("target_vs_first_row"),
    ).select(
        "class",
        "organism",
        "accession",
        "start",
        "end",
        "anchor_residues",
        "gap_residues",
        "target_residues",
        "target_vs_first_row",
        "in_pfam",
        "pfam_overlapping",
    )


def plain_log_axis(ax, which: str = "y") -> None:
    """Plain tick numbers on a log axis (1, 100, 10,000; 0.01), never 10^n."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    def fmt(v, _):
        if v >= 1:
            return f"{v:,.0f}"
        return f"{v:.10f}".rstrip("0")

    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10, numticks=12))
    axis.set_major_formatter(FuncFormatter(fmt))


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"
