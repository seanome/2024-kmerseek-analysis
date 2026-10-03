#!/usr/bin/env python3
"""Supplementary Figure 1, and the seed-reach figure that goes with it.

Supplementary Figure 1: alphabets with fewer letters carry fewer bits per letter, so each
needs a longer seed. One row per kmerseek alphabet (19), ordered by letter count:
  a  class shares in Swiss-Prot 2026_03, one segment per class, labelled with its residues
  b  bits per letter three ways: log2(n letters); B from Swiss-Prot composition,
     B = -log2(sum of squared class shares); B measured from kmerseek seed counts in the
     human proteome (notebook 250)
  c  the k-sizes to test, k_min to k_max (notebook 274), with k* = ceil(log2(N) / B)

Seed-reach figure (one column): share of Pfam-A 38.2 seed pairs at 20-30% identity whose
longest run of same-class aligned columns is at least the alphabet's k*, Wilson 95%
interval. Cohen's kappa and the SCOPe 2.08 same-superfamily version are in the values table
and the caption, not drawn.

Inputs:
  /Users/olga/data/uniprot/uniprot_sprot.2026_03.fasta.gz   counted once into
      tables/swissprot_2026_03_residue_counts.csv (pass --recount to redo)
  tables/250_two_k_per_alphabet.csv   seed counts per alphabet, copied unchanged from commit
      b5650a2 (scripts/elm_seed_floor.py, notebook 250)
  /Users/olga/data/pfam/230_pfam_seed_pair_class_agreement.parquet
  /Users/olga/data/scope/230_scope40_pair_class_agreement.parquet
      both written by scripts/pfam_seed_pair_class_agreement.py (notebook 230)
  notebooks/hp_conservation_utils.ALPHABET_CLUSTERS, checked here against kmerseek v0.4.0
      src/rust/alphabets.rs when a kmerseek checkout is at KMERSEEK

Outputs:
  figures/suppfig1.{pdf,png,svg}, figures/suppfig1_caption.md
  figures/seed_reach_at_kstar_pfam_seed_20-30pct.{pdf,png,svg} and its _caption.md
  tables/suppfig1_values.csv

Run: python scripts/plot_suppfig1_alphabets.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402
import pubfig as pf  # noqa: E402

SWISSPROT_FASTA = Path("/Users/olga/data/uniprot/uniprot_sprot.2026_03.fasta.gz")
SWISSPROT_COUNTS = REPO / "tables/swissprot_2026_03_residue_counts.csv"
SEED_COUNTS = REPO / "tables/250_two_k_per_alphabet.csv"
SEED_COUNTS_SHA256 = "42f90fe66d1731256a41008b02dbb09203fad4643fc468e82707bba4b3989d84"
# k-sizes to test per alphabet, notebook 274 (PR 101), copied unchanged into PR 100.
KSIZES = REPO / "tables/274_ksizes_to_test_per_alphabet_human_swissprot.csv"
KSIZES_SHA256 = "786fb61606ee9994d4a999a8126797edefe4d7092340695898a4fe920d00b6af"
PFAM_PAIRS = Path("/Users/olga/data/pfam/230_pfam_seed_pair_class_agreement.parquet")
SCOPE_PAIRS = Path("/Users/olga/data/scope/230_scope40_pair_class_agreement.parquet")
KMERSEEK = Path("/Users/olga/code/kmerseek")
KMERSEEK_TAG = "v0.4.0"
FIG = REPO / "figures"
VALUES = REPO / "tables/suppfig1_values.csv"
CAPTION = FIG / "suppfig1_caption.md"
REACH_STEM = "seed_reach_at_kstar_pfam_seed_20-30pct"
REACH_CAPTION = FIG / f"{REACH_STEM}_caption.md"

STANDARD = "ACDEFGHIKLMNPQRSTVWY"
BIN = "20-30%"
MIN_COLS = 50  # aligned columns, notebook 230's filter
MIN_TM = 0.5  # SCOPe pairs: the structural superposition must be real (notebook 230)
SCOPE_CATEGORIES = ["same_family", "same_superfamily_diff_family"]  # = same superfamily

# Values given with the request. The figure is not drawn if the data disagree at the shown
# precision (sum of squared shares to 3 decimals, kappa to 2), except where a value sits on a
# rounding boundary between the two Swiss-Prot sources (EXPECTED_S2_TOL).
EXPECTED_S2 = {
    "protein20": 0.060,
    "dayhoff6": 0.231,
    "gbmr4": 0.408,
    "polarity4": 0.341,
    "hp_thomas_dill2": 0.512,
}
EXPECTED_S2_TOL = (
    0.0007  # dayhoff6: 0.2304 from FASTA counts, 0.2305 from the release notes
)
EXPECTED_KAPPA = {"protein20": 0.20, "hp_thomas_dill2": 0.46}
EXPECTED_N_PFAM = 37_085

# Rows: size groups top to bottom, gap between groups; within a group, more letters first,
# ties alphabetical.
SIZE_GROUPS = [
    ("20 letters", 20, 20),
    ("12–18 letters", 12, 18),
    ("4–8 letters", 4, 8),
    ("2–3 letters", 2, 3),
]

# One meaning per colour, across the paper's figures: purple = a measured value of an
# alphabet, coral = k*, grey = the log2(n) reference. Blues in a only separate neighbours.
PURPLE = "#7B3294"
CORAL = "#F07A5A"
REF_GREY = "#8C8C8C"
JOIN = "#CFCFCF"
GUIDE = "#E6E6E6"
CELL_GREY = "#D9D9D9"  # one square per k-size to test
PRED_GREY = "#595959"  # a seed length predicted from Swiss-Prot bits
BLUE_LIGHT = "#B9CCF0"
BLUE_DARK = "#8FAAE0"
GROUP_INK = "#595959"
MONO = "Courier New"

# Layout, mm at printed size (183 mm, two columns).
W_MM = pf.TWO_COLUMN_MM
COLS = {  # left edge, width
    "a": (27.0, 51.0),
    "b": (84.0, 42.0),
    "c": (133.0, 40.0),
}
COUNT_X_MM = 7.5  # the "# k-sizes" column ends this far right of panel c
REACH_COL = (27.0, 52.0)  # the one-column seed-reach figure: left edge, width
ROW_MM = 4.6
GROUP_GAP_MM = 4.2
LEGEND_MM = 10.5  # room for the legend under each column header
TOP_MM = LEGEND_MM + 9.0  # legend, header, panel letter
TOP_MM_REACH = LEGEND_MM + 4.5
BOTTOM_MM = 9.0  # x axes
B_XMIN = 0.5  # bits axis starts here: the lowest value is 0.71
KMIN = r"$k_\mathrm{min}$"
KMAX = r"$k_\mathrm{max}$"
BAR_MM = 3.0


def check(ok: bool, msg: str) -> None:
    if not ok:
        raise SystemExit("STOP, figure not drawn: " + msg)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def alphabets_rs_clusters() -> dict[str, list[str]] | None:
    """Class lists of all 19 alphabets as written in kmerseek's alphabets.rs at KMERSEEK_TAG."""
    if not (KMERSEEK / ".git").exists():
        return None
    src = subprocess.run(
        ["git", "-C", str(KMERSEEK), "show", f"{KMERSEEK_TAG}:src/rust/alphabets.rs"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    out: dict[str, list[str]] = {}
    for name, body in re.findall(
        r"const (\w+)_CLUSTERS: &\[&str\] =\s*&\[(.*?)\];", src, re.S
    ):
        out[name.lower()] = re.findall(r'"([A-Z]+)"', body)
    statics = {
        "LEHNINGER_HP": "hp_lehninger2",
        "THOMAS_DILL_HP": "hp_thomas_dill2",
        "KYTE_DOOLITTLE_HP": "hp_kyte_doolittle2",
        "THOMAS_DILL_NO_C_HP": "hp_thomas_dill_no_c2",
        "LEHNINGER_C_NONPOLAR_HP": "hp_lehninger_c_nonpolar2",
        "LEHNINGER_HPC": "hp_lehninger_hpc3",
        "PBOTC_1ST_ED_HP": "hp_pbotc_1st_ed2",
    }
    for static, name in statics.items():
        m = re.search(rf"static {static}:.*?build_hpc?\((.*?)\)\)", src, re.S)
        out[name] = re.findall(r'b"([A-Z]+)"', m.group(1))
    out["protein20"] = list(STANDARD)
    return out


def check_alphabets() -> str:
    rs = alphabets_rs_clusters()
    for name, cl in hc.ALPHABET_CLUSTERS.items():
        check(
            "".join(sorted("".join(cl))) == STANDARD,
            f"{name} does not cover the 20 residues once each: {cl}",
        )
    if rs is None:
        return f"not checked against kmerseek (no checkout at {KMERSEEK})"
    for name, cl in hc.ALPHABET_CLUSTERS.items():
        if name == "dayhoff6":  # encoded by sourmash, not listed in alphabets.rs
            continue
        check(name in rs, f"{name} not found in alphabets.rs {KMERSEEK_TAG}")
        check(
            sorted(rs[name]) == sorted(cl),
            f"{name}: utils {cl} != alphabets.rs {rs[name]}",
        )
    return f"18 alphabets match kmerseek {KMERSEEK_TAG} src/rust/alphabets.rs; dayhoff6 is sourmash's"


def swissprot_counts(recount: bool) -> dict[str, int]:
    if recount or not SWISSPROT_COUNTS.exists():
        with gzip.open(SWISSPROT_FASTA, "rb") as fh:
            raw = fh.read()
        seq = b"".join(line for line in raw.split(b"\n") if not line.startswith(b">"))
        counts = np.bincount(np.frombuffer(seq, dtype=np.uint8), minlength=256)
        rows = [{"residue": r, "count": int(counts[ord(r)])} for r in STANDARD]
        other = int(counts.sum()) - sum(r["count"] for r in rows)
        pl.DataFrame(rows).write_csv(SWISSPROT_COUNTS)
        print(
            f"counted {SWISSPROT_FASTA.name}: {other:_} non-standard letters left out"
        )
    return dict(pl.read_csv(SWISSPROT_COUNTS).iter_rows())


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pair_columns(pairs: pl.DataFrame, kstar: dict[str, int], tag: str) -> pl.DataFrame:
    """Kappa (mean, bootstrap 95%) and share of pairs reaching k* (Wilson 95%) per alphabet."""
    kap = hc.summarise(pairs, ["alphabet"], "kappa")
    rows = []
    for a, g in pairs.group_by("alphabet"):
        a = a[0]
        n = g.height
        k = int((g["longest_run"] >= kstar[a]).sum())
        k_minus1 = int((g["longest_run"] >= kstar[a] - 1).sum())
        lo, hi = wilson(k, n)
        rows.append(
            {
                "alphabet": a,
                f"{tag}_n_pairs": n,
                f"{tag}_n_reach_kstar": k,
                f"{tag}_share_reach_kstar": k / n,
                f"{tag}_share_lo": lo,
                f"{tag}_share_hi": hi,
                f"{tag}_share_reach_kstar_minus1": k_minus1 / n,
            }
        )
    kap = kap.rename(
        {
            "n": f"{tag}_n_kappa",
            "kappa_mean": f"{tag}_kappa",
            "kappa_lo": f"{tag}_kappa_lo",
            "kappa_hi": f"{tag}_kappa_hi",
        }
    )
    return pl.DataFrame(rows).join(kap, on="alphabet")


def build_table(recount: bool) -> tuple[pl.DataFrame, dict]:
    meta = {"alphabet_check": check_alphabets()}
    counts = swissprot_counts(recount)
    n_res = sum(counts.values())
    meta["n_res"] = n_res
    check(
        hashlib.sha256(SEED_COUNTS.read_bytes()).hexdigest() == SEED_COUNTS_SHA256,
        f"{SEED_COUNTS} is not the copy from b5650a2",
    )
    seeds = {r["alphabet"]: r for r in pl.read_csv(SEED_COUNTS).iter_rows(named=True)}

    # Panel a draws the classes in the order alphabets.rs lists them (same sets, checked above).
    rs = alphabets_rs_clusters() or {}
    rows = []
    for name, cl in hc.ALPHABET_CLUSTERS.items():
        cl = rs.get(name, cl)
        shares = [sum(counts[r] for r in c) / n_res for c in cl]
        s2 = sum(q * q for q in shares)
        b_sp = -math.log2(s2)
        s = seeds[name]
        b_human = (
            math.log2(s["proteins_per_seed_k_small"] - 1)
            - math.log2(s["proteins_per_seed_k_main"] - 1)
        ) / (s["k_main"] - s["k_small"])
        rows.append(
            {
                "alphabet": name,
                "n_letters": len(cl),
                "classes": " ".join(cl),
                "class_shares": " ".join(f"{q:.6f}" for q in shares),
                "sum_sq_shares": s2,
                "bits_log2_n": math.log2(len(cl)),
                "bits_swissprot": b_sp,
                "bits_human_seed_counts": b_human,
                "kstar_exact": math.log2(n_res) / b_sp,
                "kstar": math.ceil(math.log2(n_res) / b_sp),
            }
        )
    t = pl.DataFrame(rows)
    kstar = dict(zip(t["alphabet"], t["kstar"]))

    check(
        hashlib.sha256(KSIZES.read_bytes()).hexdigest() == KSIZES_SHA256,
        f"{KSIZES} is not the copy from PR 101",
    )
    ks = pl.read_csv(KSIZES).select(
        "alphabet",
        pl.col("k_main").alias("k100_measured"),
        pl.col("k_human100").alias("k100_predicted"),
        pl.col("k_star").alias("kstar_274"),
        "k_max",
        "k_min",
        "n_ksizes",
    )
    t = t.join(ks, on="alphabet")
    check(t.height == 19, "notebook 274 table does not cover the 19 alphabets")
    check(
        bool((t["kstar"] == t["kstar_274"]).all()),
        "k* here (20 standard residues) differs from notebook 274's (all letters)",
    )
    check(
        bool(
            (
                t["k_min"]
                == t.select(
                    pl.min_horizontal("k100_measured", "k100_predicted")
                ).to_series()
            ).all()
        )
        and bool((t["n_ksizes"] == t["k_max"] - t["k_min"] + 1).all())
        and bool((t["k_min"] <= t["kstar"]).all() & (t["kstar"] <= t["k_max"]).all()),
        "notebook 274's k_min, k_max and # k-sizes do not fit together",
    )
    t = t.drop("kstar_274")

    pfam = hc.add_identity_bin(pl.read_parquet(PFAM_PAIRS)).filter(
        (pl.col("identity_bin") == BIN) & (pl.col("n_cols") >= MIN_COLS)
    )
    scope = hc.add_identity_bin(pl.read_parquet(SCOPE_PAIRS)).filter(
        (pl.col("identity_bin") == BIN)
        & (pl.col("n_cols") >= MIN_COLS)
        & (pl.max_horizontal("tm_q", "tm_t") >= MIN_TM)
        & pl.col("category").is_in(SCOPE_CATEGORIES)
    )
    for d, label in [(pfam, "Pfam"), (scope, "SCOPe")]:
        per = d.group_by("alphabet").len()
        check(
            per.height == 19 and per["len"].n_unique() == 1,
            f"{label}: not one row per pair for all 19 alphabets: {per}",
        )
    meta["pfam_n_families"] = pfam["family"].n_unique()
    meta["scope_n_superfamily_cross_family"] = scope.filter(
        (pl.col("alphabet") == "protein20")
        & (pl.col("category") == "same_superfamily_diff_family")
    ).height
    t = t.join(pair_columns(pfam, kstar, "pfam"), on="alphabet").join(
        pair_columns(scope, kstar, "scope"), on="alphabet"
    )

    # Checks against the values given with the request.
    for a, v in EXPECTED_S2.items():
        got = t.filter(pl.col("alphabet") == a)["sum_sq_shares"][0]
        check(
            abs(got - v) <= EXPECTED_S2_TOL,
            f"sum of squared shares {a} = {got:.4f}, expected {v}",
        )
    for a, v in EXPECTED_KAPPA.items():
        got = t.filter(pl.col("alphabet") == a)["pfam_kappa"][0]
        check(round(got, 2) == v, f"kappa {a} = {got:.4f}, expected {v}")
    check(
        t["pfam_n_pairs"].unique().to_list() == [EXPECTED_N_PFAM],
        f"Pfam pairs {t['pfam_n_pairs'].unique()}",
    )
    return order_rows(t), meta


def order_rows(t: pl.DataFrame) -> pl.DataFrame:
    group = pl.lit(None, dtype=pl.Int64)
    for i, (_, lo, hi) in enumerate(SIZE_GROUPS):
        group = pl.when(pl.col("n_letters").is_between(lo, hi)).then(i).otherwise(group)
    return t.with_columns(group.alias("size_group")).sort(
        ["size_group", "n_letters", "alphabet"], descending=[False, True, False]
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def row_positions(
    t: pl.DataFrame,
) -> tuple[list[float], list[tuple[str, float]], float]:
    """y (mm from the top of the row block) of each row's centre, and of each group label."""
    ys, labels, y, prev = [], [], 0.0, None
    for g in t["size_group"].to_list():
        if g != prev:
            y += GROUP_GAP_MM
            labels.append((SIZE_GROUPS[g][0], y - GROUP_GAP_MM / 2))
            prev = g
        ys.append(y + ROW_MM / 2)
        y += ROW_MM
    return ys, labels, y


def _row_labels(ax, ys, names, group_labels, width_mm: float) -> None:
    """Alphabet names and size-group labels, right-aligned left of `ax`."""
    for y, text, style in [(y, n, "row") for y, n in zip(ys, names)] + [
        (y, g, "group") for g, y in group_labels
    ]:
        ax.text(
            -1.2 / width_mm,
            y,
            text,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6 if style == "row" else 5.5,
            style="normal" if style == "row" else "italic",
            color="black" if style == "row" else GROUP_INK,
        )


def _row_axes(fig, x0_mm, w_mm, w_fig_mm, h_mm, ys, block_mm):
    a = fig.add_axes(
        [x0_mm / w_fig_mm, BOTTOM_MM / h_mm, w_mm / w_fig_mm, block_mm / h_mm]
    )
    a.set_ylim(block_mm, 0)
    a.set_yticks(ys)
    a.set_yticklabels([])
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    return a


def _column_legend(ax, handles, block_mm: float) -> None:
    """A legend directly under the column header, above the marks it explains."""
    ax.legend(
        [h for h, _ in handles],
        [lab for _, lab in handles],
        loc="upper left",
        bbox_to_anchor=(0, 1 + LEGEND_MM / block_mm),
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0,
        borderpad=0,
    )


def _mark(**kw):
    return Line2D([], [], ls="", **kw)


def draw_suppfig1(t: pl.DataFrame, stem: Path) -> None:
    """Supplementary Figure 1: a class shares, b bits per letter, c seed lengths to test."""
    pf.use_style()
    ys, group_labels, block_mm = row_positions(t)
    h_mm = TOP_MM + block_mm + BOTTOM_MM
    fig = plt.figure(figsize=(W_MM * pf.MM, h_mm * pf.MM))
    fig.set_layout_engine(None)
    axes = {k: _row_axes(fig, *COLS[k], W_MM, h_mm, ys, block_mm) for k in COLS}
    _row_labels(axes["a"], ys, t["alphabet"].to_list(), group_labels, COLS["a"][1])
    for k in "bc":  # row guides from each row's tick (7c)
        for y in ys:
            axes[k].axhline(y, color=GUIDE, lw=0.3, zorder=0)

    # a: class shares.
    ax_a = axes["a"]
    ax_a.set_xlim(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Share of Swiss-Prot 2026_03 residues (%)")
    mm_per_pct = COLS["a"][1] / 100
    char_mm = 5.5 / 72 * 25.4 * 0.6  # Courier New advance is 0.6 em
    out_char_mm = 5 / 72 * 25.4 * 0.6
    for y, cls, shares in zip(ys, t["classes"].to_list(), t["class_shares"].to_list()):
        x = 0.0
        prev_right = -1.0  # right edge (in %) of the last label printed above this bar
        for i, (c, q) in enumerate(zip(cls.split(), map(float, shares.split()))):
            w = 100 * q
            ax_a.add_patch(
                Rectangle(
                    (x, y - BAR_MM / 2),
                    w,
                    BAR_MM,
                    facecolor=BLUE_LIGHT if i % 2 == 0 else BLUE_DARK,
                    edgecolor="white",
                    lw=0.4,
                )
            )
            if len(c) * char_mm + 0.3 <= w * mm_per_pct:
                ax_a.text(
                    x + w / 2, y, c, ha="center", va="center", family=MONO, fontsize=5.5
                )
            else:
                # Too narrow: print the residues just above the segment, moved right only
                # as far as needed to clear the previous label above this bar.
                half = len(c) * out_char_mm / mm_per_pct / 2
                cx = min(
                    max(x + w / 2, prev_right + 0.25 / mm_per_pct + half), 100 - half
                )
                shift_mm = abs(cx - (x + w / 2)) * mm_per_pct
                check(
                    shift_mm <= 1.5,
                    f"label {c} in row {y} would sit {shift_mm:.1f} mm off",
                )
                ax_a.text(
                    cx,
                    y - BAR_MM / 2 - 0.15,
                    c,
                    ha="center",
                    va="bottom",
                    family=MONO,
                    fontsize=5,
                )
                prev_right = cx + half
            x += w
        check(abs(x - 100) < 1e-3, f"class shares do not sum to 100% in row {y}")
    _column_legend(
        ax_a,
        [
            (
                Patch(facecolor=BLUE_LIGHT, edgecolor="white"),
                "one class; the two blues only separate neighbours",
            )
        ],
        block_mm,
    )

    # b: bits per letter, three values on the alphabet's row.
    ax_b = axes["b"]
    lo = t.select(
        pl.min_horizontal("bits_human_seed_counts", "bits_swissprot")
    ).to_series()
    ax_b.hlines(ys, lo, t["bits_log2_n"], color=JOIN, lw=1.0, zorder=1)
    ax_b.scatter(
        t["bits_log2_n"],
        ys,
        s=14,
        facecolor="white",
        edgecolor=REF_GREY,
        lw=0.8,
        zorder=2,
    )
    ax_b.scatter(t["bits_swissprot"], ys, s=4, color=PURPLE, lw=0, zorder=3)
    ax_b.scatter(
        t["bits_human_seed_counts"],
        ys,
        s=7,
        marker="D",
        facecolor=PURPLE,
        edgecolor="white",
        lw=0.3,
        zorder=4,
    )
    ax_b.set_xlim(B_XMIN, 4.5)
    ax_b.set_xticks([1, 2, 3, 4])
    ax_b.set_xlabel("Bits per letter")
    _column_legend(
        ax_b,
        [
            (
                _mark(marker="o", ms=3.8, mfc="white", mec=REF_GREY, mew=0.8),
                "log2(n letters): all equally common",
            ),
            (_mark(marker="o", ms=2.2, color=PURPLE), "from Swiss-Prot composition"),
            (
                _mark(marker="D", ms=2.6, mfc=PURPLE, mec="white", mew=0.3),
                "measured in the human proteome",
            ),
            (Line2D([], [], color=JOIN, lw=1.0), "joins the values of one alphabet"),
        ],
        block_mm,
    )

    # c: the k-sizes to test, k_min to k_max, and k*.
    ax_c = axes["c"]
    for y, lo_k, hi_k in zip(ys, t["k_min"], t["k_max"]):
        for k in range(lo_k, hi_k + 1):
            ax_c.add_patch(
                Rectangle(
                    (k - 0.42, y - 0.75), 0.84, 1.5, facecolor=CELL_GREY, lw=0, zorder=1
                )
            )
    ax_c.scatter(t["kstar"], ys, s=9, color=CORAL, lw=0, zorder=3)
    for y, n in zip(ys, t["n_ksizes"]):
        ax_c.text(
            1 + COUNT_X_MM / COLS["c"][1],
            y,
            str(n),
            transform=ax_c.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6,
        )
    ax_c.set_xlim(0, 40)
    ax_c.set_xticks([0, 10, 20, 30, 40])
    ax_c.set_xlabel("Seed length k (letters)")
    _column_legend(
        ax_c,
        [
            (Patch(facecolor=CELL_GREY), "one k-size to test, " + KMIN + " to " + KMAX),
            (
                _mark(marker="o", ms=2.6, color=CORAL),
                "k*: one chance match in Swiss-Prot",
            ),
        ],
        block_mm,
    )

    headers = {
        "a": "Classes, width = share of residues",
        "b": "Information per letter",
        "c": "Seed lengths to test",
    }
    _headers(fig, headers, {k: v[0] for k, v in COLS.items()}, block_mm, h_mm, W_MM)
    fig.text(
        (COLS["c"][0] + COLS["c"][1] + COUNT_X_MM) / W_MM,
        (BOTTOM_MM + block_mm + LEGEND_MM + 0.8) / h_mm,
        "# k-sizes",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    assert_no_text_collisions(fig)
    pf.save(fig, stem)
    print(f"wrote {stem}.pdf/.png/.svg ({W_MM} x {h_mm:.0f} mm)")


def _headers(fig, headers, x0s, block_mm, h_mm, w_fig_mm) -> None:
    """Panel letter, then the column header, then (in the axes) the column legend."""
    y_head = (BOTTOM_MM + block_mm + LEGEND_MM + 0.8) / h_mm
    y_letter = (BOTTOM_MM + block_mm + LEGEND_MM + 4.0) / h_mm
    for k, x0 in x0s.items():
        fig.text(
            x0 / w_fig_mm, y_head, headers[k], ha="left", va="bottom", fontsize=6.5
        )
        if len(x0s) > 1:
            fig.text(
                (1.0 if k == "a" else x0 - 2.5) / w_fig_mm,
                y_letter,
                k,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )


def draw_seed_reach(t: pl.DataFrame, stem: Path) -> None:
    """Share of Pfam seed pairs whose longest same-class run along the alignment is >= k*."""
    pf.use_style()
    ys, group_labels, block_mm = row_positions(t)
    h_mm = TOP_MM_REACH + block_mm + BOTTOM_MM
    x0, w = REACH_COL
    fig = plt.figure(figsize=(pf.ONE_COLUMN_MM * pf.MM, h_mm * pf.MM))
    fig.set_layout_engine(None)
    ax = _row_axes(fig, x0, w, pf.ONE_COLUMN_MM, h_mm, ys, block_mm)
    _row_labels(ax, ys, t["alphabet"].to_list(), group_labels, w)
    for y in ys:
        ax.axhline(y, color=GUIDE, lw=0.3, zorder=0)
    sh, lo, hi = (
        100 * t[c] for c in ("pfam_share_reach_kstar", "pfam_share_lo", "pfam_share_hi")
    )
    ax.hlines(ys, lo, hi, color=PURPLE, lw=0.8, zorder=2)
    ax.scatter(sh, ys, s=7, color=PURPLE, lw=0, zorder=3)
    x_max = 8
    check(
        float(hi.max()) < x_max - 1.2,
        "axis too short for the intervals and their numbers",
    )
    for y, v, h in zip(ys, sh, hi):
        ax.text(h + 0.2, y, f"{v:.1f}", va="center", ha="left", fontsize=6)
    ax.set_xlim(0, x_max)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("Pairs with a run ≥ the alphabet's k* (%)")
    _column_legend(
        ax,
        [
            (_mark(marker="o", ms=2.6, color=PURPLE), "measured share of pairs"),
            (Line2D([], [], color=PURPLE, lw=0.8), "Wilson 95% interval"),
            (_mark(), "run: aligned columns in a row, no gap, same class in both"),
        ],
        block_mm,
    )
    n = t["pfam_n_pairs"][0]
    _headers(
        fig,
        {"e": f"Pfam-A 38.2 seed pairs, 20–30% identity (n = {n:,})"},
        {"e": x0},
        block_mm,
        h_mm,
        pf.ONE_COLUMN_MM,
    )
    assert_no_text_collisions(fig)
    pf.save(fig, stem)
    print(f"wrote {stem}.pdf/.png/.svg ({pf.ONE_COLUMN_MM} x {h_mm:.0f} mm)")


def assert_no_text_collisions(fig) -> None:
    """Every text inside the figure, and no two texts overlapping (7, 7b)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        check(
            bb.x0 >= fb.x0 - 0.5
            and bb.x1 <= fb.x1 + 0.5
            and bb.y0 >= fb.y0 - 0.5
            and bb.y1 <= fb.y1 + 0.5,
            f"text leaves the canvas: {t.get_text()!r}",
        )
        boxes.append((t.get_text(), bb.shrunk(0.97, 0.80)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise SystemExit(
                    f"STOP: text overlaps: {boxes[i][0]!r} and {boxes[j][0]!r}"
                )


def write_captions(t: pl.DataFrame, meta: dict) -> None:
    """Both captions, formatted from the table; each sentence's claim is checked first."""
    row = {r["alphabet"]: r for r in t.iter_rows(named=True)}
    small = t.filter(pl.col("n_letters") <= 3)
    two = t.filter(pl.col("n_letters") == 2)
    reduced = t.filter(pl.col("alphabet") != "protein20")
    pct2 = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    neg = lambda v: f"{v:.2f}".replace("-", "−")  # noqa: E731
    spear = lambda a, b: t.select(pl.corr(a, b, method="spearman")).item()  # noqa: E731

    # Supplementary Figure 1 claims.
    rho_nb, rho_nks = spear("n_letters", "bits_swissprot"), spear("n_letters", "kstar")
    check(
        rho_nb > 0 and rho_nks < 0,
        "letter count does not order bits and k* as the title says",
    )
    check(
        bool((t["bits_human_seed_counts"] < t["bits_swissprot"]).all()),
        "human B not below Swiss-Prot B",
    )
    check(
        bool((t["bits_swissprot"] < t["bits_log2_n"]).all()),
        "Swiss-Prot B not below log2(n)",
    )
    check(
        int(reduced["kstar"].min()) > row["protein20"]["kstar"],
        "a reduced alphabet has k* <= protein20",
    )
    check(
        row["gbmr7"]["kstar"] > row["gbmr4"]["kstar"], "gbmr7 k* no longer above gbmr4"
    )
    check(
        row["hp_lehninger_hpc3"]["kstar"] < int(two["kstar"].min()),
        "hpc3 k* no longer below 2-letter",
    )
    check(
        float(t["bits_human_seed_counts"].min()) > B_XMIN,
        "a bits value is left of panel b's axis",
    )
    gbmr7_top = sorted(map(float, row["gbmr7"]["class_shares"].split()))[-1]
    longest = t.sort("kstar").row(-1, named=True)["alphabet"]
    most_k = t.sort("n_ksizes").row(-1, named=True)["alphabet"]

    supp = f"""**Supplementary Figure 1 | Alphabets with fewer letters carry fewer bits per letter, so each needs a longer seed.**

One row per kmerseek alphabet (19), grouped by letter count; the rows are in the same order in every column. Classes are those of kmerseek {KMERSEEK_TAG} `src/rust/alphabets.rs` (dayhoff6 is encoded by sourmash). Across the 19 alphabets, fewer letters go with fewer bits per letter (Spearman ρ = {neg(rho_nb)} between letter count and bits) and a longer k* (ρ = {neg(rho_nks)}). Every reduced alphabet needs a longer seed than the 20 amino acids. The order is not strict: gbmr7 (7 letters) needs k* = {row["gbmr7"]["kstar"]}, more than gbmr4 (4 letters, {row["gbmr4"]["kstar"]}), because one of its classes holds {100 * gbmr7_top:.0f}% of residues; hp_lehninger_hpc3 (3 letters) needs {row["hp_lehninger_hpc3"]["kstar"]}, less than every 2-letter alphabet ({int(two["kstar"].min())}–{int(two["kstar"].max())}).

**a**, Classes of each alphabet, labelled with their residues. Segment width is the class's share of the {meta["n_res"]:,} standard residues in UniProtKB/Swiss-Prot release 2026_03. The residues of a class too narrow for its letters are printed just above its segment.

**b**, Bits of information in one matching letter, three ways. Open grey circle: log2 of the number of letters, the value if all letters were equally common. Filled purple circle: B = −log2(Σ q²), where q is each class's share of Swiss-Prot residues, so Σ q² is the chance that two residues drawn at random fall in the same class. Purple diamond: B measured from kmerseek seed counts in the human proteome (QfO 2020_04, UP000005640). P(k), the mean number of proteins sharing a seed of k letters, falls by a factor 2^B for each added letter, so B = [log2(P(k_small) − 1) − log2(P(k_main) − 1)] / (k_main − k_small) (notebook 250). In all 19 alphabets the human value is below the Swiss-Prot value, and both are below log2 of the letter count. For hp_thomas_dill2 the three values are {row["hp_thomas_dill2"]["bits_log2_n"]:.3f}, {row["hp_thomas_dill2"]["bits_swissprot"]:.3f} and {row["hp_thomas_dill2"]["bits_human_seed_counts"]:.3f} bits. The axis starts at {B_XMIN} bits.

**c**, The seed lengths to test for each alphabet, one grey square per k-size from k_min to k_max; the column on the right counts them (notebook 274). Seed lengths come from k = ⌈(log2 N − log2 m) / B⌉, the shortest seed with at most m chance matches among N residues when one letter carries B bits. Coral dot: k*, with N = {meta["n_res"]:,} Swiss-Prot residues, m = 1 and B from Swiss-Prot composition, the seed length at which one match by chance is expected across Swiss-Prot. k_max is the same with B measured in the human proteome. k_min is the k at which a seed is shared by about 100 human proteins: the smaller of the value measured from kmerseek seed counts and the value from the formula with N = human proteome residues and m = 100. k* runs from {row["protein20"]["kstar"]} (protein20) to {int(t["kstar"].max())} ({longest}). The number of k-sizes runs from {row["protein20"]["n_ksizes"]} (protein20) to {int(t["n_ksizes"].max())} ({most_k}), and is {int(small["n_ksizes"].min())}–{int(small["n_ksizes"].max())} for the 2–3 letter alphabets.

All numbers, including both k_min estimates and k_max: `tables/suppfig1_values.csv`, written with the figure by `scripts/plot_suppfig1_alphabets.py`.
"""

    # Seed-reach figure claims.
    top = {
        ds: t.sort(f"{ds}_share_reach_kstar", descending=True).row(0, named=True)
        for ds in ("pfam", "scope")
    }
    for ds in ("pfam", "scope"):
        check(
            float(t[f"{ds}_share_reach_kstar"].max()) < 0.06,
            f"{ds}: an alphabet reaches 6% or more",
        )
    rho_ke = spear("pfam_kappa", "pfam_share_reach_kstar")
    check(
        rho_ke < 0,
        "kappa and the seed-reach share no longer rank the alphabets in opposite ways",
    )
    check(
        small["pfam_kappa"].min() > row["protein20"]["pfam_kappa"],
        "a 2-3 letter alphabet has kappa <= protein20",
    )
    check(
        small["pfam_share_reach_kstar"].max()
        < row["protein20"]["pfam_share_reach_kstar"],
        "a 2-3 letter alphabet reaches more than protein20",
    )
    ratio = t["pfam_share_reach_kstar_minus1"] / t["pfam_share_reach_kstar"]
    m1 = t.sort("pfam_share_reach_kstar_minus1", descending=True).row(0, named=True)
    rho_e = spear("pfam_share_reach_kstar", "scope_share_reach_kstar")
    d_e = (
        (100 * (t["scope_share_reach_kstar"] - t["pfam_share_reach_kstar"])).abs().max()
    )
    check(d_e < 1, "SCOPe and Pfam shares differ by 1 point or more")
    hp = small.sort("pfam_share_reach_kstar")
    reach = f"""**Seed reach at k* | At its own seed length k*, no alphabet shares a seed along the alignment in more than {100 * top["pfam"]["pfam_share_reach_kstar"]:.1f}% of Pfam seed pairs at 20–30% identity.**

One row per kmerseek alphabet, in the order of Supplementary Figure 1; k* is from its panel c. Pairs are two sequences from the same Pfam-A 38.2 seed alignment, 20–30% identity over at least {MIN_COLS} aligned columns ({row["protein20"]["pfam_n_pairs"]:,} pairs from {meta["pfam_n_families"]:,} families). A run is a stretch of consecutive aligned columns with no gap in either sequence and the same class in both: an exact seed the two sequences share along their alignment. A seed shared elsewhere, off the alignment, is not counted. Dot: share of pairs whose longest run is at least k*; line: Wilson 95% interval. The highest is {top["pfam"]["alphabet"]}, {pct2(top["pfam"]["pfam_share_reach_kstar"])} ({top["pfam"]["pfam_n_reach_kstar"]:,} pairs); protein20 {pct2(row["protein20"]["pfam_share_reach_kstar"])}; the 2–3 letter alphabets {pct2(hp["pfam_share_reach_kstar"][0])} ({hp["alphabet"][0]}) to {pct2(hp["pfam_share_reach_kstar"][-1])} ({hp["alphabet"][-1]}); gbmr7 {pct2(row["gbmr7"]["pfam_share_reach_kstar"])}.

The share depends on rounding k* up to a whole letter. At one letter less, it is {ratio.min():.1f} to {ratio.max():.1f} times higher, and the highest is {m1["alphabet"]}, {pct2(m1["pfam_share_reach_kstar_minus1"])}.

Cohen's κ for the class of aligned residues ranks the alphabets partly in the opposite order (Spearman ρ = {neg(rho_ke)} against this share): κ is {row["protein20"]["pfam_kappa"]:.2f} for protein20 and {small["pfam_kappa"].min():.2f}–{small["pfam_kappa"].max():.2f} for the 2–3 letter alphabets, which keep classes better but carry fewer bits per letter. κ values are in `tables/suppfig1_values.csv`.

On SCOPe 2.08 (40% set) domain pairs in the same superfamily, aligned by structure with USalign (TM-score ≥ {MIN_TM}, {row["protein20"]["scope_n_pairs"]:,} pairs), the alphabets come out in nearly the same order (Spearman ρ = {rho_e:.2f}), each share moves by at most {d_e:.1f} percentage point, and the highest is {top["scope"]["alphabet"]}, {pct2(top["scope"]["scope_share_reach_kstar"])}.
"""
    CAPTION.write_text(supp)
    REACH_CAPTION.write_text(reach)
    print(f"wrote {CAPTION} and {REACH_CAPTION}")
    print(
        f"kappa vs reach rho {rho_ke:.3f}; k*-1 ratio {ratio.min():.2f}-{ratio.max():.2f}; "
        f"SCOPe-Pfam kappa diff {(t['scope_kappa'] - t['pfam_kappa']).min():.3f} to {(t['scope_kappa'] - t['pfam_kappa']).max():.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--recount",
        action="store_true",
        help="recount Swiss-Prot residues from the FASTA",
    )
    args = ap.parse_args()
    t, meta = build_table(args.recount)
    print(meta)
    with pl.Config(tbl_rows=25, tbl_cols=20, tbl_width_chars=250):
        print(
            t.select(
                "alphabet",
                "n_letters",
                pl.col("sum_sq_shares").round(4),
                pl.col("bits_swissprot").round(3),
                pl.col("bits_human_seed_counts").round(3),
                "kstar",
                pl.col("pfam_kappa").round(3),
                (100 * pl.col("pfam_share_reach_kstar")).round(2).alias("pfam_%"),
                "pfam_n_reach_kstar",
                pl.col("scope_kappa").round(3),
                (100 * pl.col("scope_share_reach_kstar")).round(2).alias("scope_%"),
                "scope_n_reach_kstar",
            )
        )
    t.drop("size_group").write_csv(VALUES, float_precision=5)
    draw_suppfig1(t, FIG / "suppfig1")
    draw_seed_reach(t, FIG / REACH_STEM)
    write_captions(t, meta)


if __name__ == "__main__":
    main()
