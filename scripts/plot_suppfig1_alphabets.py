#!/usr/bin/env python3
"""Supplementary Figure 1: what each of kmerseek's 19 alphabets keeps, and what seed it needs.

One row per alphabet, ordered by letter count, the same rows in every column:
  a  class shares in Swiss-Prot 2026_03, one segment per class, labelled with its residues
  b  bits per letter three ways: log2(n letters); B from Swiss-Prot composition,
     B = -log2(sum of squared class shares); B measured from kmerseek seed counts in the
     human proteome (notebook 250)
  c  k* = ceil(log2(N) / B), N = standard residues in Swiss-Prot 2026_03, B from composition:
     the seed length at which one chance match is expected across Swiss-Prot
  d  Cohen's kappa of the aligned residues' classes at 20-30% identity, mean over pairs,
     bootstrap 95% interval
  e  share of those pairs whose longest class-identical run (no gap, same class in both
     sequences) is at least the alphabet's own k*, Wilson 95% interval

Two versions: Pfam-A 38.2 seed pairs (figures/suppfig1.*) and SCOPe 2.08 40% pairs in the
same superfamily, structurally aligned by USalign (figures/suppfig1_scope.*). Columns a-c are
the same in both.

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
  figures/suppfig1.{pdf,png,svg}, figures/suppfig1_scope.{pdf,png,svg}
  figures/suppfig1_caption.md, tables/suppfig1_values.csv

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
PFAM_PAIRS = Path("/Users/olga/data/pfam/230_pfam_seed_pair_class_agreement.parquet")
SCOPE_PAIRS = Path("/Users/olga/data/scope/230_scope40_pair_class_agreement.parquet")
KMERSEEK = Path("/Users/olga/code/kmerseek")
KMERSEEK_TAG = "v0.4.0"
FIG = REPO / "figures"
VALUES = REPO / "tables/suppfig1_values.csv"
CAPTION = FIG / "suppfig1_caption.md"

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
BLUE_LIGHT = "#B9CCF0"
BLUE_DARK = "#8FAAE0"
GROUP_INK = "#595959"
MONO = "Courier New"

# Layout, mm at printed size (183 mm, two columns).
W_MM = pf.TWO_COLUMN_MM
COLS = {  # left edge, width
    "a": (27.0, 57.0),
    "b": (89.0, 27.0),
    "c": (121.0, 14.0),
    "d": (141.0, 18.0),
    "e": (164.0, 18.0),
}
ROW_MM = 4.6
GROUP_GAP_MM = 4.2
TOP_MM = 24.5  # legend and column headers
BOTTOM_MM = 9.0  # x axes
BAR_MM = 3.0
HUMAN_DY_MM = 1.2


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
        lo, hi = wilson(k, n)
        rows.append(
            {
                "alphabet": a,
                f"{tag}_n_pairs": n,
                f"{tag}_n_reach_kstar": k,
                f"{tag}_share_reach_kstar": k / n,
                f"{tag}_share_lo": lo,
                f"{tag}_share_hi": hi,
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

    rows = []
    for name, cl in hc.ALPHABET_CLUSTERS.items():
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


def draw(t: pl.DataFrame, meta: dict, ds: str, stem: Path) -> None:
    pf.use_style()
    ys, group_labels, block_mm = row_positions(t)
    h_mm = TOP_MM + block_mm + BOTTOM_MM
    fig = plt.figure(figsize=(W_MM * pf.MM, h_mm * pf.MM))
    fig.set_layout_engine(None)

    def ax_for(key: str):
        x0, w = COLS[key]
        a = fig.add_axes([x0 / W_MM, BOTTOM_MM / h_mm, w / W_MM, block_mm / h_mm])
        a.set_ylim(block_mm, 0)
        a.set_yticks(ys)
        a.set_yticklabels([])
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        return a

    axes = {k: ax_for(k) for k in COLS}
    names = t["alphabet"].to_list()

    # Row labels and size groups, left of a.
    ax_a = axes["a"]
    for y, name in zip(ys, names):
        ax_a.text(
            -1.2 / COLS["a"][1],
            y,
            name,
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6,
        )
    for text, y in group_labels:
        ax_a.text(
            -1.2 / COLS["a"][1],
            y,
            text,
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            style="italic",
            color=GROUP_INK,
        )

    # Row guides from each row across b, d, e (7c): every mark reads back to its label.
    for k in "bde":
        for y in ys:
            axes[k].axhline(y, color=GUIDE, lw=0.3, zorder=0)

    # a: class shares.
    ax_a.set_xlim(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Share of Swiss-Prot residues (%)")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mm_per_pct = COLS["a"][1] / 100
    char_mm = 5.5 / 72 * 25.4 * 0.6  # Courier New advance is 0.6 em
    for y, cls, shares in zip(ys, t["classes"].to_list(), t["class_shares"].to_list()):
        x = 0.0
        prev_out = None  # 'above' / 'below' of the previous narrow class, to alternate
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
            need_mm = len(c) * char_mm + 0.3
            if need_mm <= w * mm_per_pct:
                ax_a.text(
                    x + w / 2, y, c, ha="center", va="center", family=MONO, fontsize=5.5
                )
                prev_out = None
            else:
                side = "below" if prev_out == "above" else "above"
                dy = (BAR_MM / 2 + 0.15) * (-1 if side == "above" else 1)
                ax_a.text(
                    x + w / 2,
                    y + dy,
                    c,
                    ha="center",
                    va="bottom" if side == "above" else "top",
                    family=MONO,
                    fontsize=5,
                )
                prev_out = side
            x += w
        check(abs(x - 100) < 1e-3, f"class shares do not sum to 100% in row {y}")

    # b: bits per letter.
    ax_b = axes["b"]
    # The human-proteome value sits HUMAN_DY_MM under the row line: in the 2-letter rows the
    # three values lie within 0.3 bits and would otherwise cover one another.
    y_h = [y + HUMAN_DY_MM for y in ys]
    ax_b.hlines(ys, t["bits_swissprot"], t["bits_log2_n"], color=JOIN, lw=1.0, zorder=1)
    ax_b.hlines(
        y_h,
        t["bits_human_seed_counts"],
        t["bits_swissprot"],
        color=JOIN,
        lw=0.6,
        zorder=1,
    )
    ax_b.scatter(
        t["bits_log2_n"],
        ys,
        s=16,
        facecolor="white",
        edgecolor=REF_GREY,
        lw=0.8,
        zorder=2,
    )
    ax_b.scatter(t["bits_swissprot"], ys, s=6, color=PURPLE, lw=0, zorder=3)
    ax_b.scatter(
        t["bits_human_seed_counts"],
        y_h,
        s=9,
        marker="D",
        facecolor="white",
        edgecolor=PURPLE,
        lw=0.7,
        zorder=4,
    )
    ax_b.set_xlim(0, 4.5)
    ax_b.set_xticks([0, 1, 2, 3, 4])
    ax_b.set_xlabel("Bits per letter")

    # c: k*.
    ax_c = axes["c"]
    ax_c.barh(ys, t["kstar"], height=BAR_MM * 0.8, color=CORAL, lw=0)
    for y, k in zip(ys, t["kstar"]):
        ax_c.text(k + 0.8, y, str(k), va="center", ha="left", fontsize=6)
    ax_c.set_xlim(0, 36)
    ax_c.set_xticks([0, 10, 20, 30])
    ax_c.set_xlabel("k* (letters)")

    # d: kappa.
    ax_d = axes["d"]
    kap, klo, khi = (t[f"{ds}_kappa"], t[f"{ds}_kappa_lo"], t[f"{ds}_kappa_hi"])
    ax_d.hlines(ys, klo, khi, color=PURPLE, lw=0.8, zorder=2)
    ax_d.scatter(kap, ys, s=6, color=PURPLE, lw=0, zorder=3)
    ax_d.set_xlim(0, 0.6)
    ax_d.set_xticks([0, 0.2, 0.4, 0.6])
    ax_d.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax_d.set_xlabel("Cohen's κ")

    # e: share of pairs with a run >= k*.
    ax_e = axes["e"]
    sh = 100 * t[f"{ds}_share_reach_kstar"]
    ax_e.hlines(
        ys,
        100 * t[f"{ds}_share_lo"],
        100 * t[f"{ds}_share_hi"],
        color=PURPLE,
        lw=0.8,
        zorder=2,
    )
    ax_e.scatter(sh, ys, s=6, color=PURPLE, lw=0, zorder=3)
    e_max = 8
    check(
        float((100 * t[f"{ds}_share_hi"]).max()) < e_max - 1.6,
        "panel e axis too short for its intervals and numbers",
    )
    for y, v, hi in zip(ys, sh, 100 * t[f"{ds}_share_hi"]):
        ax_e.text(
            hi + 0.25,
            y,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=5.5,
            color="black",
        )
    ax_e.set_xlim(0, e_max)
    ax_e.set_xticks([0, 2, 4, 6, 8])
    ax_e.set_xlabel("Pairs (%)")

    # Column headers, panel letters.
    n_pairs = t[f"{ds}_n_pairs"][0]
    src = (
        "Pfam-A 38.2 seed alignments"
        if ds == "pfam"
        else "SCOPe 2.08 40%, same superfamily"
    )
    headers = {
        "a": "Classes; width = share of\nSwiss-Prot 2026_03 residues",
        "b": "Information\nper letter",
        "c": "Seed length\nk*",
        "d": "Class agreement\nabove chance",
        "e": "Pairs with a\nrun ≥ own k*",
    }
    y_head = (BOTTOM_MM + block_mm + 1.2) / h_mm
    y_letter = (BOTTOM_MM + block_mm + 6.6) / h_mm
    for k in axes:
        x0 = COLS[k][0]
        fig.text(
            x0 / W_MM,
            y_head,
            headers[k],
            ha="left",
            va="bottom",
            fontsize=6,
            linespacing=1.1,
        )
        fig.text(
            (1.0 if k == "a" else x0 - 2.5) / W_MM,
            y_letter,
            k,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    # Data source for d and e, above their panel letters, with a rule spanning both columns.
    y_src = (BOTTOM_MM + block_mm + 11.0) / h_mm
    x_d, x_e_end = COLS["d"][0] / W_MM, (COLS["e"][0] + COLS["e"][1]) / W_MM
    fig.add_artist(
        Line2D([x_d, x_e_end], [y_src - 0.5 / h_mm] * 2, color=GROUP_INK, lw=0.4)
    )
    fig.text(
        x_d,
        y_src,
        f"{src}\n20–30% identity, n = {n_pairs:,} pairs",
        fontsize=6,
        ha="left",
        va="bottom",
        color=GROUP_INK,
        linespacing=1.1,
    )

    # Legend, above everything (read before the marks).
    handles = [
        (
            Patch(facecolor=BLUE_LIGHT, edgecolor="white"),
            "class of residues (a); light and dark blue only separate neighbouring classes",
        ),
        (
            Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=REF_GREY, mew=0.8),
            "log2(n letters): bits if every letter were equally common (b)",
        ),
        (
            Line2D([], [], ls="", marker="o", ms=2.6, color=PURPLE),
            "measured: from Swiss-Prot composition (b), from aligned pairs (d, e)",
        ),
        (
            Line2D([], [], ls="", marker="D", ms=2.8, mfc="white", mec=PURPLE, mew=0.7),
            "measured: from kmerseek seed counts in the human proteome (b, set just under its row)",
        ),
        (
            Patch(facecolor=CORAL),
            "k*: seed length at which one chance match is expected in Swiss-Prot (c)",
        ),
        (
            Line2D([], [], color=PURPLE, lw=0.8),
            "95% interval: bootstrap over pairs (d, narrower than the dot), Wilson (e)",
        ),
    ]
    fig.legend(
        [h for h, _ in handles],
        [l for _, l in handles],
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(COLS["a"][0] / W_MM - 0.12, 1 - 0.6 / h_mm),
        frameon=False,
        fontsize=6,
        handlelength=1.4,
        columnspacing=1.5,
        borderaxespad=0,
    )

    assert_no_text_collisions(fig)
    pf.save(fig, stem)
    print(f"wrote {stem}.pdf/.png/.svg ({W_MM} x {h_mm:.0f} mm)")


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


def write_caption(t: pl.DataFrame, meta: dict) -> None:
    """Caption and legend text, formatted from the table; each sentence's claim is checked first."""
    row = {r["alphabet"]: r for r in t.iter_rows(named=True)}
    small = t.filter(pl.col("n_letters") <= 3)
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    pct2 = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    top = {
        ds: t.sort(f"{ds}_share_reach_kstar", descending=True).row(0, named=True)
        for ds in ("pfam", "scope")
    }
    x_pfam = top["pfam"]["pfam_share_reach_kstar"]
    rho = t.select(pl.corr("pfam_kappa", "scope_kappa", method="spearman")).item()
    rho_e = t.select(
        pl.corr("pfam_share_reach_kstar", "scope_share_reach_kstar", method="spearman")
    ).item()
    rho_nk = t.select(pl.corr("n_letters", "pfam_kappa", method="spearman")).item()
    rho_nb = t.select(pl.corr("n_letters", "bits_swissprot", method="spearman")).item()
    rho_nks = t.select(pl.corr("n_letters", "kstar", method="spearman")).item()

    check(
        bool((t["bits_human_seed_counts"] < t["bits_swissprot"]).all()),
        "human B not below Swiss-Prot B everywhere",
    )
    check(
        bool((t["bits_swissprot"] < t["bits_log2_n"]).all()),
        "Swiss-Prot B not below log2(n) everywhere",
    )
    check(
        rho_nk < 0 and rho_nb > 0 and rho_nks < 0,
        "letter count does not order kappa, bits and k* as the title says",
    )
    check(
        small["pfam_kappa"].min() > row["protein20"]["pfam_kappa"],
        "a 2-3 letter alphabet has kappa below protein20",
    )
    check(
        row["gbmr7"]["pfam_kappa"] < row["sdm12"]["pfam_kappa"],
        "gbmr7 is no longer below sdm12",
    )
    for ds in ("pfam", "scope"):
        check(
            float(t[f"{ds}_share_reach_kstar"].max()) < 0.06,
            f"{ds}: an alphabet reaches 6% or more",
        )

    hp = small.sort("pfam_share_reach_kstar")
    rho_nk, rho_nb, rho_nks = (
        f"{v:.2f}".replace("-", "−") for v in (rho_nk, rho_nb, rho_nks)
    )
    text = f"""**Supplementary Figure 1 | Fewer letters keep more of each residue's class between related proteins but carry fewer bits per letter, so every alphabet needs a longer seed; at its own seed length, no alphabet reaches more than {pct(x_pfam)} of 20–30% identity pairs.**

One row per kmerseek alphabet (19), grouped by letter count; the rows are in the same order in every column. Classes are those of kmerseek {KMERSEEK_TAG} `src/rust/alphabets.rs` (dayhoff6 is encoded by sourmash). Across the 19 alphabets, fewer letters go with higher κ (Spearman ρ = {rho_nk} between letter count and κ), fewer bits per letter (ρ = {rho_nb}) and a longer k* (ρ = {rho_nks}).

**a**, Classes of each alphabet, labelled with their residues. Segment width is the class's share of the {meta["n_res"]:,} standard residues in UniProtKB/Swiss-Prot release 2026_03. The residues of a class too narrow for its letters are printed above or below its segment. The two shades of blue only separate neighbouring classes.

**b**, Bits of information in one matching letter. Open grey circle: log2 of the number of letters, the value if all letters were equally common. Filled purple circle: B = −log2(Σ q²), where q is each class's share of Swiss-Prot residues, so Σ q² is the chance that two residues drawn at random fall in the same class. Purple diamond, set just under its row: B measured from kmerseek seed counts in the human proteome. P(k), the mean number of proteins sharing a seed of k letters, falls by a factor 2^B for each added letter, so B = [log2(P(k_small) − 1) − log2(P(k_main) − 1)] / (k_main − k_small) (notebook 250). In all 19 alphabets the human value is below the Swiss-Prot value, and both are below log2 of the letter count. For hp_thomas_dill2 the three values are {row["hp_thomas_dill2"]["bits_log2_n"]:.3f}, {row["hp_thomas_dill2"]["bits_swissprot"]:.3f} and {row["hp_thomas_dill2"]["bits_human_seed_counts"]:.3f} bits.

**c**, k* = ⌈log2 N / B⌉, with N = {meta["n_res"]:,} residues and B from Swiss-Prot composition (b, filled circle): the seed length at which one match by chance is expected across Swiss-Prot. k* runs from {row["protein20"]["kstar"]} (protein20) to {int(t["kstar"].max())} ({t.sort("kstar").row(-1, named=True)["alphabet"]}).

**d**, Cohen's κ for the classes of aligned residues: (observed agreement − agreement expected from the two sequences' class compositions) / (1 − expected). κ = 0 means no more agreement than chance; higher means the class is kept more often. Pairs from the same Pfam-A 38.2 seed alignment with 20–30% identity over at least {MIN_COLS} aligned columns ({row["protein20"]["pfam_n_pairs"]:,} pairs from {meta["pfam_n_families"]:,} families). Mean over pairs with a bootstrap 95% interval (500 resamples; the interval is narrower than the dot). κ is {row["protein20"]["pfam_kappa"]:.2f} for protein20 and {small["pfam_kappa"].min():.2f}–{small["pfam_kappa"].max():.2f} for the 2–3 letter alphabets. gbmr7 ({row["gbmr7"]["pfam_kappa"]:.2f}) is below sdm12 ({row["sdm12"]["pfam_kappa"]:.2f}) although it has fewer letters.

**e**, Share of the same pairs whose longest class-identical run is at least the alphabet's own k* (c). A class-identical run is a stretch of consecutive aligned columns with no gap in either sequence and the same class in both: the longest exact shared seed the pair has in that alphabet. Wilson 95% interval. The highest is {top["pfam"]["alphabet"]}, {pct2(x_pfam)} ({top["pfam"]["pfam_n_reach_kstar"]:,} pairs); protein20 {pct2(row["protein20"]["pfam_share_reach_kstar"])}; the 2–3 letter alphabets {pct2(hp["pfam_share_reach_kstar"][0])} ({hp["alphabet"][0]}) to {pct2(hp["pfam_share_reach_kstar"][-1])} ({hp["alphabet"][-1]}); gbmr7 {pct2(row["gbmr7"]["pfam_share_reach_kstar"])}.

**Second version (`suppfig1_scope`), a check on the alignment source.** Columns a–c are unchanged. Columns d and e use SCOPe 2.08 domain pairs (40% identity set) in the same superfamily, from the same or a different family, aligned by structure with USalign (TM-score ≥ {MIN_TM} for at least one of the two domains), 20–30% identity over at least {MIN_COLS} columns: {row["protein20"]["scope_n_pairs"]:,} pairs, {meta["scope_n_superfamily_cross_family"]:,} of them from different families. The order of the alphabets is close to the Pfam one (Spearman ρ = {rho:.2f} for κ, {rho_e:.2f} for e). The highest share in e is {top["scope"]["alphabet"]}, {pct2(top["scope"]["scope_share_reach_kstar"])} ({top["scope"]["scope_n_reach_kstar"]} pairs); hp_thomas_dill2 {pct2(row["hp_thomas_dill2"]["scope_share_reach_kstar"])} ({row["hp_thomas_dill2"]["scope_n_reach_kstar"]} pairs). κ for hp_thomas_dill2 is {row["hp_thomas_dill2"]["scope_kappa"]:.2f} and for protein20 {row["protein20"]["scope_kappa"]:.2f}.

All numbers: `tables/suppfig1_values.csv`, written with the figure by `scripts/plot_suppfig1_alphabets.py`.
"""
    CAPTION.write_text(text)
    print(f"wrote {CAPTION}")


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
    draw(t, meta, "pfam", FIG / "suppfig1")
    draw(t, meta, "scope", FIG / "suppfig1_scope")
    write_caption(t, meta)


if __name__ == "__main__":
    main()
