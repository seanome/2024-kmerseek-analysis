"""Computations and figures for notebook 245 (the bits ceiling on an exact reduced-alphabet seed).

Nothing here searches a database. Every input number is in tables/245_*.tsv, written by
scripts/make_245_inputs.py from the executed notebooks that measured it. What this module
adds is arithmetic on those numbers and the figures.

Colours carry one meaning each, in every figure of the notebook:
  HAVE (blue)  bits the BCL2/CED-9 pair has, or could have at most
  NEED (red)   bits a search of the database needs before a hit stands out from chance
Other quantities (conservation, reach, the chance score) are drawn in black and grey.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIG = ROOT / "figures"

HAVE = "#2c6fbb"
NEED = "#c23b22"
INK = "#222222"
GREY = "#8c8c8c"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

# The two hydrophobic/polar partitions used here, copied from kmerseek src/rust/alphabets.rs
# (LEHNINGER_HP and THOMAS_DILL_HP). Letters listed are the hydrophobic class; every other
# residue is polar.
HYDROPHOBIC = {
    "hp_lehninger2": set("AFGILMPVWY"),
    "hp_thomas_dill2": set("ACFILMVWY"),
}

# BH1 window: BCL2 130-166 against CED-9 154-190, one gapless diagonal (kmerseek docs Part 2).
BH1 = {"bcl2_start": 130, "ced9_start": 154, "length": 37}


# ------------------------------------------------------------------ inputs -------------
def inputs() -> dict[str, float]:
    t = pl.read_csv(TABLES / "245_inputs.tsv", separator="\t")
    return dict(zip(t["quantity"], t["value"]))


def inputs_table() -> pl.DataFrame:
    return pl.read_csv(TABLES / "245_inputs.tsv", separator="\t")


def sequences() -> dict[str, str]:
    seqs, name = {}, None
    for line in (TABLES / "245_bcl2_ced9.fasta").read_text().splitlines():
        if line.startswith(">"):
            name = re.search(r"GN=(\S+)", line).group(1).upper().replace("-", "")
            seqs[name] = ""
        else:
            seqs[name] += line.strip()
    return seqs  # keys BCL2, CED9


def chance_scores() -> pl.DataFrame:
    return pl.read_csv(TABLES / "245_chance_score_by_alphabet.tsv", separator="\t")


def reach_by_k() -> pl.DataFrame:
    return pl.read_csv(TABLES / "245_reach_by_k.tsv", separator="\t")


# ------------------------------------------------------------------ the BH1 block ------
def bh1_block() -> dict:
    """Residues, classes and match lines of the BH1 window, all counted here."""
    s = sequences()
    L = BH1["length"]
    q = s["BCL2"][BH1["bcl2_start"] - 1: BH1["bcl2_start"] - 1 + L]
    t = s["CED9"][BH1["ced9_start"] - 1: BH1["ced9_start"] - 1 + L]
    out = {"bcl2": q, "ced9": t, "length": L,
           "identity_line": "".join("|" if a == b else " " for a, b in zip(q, t)),
           "identical": sum(a == b for a, b in zip(q, t)),
           "ced9_length": len(s["CED9"]), "bcl2_length": len(s["BCL2"])}
    for name, h in HYDROPHOBIC.items():
        cq = "".join("h" if c in h else "p" for c in q)
        ct = "".join("h" if c in h else "p" for c in t)
        line = "".join("|" if a == b else " " for a, b in zip(cq, ct))
        runs = [(m.start(), m.end()) for m in re.finditer(r"\|+", line)]
        a, b = max(runs, key=lambda r: r[1] - r[0])
        out[name] = {"bcl2": cq, "ced9": ct, "line": line, "same_class": line.count("|"),
                     "run_start": a, "run_length": b - a}
    return out


def print_bh1(block: dict) -> None:
    """Fixed-width alignment with coordinates, the way the paper shows it."""
    b0, c0, L = BH1["bcl2_start"], BH1["ced9_start"], block["length"]
    w = 16
    print(f"{'BCL2':<{w}}{b0:>4} {block['bcl2']} {b0 + L - 1}")
    print(f"{'':<{w}}     {block['identity_line']}   {block['identical']} of {L} identical")
    print(f"{'CED-9':<{w}}{c0:>4} {block['ced9']} {c0 + L - 1}")
    for name in HYDROPHOBIC:
        a = block[name]
        s, n = a["run_start"], a["run_length"]
        print()
        print(f"{name} classes (h = {''.join(sorted(HYDROPHOBIC[name]))}, p = the rest)")
        print(f"{'BCL2':<{w}}{b0:>4} {a['bcl2']} {b0 + L - 1}")
        print(f"{'':<{w}}     {a['line']}   {a['same_class']} of {L} same class")
        print(f"{'CED-9':<{w}}{c0:>4} {a['ced9']} {c0 + L - 1}")
        print(f"{'':<{w}}     {' ' * s}{'^' * n}   longest class-identical run: {n}, "
              f"BCL2 {b0 + s}-{b0 + s + n - 1}, CED-9 {c0 + s}-{c0 + s + n - 1}")


def usalign_agreement() -> tuple[int, int]:
    """How many BH1 positions the USalign structure alignment puts on the same diagonal."""
    p = pl.read_csv(TABLES / "245_bcl2_ced9_usalign_pairs.tsv", separator="\t")
    off = BH1["ced9_start"] - BH1["bcl2_start"]
    w = p.filter(pl.col("bcl2_pos").is_between(BH1["bcl2_start"],
                                               BH1["bcl2_start"] + BH1["length"] - 1))
    return int((w["ced9_pos"] - w["bcl2_pos"] == off).sum()), BH1["length"]


# ------------------------------------------------------------------ information --------
def mutual_information(joint: np.ndarray) -> float:
    """Bits of mutual information in a joint table of two aligned class labels."""
    joint = joint / joint.sum()
    fq, ft = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (fq @ ft)[nz])).sum())


def copy_rate_joint(shares: np.ndarray, kappa: float) -> np.ndarray:
    """Pr(query class i, target class j) under the copy-rate model of kmerseek docs Part 2.

    With probability kappa the target copies the query's class; otherwise it draws its
    class from the same shares, independently of the query.
    """
    f = np.asarray(shares, float)
    return kappa * np.diag(f) + (1 - kappa) * np.outer(f, f)


def two_letter_shares(pr_match_unrelated: float) -> np.ndarray:
    """Class shares (h, p) of a two-letter alphabet from its chance match rate h^2 + p^2."""
    h = 0.5 - math.sqrt(max(0.0, (pr_match_unrelated - 0.5) / 2))
    return np.array([h, 1 - h])


def bits_needed_region(K: float, m: float, n: float, E: float = 1.0) -> float:
    """log2(K m n / E): bits a scored region needs for an E-value of E (Karlin-Altschul)."""
    return math.log2(K * m * n / E)


def bits_needed_seed(m: float, n: float) -> float:
    """log2(m n): bits an exact seed needs to be expected once by chance in the search."""
    return math.log2(m * n)


def seed_bits(k: int, pr_match_unrelated: float) -> float:
    """-log2 Pr(an exact k-residue class match by chance), positions independent."""
    return -k * math.log2(pr_match_unrelated)


# ------------------------------------------------------------------ figures ------------
def _legend_top(ax, handles, ncol=2):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol,
              frameon=False, fontsize=9, handlelength=1.6, borderaxespad=0)


def fig_bits_have_vs_need(v: dict, path: Path) -> None:
    """Figure 1: one bar per quantity; blue = what the pair has, red = what is needed."""
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), gridspec_kw={"height_ratios": [3, 2],
                                                                     "hspace": 0.9})
    panels = [
        (axes[0], "a  Evidence in the 37-residue BH1 block, against the cost of E = 1",
         [(v["label_ceiling20"], v["bits_block_20"], HAVE),
          (v["label_coin2"], v["bits_block_hp"], HAVE),
          (v["label_need_region"], v["bits_needed_region"], NEED)]),
        (axes[1], "b  Rarity of the longest exact seed, against one chance seed per search",
         [(v["label_seed_have"], v["seed_bits_bh1"], HAVE),
          (v["label_need_seed"], v["bits_needed_seed"], NEED)]),
    ]
    xmax = max(val for _, _, bars in panels for _, val, _ in bars) * 1.18
    for ax, title, bars in panels:
        y = np.arange(len(bars))[::-1]
        for yi, (lab, val, col) in zip(y, bars):
            ax.barh(yi, val, color=col, height=0.62)
            ax.text(val + 0.4, yi, f"{val:.1f} bits", va="center", fontsize=9)
        ax.set_yticks(y, [b[0] for b in bars], fontsize=9)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=10, pad=6)
        ax.set_xlabel("bits")
    fig.legend(handles=[Patch(color=HAVE, label="bits the BCL2/CED-9 pair has, or can have at most"),
                        Patch(color=NEED, label="bits the human-proteome search needs")],
               loc="lower left", bbox_to_anchor=(0.01, 0.97), ncol=2, frameon=False, fontsize=9)
    fig.savefig(path)
    plt.show()


def fig_scaling(v: dict, dbs: list[dict], path: Path) -> None:
    """Figure 2: bits needed against database size; one extra bit per doubling."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    n = np.logspace(6, 10.3, 50)
    ax.plot(n, np.log2(v["m"] * n), color=NEED, ls="--", lw=1.6,
            label=f"one chance exact seed per search, log$_2$(m·n), m = {v['m']:.0f}")
    for d in dbs:
        ax.plot(d["n"], d["bits"], "o", color=NEED, ms=7)
        ax.annotate(d["name"], (d["n"], d["bits"]), xytext=(6, -12), textcoords="offset points",
                    fontsize=9)
    ax.plot([], [], "o", color=NEED, label="E = 1 for a scored region, log$_2$(K·m·n), K fitted per database")
    ax.axhline(v["bits_block_20"], color=HAVE, lw=1.6,
               label=f"BH1 block, 20-letter ceiling: {v['bits_block_20']:.1f} bits")
    ax.axhline(v["bits_block_hp"], color=HAVE, lw=1.6, ls=":",
               label=f"BH1 block, two letters (copy-rate model): {v['bits_block_hp']:.1f} bits")
    ax.set_xscale("log")
    ax.set_xlabel("residues in the database, n (log scale)")
    ax.set_ylabel("bits")
    ax.set_ylim(0, 45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
              fontsize=8.5, ncol=1, borderaxespad=0)
    fig.savefig(path)
    plt.show()


def fig_conservation_vs_reach(reach: pl.DataFrame, v: dict, path: Path) -> None:
    """Figure 3: per-position conservation does not depend on k; exact-seed reach does."""
    r = reach.filter((pl.col("dataset") == "Pfam") & (pl.col("identity_bin") == "20-30%")
                     & (pl.col("alphabet") == "hp_thomas_dill2")).sort("k")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(r["k"], r["fraction_of_pairs_reachable"], "-o", color=INK, ms=4,
            label="share of pairs whose longest exact class-identical run is at least k")
    ax.axhline(v["kappa_hp_thomas_dill2_pfam_20_30"], color=GREY, ls="--", lw=1.5,
               label=f"Cohen's κ per aligned position, {v['kappa_hp_thomas_dill2_pfam_20_30']:.3f} "
                     "(does not depend on k)")
    ax.axvline(v["longest_run_mean_hp_thomas_dill2_pfam_20_30"], color=GREY, lw=1, ls=":",
               label=f"mean longest exact run, {v['longest_run_mean_hp_thomas_dill2_pfam_20_30']:.1f} residues")
    marks = []
    for k in (23, 26, 30):
        f = r.filter(pl.col("k") == k)["fraction_of_pairs_reachable"][0]
        marks.append(f"k = {k}: {100 * f:.1f}% of pairs")
    ax.text(0.99, 0.30, "\n".join(marks), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK)
    ax.set_xlabel("k, exact seed length (residues)")
    ax.set_ylabel("value (0 to 1)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8.5,
              borderaxespad=0)
    ax.text(0.99, 0.55, "hp_thomas_dill2\nPfam seed pairs, 20-30% identity\n"
            "(n = 37,085 pairs, notebook 230)", transform=ax.transAxes, ha="right", fontsize=8.5,
            color=INK)
    fig.savefig(path)
    plt.show()


def fig_bh1(block: dict, path: Path) -> None:
    """Figure 4: the BH1 residues, identity line and class strings, longest run boxed."""
    L = block["length"]
    rows = [("BCL2", BH1["bcl2_start"], block["bcl2"]),
            ("", None, block["identity_line"]),
            ("CED-9", BH1["ced9_start"], block["ced9"])]
    for name in HYDROPHOBIC:
        a = block[name]
        rows += [(None, None, None),
                 (f"{name}", None, None),
                 ("BCL2", BH1["bcl2_start"], a["bcl2"]),
                 ("", None, a["line"]),
                 ("CED-9", BH1["ced9_start"], a["ced9"])]
    cw, lh = 0.16, 0.26
    fig = plt.figure(figsize=(cw * (L + 18), lh * (len(rows) + 2)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, L + 18)
    ax.set_ylim(-len(rows) - 1.6, 1.2)
    ax.axis("off")
    x0 = 8
    ax.text(0, 0.5, f"BH1: BCL2 {BH1['bcl2_start']}-{BH1['bcl2_start'] + L - 1} against CED-9 "
            f"{BH1['ced9_start']}-{BH1['ced9_start'] + L - 1}, {block['identical']} of {L} residues "
            "identical", fontsize=10, weight="bold", va="center")
    ax.text(0, -0.35, "| = same residue (top block) or same class (class blocks).   "
            "Box = the longest run of consecutive same-class positions.", fontsize=8.5, va="center",
            color=INK)
    y = -1.3
    run_rows = {}
    current = None
    for label, start, s in rows:
        if s is None and label is None:
            y -= 0.5
            continue
        if s is None:
            current = label
            h = "".join(sorted(HYDROPHOBIC[label]))
            ax.text(0, y, f"{label} classes: h = {h}, p = the other residues;  "
                    f"{block[label]['same_class']} of {L} positions same class", fontsize=9,
                    va="center", style="italic")
            y -= 1
            continue
        ax.text(0, y, label, fontsize=9, va="center", family="monospace")
        if start is not None:
            ax.text(x0 - 0.4, y, str(start), fontsize=9, va="center", ha="right", family="monospace")
            ax.text(x0 + L + 0.6, y, str(start + L - 1), fontsize=9, va="center", family="monospace")
            if current:
                run_rows.setdefault(current, []).append(y)
        for i, c in enumerate(s):
            ax.text(x0 + i + 0.5, y, c, fontsize=9.5, va="center", ha="center", family="monospace")
        y -= 1
    for name, ys in run_rows.items():
        a = block[name]
        ax.add_patch(Rectangle((x0 + a["run_start"], min(ys) - 0.5), a["run_length"],
                               max(ys) - min(ys) + 1, fill=False, ec=HAVE, lw=1.4))
        ax.text(x0 + a["run_start"] + a["run_length"] + 0.3, min(ys) - 0.9,
                f"longest class-identical run: {a['run_length']} positions", fontsize=8.5,
                color=HAVE, va="center")
    fig.savefig(path)
    plt.show()


def fig_chance_score(cs: pl.DataFrame, path: Path) -> None:
    """Figure 5: expected score of one chance position per alphabet; lambda needs it below 0."""
    cs = cs.sort("expected_score_chance_position")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y = np.arange(cs.height)
    vals = cs["expected_score_chance_position"].to_numpy()
    at_or_above = vals >= 0
    ax.barh(y[~at_or_above], vals[~at_or_above], color=GREY, height=0.7,
            label="below 0: a chance match loses score as it runs, λ exists")
    ax.barh(y[at_or_above], vals[at_or_above], color=INK, hatch="///", height=0.7,
            label="at or above 0: no λ at any k, so no E-value")
    ax.axvline(0, color=INK, lw=1)
    for yi, (a, val) in enumerate(zip(cs["alphabet"], vals)):
        lab = f"{val:+.4f}" if abs(val) < 0.01 else f"{val:+.2f}"
        if val < -0.35:
            lab += " (bar cut at the left edge)"
        if a == "gbmr4":
            lab += " (bar too short to see; just below 0)"
        ax.text(0.02 if val < 0 else val + 0.02, yi, lab, va="center", fontsize=8)
    ax.set_yticks(y, cs["alphabet"].to_list(), fontsize=8.5)
    ax.set_xlabel("expected score of one position between unrelated sequences,\n"
                  "Pr(match | unrelated) − C · (1 − Pr(match | unrelated))  (points)")
    ax.set_xlim(-0.35, 0.45)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8.5,
              borderaxespad=0)
    fig.savefig(path)
    plt.show()
