"""Helpers for notebook 346: known protein pairs searched against whole databases.

Ranks and decoy rates from the score tables of scripts/346_database_scan.py, the
fixed-width residue blocks of section 1, and the residue panels of Figure 1. The scores
themselves come from linked_pairs_utils.diagonal_scan.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

import linked_pairs_utils as lp

SCORES = ["B2", "linked", "randstrobe", "B3"]
SCORE_LABEL = {
    "B2": "B2: one 8-letter word",
    "linked": "linked key: two 6-letter words",
    "randstrobe": "randstrobe: one partner per word",
    "B3": "B3: 8-letter word + second segment",
}
DECOYS = ["uniform", "dipeptide"]
HYDROPHOBIC = "ACFILMVWY"  # hp_thomas_dill2's H class; every other residue is P

# One colour per meaning, in every figure of notebook 346 (Okabe-Ito, pubfig.OKABE_ITO).
COLOR = {
    "H": "#F0E442",  # hydrophobic residue (square fill)
    "P": "#56B4E9",  # polar residue (square fill)
    "linked": "#D55E00",  # the linked key (its two words, its key counts)
    "B3": "#CC79A7",  # B3 (its two segments, its key counts)
    "B2": "#E69F00",  # B2's 8-letter words
    "real": "#999999",  # real database proteins
    "uniform": "#009E73",  # decoys: uniform residue shuffle
    "dipeptide": "#0072B2",  # decoys: dipeptide-preserving shuffle
    "partner": "#000000",  # the true partner, and identical residues
}


def read_fasta(path) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
                out[name] = []
            elif name is not None:
                out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def classes(seq: str) -> str:
    """hp_thomas_dill2 class string: h for ACFILMVWY, p for the other 11 residues, '-' for a
    gap and '.' for any other letter."""
    return "".join(
        (
            "-"
            if c == "-"
            else (
                "h" if c in HYDROPHOBIC else ("p" if c.isalpha() and c != "X" else ".")
            )
        )
        for c in seq.upper()
    )


def search_table(
    scan: pl.DataFrame, query: str, partner: str, keep=None
) -> pl.DataFrame:
    """One row per score: the true partner's score and rank among the real proteins (ties
    count against it: rank = 1 + proteins other than the partner scoring at or above it),
    the decoy rate under each shuffle (share of decoy proteins scoring at or above the
    partner; a decoy whose dipeptide shuffle failed is left out), the number of real
    proteins at or above it, the partner's key count and the median key count.

    keep: optional list of target names that form the database (the rest are left out).
    """
    d = scan.filter(pl.col("query") == query)
    if keep is not None:
        d = d.filter(pl.col("target").is_in(keep))
    real = d.filter(pl.col("kind") == "real")
    p = real.filter(pl.col("target") == partner)
    if p.height != 1:
        raise ValueError(f"{p.height} rows for partner {partner[:60]!r}")
    others = real.filter(pl.col("target") != partner)
    rows = []
    for s in SCORES:
        ps = int(p[s][0])
        row = {
            "score": s,
            "partner_score": ps,
            "rank": 1 + int((others[s] >= ps).sum()),
            "n_real": real.height,
            "n_real_at_or_above": int((others[s] >= ps).sum()),
        }
        for k in DECOYS:
            dk = d.filter((pl.col("kind") == k) & (pl.col("shuffle_failed") == 0))[s]
            row[f"decoy_rate_{k}"] = float((dk >= ps).mean())
            row[f"n_decoys_{k}"] = dk.len()
        row["partner_keys"] = int(p[f"{s}_count"][0])
        row["median_keys"] = float(real[f"{s}_count"].median())
        rows.append(row)
    return pl.DataFrame(rows)


def longest_run(mask) -> tuple[int, int]:
    """Length and start of the longest run of True."""
    best, start, cur, cur_start = 0, -1, 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, start = cur, cur_start
        else:
            cur = 0
    return best, start


def columns_of(row: str) -> np.ndarray:
    """Column of each residue of a gapped row."""
    return np.flatnonzero(np.frombuffer(row.encode(), np.uint8) != ord("-"))


def residue_marks(qrow, trow, best):
    """Columns of a best key's anchors and segments (dict from best_keys_with_segments, in
    residue indices of the rows' gap-stripped sequences): list of (kind, first column,
    last column). kind is anchor1 or anchor2, then one entry per segment: for a B3 key
    (one anchor of DB_SEED residues, the other of one) "seed segment" holds the 8-letter
    word and "second segment" the other; for a linked key "segment"."""
    if best is None:
        return []
    qc = columns_of(qrow)
    out = [
        ("anchor1", qc[best["q1"]], qc[best["q1"] + best["a1"] - 1]),
        ("anchor2", qc[best["q2"]], qc[best["q2"] + best["a2"] - 1]),
    ]
    b3 = sorted([best["a1"], best["a2"]]) == [1, lp.DB_SEED]
    seed_q = best["q1"] if best["a1"] == lp.DB_SEED else best["q2"]
    for s, e, _ in best["segments"]:
        kind = (
            ("seed segment" if s <= seed_q < e else "second segment")
            if b3
            else "segment"
        )
        out.append((kind, qc[s], qc[e - 1]))
    return out


def print_block(qrow, trow, qname, tname, qstart, tstart, marks=None, width=60):
    """Print an alignment in blocks: residues with numbers, an identity line, the HP class
    strings with a class-match line, a line marking every column inside an exact shared
    6-letter HP word (=), the longest class-identical run (^), and the best linked key's
    anchors (L) and B3's seed segment (S) and second segment (2). Returns the counts printed in the header.
    """
    n = len(qrow)
    qc, tc = classes(qrow), classes(trow)
    both = np.array([a != "-" and b != "-" for a, b in zip(qrow, trow)])
    ident = np.array([a == b and a != "-" for a, b in zip(qrow, trow)])
    same = np.array([a == b and a in "hp" for a, b in zip(qc, tc)])
    run, run_start = longest_run(same)
    in_word = np.zeros(n, bool)
    for i in range(n - 5):
        if same[i : i + 6].all():
            in_word[i : i + 6] = True
    lab = max(len(qname), len(tname), 12)
    marks = marks or {}
    mline = [" "] * n
    for kind, a, b in marks.get("linked", []):
        if kind.startswith("anchor"):
            for c in range(a, b + 1):
                mline[c] = "L"
    bline = [" "] * n
    for kind, a, b in marks.get("B3", []):
        if kind.endswith("segment"):
            for c in range(a, b + 1):
                bline[c] = "S" if kind == "seed segment" else "2"
    qpos, tpos = qstart, tstart
    print(
        f"{int(ident.sum())} of {int(both.sum())} aligned positions identical; "
        f"{int(same.sum())} of {int(both.sum())} the same HP class; "
        f"longest exact shared HP word {run} residues"
    )
    for s in range(0, n, width):
        e = min(n, s + width)
        qseg, tseg = qrow[s:e], trow[s:e]
        qe = qpos + sum(c != "-" for c in qseg) - 1
        te = tpos + sum(c != "-" for c in tseg) - 1
        print(f"{qname:<{lab}} {qpos:>5} {qseg} {qe}")
        print(f"{'':<{lab}} {'':>5} {''.join('|' if x else ' ' for x in ident[s:e])}")
        print(f"{tname:<{lab}} {tpos:>5} {tseg} {te}")
        print(f"{'':<{lab}} {'class':>5} {qc[s:e]}")
        print(f"{'':<{lab}} {'':>5} {''.join('|' if x else ' ' for x in same[s:e])}")
        print(f"{'':<{lab}} {'class':>5} {tc[s:e]}")
        print(
            f"{'6-word':<{lab}} {'':>5} {''.join('=' if x else ' ' for x in in_word[s:e])}"
        )
        run_line = "".join(
            "^" if run_start <= c < run_start + run else " " for c in range(s, e)
        )
        if run_line.strip():
            print(f"{'longest':<{lab}} {'':>5} {run_line}")
        if "".join(mline[s:e]).strip():
            print(f"{'linked':<{lab}} {'':>5} {''.join(mline[s:e])}")
        if "".join(bline[s:e]).strip():
            print(f"{'B3':<{lab}} {'':>5} {''.join(bline[s:e])}")
        print()
        qpos, tpos = qe + 1, te + 1
    rq = [qstart + sum(c != "-" for c in qrow[:run_start])] if run else [None]
    rt = [tstart + sum(c != "-" for c in trow[:run_start])] if run else [None]
    if run:
        print(
            f"longest class-identical run: {run} residues, {qname} {rq[0]}-{rq[0] + run - 1}, "
            f"{tname} {rt[0]}-{rt[0] + run - 1}"
        )
    return {
        "identical": int(ident.sum()),
        "aligned": int(both.sum()),
        "same_class": int(same.sum()),
        "longest_run": run,
    }


def key_rows(query: str, target: str, best: dict, flank: int = 8):
    """The local alignment a best key implies between two whole sequences (no alignment
    given): flank residues before the first segment, the first segment on its diagonal,
    the residues between the segments with the indel placed right after the first segment,
    the second segment, flank residues after. Returns (query row, target row, query start
    and target start, 1-based) and the key re-indexed to the rows' gap-stripped sequences.
    """
    segs = best["segments"]
    qs1, qe1, ts1 = segs[0]
    qs2, qe2, ts2 = segs[-1]
    te1 = ts1 + (qe1 - qs1)
    f0 = min(flank, qs1, ts1)
    te2 = ts2 + (qe2 - qs2)
    f1 = min(flank, len(query) - qe2, len(target) - te2)
    qrow, trow = [query[qs1 - f0 : qe1]], [target[ts1 - f0 : te1]]
    if len(segs) == 2:
        mq, mt = query[qe1:qs2], target[te1:ts2]
        k = min(len(mq), len(mt))
        # pair the residues between the segments from the right end, the extra ones of the
        # longer side become gap columns right after the first segment
        qrow.append("-" * (len(mt) - k) + mq)
        trow.append("-" * (len(mq) - k) + mt)
        qrow.append(query[qs2:qe2])
        trow.append(target[ts2:te2])
    qrow.append(query[qe2 : qe2 + f1])
    trow.append(target[te2 : te2 + f1])
    q0, t0 = qs1 - f0, ts1 - f0
    shifted = dict(best)
    shifted.update(
        q1=best["q1"] - q0,
        q2=best["q2"] - q0,
        t1=best["t1"] - t0,
        t2=best["t2"] - t0,
        segments=[(s - q0, e - q0, t - t0) for s, e, t in best["segments"]],
    )
    return "".join(qrow), "".join(trow), q0 + 1, t0 + 1, shifted


def draw_pair(
    ax, qrow, trow, qname, tname, qstart, tstart, linked=None, b3=None, title=""
):
    """Residue panel: the two sequences as rows of squares coloured by HP class, identical
    residues outlined in black, exact shared HP words of 6 or more as black bars, the best
    linked key's two words as vermillion bars joined by a bracket labelled with both gaps,
    and B3's two segments as reddish-purple bars (seed segment filled, second segment open).
    linked, b3: marks from residue_marks."""
    from matplotlib.patches import Rectangle

    n = len(qrow)
    qc, tc = classes(qrow), classes(trow)
    for row, cls, y, label, start in [
        (qrow, qc, 1.0, qname, qstart),
        (trow, tc, 0.0, tname, tstart),
    ]:
        for x, (aa, c) in enumerate(zip(row, cls)):
            if aa == "-":
                continue
            ident = aa != "-" and qrow[x] == trow[x]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    0.92,
                    0.92,
                    facecolor=(
                        COLOR["H"]
                        if c == "h"
                        else (COLOR["P"] if c == "p" else "white")
                    ),
                    edgecolor=COLOR["partner"] if ident else "none",
                    linewidth=0.8 if ident else 0,
                )
            )
            ax.text(
                x + 0.46,
                y + 0.42,
                aa,
                ha="center",
                va="center",
                family="monospace",
                fontsize=5,
            )
        ax.text(
            -0.8, y + 0.46, f"{label} {start}", ha="right", va="center", fontsize=5.5
        )
    same = np.array([a == b and a in "hp" for a, b in zip(qc, tc)])
    x = 0
    while x < n:
        if same[x]:
            e = x
            while e < n and same[e]:
                e += 1
            if e - x >= 6:
                ax.plot(
                    [x, e - 0.08],
                    [-0.35, -0.35],
                    color="black",
                    lw=1.4,
                    solid_capstyle="butt",
                )
            x = e
        else:
            x += 1
    if linked:
        a = {k: (s, e) for k, s, e in linked}
        for k in ["anchor1", "anchor2"]:
            s, e = a[k]
            ax.plot(
                [s, e + 0.92],
                [2.25, 2.25],
                color=COLOR["linked"],
                lw=2.2,
                solid_capstyle="butt",
            )
        (s1, e1), (s2, e2) = a["anchor1"], a["anchor2"]
        ax.plot(
            [e1 + 0.92, e1 + 0.92, s2, s2],
            [2.25, 2.7, 2.7, 2.25],
            color=COLOR["linked"],
            lw=0.6,
        )
        gq = sum(c != "-" for c in qrow[e1 + 1 : s2])
        gt = sum(c != "-" for c in trow[e1 + 1 : s2])
        gap_text = f"gap {gq}" if gq == gt else f"gap {gq} in {qname}, {gt} in {tname}"
        ax.text(
            (e1 + s2) / 2 + 0.46,
            2.85,
            gap_text,
            ha="center",
            va="bottom",
            fontsize=5,
            color=COLOR["linked"],
        )
    if b3:
        for k, s, e in b3:
            if not k.endswith("segment"):
                continue
            seed = k == "seed segment"
            ax.add_patch(
                Rectangle(
                    (s, -0.95),
                    e - s + 0.92,
                    0.3,
                    facecolor=COLOR["B3"] if seed else "white",
                    edgecolor=COLOR["B3"],
                    linewidth=0.8,
                )
            )
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-1.2, 3.4)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=6)
