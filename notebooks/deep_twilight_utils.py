"""Helpers for notebook 252 (deep-twilight controls).

Reads the tables nextflow-runs/deep-twilight-controls writes, draws the outcome heatmap
and prints the aligned residues of each functional label.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "nextflow-runs" / "deep-twilight-controls" / "results"
LABELS = REPO / "tables" / "deep_twilight_pairs.tsv"
FAMILY_FASTA = (
    REPO
    / "nextflow-runs"
    / "deep-twilight-controls"
    / "assets"
    / "deep_twilight_proteins.fasta"
)
ARMS = REPO / "nextflow-runs" / "deep-twilight-controls" / "assets" / "arms.tsv"

FAMILIES = ["globin", "lysozyme_lactalbumin", "cystatin"]
FAMILY_TITLE = {
    "globin": "Globins",
    "lysozyme_lactalbumin": "Lysozyme /\nalpha-lactalbumin",
    "cystatin": "Cystatins",
}
BASELINES = ["phmmer", "mmseqs2", "foldseek"]
TOOL_TITLE = {
    "phmmer": "phmmer",
    "mmseqs2": "MMseqs2",
    "foldseek": "Foldseek",
    "kmerseek": "kmerseek",
}

#: One colour and one hatch per outcome, and one outcome per colour.
OUTCOMES = ["correct", "not carried", "misplaced", "carried wrongly", "no call"]
OUTCOME_STYLE = {
    "correct": dict(color="#0F6E56", hatch=""),
    "not carried": dict(color="#9FE1CB", hatch=""),
    "misplaced": dict(color="#EF9F27", hatch="////"),
    "carried wrongly": dict(color="#7F77DD", hatch="xxxx"),
    "no call": dict(color="#FFFFFF", hatch=""),
}
OUTCOME_TEXT = {
    "correct": "correct: a labelled query residue lands on the same label in the target",
    "not carried": "correct: the target has no such label; the tool aligns the pair but leaves it off",
    "misplaced": "wrong: labelled query residues land on the target, none on its label",
    "carried wrongly": "wrong: the label is carried onto a target that does not have it",
    "no call": "no call with E <= 1000 covers a labelled query residue",
}

HP_GROUPS = {"hp_pbotc_1st_ed2": ("ACFILMPVWY", "DEGHKNQRST")}


def read_fasta(path: Path = FAMILY_FASTA) -> dict[str, str]:
    seqs, acc = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            acc = line[1:].split()[0].split("|")[1]
            seqs[acc] = ""
        elif acc:
            seqs[acc] += line.strip()
    return seqs


def load_labels() -> pl.DataFrame:
    return pl.read_csv(LABELS, separator="\t", infer_schema_length=0).with_columns(
        pl.col("length").cast(pl.Int64)
    )


def load_arms() -> pl.DataFrame:
    """The 152 kmerseek arms, ordered by bits per residue of the alphabet, then k."""
    arms = pl.read_csv(ARMS, separator="\t")
    per_res = arms.group_by("alphabet").agg(
        (pl.col("bits") / pl.col("ksize")).mean().alias("bits_per_residue")
    )
    return arms.join(per_res, on="alphabet").sort("bits_per_residue", "ksize")


def pair_order(outcomes: pl.DataFrame, family: str) -> pl.DataFrame:
    """Columns of one family's heatmap: (query, target, label), lowest identity first."""
    return (
        outcomes.filter(pl.col("family") == family)
        .select("query", "target", "label", "needle_identity_pct", "target_has_label")
        .unique()
        .sort("needle_identity_pct", "label", "query", "target")
        .with_row_index("col")
    )


def _grid(sub: pl.DataFrame, cols: pl.DataFrame, rows: list, row_key) -> np.ndarray:
    idx = {o: i for i, o in enumerate(OUTCOMES)}
    grid = np.full((len(rows), cols.height), idx["no call"])
    lookup = {
        (r["query"], r["target"], r["label"]): r["col"]
        for r in cols.iter_rows(named=True)
    }
    rpos = {k: i for i, k in enumerate(rows)}
    for r in sub.iter_rows(named=True):
        grid[rpos[row_key(r)], lookup[(r["query"], r["target"], r["label"])]] = idx[
            r["outcome"]
        ]
    return grid


def _paint(ax, grid: np.ndarray):
    for (i, j), v in np.ndenumerate(grid):
        st = OUTCOME_STYLE[OUTCOMES[v]]
        ax.add_patch(
            plt.Rectangle(
                (j, i),
                1,
                1,
                facecolor=st["color"],
                hatch=st["hatch"],
                edgecolor="#FFFFFF" if not st["hatch"] else "#444444",
                linewidth=0.0 if not st["hatch"] else 0.0,
            )
        )
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(grid.shape[0], 0)
    for s in ax.spines.values():
        s.set_color("#888888")
        s.set_linewidth(0.6)


def draw_heatmap(outcomes: pl.DataFrame, arms: pl.DataFrame):
    """Families side by side; within each, phmmer, MMseqs2 and Foldseek as one row each
    and kmerseek as one row per alphabet and k, all sharing the pair axis."""
    arm_rows = list(zip(arms["alphabet"], arms["ksize"]))
    widths = [pair_order(outcomes, f).height for f in FAMILIES]
    fig = plt.figure(figsize=(3 + 0.16 * sum(widths), 17))
    gs = fig.add_gridspec(
        5,
        3,
        width_ratios=widths,
        height_ratios=[3.2, 1, 1, 1, 42],
        hspace=0.12,
        wspace=0.08,
        left=0.14,
        right=0.99,
        top=0.97,
        bottom=0.07,
    )
    handles = [
        Patch(
            facecolor=OUTCOME_STYLE[o]["color"],
            hatch=OUTCOME_STYLE[o]["hatch"],
            edgecolor="#444444",
            label=OUTCOME_TEXT[o],
        )
        for o in OUTCOMES
    ]
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis("off")
    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        handleheight=1.4,
    )
    axes = {}
    for c, fam in enumerate(FAMILIES):
        cols = pair_order(outcomes, fam)
        fsub = outcomes.filter(pl.col("family") == fam)
        for r, tool in enumerate(BASELINES):
            ax = fig.add_subplot(gs[1 + r, c])
            g = _grid(
                fsub.filter(pl.col("tool") == tool), cols, [tool], lambda x: x["tool"]
            )
            _paint(ax, g)
            ax.set_xticks([])
            ax.set_yticks([0.5])
            ax.set_yticklabels([TOOL_TITLE[tool]] if c == 0 else [], fontsize=10)
            if c > 0:
                ax.tick_params(left=False)
            if r == 0:
                ax.set_title(
                    f"{FAMILY_TITLE[fam]}\n{cols.height} query, target, label",
                    fontsize=11,
                )
            axes[(fam, tool)] = ax
        ax = fig.add_subplot(gs[4, c])
        g = _grid(
            fsub.filter(pl.col("tool") == "kmerseek"),
            cols,
            arm_rows,
            lambda x: (x["alphabet"], x["ksize"]),
        )
        _paint(ax, g)
        # one label per alphabet, at the middle of its rows; a line between alphabets
        bounds, names = [], []
        for a in arms["alphabet"].unique(maintain_order=True):
            ix = [i for i, (al, _) in enumerate(arm_rows) if al == a]
            bounds.append(ix[-1] + 1)
            names.append((a, (ix[0] + ix[-1] + 1) / 2))
        for b in bounds[:-1]:
            ax.axhline(b, color="#888888", lw=0.5)
        ax.set_yticks([m for _, m in names])
        ax.set_yticklabels([n for n, _ in names] if c == 0 else [], fontsize=8)
        if c > 0:
            ax.tick_params(left=False)
        ax.set_xticks(np.arange(cols.height) + 0.5)
        ax.set_xticklabels(
            [f"{v:.0f}" for v in cols["needle_identity_pct"]], fontsize=7, rotation=90
        )
        if c == 0:
            ax.set_ylabel(
                "kmerseek, one row per alphabet and k\n(within an alphabet, k rises downward)",
                fontsize=10,
            )
        axes[(fam, "kmerseek")] = ax
    fig.supxlabel(
        "global identity of the pair, % (EMBOSS needle); one column per query, target and label",
        fontsize=11,
        y=0.035,
    )
    return fig, axes


# ---------------------------------------------------------------------------
# Residue printout
# ---------------------------------------------------------------------------
def class_string(
    seq: str, alphabet: str = "hp_pbotc_1st_ed2", letters: str = "HP"
) -> str:
    table = {r: letters[i] for i, g in enumerate(HP_GROUPS[alphabet]) for r in g}
    return "".join(table.get(c, "-" if c == "-" else "?") for c in seq)


def columns(qstart: int, tstart: int, qaln: str, taln: str) -> list[tuple]:
    """(query position or None, query residue, target position or None, target residue)."""
    q, t, out = qstart - 1, tstart - 1, []
    for a, b in zip(qaln, taln):
        q += a != "-"
        t += b != "-"
        out.append((q if a != "-" else None, a, t if b != "-" else None, b))
    return out


def format_label_window(
    call: dict,
    q_label: set[int],
    t_label: set[int],
    q_name: str,
    t_name: str,
    flank: int = 12,
) -> str:
    """The aligned columns around the query's labelled residues, from one call.

    '|' under an identical residue, '*' over a labelled residue on either protein. Then
    the hp_pbotc_1st_ed2 class strings of the same columns with their own match line.
    """
    cols = columns(call["qstart"], call["tstart"], call["qaln"], call["taln"])
    hit = [i for i, c in enumerate(cols) if c[0] in q_label]
    if not hit:
        return "(this call does not cover a labelled query residue)"
    lo, hi = max(0, min(hit) - flank), min(len(cols), max(hit) + flank + 1)
    win = cols[lo:hi]
    qs = "".join(c[1] for c in win)
    ts = "".join(c[3] for c in win)
    match = "".join("|" if a == b and a != "-" else " " for a, b in zip(qs, ts))
    both = [(a, b) for a, b in zip(qs, ts) if a != "-" and b != "-"]
    n_id = sum(a == b for a, b in both)
    qmark = "".join("*" if c[0] in q_label else " " for c in win)
    tmark = "".join("*" if c[2] in t_label else " " for c in win)
    qc, tc = class_string(qs), class_string(ts)
    cmatch = "".join("|" if a == b and a != "-" else " " for a, b in zip(qc, tc))
    n_cid = sum(a == b for a, b in zip(qc, tc) if a != "-" and b != "-")
    q0 = next(c[0] for c in win if c[0] is not None)
    q1 = max(c[0] for c in win if c[0] is not None)
    t_pos = [c[2] for c in win if c[2] is not None]
    t0, t1 = (min(t_pos), max(t_pos)) if t_pos else (None, None)
    w = max(len(q_name), len(t_name)) + 9
    return "\n".join(
        [
            f"{n_id} of {len(both)} aligned residues identical; "
            f"{n_cid} of {len(both)} in the same hp_pbotc_1st_ed2 class",
            f"{'':<{w}}{'':>5} {qmark}   (* labelled query residue)",
            f"{q_name:<{w}}{q0:>5} {qs} {q1}",
            f"{'':<{w}}{'':>5} {match}",
            f"{t_name:<{w}}{t0 or '':>5} {ts} {t1 or ''}",
            f"{'':<{w}}{'':>5} {tmark}   (* labelled target residue)",
            f"{q_name + ' classes':<{w}}{'':>5} {qc}",
            f"{'':<{w}}{'':>5} {cmatch}",
            f"{t_name + ' classes':<{w}}{'':>5} {tc}",
        ]
    )
