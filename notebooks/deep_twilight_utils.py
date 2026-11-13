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
    """The kmerseek arms in assets/arms.tsv, ordered by bits per residue of the alphabet, then k."""
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


# ---- figures for sections 3-5, drawn with the paper style in pubfig.py ----

#: One colour and one marker per family, the same in every section 3-5 figure.
FAMILY_STYLE = {
    "globin": dict(color="#0072B2", marker="o"),
    "lysozyme_lactalbumin": dict(color="#D55E00", marker="s"),
    "cystatin": dict(color="#009E73", marker="^"),
}
FAMILY_NAME = {f: t.replace("/\n", "/ ") for f, t in FAMILY_TITLE.items()}
KMERSEEK_BEST = "kmerseek, best of 150"


def alphabet_size(alphabet: str) -> int:
    """Letters in a reduced alphabet, read from the number its name ends in."""
    digits = "".join(ch for ch in reversed(alphabet) if ch.isdigit())[::-1]
    return int(digits)


def alphabet_rows(alphabets) -> list[tuple[float, str]]:
    """(y position, alphabet) grouped by size, 2-3 / 4-8 / 12-18 / 20, a gap between groups."""

    def group(n):
        return 0 if n <= 3 else 1 if n <= 8 else 2 if n <= 18 else 3

    order = sorted(set(alphabets), key=lambda a: (alphabet_size(a), a))
    rows, y, last = [], 0.0, None
    for a in order:
        g = group(alphabet_size(a))
        if last is not None and g != last:
            y += 0.8
        rows.append((y, a))
        y += 1
        last = g
    return rows


def fig_lowest_identity(lowest: pl.DataFrame, share: pl.DataFrame):
    """a: lowest identity at which each tool places the label correctly, per family.
    b: share of the 150 kmerseek alphabet-k combinations right on each labelled pair."""
    import pubfig as pf

    pf.use_style()
    fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, width_ratios=[1, 1.25])
    tools = [TOOL_TITLE[t] for t in BASELINES] + [KMERSEEK_BEST]
    ypos = {t: i for i, t in enumerate(tools)}
    offset = {"globin": -0.2, "lysozyme_lactalbumin": 0.0, "cystatin": 0.2}
    for f in FAMILIES:
        sub = lowest.filter(pl.col("family") == f)
        a.scatter(
            sub["lowest_identity_correct_pct"],
            [ypos[t] + offset[f] for t in sub["tool"]],
            s=14,
            label=FAMILY_NAME[f],
            **FAMILY_STYLE[f],
        )
        s = share.filter(pl.col("family") == f)
        b.scatter(
            s["needle_identity_pct"],
            100 * s["share_of_150_correct"],
            s=10,
            alpha=0.75,
            linewidths=0,
            **FAMILY_STYLE[f],
        )
    a.set_yticks(range(len(tools)), tools)
    a.invert_yaxis()
    for y in range(len(tools)):
        a.axhline(y, color="#dddddd", lw=0.3, zorder=0)
    a.set_xlim(0, 65)
    a.set_xlabel("Lowest global identity with the label placed correctly (%)")
    b.set_xlim(0, 100)
    b.set_ylim(-3, 103)
    b.set_xlabel("Global identity of the pair (%)")
    b.set_ylabel(
        "kmerseek alphabet-k combinations\nplacing the label correctly (% of 150)"
    )
    pf.shared_legend(fig)
    pf.panel_label(a, "a", dx_pt=-75)
    pf.panel_label(b, "b", dx_pt=-30)
    return fig


def fig_ka_fit_and_placement(
    ext: pl.DataFrame, km_calls: pl.DataFrame, placed: pl.DataFrame
):
    """a: alphabet-k combinations whose index has a Karlin-Altschul fit, per alphabet.
    b: kmerseek calls by where their E-value came from.
    c: chance a correct call is correct by its placement alone, one point per call."""
    import pubfig as pf

    pf.use_style()
    fig, (a, b, c) = pf.figure(
        pf.TWO_COLUMN_MM, 80, ncols=3, width_ratios=[1, 0.45, 1.3]
    )
    per = ext.group_by("alphabet").agg(
        pl.col("has_ka_fit").sum().alias("fit"),
        (~pl.col("has_ka_fit")).sum().alias("nofit"),
    )
    n = {r["alphabet"]: r for r in per.iter_rows(named=True)}
    rows = alphabet_rows(per["alphabet"])
    ys = [y for y, _ in rows]
    fit = [n[al]["fit"] for _, al in rows]
    nofit = [n[al]["nofit"] for _, al in rows]
    a.barh(
        ys, fit, color="#0072B2", height=0.8, label="index has a Karlin-Altschul fit"
    )
    a.barh(
        ys,
        nofit,
        left=fit,
        color="#E69F00",
        height=0.8,
        label="no fit: E-value from the run length",
    )
    a.set_yticks(ys, [f"{al} ({alphabet_size(al)})" for _, al in rows])
    a.invert_yaxis()
    a.set_xlabel("k values searched (n)")
    a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    src = dict(km_calls.group_by("evalue_source").len().iter_rows())
    b.bar(
        [0, 1],
        [src.get("ka", 0), src.get("run", 0)],
        color=["#0072B2", "#E69F00"],
        width=0.7,
    )
    b.set_xticks([0, 1], ["Karlin-\nAltschul", "run\nlength"])
    b.set_ylabel("kmerseek calls (n)")
    b.set_xlabel("E-value source")

    groups = placed.select("family", "label").unique().sort("family", "label").rows()
    rng = np.random.default_rng(0)
    for i, (f, lab) in enumerate(groups):
        v = placed.filter((pl.col("family") == f) & (pl.col("label") == lab))[
            "placement_null"
        ]
        c.scatter(
            v,
            i + rng.uniform(-0.3, 0.3, len(v)),
            s=3,
            alpha=0.35,
            linewidths=0,
            color=FAMILY_STYLE[f]["color"],
            marker=FAMILY_STYLE[f]["marker"],
        )
    c.axvline(0.05, color="#000000", lw=0.6, ls="--", label="chance = 0.05")
    c.set_yticks(range(len(groups)), [f"{FAMILY_NAME[f]}:\n{lab}" for f, lab in groups])
    c.invert_yaxis()
    c.set_xlim(-0.02, 1.02)
    c.set_xlabel(
        "Chance of being correct by placement alone\n(1 = call spans the whole target)"
    )
    pf.shared_legend(fig, ncol=3)
    for ax, letter, dx in [(a, "a", -95), (b, "b", -30), (c, "c", -95)]:
        pf.panel_label(ax, letter, dx_pt=dx)
    return fig


def fig_shared_kmers(pk_long: pl.DataFrame, alphabet: str, k_shown: int):
    """a: longest identical-class run per pair against identity. b: shared k-mers at k_shown."""
    import pubfig as pf

    pf.use_style()
    fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)
    one_k = pk_long.filter(pl.col("ksize") == k_shown)
    for f in FAMILIES:
        s = one_k.filter(pl.col("family") == f)
        kw = dict(s=10, alpha=0.75, linewidths=0, **FAMILY_STYLE[f])
        a.scatter(
            s["needle_identity_pct"],
            s["longest_identical_class_run"],
            label=FAMILY_NAME[f],
            **kw,
        )
        b.scatter(s["needle_identity_pct"], s["n_shared_kmers"], **kw)
    a.axhline(
        k_shown,
        color="#000000",
        lw=0.6,
        ls="--",
        label=f"k = {k_shown}: a shorter run shares no k-mer",
    )
    for ax in (a, b):
        ax.set_xlim(0, 100)
        ax.set_xlabel("Global identity of the pair (%)")
    a.set_ylabel(f"Longest identical-class run (aa)\n{alphabet}")
    b.set_ylabel(f"Shared k-mers (n)\n{alphabet}, k = {k_shown}")
    pf.shared_legend(fig, ncol=4)
    pf.panel_label(a, "a", dx_pt=-35)
    pf.panel_label(b, "b", dx_pt=-35)
    return fig
