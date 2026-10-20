"""Draw SF1-SF3 for notebook 351 from results/boundary_errors.csv.

SF1  cumulative share of boundaries within x residues, one line per method
SF2  20 random instances: the protein as a line, the true feature as a box, and each
     method's scored placement as a bar under it
SF3  boundary error against the length of the true feature

Panel a of SF1 and SF3 holds the preregistered runs; panel b holds the extended kmerseek
runs, which were not preregistered (report.md, change 2).

Usage: python make_figures.py <analysis dir>
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks"))
import pubfig as pf  # noqa: E402

WITHIN = 10
SEED = 351
X_MAX = 100  # residues shown on the error axes
WINDOW = 60
MIN_N = 10  # SF3 shows a median only for feature lengths with at least this many instances  # residues shown either side of the true feature in SF2

# One colour per method and setting, the same in every figure. An extended kmerseek run
# keeps its setting's colour; it is drawn in its own panel, never next to the default run.
COLOURS = {
    "protein20_k5": pf.OKABE_ITO["blue"],
    "dayhoff6_k7": pf.OKABE_ITO["bluish_green"],
    "hp_lehninger2_k12": pf.OKABE_ITO["reddish_purple"],
    "pfam_hmmscan": pf.OKABE_ITO["vermillion"],
    "blastp": pf.OKABE_ITO["orange"],
}
LABELS = {
    "protein20_k5": "kmerseek protein20 k=5",
    "dayhoff6_k7": "kmerseek dayhoff6 k=7",
    "hp_lehninger2_k12": "kmerseek hp_lehninger2 k=12",
    "pfam_hmmscan": "hmmscan, Pfam-A envelope",
    "blastp": "blastp, best HSP per target",
}
KMERSEEK = ["protein20_k5", "dayhoff6_k7", "hp_lehninger2_k12"]
OTHERS = ["pfam_hmmscan", "blastp"]
EXT = "_extend2"


def base(setting):
    return setting[: -len(EXT)] if setting.endswith(EXT) else setting


def label(setting):
    return LABELS[base(setting)] + (", extended" if setting.endswith(EXT) else "")


def short(setting):
    """Track name on SF2's left axis."""
    b = base(setting)
    name = {
        "protein20_k5": "protein20 k=5",
        "dayhoff6_k7": "dayhoff6 k=7",
        "hp_lehninger2_k12": "hp_lehninger2 k=12",
        "pfam_hmmscan": "hmmscan Pfam-A",
        "blastp": "blastp",
    }[b]
    if b in KMERSEEK:
        name = "kmerseek " + name + (", extended" if setting.endswith(EXT) else "")
    return name


def pooled(df):
    return np.concatenate(
        [df["start_error"].to_numpy(float), df["end_error"].to_numpy(float)]
    )


def cdf_panel(ax, err, ur, settings):
    x = np.arange(0, X_MAX + 1)
    e = pooled(ur)
    ax.plot(
        x,
        [(e <= v).mean() for v in x],
        color=pf.GREY,
        label="random placement in the protein",
    )
    for s in settings:
        sub = err[err["setting"] == s]
        if sub.empty:
            continue
        e = pooled(sub)
        none = " (placed nothing; line at 0)" if not np.isfinite(e).any() else ""
        ax.plot(
            x,
            [(e <= v).mean() for v in x],
            color=COLOURS[base(s)],
            label=label(s) + none,
        )
    ax.axvline(WITHIN, color="black", lw=0.5, ls=":", label="10 residues")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.03, 1)  # a line at 0 stays above the axis, not hidden in it
    ax.set_xlabel("Absolute boundary error (residues)")
    ax.set_ylabel("Share of starts and ends\nwithin this error")


def sf1(err, ur, out):
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2, layout=None)
    cdf_panel(axs[0], err, ur, KMERSEEK + OTHERS)
    cdf_panel(axs[1], err, ur, [s + EXT for s in KMERSEEK] + OTHERS)
    for ax, letter, text in zip(
        axs,
        "ab",
        ["preregistered settings", "kmerseek with extension (not preregistered)"],
    ):
        pf.panel_label(ax, letter)
        ax.text(
            0.0,
            1.04,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            weight="bold",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.5), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.66, wspace=0.3)
    pf.save(fig, str(out / "SF1_boundary_error_cdf_elm_swissprot"))


def sf2(err, inst, lengths, out):
    rng = np.random.default_rng(SEED)
    pick = inst.iloc[
        np.sort(rng.choice(len(inst), size=min(20, len(inst)), replace=False))
    ]
    placed_any = set(err.loc[err["placed_start"].notna(), "setting"])
    # A run that placed nothing on any protein would be a track of grey dots; it is left out
    # and named in the text line instead.
    empty = [
        s
        for s in KMERSEEK + [k + EXT for k in KMERSEEK] + OTHERS
        if s in set(err["setting"]) and s not in placed_any
    ]
    tracks = [s for k in KMERSEEK for s in (k, k + EXT) if s in placed_any] + [
        s for s in OTHERS if s in placed_any
    ]
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 170, nrows=5, ncols=4, layout=None)
    axs = np.asarray(axs).ravel()
    for ax, r in zip(axs, pick.itertuples()):
        L = lengths[r.protein]
        lo, hi = max(1, r.start - WINDOW), min(L, r.end + WINDOW)
        y0 = len(tracks)
        ax.plot([lo, hi], [y0, y0], color="black", lw=0.8)
        ax.add_patch(
            plt.Rectangle(
                (r.start - 0.5, y0 - 0.35), r.end - r.start + 1, 0.7, color="black"
            )
        )
        for i, s in enumerate(tracks):
            y = y0 - 1 - i
            row = err[(err["setting"] == s) & (err["instance_id"] == r.instance_id)]
            if row.empty or pd.isna(row["placed_start"].iloc[0]):
                ax.plot([lo], [y], marker=".", color=pf.GREY, ms=2)
                continue
            a, b = int(row["placed_start"].iloc[0]), int(row["placed_end"].iloc[0])
            c = COLOURS[base(s)]
            # Filled box: kmerseek with extension, or another tool. Open box: kmerseek default.
            filled = not (base(s) in KMERSEEK and not s.endswith(EXT))
            ax.add_patch(
                plt.Rectangle(
                    (max(a, lo) - 0.5, y - 0.3),
                    min(b, hi) - max(a, lo) + 1,
                    0.6,
                    facecolor=c if filled else "white",
                    edgecolor=c,
                    lw=0.6,
                )
            )
            if a < lo:
                ax.plot([lo], [y], marker="<", color="black", ms=2)
            if b > hi:
                ax.plot([hi], [y], marker=">", color="black", ms=2)
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_ylim(-0.8, y0 + 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if ax in axs[::4]:
            ax.set_yticks(
                range(y0, -1, -1), ["true ELM feature"] + [short(t) for t in tracks]
            )
        ax.set_title(
            f"{r.protein} {r.elm_class}\nfeature {r.start}-{r.end}, protein {L} aa",
            loc="left",
        )
    for ax in axs[len(pick) :]:
        ax.set_visible(False)
    for ax in axs[16:20]:
        ax.set_xlabel("Position in protein (residue)")
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color="black",
            label="protein (line) and true ELM feature (box)",
        )
    ]
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=COLOURS[b], label=LABELS[b])
        for b in KMERSEEK + OTHERS
        if b in {base(t) for t in tracks}
    ]
    if any(t in KMERSEEK for t in tracks):
        handles += [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                lw=0.6,
                label="open box: kmerseek default search (preregistered)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color="black",
                label="filled box: kmerseek with extension (not preregistered), or another tool",
            ),
        ]
    handles += [
        plt.Line2D(
            [],
            [],
            ls="",
            marker=".",
            color=pf.GREY,
            label="no placement passing E <= 0.01",
        ),
        plt.Line2D(
            [],
            [],
            ls="",
            marker=">",
            color="black",
            label="placement continues past the window",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.998),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.01,
        0.885,
        f"Each panel shows {WINDOW} residues either side of the true feature. "
        "Placement = the reported match nearest the true feature, E <= 0.01."
        + (
            f" Not drawn, as they placed nothing on any protein: {', '.join(label(e) for e in empty)}."
            if empty
            else ""
        ),
        wrap=True,
    )
    fig.subplots_adjust(
        top=0.84, hspace=0.9, wspace=0.12, left=0.13, right=0.99, bottom=0.04
    )
    pf.save(fig, str(out / "SF2_placements_20_random_elm_instances_swissprot"))


def length_panel(ax, err, settings):
    for i, s in enumerate(settings):
        sub = err[err["setting"] == s].copy()
        if sub.empty:
            continue
        sub["err"] = sub[["start_error", "end_error"]].max(axis=1)
        g = sub.groupby("true_length")["err"]
        med = g.median()[g.size() >= MIN_N]
        c = COLOURS[base(s)]
        # Lengths are consecutive integers; a missing or clipped length breaks the line.
        full = (
            med.reindex(range(int(med.index.min()), int(med.index.max()) + 1))
            if len(med)
            else med
        )
        ax.plot(
            full.index,
            full.where(full < X_MAX).values,
            color=c,
            marker="o",
            ms=2,
            label=label(s),
        )
        top = med[med >= X_MAX]
        # each method's row of triangles sits a little higher, so none hides another
        ax.plot(top.index, [X_MAX + 3 * i] * len(top), ls="", marker="^", color=c, ms=3)
    ax.set_ylim(0, X_MAX + 3 * len(settings) + 2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.set_xlabel("Length of the true ELM feature (residues)")
    ax.set_ylabel("Median of the larger of start\nand end error (residues)")


def sf3(err, inst, out):
    fig, axs = pf.figure(
        pf.TWO_COLUMN_MM,
        95,
        nrows=2,
        ncols=2,
        layout=None,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 3]},
    )
    counts = inst["length"].value_counts().sort_index()
    for col, settings, text in [
        (0, KMERSEEK + OTHERS, "preregistered settings"),
        (
            1,
            [s + EXT for s in KMERSEEK] + OTHERS,
            "kmerseek with extension (not preregistered)",
        ),
    ]:
        top, ax = axs[0][col], axs[1][col]
        top.bar(counts.index, counts.values, color=pf.GREY, width=0.8)
        top.set_ylabel("ELM instances\n(n)")
        length_panel(ax, err, settings)
        ax.plot(
            [],
            [],
            ls="",
            marker="^",
            color="black",
            ms=3,
            label="median 100 or more,\nor unplaced (a row per method)",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.85), ncol=2, frameon=False)
        top.text(
            0.0,
            1.06,
            text,
            transform=top.transAxes,
            ha="left",
            va="bottom",
            weight="bold",
        )
        top.text(
            0.99,
            0.95,
            f"medians below: lengths with\n{MIN_N} or more instances only",
            transform=top.transAxes,
            ha="right",
            va="top",
        )
        ax.set_xlim(0, counts[counts >= MIN_N].index.max() + 1)
        pf.panel_label(top, "ab"[col])
    fig.subplots_adjust(top=0.74, hspace=0.15, wspace=0.35, right=0.92)
    pf.save(fig, str(out / "SF3_boundary_error_by_feature_length_elm_swissprot"))


def read_lengths(path):
    lengths, name = {}, None
    for line in open(path):
        if line.startswith(">"):
            name = line[1:].split()[0]
            lengths[name] = 0
        else:
            lengths[name] += len(line.strip())
    return lengths


def main(analysis_dir):
    a = Path(analysis_dir)
    pf.use_style()
    err = pd.read_csv(a / "results/boundary_errors.csv")
    ur = pd.read_csv(a / "results/uniform_random_errors.csv")
    inst = pd.read_csv(a / "data/filtered/instances_kept.tsv", sep="\t")
    lengths = read_lengths(a / "data/filtered/query_proteins.fasta")
    out = a / "figures"
    out.mkdir(exist_ok=True)
    sf1(err, ur, out)
    sf2(err, inst, lengths, out)
    sf3(err, inst, out)


if __name__ == "__main__":
    main(sys.argv[1])
