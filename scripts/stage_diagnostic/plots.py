"""One figure per endpoint, one panel per dataset (a ice binding, b toxin scaffold,
c toxin target). Held-out queries unless the axis says otherwise. Figures go to
figures/350/ as PDF, SVG and PNG through notebooks/pubfig.py (Nature style)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf  # noqa: E402

DATASETS = ["ice_binding", "toxin_scaffold", "toxin_target"]
TITLES = {
    "ice_binding": "Ice binding, other origin",
    "toxin_scaffold": "Toxin, same cysteine framework",
    "toxin_target": "Toxin, same target class",
}
# one colour per method, the same in every figure
METHOD = {
    "A": ("A  full-residue alignment, every reference", "#000000"),
    "B": ("B  reduced alignment, every reference", "#0072B2"),
    "C_B": ("C  kmerseek seeds, scaled 1, reduced alignment", "#009E73"),
    "C_A": ("C  kmerseek seeds, scaled 1, full-residue alignment", "#CC79A7"),
    "D_B": ("D  kmerseek seeds, scaled 10, reduced alignment", "#56B4E9"),
    "E1": ("E  kmerseek end to end, scaled 1", "#D55E00"),
    "E10": ("E  kmerseek end to end, scaled 10", "#E69F00"),
}
ORDER = list(METHOD)


def _fig(height=62):
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, height, ncols=3)
    return fig, axs


def _finish(fig, axs, path):
    for ax, letter in zip(axs, "abc"):
        pf.panel_label(ax, letter)
    pf.save(fig, str(path))


def candidate_recall(res, out, chosen):
    fig, axs = _fig(58)
    for ax, ds in zip(axs, DATASETS):
        c = pl.read_csv(res / ds / "candidate_recall.tsv", separator="\t").filter(
            pl.col("split") == "heldout"
        )
        vals = [1.0] + [c.filter(pl.col("stage") == s)["candidate_recall"][0] for s in ("C", "D")]
        lo = [0] + [c.filter(pl.col("stage") == s)["ci_low"][0] for s in ("C", "D")]
        hi = [0] + [c.filter(pl.col("stage") == s)["ci_high"][0] for s in ("C", "D")]
        cols = [METHOD["B"][1], METHOD["C_B"][1], METHOD["D_B"][1]]
        x = np.arange(3)
        ax.bar(x, vals, color=cols, width=0.6)
        ax.errorbar(
            x[1:],
            vals[1:],
            yerr=[np.array(vals[1:]) - lo[1:], np.array(hi[1:]) - vals[1:]],
            fmt="none",
            ecolor="black",
        )
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom")
        ax.set_xticks(x, ["A, B\n(no seeds)", "C\nscaled 1", "D\nscaled 10"])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Positives kept as candidates\n(mean over held-out queries, 95% CI)")
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    _finish(fig, axs, out / "350_candidate_recall_heldout")


def recall_at_budget(res, out, chosen, budgets):
    fig, axs = _fig(70)
    for ax, ds in zip(axs, DATASETS):
        r = pl.read_csv(res / ds / "recall.tsv", separator="\t").filter(
            pl.col("split") == "heldout"
        )
        for i, m in enumerate(ORDER):
            s = r.filter(pl.col("method") == m).sort("budget")
            x = np.arange(len(budgets)) + (i - 3) * 0.07
            y = s["recall"].to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=[y - s["ci_low"].to_numpy(), s["ci_high"].to_numpy() - y],
                color=METHOD[m][1],
                marker="o",
                markersize=3,
                label=METHOD[m][0],
            )
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("Negatives allowed above a positive, per query")
        ax.set_ylabel("Positives found (mean recall over\nheld-out queries, 95% CI)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=2)
    _finish(fig, axs, out / "350_recall_at_false_hit_budget_heldout")


def pr_curves(res, out, chosen):
    fig, axs = _fig(70)
    for ax, ds in zip(axs, DATASETS):
        pr = pl.read_parquet(res / ds / "pr_heldout.parquet")
        for m in ORDER:
            s = pr.filter(pl.col("method") == m)
            if s.height:
                ax.plot(
                    s["recall"],
                    s["precision"],
                    color=METHOD[m][1],
                    label=METHOD[m][0],
                    drawstyle="steps-post",
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall (all held-out positives)")
        ax.set_ylabel("Precision")
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=2)
    _finish(fig, axs, out / "350_precision_recall_heldout")


STAGES = ["found", "rank", "filter", "sketch", "seed"]
STAGE_LABEL = {
    "seed": "no shared k-mer (seed)",
    "sketch": "k-mer dropped by the sketch",
    "filter": "removed by kmerseek's filters",
    "rank": "reported, ranked below the budget",
    "found": "found within the budget",
}


def loss_by_stage(res, out, chosen):
    import seaborn as sns

    cols = dict(zip(STAGES, sns.color_palette("crest", len(STAGES))))
    fig, axs = _fig(55)
    for ax, ds in zip(axs, DATASETS):
        d = pl.read_csv(res / ds / "loss_by_stage.tsv", separator="\t").filter(
            pl.col("split") == "heldout"
        )
        for yi, path in enumerate(["scaled 1", "scaled 10"]):
            s = d.filter(pl.col("path") == path)
            tot = s["len"].sum()
            left = 0
            for st in STAGES:
                n = s.filter(pl.col("stage") == st)["len"].sum()
                if path == "scaled 1" and st == "sketch":
                    continue
                ax.barh(
                    yi,
                    n / tot,
                    left=left,
                    color=cols[st],
                    label=STAGE_LABEL[st] if yi == 1 else None,
                )
                if n / tot > 0.06:
                    ax.text(left + n / tot / 2, yi, str(n), ha="center", va="center", color="white")
                left += n / tot
        sc = d.filter(pl.col("path") == "scorers")
        red = sc.filter(pl.col("stage") == "A found, B not")["len"].sum()
        gain = sc.filter(pl.col("stage") == "B found, A not")["len"].sum()
        ax.set_yticks([0, 1], ["E, scaled 1", "E, scaled 10"])
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            f"Share of {tot} held-out positive pairs\n(A finds, B misses: {red}; B finds, A misses: {gain})"
        )
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=5)
    _finish(fig, axs, out / "350_positives_by_stage_lost_heldout")


def by_group(res, out, chosen):
    import seaborn as sns  # noqa: F401
    import matplotlib as mpl

    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 90, ncols=3, gridspec_kw=dict(wspace=1.1))
    cmap = mpl.colormaps["mako"].copy()
    cmap.set_bad("#dddddd")
    for ax, ds in zip(axs, DATASETS):
        g = pl.read_csv(res / ds / "by_group.tsv", separator="\t")
        groups = (
            g.select("split", "group", "n_queries")
            .unique(["split", "group"])
            .sort("split", "group")
        )
        mat = np.full((groups.height, len(ORDER)), np.nan)
        for i, (sp, gr, _) in enumerate(groups.iter_rows()):
            for j, m in enumerate(ORDER):
                v = g.filter(
                    (pl.col("split") == sp) & (pl.col("group") == gr) & (pl.col("method") == m)
                )["recall"]
                if len(v):
                    mat[i, j] = v[0]
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ORDER)), [m.replace("_", " ") for m in ORDER], rotation=90)
        ax.set_yticks(
            range(groups.height),
            [
                f"{'held-out' if sp == 'heldout' else 'dev'} | {gr[:28]} (n={n})"
                for sp, gr, n in groups.iter_rows()
            ],
        )
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    cb = fig.colorbar(im, ax=axs, shrink=0.6)
    cb.set_label("Recall at 1 negative allowed (mean over the group's queries)")
    _finish(fig, axs, out / "350_recall_by_origin_framework_class")


def collisions(res, out, chosen):
    fig, axs = _fig(55)
    for ax, ds in zip(axs, DATASETS):
        c = pl.read_csv(res / ds / "collisions.tsv", separator="\t")
        rels = [
            r
            for r in ("positive", "negative_annotated", "negative_hard", "background")
            if r in c["relation"].to_list()
        ]
        x = np.arange(len(rels))
        both = [
            c.filter((pl.col("relation") == r) & pl.col("shares_full_residue_kmer"))["len"].sum()
            for r in rels
        ]
        red = [
            c.filter((pl.col("relation") == r) & ~pl.col("shares_full_residue_kmer"))["len"].sum()
            for r in rels
        ]
        ax.bar(x, both, color=pf.GREY, label="also shares a full-residue k-mer of the same length")
        ax.bar(
            x,
            red,
            bottom=both,
            color="#0072B2",
            label="shares only reduced k-mers (same reduced letters, different residues)",
        )
        for xi, b, r in zip(x, both, red):
            ax.text(xi, b + r, f"{r}/{b + r}", ha="center", va="bottom")
        ax.set_xticks(x, [r.replace("negative_", "").replace("_", " ") for r in rels])
        ax.set_yscale("log")
        ax.set_ylabel("Candidate pairs at scaled 1 (all queries)")
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=1)
    _finish(fig, axs, out / "350_reduced_only_seed_matches")


def cost(res, out, chosen):
    c = pl.read_csv(res / "cost.tsv", separator="\t", infer_schema_length=None)
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 110, nrows=2, ncols=3)
    for j, ds in enumerate(DATASETS):
        s = c.filter((pl.col("dataset") == ds) & pl.col("kind").is_in(["stage", "kmerseek"]))
        labels = s["stage"].to_list()
        ax = axs[0, j]
        ax.barh(range(len(labels)), s["cpu_seconds"].to_numpy(), color=pf.GREY)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Time (s): A, B summed CPU over queries;\nkmerseek wall clock per step")
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
        ax = axs[1, j]
        v = c.filter((pl.col("dataset") == ds) & pl.col("kind").is_in(["volume", "stage"]))
        lab = v["stage"].to_list()
        ax.barh(range(len(lab)), v["candidates_per_query"].to_numpy(), color=pf.GREY)
        ax.set_yticks(range(len(lab)), lab)
        ax.set_xlabel("References examined or reported per query")
    for ax, letter in zip(axs.flat, "abcdef"):
        pf.panel_label(ax, letter)
    pf.save(fig, str(out / "350_runtime_memory_candidates"))


def run(cfg, res: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    pf.use_style()
    sel = pl.read_csv(res / "selection.tsv", separator="\t").filter(pl.col("chosen"))
    chosen = dict(zip(sel["dataset"], sel["setting"]))
    candidate_recall(res, out, chosen)
    recall_at_budget(res, out, chosen, cfg["evaluation"]["false_hit_budgets"])
    pr_curves(res, out, chosen)
    loss_by_stage(res, out, chosen)
    by_group(res, out, chosen)
    collisions(res, out, chosen)
    cost(res, out, chosen)
