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
    fig, axs = _fig(62)
    for ax, ds in zip(axs, DATASETS):
        c = pl.read_csv(res / ds / "candidate_recall.tsv", separator="\t").filter(
            pl.col("split") == "heldout"
        )
        for i, (st, colr) in enumerate((("C", METHOD["C_B"][1]), ("D", METHOD["D_B"][1]))):
            r = c.filter(pl.col("stage") == st).row(0, named=True)
            pos, neg = r["candidate_recall"], r["negatives_kept"]
            ax.bar(
                i - 0.18, pos, width=0.34, color=colr, label="positives kept" if i == 0 else None
            )
            ax.errorbar(
                i - 0.18,
                pos,
                yerr=[[pos - r["ci_low"]], [r["ci_high"] - pos]],
                fmt="none",
                ecolor="black",
            )
            ax.bar(
                i + 0.18, neg, width=0.34, color=pf.GREY, label="negatives kept" if i == 0 else None
            )
            for xx, v in ((i - 0.18, pos), (i + 0.18, neg)):
                inside = v > 0.12
                ax.text(
                    xx,
                    0.02 if inside else v + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    color="white" if inside else "black",
                )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.text(1.5, 1.01, "A and B examine every reference", ha="right", va="bottom")
        ax.set_xticks([0, 1], ["C  scaled 1", "D  scaled 10"])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(
            "Share kept as kmerseek candidates\n(mean over held-out queries;\n95% CI for positives)"
        )
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    handles = [
        __import__("matplotlib").patches.Patch(
            color=METHOD["C_B"][1], label="positives kept, scaled 1"
        ),
        __import__("matplotlib").patches.Patch(
            color=METHOD["D_B"][1], label="positives kept, scaled 10"
        ),
        __import__("matplotlib").patches.Patch(color=pf.GREY, label="negatives kept (same stage)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
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
        ax.set_xlabel("Negatives allowed above a positive")
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
                    # A is drawn first, wide and pale, so a curve on top of it stays visible
                    **({"linewidth": 3, "alpha": 0.35, "zorder": 1} if m == "A" else {"zorder": 2}),
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
            f"Share of {tot} held-out positive pairs\n(A finds only: {red}; B finds only: {gain})"
        )
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=5)
    _finish(fig, axs, out / "350_positives_by_stage_lost_heldout")


def by_group(res, out, chosen):
    import matplotlib as mpl
    import seaborn as sns  # noqa: F401

    tables = {}
    for ds in DATASETS:
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
        tables[ds] = (groups, mat)
    ratios = [tables[ds][1].shape[0] + 3 for ds in DATASETS]
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 170, nrows=3, gridspec_kw=dict(height_ratios=ratios))
    cmap = mpl.colormaps["mako"].copy()
    cmap.set_bad("#dddddd")
    lines = [f"{m.replace('_', ' ')}: {METHOD[m][0][3:]}" for m in ORDER]
    key = "\n".join("     ".join(lines[i : i + 3]) for i in range(0, len(lines), 3))
    fig.suptitle(key)
    for ax, ds in zip(axs, DATASETS):
        groups, mat = tables[ds]
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if mat[i, j] > 0.55 else "white",
                    )
        ax.set_xticks(range(len(ORDER)), [m.replace("_", " ") for m in ORDER])
        ax.xaxis.tick_top()
        ax.set_yticks(
            range(groups.height),
            [
                f"{'held-out' if sp == 'heldout' else 'dev'}  {gr[:40]}  (n={n})"
                for sp, gr, n in groups.iter_rows()
            ],
        )
        ax.set_ylabel(f"{TITLES[ds]}\n{chosen[ds]}")
    cb = fig.colorbar(im, ax=axs, shrink=0.4)
    cb.set_label("Recall at 1 negative allowed\n(mean over the group's queries)")
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
            ax.text(xi, (b + r) * 1.1, f"{r}\nof {b + r}", ha="center", va="bottom")
        ax.set_xticks(x, [r.replace("negative_", "").replace("_", " ") for r in rels])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            __import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        ax.set_ylabel("Candidate pairs, scaled 1\n(all queries)")
        ax.set_ylim(1, max(b + r for b, r in zip(both, red)) * 12)
        ax.set_title(f"{TITLES[ds]}\n{chosen[ds]}")
    pf.shared_legend(fig, ncol=1)
    _finish(fig, axs, out / "350_reduced_only_seed_matches")


def cost(res, out, chosen):
    c = pl.read_csv(res / "cost.tsv", separator="\t", infer_schema_length=None)
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 150, nrows=3, ncols=3)
    for j, ds in enumerate(DATASETS):
        s = c.filter((pl.col("dataset") == ds) & pl.col("kind").is_in(["stage", "kmerseek"])).sort(
            ["kind", "stage"], descending=[True, True]
        )
        labels = [x.replace("e2e", "end to end") for x in s["stage"].to_list()]
        for row, col, xl in (
            (
                0,
                "cpu_seconds",
                "Time (s): A, B CPU summed over queries;\nkmerseek wall clock per step",
            ),
            (1, "peak_mb", "Peak memory (MB): A, B per worker;\nkmerseek per step"),
        ):
            ax = axs[row, j]
            ax.barh(range(len(labels)), s[col].to_numpy(), color=pf.GREY)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel(xl)
        axs[0, j].set_title(f"{TITLES[ds]}\n{chosen[ds]}")
        ax = axs[2, j]
        v = c.filter((pl.col("dataset") == ds) & pl.col("kind").is_in(["volume", "stage"])).sort(
            ["kind", "stage"], descending=[True, True]
        )
        lab = v["stage"].to_list()
        ax.barh(range(len(lab)), v["candidates_per_query"].to_numpy(), color=pf.GREY)
        ax.set_yticks(range(len(lab)), lab)
        ax.set_xlabel("References examined (A-D) or\nreported (E) per query")
    for ax, letter in zip(axs.flat, "abcdefghi"):
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
