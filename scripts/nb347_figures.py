"""Figures for notebook 347 (called by run_nb347_invisible_mutations.py)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

ENC_COLOUR = {
    "hp_lehninger2": pf.OKABE_ITO["vermillion"],
    "dayhoff6": pf.OKABE_ITO["blue"],
    "composition": pf.OKABE_ITO["bluish_green"],
    "identity": "black",
}
ENC_LABEL = {
    "hp_lehninger2": "hp_lehninger2 (2 letters)",
    "dayhoff6": "dayhoff6 (6 letters)",
    "composition": "amino-acid counts only",
    "identity": "protein20 (identity)",
}
ROWS = ["hp_lehninger2", "dayhoff6", "composition", "identity"]
HP_CLASSES = "hydrophobic AFGILMPVWY, polar CDEHKNQRST"
DAYHOFF_CLASSES = "C | AGPST | DENQ | HKR | ILMV | FWY"

AA1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
       "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
       "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}  # fmt: skip


def short(hgvs: str) -> str:
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvs)
    return f"{AA1[m.group(1)]}{m.group(2)}{AA1[m.group(3)]}" if m else hgvs


def header(fig, text):
    """One line above everything; the style's constrained layout makes room for it."""
    fig.suptitle(text, x=0.01, ha="left", fontsize=6)


def fig_where_variants_fall(summary, groups_df, out):
    """a: sizes of collision groups other than the wild-type-sequence group;
    b: share of variants that are alone, in other groups, or in the wild-type group."""
    fig, (a, b) = pf.figure(
        pf.TWO_COLUMN_MM, 62, ncols=2, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    for enc in ["hp_lehninger2", "dayhoff6", "composition"]:
        sizes = groups_df.filter(
            (pl.col("encoding") == enc) & ~pl.col("is_reference_group")
        )["n_variants"].to_numpy()
        vals, counts = np.unique(sizes, return_counts=True)
        a.plot(
            vals, counts, "o-", color=ENC_COLOUR[enc], ms=3, lw=1, label=ENC_LABEL[enc]
        )
    a.set_xlabel("Variants in the group (n)")
    a.set_ylabel("Collision groups (n)")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.legend(frameon=False, loc="upper right")
    a.set_title("Groups of variants with the same encoded sequence,\n"
                "not counting the group equal to wild type", fontsize=6, loc="left")  # fmt: skip

    s = summary.filter(pl.col("grouping") == "full").pivot(
        on="scope", index="encoding", values="n_variants_collided"
    )
    N = summary["N"][0]
    parts = [("reference_group", "same encoded sequence as wild type", "black"),
             ("other_groups", "in another group of 2 or more", "#7f7f7f"),
             ("singleton", "alone (no other variant shares it)", "#d9d9d9")]  # fmt: skip
    y = np.arange(len(ROWS))[::-1]
    for yi, enc in zip(y, ROWS):
        r = s.filter(pl.col("encoding") == enc).row(0, named=True)
        vals = {
            "reference_group": r["reference_group"],
            "other_groups": r["other_groups"],
        }
        vals["singleton"] = N - vals["reference_group"] - vals["other_groups"]
        left = 0.0
        for key, label, col in parts:
            frac = vals[key] / N
            b.barh(yi, frac, left=left, color=col, edgecolor="white", lw=0.3,
                   label=label if enc == ROWS[0] else None)  # fmt: skip
            left += frac
    b.set_yticks(y, [ENC_LABEL[e] for e in ROWS])
    b.set_xlim(0, 1)
    b.set_xlabel(f"Share of the {int(N)} missense variants")
    b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    for ax, letter in zip((a, b), "ab"):
        pf.panel_label(ax, letter)
    header(fig, f"PTEN VAMP-seq (MaveDB urn:mavedb:00000013-a-1), {int(N)} single missense "
                f"variants. HP: {HP_CLASSES}. Dayhoff: {DAYHOFF_CLASSES}.")  # fmt: skip
    pf.save(fig, out)


def fig_error_floors(summary, out):
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2, sharey=True)
    sd_all = float(np.sqrt(summary["total_variance"][0]))
    y = np.arange(len(ROWS))[::-1]
    titles = {
        "full": "Groups = same encoded sequence",
        "same_position": "Groups = same encoded sequence and same position",
    }
    for ax, grouping in zip(axs, ["full", "same_position"]):
        ax.axvline(sd_all, color=pf.GREY, lw=1, zorder=0)
        for yi, enc in zip(y, ROWS):
            r = summary.filter((pl.col("encoding") == enc) & (pl.col("grouping") == grouping)
                               & (pl.col("scope") == "all")).row(0, named=True)  # fmt: skip
            c = ENC_COLOUR[enc]
            lo, hi = r["floor_rmse_lo"], r["floor_rmse_hi"]
            if lo is not None and not np.isnan(lo):
                ax.plot([lo, hi], [yi + 0.15] * 2, color=c, lw=1)
            ax.plot(r["floor_rmse"], yi + 0.15, "o", color=c, ms=4)
            ax.plot(
                np.sqrt(r["expected_noise_mse"]),
                yi - 0.15,
                "o",
                mfc="white",
                mec=c,
                ms=4,
            )
            if r["n_groups_collided"] == 0:
                ax.text(
                    0.012,
                    yi,
                    "no two variants share this encoding",
                    va="center",
                    fontsize=5,
                )
        ax.set_yticks(y, [ENC_LABEL[e] for e in ROWS])
        for yi in y:
            ax.axhline(yi, color="#e0e0e0", lw=0.3, zorder=-1)
        ax.set_xlim(-0.005, sd_all * 1.08)
        ax.set_xlabel(
            "Root mean squared error over all variants (abundance score units)"
        )
        ax.set_title(titles[grouping], fontsize=6, loc="left")
    handles = [
        Line2D([], [], marker="o", color="k", lw=1, ms=4,
               label="lowest error any predictor reading only the encoding can reach, "
                     "on these variants (95% interval: resampling positions)"),
        Line2D([], [], marker="o", mfc="white", mec="k", lw=0, ms=4,
               label="error expected from replicate noise alone, if group members had equal true scores"),
        Line2D([], [], color=pf.GREY, lw=1,
               label=f"spread of all variants (SD {sd_all:.3f}): error of predicting the mean for every variant"),
    ]  # fmt: skip
    # The header goes in the legend's title: a suptitle and an outside legend overlap.
    fig.legend(
        handles=handles,
        loc="outside upper left",
        frameon=False,
        alignment="left",
        title="Lower is better for the encoding; 0 means it hides nothing. Score: "
        "1 = wild-type abundance, 0 = mean of nonsense variants.",
        title_fontsize=6,
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, out)


def fig_examples(groups_df, variants, examples, out, max_members=8, seed=347):
    rng = np.random.default_rng(seed)
    by_hgvs = {r["hgvs_pro"]: r for r in variants.iter_rows(named=True)}
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 165, ncols=3,
                         gridspec_kw={"width_ratios": [0.9, 1, 1]})  # fmt: skip
    # a: every collision group's range, both encodings, with the shown ones marked
    a = axs[0]
    for enc in ["hp_lehninger2", "dayhoff6"]:
        rng_vals = np.sort(
            groups_df.filter(pl.col("encoding") == enc)["range"].to_numpy()
        )
        a.step(rng_vals, np.arange(1, len(rng_vals) + 1) / len(rng_vals), where="post",
               color=ENC_COLOUR[enc], lw=1, label=f"{ENC_LABEL[enc]}, {len(rng_vals)} groups")  # fmt: skip
        ex = examples.filter(pl.col("encoding") == enc)
        for r in ex.iter_rows(named=True):
            q = np.searchsorted(rng_vals, r["range"], side="right") / len(rng_vals)
            mk = "v" if r["selection"].startswith("selected") else "D"
            a.plot(r["range"], q, mk, mfc="white", mec=ENC_COLOUR[enc], ms=4)
    h, _ = a.get_legend_handles_labels()
    h += [Line2D([], [], marker="v", mfc="white", mec="k", lw=0, ms=4,
                 label="shown at right: largest ranges (selected extremes)"),
          Line2D([], [], marker="D", mfc="white", mec="k", lw=0, ms=4,
                 label="shown at right: drawn at random")]  # fmt: skip
    a.legend(handles=h, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    a.set_xlabel("Range of scores within the group\n(highest minus lowest)")
    a.set_ylabel("Share of collision groups with this range or less")

    for ax, enc in zip(axs[1:], ["hp_lehninger2", "dayhoff6"]):
        ex = examples.filter(pl.col("encoding") == enc)
        yt, yl, ypos = [], [], 0.0
        for r in ex.iter_rows(named=True):
            members = r["members"].split(";")
            shown = (
                members
                if len(members) <= max_members
                else list(rng.choice(members, max_members, replace=False))
            )
            shown = sorted(shown, key=lambda m: by_hgvs[m]["y"])
            tag = "largest range" if r["selection"].startswith("selected") else "random"
            where = (
                "same as wild type, many positions"
                if r["is_reference_group"]
                else "one position"
            )
            ax.text(-0.38, ypos - 0.2, f"{tag}; {where}; {len(shown)} of {r['n_variants']} shown",
                    fontsize=5, va="bottom", ha="left")  # fmt: skip
            ypos -= 1.0
            for m in shown:
                v = by_hgvs[m]
                ax.plot(v["replicates"], [ypos] * len(v["replicates"]), ".", color="#b0b0b0",
                        ms=2, zorder=1)  # fmt: skip
                ax.errorbar(v["y"], ypos, xerr=v["se"], fmt="o", color="black", ms=2.5,
                            lw=0.8, capsize=0, zorder=2)  # fmt: skip
                yt.append(ypos)
                yl.append(short(m))
                ypos -= 1.0
            ypos -= 0.8
        ax.set_yticks(yt, yl, fontsize=5)
        ax.set_xlim(-0.4, 1.8)
        ax.axvline(1, color=pf.GREY, lw=0.5, zorder=0)
        ax.axvline(0, color=pf.GREY, lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Abundance score")
        ax.set_title(ENC_LABEL[enc], fontsize=6, loc="left", color="black")
        handles = [Line2D([], [], marker=".", color="#b0b0b0", lw=0, ms=4, label="one replicate"),
                   Line2D([], [], marker="o", color="black", lw=0.8, ms=2.5,
                          label="mean ± standard error"),
                   Line2D([], [], color=pf.GREY, lw=0.5, label="wild type (1)"),
                   Line2D([], [], color=pf.GREY, lw=0.5, ls=":", label="nonsense mean (0)")]  # fmt: skip
        ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=2)  # fmt: skip
    for ax, letter in zip(axs, "abc"):
        pf.panel_label(ax, letter)
    header(fig, "Variants in one group have the same encoded sequence, so any predictor reading "
                "only the encoding gives them one value.")  # fmt: skip
    pf.save(fig, out)


def make_all(summary, groups_df, variants, examples, figures: Path, T_FIG="347"):
    pf.use_style()
    fig_where_variants_fall(
        summary, groups_df, figures / f"{T_FIG}_SF1_collision_groups_pten_vamp_seq"
    )
    fig_error_floors(summary, figures / f"{T_FIG}_SF2_error_floors_pten_vamp_seq")
    fig_examples(groups_df, variants, examples,
                 figures / f"{T_FIG}_SF3_example_groups_pten_vamp_seq")  # fmt: skip
