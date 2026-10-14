"""Tables and figures for notebook 292: keeping nanopore reads by translated reduced-alphabet
k-mers. The scripts in scripts/adaptive/ do the computation; this module reads their output,
computes the per-setting numbers and draws the figures."""

from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pubfig as pf

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "adaptive"
    ),
)

DATA = os.environ.get("NB292_DIR", "/Users/olga/data/adaptive_sampling")
PFAM = "PF02950"
PFAM_NAME = "Conotoxin"
RUN = "SRR32537183"
DIAMOND = "/Users/olga/anaconda3/envs/adaptive-sampling/bin/diamond"
MIN_KMERS = [1, 2, 3]
FALSE_KEEP_MAX = 0.05
# Brief's pore-time model, an assumption and not a measurement: a kept read costs its full
# length in bases; an ejected read costs the chunk plus this many bases of lost pore time.
EJECT_LOST_BASES = 500
F1_FLOOR = 1e-4  # zero is drawn here on the log axis, with an x
HOLDOUT_LABEL = {"ii": "genus held out", "iii": "family held out"}
ALPHABET_ORDER = ["hp", "dayhoff", "protein"]
ALPHABET_CLASSES = {"protein": 20, "dayhoff": 6, "hp": 2}

# One colour per rule, used in every figure of the notebook.
RULE_COLOUR = {
    "k-mer rule": pf.OKABE_ITO["blue"],
    "DNA rule (minimap2)": pf.OKABE_ITO["vermillion"],
    "DIAMOND blastx (slow bound)": pf.OKABE_ITO["bluish_green"],
}
RULE_MARKER = {
    "k-mer rule": "o",
    "DNA rule (minimap2)": "s",
    "DIAMOND blastx (slow bound)": "^",
}
SHUFFLED_COLOUR = "#999999"


def truth() -> pl.DataFrame:
    """Per-read truth labels from scripts/adaptive/truth_table.py."""
    return pl.read_parquet(f"{DATA}/truth/reads_truth_{PFAM}.parquet")


def scores() -> pl.LazyFrame:
    """Per read x copy x chunk x hold-out x alphabet x k counts from score_chunks.py."""
    return pl.scan_parquet(f"{DATA}/scores_{PFAM}/part*.parquet")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _rates(
    kept: pl.DataFrame,
    by: list[str],
    tr: pl.DataFrame,
    n_bases_col: str = "chunk_bases",
) -> pl.DataFrame:
    """Sensitivity, false keep, enrichment and pore-time enrichment from a table of
    real-read (copy 0) rows with a boolean `keep` column, grouped by `by`."""
    lab = tr.select("read_id", "label", "read_length")
    d = kept.join(lab, on="read_id", how="inner")
    n_fam = tr.filter(pl.col("label") == "family").height
    n_non = tr.filter(pl.col("label") == "non_family").height
    n_all = tr.height
    fam_bases_all = tr.filter(pl.col("label") == "family")["read_length"].sum()
    bases_all = tr["read_length"].sum()
    out = d.group_by(by).agg(
        n_family_kept=((pl.col("label") == "family") & pl.col("keep")).sum(),
        n_non_family_kept=((pl.col("label") == "non_family") & pl.col("keep")).sum(),
        n_unassigned_kept=((pl.col("label") == "unassigned") & pl.col("keep")).sum(),
        n_kept=pl.col("keep").sum(),
        family_bases_kept=pl.when(pl.col("keep") & (pl.col("label") == "family"))
        .then(pl.col("read_length"))
        .otherwise(0)
        .sum(),
        pore_bases=pl.when(pl.col("keep"))
        .then(pl.col("read_length"))
        .otherwise(
            pl.min_horizontal(pl.col("read_length"), pl.col(n_bases_col))
            + EJECT_LOST_BASES
        )
        .sum(),
    )
    return out.with_columns(
        n_family=pl.lit(n_fam),
        n_non_family=pl.lit(n_non),
        n_reads=pl.lit(n_all),
        sensitivity=pl.col("n_family_kept") / n_fam,
        false_keep=pl.col("n_non_family_kept") / n_non,
        enrichment=(pl.col("n_family_kept") / pl.col("n_kept")) / (n_fam / n_all),
        pore_time_enrichment=(pl.col("family_bases_kept") / pl.col("pore_bases"))
        / (fam_bases_all / bases_all),
    )


def kmer_settings_table(tr: pl.DataFrame) -> pl.DataFrame:
    """One row per hold-out x chunk x alphabet x k x m: rates on real reads and the keep
    rate on the 10 shuffled copies of every read (all reads, every label)."""
    s = (
        with_requested_chunk(scores())
        .with_columns(pl.lit(MIN_KMERS).alias("min_kmers"))
        .explode("min_kmers")
        .with_columns(keep=pl.col("n_kmers_in_index") >= pl.col("min_kmers"))
    )
    by = ["holdout", "chunk_bases_requested", "alphabet", "k", "min_kmers"]
    real = s.filter(pl.col("copy") == 0).collect()
    shuf = (
        s.filter(pl.col("copy") > 0)
        .group_by(by)
        .agg(shuffled_keep=pl.col("keep").mean(), n_shuffled=pl.len())
        .collect()
    )
    return (
        _rates(real, by, tr)
        .join(shuf, on=by, how="left")
        .with_columns(setting=pl.format("{} k={} m={}", "alphabet", "k", "min_kmers"))
        .sort(by)
    )


CHUNKS = [200, 400, 800]


def with_requested_chunk(s: pl.LazyFrame) -> pl.LazyFrame:
    """Add N, the chunk length asked for. score_chunks.py stores the chunk actually used,
    which is the whole read when the read is shorter than N, and writes the rows of one
    read, copy, hold-out, alphabet and k in the order N = 200, 400, 800. The row's place in
    that group gives N back."""
    g = ["read_id", "copy", "holdout", "alphabet", "k"]
    return s.with_columns(
        chunk_bases_requested=pl.int_range(pl.len()).over(g)
    ).with_columns(
        chunk_bases_requested=pl.col("chunk_bases_requested").replace_strict(
            dict(enumerate(CHUNKS)), return_dtype=pl.Int32
        )
    )


def comparator_table(tr: pl.DataFrame, chunks=(200, 400, 800)) -> pl.DataFrame:
    """Keep rates of the DNA rule and the slow DIAMOND bound, on real reads and shuffled
    copies, from run_comparators.sh output."""
    rows = []
    ids = tr.select("read_id")
    for n in chunks:
        for lvl in ["ii", "iii"]:
            for rule, path, fmt in [
                (
                    "DNA rule (minimap2)",
                    f"{DATA}/comparators/N{n}.minimap2_{lvl}.paf",
                    "paf",
                ),
                (
                    "DIAMOND blastx (slow bound)",
                    f"{DATA}/comparators/N{n}.diamond_{lvl}.tsv",
                    "tsv",
                ),
            ]:
                if not os.path.exists(path):
                    continue
                if os.path.getsize(path) == 0:
                    hit = pl.DataFrame({"qid": []}, schema={"qid": pl.Utf8})
                else:
                    hit = pl.read_csv(
                        path,
                        separator="\t",
                        has_header=False,
                        columns=[0],
                        new_columns=["qid"],
                        truncate_ragged_lines=True,
                    )
                hit = hit.unique().with_columns(
                    read_id=pl.col("qid").str.split("|").list.get(0),
                    copy=pl.col("qid").str.split("|").list.get(1).cast(pl.Int32),
                )
                real_ids = set(hit.filter(pl.col("copy") == 0)["read_id"].to_list())
                real = ids.with_columns(
                    keep=pl.col("read_id").is_in(list(real_ids)),
                    chunk_bases_requested=pl.lit(n),
                )
                r = _rates(
                    real,
                    ["chunk_bases_requested"],
                    tr,
                    n_bases_col="chunk_bases_requested",
                )
                shuffled_keep = hit.filter(pl.col("copy") > 0).height / (10 * tr.height)
                rows.append(
                    r.with_columns(
                        rule=pl.lit(rule),
                        holdout=pl.lit(lvl),
                        shuffled_keep=pl.lit(shuffled_keep),
                    )
                )
    return pl.concat(rows, how="diagonal_relaxed")


def best_setting(tab: pl.DataFrame, holdout: str = "ii", n: int = 400) -> dict:
    """Highest sensitivity at false keep <= 5% on hold-out ii at chunk N; ties broken by lower
    false keep, then the setting name, so the pick is stable."""
    ok = tab.filter(
        (pl.col("holdout") == holdout)
        & (pl.col("chunk_bases_requested") == n)
        & (pl.col("false_keep") <= FALSE_KEEP_MAX)
    )
    if ok.height == 0:
        return {}
    return ok.sort(
        ["sensitivity", "false_keep", "setting"], descending=[True, False, False]
    ).row(0, named=True)


HOLDOUT_SHORT = {"ii": "genus", "iii": "family"}


def verdict(holdout: str, n: int, best_row: dict, dna_row: dict, n_family: int) -> str:
    """The brief's verdict sentence and its two branches, plus a third branch the brief did
    not foresee: the k-mer rule keeps family reads at no higher a rate than non-family
    reads, so it does not pick out the family at all and "reaches further" means nothing.
    """
    s = (
        f"At the {HOLDOUT_SHORT[holdout]} hold-out, the k-mer rule ({best_row['setting']}, N = {n}) keeps "
        f"{best_row['sensitivity']:.1%} of {n_family:_} family reads at a false keep of {best_row['false_keep']:.1%}; "
        f"DNA mapping keeps {dna_row['sensitivity']:.1%}."
    )
    if best_row["sensitivity"] <= 2 * best_row["false_keep"]:
        return s + (
            " The k-mer rule keeps family reads at about the rate it keeps other reads, so it does not"
            " pick out the family; neither rule works at this hold-out."
        )
    if dna_row["sensitivity"] >= best_row["sensitivity"]:
        return (
            s
            + " DNA mapping keeps as many; the cross-species claim is dead at this hold-out."
        )
    return s + " The translated rule reaches where DNA mapping does not."


# ---------------------------------------------------------------------------------------
# Hero example: residues, frames and alignments.
# ---------------------------------------------------------------------------------------


def read_sequence(read_id: str) -> str:
    """The full sampled read, as sequenced."""
    with open(f"{DATA}/reads/{RUN}_sample200k_seed20261014.fastq") as f:
        for i, line in enumerate(f):
            if i % 4 == 0 and line[1:].split()[0] == read_id:
                return next(f).strip()
    raise KeyError(read_id)


def reference_proteins(level: str) -> dict[str, str]:
    """Accession -> sequence for one hold-out reference."""
    out, acc = {}, None
    with open(f"{DATA}/ref/{PFAM}/ref_{level}.fasta") as f:
        for line in f:
            if line.startswith(">"):
                acc = line[1:].strip()
            else:
                out[acc] = line.strip()
    return out


def match_line(a: str, b: str) -> str:
    """'|' where two equal-length strings agree, ' ' where they differ."""
    return "".join("|" if x == y else " " for x, y in zip(a, b))


def print_block(
    label_a: str,
    a: str,
    start_a: int,
    label_b: str,
    b: str,
    start_b: int,
    width: int = 60,
) -> None:
    """Print two equal-length strings with 1-based coordinates and a match line."""
    w = max(len(label_a), len(label_b))
    for i in range(0, len(a), width):
        x, y = a[i : i + width], b[i : i + width]
        print(f"{label_a:<{w}} {start_a + i:>5} {x} {start_a + i + len(x) - 1}")
        print(f"{'':<{w}} {'':>5} {match_line(x, y)}")
        print(f"{label_b:<{w}} {start_b + i:>5} {y} {start_b + i + len(y) - 1}")
        print()
    n_id = sum(p == q for p, q in zip(a, b))
    print(f"{n_id} of {len(a)} identical")


# ---------------------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------------------


def header(fig, tools: str, conclusion: str) -> None:
    """Tool line and conclusion line above the panels (notebook version of the figure)."""
    fig.text(0.01, 1.0, tools, ha="left", va="bottom", fontsize=6)
    fig.text(
        0.01,
        1.0 + 0.045,
        conclusion,
        ha="left",
        va="bottom",
        fontsize=6,
        fontweight="bold",
    )


def fig_read_lengths(all_lengths: np.ndarray, sample_lengths: np.ndarray, path: str):
    """SF1: read lengths of the whole run and of the 200_000-read sample."""
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
    bins = np.logspace(
        np.log10(max(all_lengths.min(), 10)), np.log10(all_lengths.max()), 80
    )
    ax.hist(
        all_lengths,
        bins=bins,
        density=True,
        histtype="step",
        color=pf.GREY,
        linewidth=1.5,
        label=f"whole run ({len(all_lengths):_} reads)",
    )
    ax.hist(
        sample_lengths,
        bins=bins,
        density=True,
        histtype="step",
        color=pf.OKABE_ITO["blue"],
        label=f"sample ({len(sample_lengths):_} reads)",
    )
    for n in CHUNKS:
        ax.axvline(n, color="black", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ticks = [50, 100, 200, 400, 800, 2000, 5000, 10_000]
    ax.set_xticks(ticks, [f"{t:_}" for t in ticks])
    ax.set_xlabel("Read length (bases); dotted lines at chunk N = 200, 400, 800")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=False)
    pf.save(fig, path)
    return fig


def fig_rules_by_holdout(panels: list[tuple[pl.DataFrame, str]], n: int, path: str):
    """F1: sensitivity at each hold-out level, one line per rule, at chunk N, one panel per
    truth. Each rule's keep rate on shuffled reads is a dashed line of the same colour. The
    legend gives each rule's false keep on non-family reads."""
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 75, ncols=len(panels), sharey=True)
    axs = np.atleast_1d(axs)
    x = {"ii": 0, "iii": 1}
    for ax, (rows, title) in zip(axs, panels):
        for rule in RULE_COLOUR:
            r = rows.filter(pl.col("rule") == rule).sort("holdout")
            if r.height == 0:
                continue
            xs = (
                np.array([x[h] for h in r["holdout"]])
                + {
                    "k-mer rule": -0.04,
                    "DNA rule (minimap2)": 0.0,
                    "DIAMOND blastx (slow bound)": 0.04,
                }[rule]
            )
            sens = np.array(r["sensitivity"])
            lo = np.clip(np.array(r["sensitivity_ci_low"]), F1_FLOOR, None)
            hi = np.array(r["sensitivity_ci_high"])
            ys = np.clip(sens, F1_FLOOR, None)
            ax.plot(xs, ys, color=RULE_COLOUR[rule], linewidth=1, zorder=2)
            for xi, yi, si, l_, h_ in zip(xs, ys, sens, lo, hi):
                if si > 0:
                    ax.errorbar(
                        [xi],
                        [yi],
                        yerr=[[yi - l_], [h_ - yi]],
                        color=RULE_COLOUR[rule],
                        marker=RULE_MARKER[rule],
                        capsize=2,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        [xi],
                        [yi],
                        marker="x",
                        color=RULE_COLOUR[rule],
                        markersize=5,
                        zorder=3,
                    )
            ax.plot(
                [],
                [],
                color=RULE_COLOUR[rule],
                marker=RULE_MARKER[rule],
                label=r["label"][0],
            )
            ax.plot(
                xs,
                np.clip(np.array(r["shuffled_keep"]), F1_FLOOR, None),
                color=RULE_COLOUR[rule],
                linestyle="--",
                linewidth=0.8,
            )
        ax.set_xticks(
            [0, 1],
            [
                "genus held out\n(440 proteins; 8 from\nother tarantula genera)",
                "family held out\n(432 proteins; cone snails\nand one sea anemone)",
            ],
        )
        ax.set_xlim(-0.4, 1.4)
        ax.set_yscale("log")
        ax.set_ylim(F1_FLOOR * 0.7, 1.3)
        yt = [F1_FLOOR, 1e-3, 1e-2, 0.1, 0.5, 1]
        ax.set_yticks(yt, ["0 or <0.01%", "0.1%", "1%", "10%", "50%", "100%"])
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(title, fontsize=6)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=5)
    axs[0].plot(
        [],
        [],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="dashed: same rule on shuffled reads",
    )
    axs[0].plot(
        [],
        [],
        color="black",
        marker="x",
        linestyle="",
        label="x: no family read kept (0)",
    )
    axs[0].plot(
        [],
        [],
        color="black",
        linestyle=":",
        linewidth=0.5,
        label="dotted: 50%, the decision rule's bar",
    )
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=5)
    axs[0].set_ylabel(f"Family reads kept (sensitivity), N = {n} bases")
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, path)
    return fig


def fig_by_chunk(
    kmer_rows: pl.DataFrame, dna_rows: pl.DataFrame, holdout: str, path: str
):
    """F2: sensitivity (left) and false keep (right) against chunk N for the best k-mer
    setting and the DNA rule, at one hold-out. Shuffled keep rate dashed, same colour.
    """
    fig, axs = pf.figure(pf.TWO_COLUMN_MM * 0.62, 55, ncols=2)
    for rule, r in [("k-mer rule", kmer_rows), ("DNA rule (minimap2)", dna_rows)]:
        r = r.sort("chunk_bases_requested")
        for ax, col in zip(axs, ["sensitivity", "false_keep"]):
            xs = np.array(r["chunk_bases_requested"])
            v = np.array(r[col])
            ax.plot(
                xs,
                np.clip(v, F1_FLOOR, None),
                color=RULE_COLOUR[rule],
                linewidth=1,
                label=r["label"][0] if col == "sensitivity" else None,
                marker=None,
            )
            ax.plot(
                xs[v > 0],
                v[v > 0],
                color=RULE_COLOUR[rule],
                marker=RULE_MARKER[rule],
                linestyle="",
            )
            ax.plot(
                xs[v == 0],
                np.full((v == 0).sum(), F1_FLOOR),
                color=RULE_COLOUR[rule],
                marker="x",
                linestyle="",
            )
            ax.plot(
                xs,
                np.clip(np.array(r["shuffled_keep"]), F1_FLOOR, None),
                color=RULE_COLOUR[rule],
                linestyle="--",
                linewidth=0.8,
            )
    axs[0].plot(
        [],
        [],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="same rule on shuffled reads",
    )
    axs[0].plot(
        [], [], color="black", marker="x", linestyle="", label="0 (drawn at the floor)"
    )
    axs[0].plot(
        [],
        [],
        color="black",
        linestyle=":",
        linewidth=0.5,
        label="dotted: 50% (a), 5% (b), the decision rule's bars",
    )
    for ax, lab in zip(
        axs, ["Family reads kept (sensitivity)", "Non-family reads kept (false keep)"]
    ):
        ax.set_xticks(CHUNKS)
        ax.set_xlabel("Chunk N (bases from the read start)")
        ax.set_ylabel(lab)
        ax.set_yscale("log")
        ax.set_ylim(F1_FLOOR * 0.7, 1.3)
        ax.set_yticks(
            [F1_FLOOR, 1e-3, 1e-2, 0.1, 1], ["0 or <0.01%", "0.1%", "1%", "10%", "100%"]
        )
    axs[0].axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    axs[1].axhline(FALSE_KEEP_MAX, color="black", linewidth=0.5, linestyle=":")
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.08, 1.0),
        ncol=1,
        frameon=False,
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, path)
    return fig


def fig_all_settings(tab: pl.DataFrame, n: int, best: dict, path: str):
    """SF2: every k-mer setting at chunk N, false keep (x) against sensitivity (y), one
    panel per hold-out; colour = alphabet, marker fill = m. The chosen setting is circled.
    """
    fig, axs = pf.figure(pf.TWO_COLUMN_MM * 0.75, 70, ncols=2, sharey=True)
    alpha_colour = {
        "protein": pf.OKABE_ITO["blue"],
        "dayhoff": pf.OKABE_ITO["orange"],
        "hp": pf.OKABE_ITO["reddish_purple"],
    }
    m_marker = {1: "o", 2: "s", 3: "^"}
    for ax, lvl in zip(axs, ["ii", "iii"]):
        r = tab.filter(
            (pl.col("holdout") == lvl) & (pl.col("chunk_bases_requested") == n)
        )
        for row in r.iter_rows(named=True):
            ax.scatter(
                row["false_keep"],
                row["sensitivity"],
                color=alpha_colour[row["alphabet"]],
                marker=m_marker[row["min_kmers"]],
                s=10,
                zorder=3,
            )
        ax.axvline(FALSE_KEEP_MAX, color="black", linewidth=0.5, linestyle=":")
        ax.plot([0, 1], [0, 1], color=pf.GREY, linewidth=0.5, linestyle="--")
        if best and lvl == best["holdout"]:
            ax.scatter(
                best["false_keep"],
                best["sensitivity"],
                s=80,
                facecolors="none",
                edgecolors="black",
                zorder=4,
            )
            ax.annotate(
                best["setting"],
                (best["false_keep"], best["sensitivity"]),
                xytext=(12, 6),
                textcoords="offset points",
                fontsize=5,
            )
        ax.set_xlabel("Non-family reads kept (false keep)")
        ax.set_title(HOLDOUT_LABEL[lvl], fontsize=6)
    axs[0].set_ylabel(f"Family reads kept (sensitivity), N = {n}")
    handles = [
        plt.Line2D(
            [],
            [],
            color=c,
            marker="o",
            linestyle="",
            label=f"{a} ({ALPHABET_CLASSES[a]} letters)",
        )
        for a, c in alpha_colour.items()
    ]
    handles += [
        plt.Line2D(
            [], [], color="black", marker=mk, linestyle="", label=f"m = {m} k-mers"
        )
        for m, mk in m_marker.items()
    ]
    handles += [
        plt.Line2D([], [], color="black", linestyle=":", label="5% false keep limit"),
        plt.Line2D(
            [], [], color=pf.GREY, linestyle="--", label="family kept = non-family kept"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 1.0),
        ncol=4,
        frameon=False,
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, path)
    return fig


def fig_decision_time(times: pl.DataFrame, path: str):
    """SF3: time per decision in Python, per alphabet and k, against readfish's ~0.8 s chunk."""
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
    times = times.sort(["alphabet", "k"])
    labels = []
    for i, row in enumerate(times.iter_rows(named=True)):
        ax.plot([row["p05_ms"], row["p95_ms"]], [i, i], color=pf.GREY, linewidth=2)
        ax.plot(row["median_ms"], i, "o", color=pf.OKABE_ITO["blue"])
        labels.append(f"{row['alphabet']} k={row['k']}")
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(800, color="black", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ticks = [0.1, 1, 10, 100, 800]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Time per read decision (ms); dotted line = 800 ms chunk")
    ax.plot([], [], "o", color=pf.OKABE_ITO["blue"], label="median")
    ax.plot([], [], color=pf.GREY, linewidth=2, label="5th to 95th percentile")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    pf.save(fig, path)
    return fig


def fig_shuffled(tab: pl.DataFrame, n: int, path: str):
    """SF4: for every k-mer setting, keep rate on real non-family reads (x) against the keep
    rate on shuffled reads (y). Points on the diagonal keep real reads at the chance rate.
    """
    fig, axs = pf.figure(pf.TWO_COLUMN_MM * 0.75, 70, ncols=2, sharey=True)
    alpha_colour = {
        "protein": pf.OKABE_ITO["blue"],
        "dayhoff": pf.OKABE_ITO["orange"],
        "hp": pf.OKABE_ITO["reddish_purple"],
    }
    for ax, lvl in zip(axs, ["ii", "iii"]):
        r = tab.filter(
            (pl.col("holdout") == lvl) & (pl.col("chunk_bases_requested") == n)
        )
        for row in r.iter_rows(named=True):
            ax.scatter(
                max(row["false_keep"], 1e-6),
                max(row["shuffled_keep"], 1e-6),
                color=alpha_colour[row["alphabet"]],
                s=10,
                zorder=3,
            )
        ax.plot([1e-6, 1], [1e-6, 1], color="black", linewidth=0.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [1e-6, 1e-4, 1e-2, 1]
        lab = ["0 (or below 1 in a million)", "1 in 10_000", "1 in 100", "all"]
        ax.set_xticks(ticks, ["≤0.0001%", "0.01%", "1%", "100%"])
        ax.set_yticks(ticks, ["≤0.0001%", "0.01%", "1%", "100%"])
        ax.set_xlabel("Real non-family reads kept")
        ax.set_title(HOLDOUT_LABEL[lvl], fontsize=6)
    axs[0].set_ylabel(f"Shuffled reads kept, N = {n}")
    handles = [
        plt.Line2D(
            [],
            [],
            color=c,
            marker="o",
            linestyle="",
            label=f"{a} ({ALPHABET_CLASSES[a]} letters)",
        )
        for a, c in alpha_colour.items()
    ]
    handles += [plt.Line2D([], [], color="black", linestyle=":", label="equal rates")]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 1.0),
        ncol=4,
        frameon=False,
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, path)
    return fig


def fig_family_choice(fam: pl.DataFrame, comp: pl.DataFrame, path: str):
    """(a) Candidate toxin families: Swiss-Prot entries against reads with a best hit in the
    family; (b) genera in each held-out reference."""
    fig, axs = pf.figure(
        pf.TWO_COLUMN_MM * 0.8, 65, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax = axs[0]
    for row in fam.iter_rows(named=True):
        picked = row["pfam"] == PFAM
        ax.scatter(
            row["n_sprot_entries"],
            max(row["n_reads_best_hit"], 1),
            color=pf.OKABE_ITO["blue"] if picked else pf.GREY,
            s=18 if picked else 8,
            zorder=3,
        )
        ax.annotate(
            row["pfam_name"],
            (row["n_sprot_entries"], max(row["n_reads_best_hit"], 1)),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5,
        )
    ax.axvline(50, color="black", linewidth=0.5, linestyle=":")
    ax.set_yscale("log")
    yt = [1, 10, 100, 1000, 10_000]
    ax.set_yticks(yt, [f"{t:_}" for t in yt])
    ax.set_xlabel("Swiss-Prot 2026_03 entries carrying the family")
    ax.set_ylabel("Sampled reads whose best hit carries it")
    ax.scatter(
        [], [], color=pf.OKABE_ITO["blue"], s=18, label=f"picked: {PFAM_NAME} ({PFAM})"
    )
    ax.scatter([], [], color=pf.GREY, s=8, label="other candidate toxin families")
    ax.plot(
        [], [], color="black", linestyle=":", linewidth=0.5, label="50-entry minimum"
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1)
    ax = axs[1]
    c = comp.sort(["holdout", "n_proteins"], descending=[False, False])
    fam_colour = {
        "Theraphosidae": pf.OKABE_ITO["vermillion"],
        "Conidae": pf.OKABE_ITO["blue"],
        "Metridiidae": pf.OKABE_ITO["bluish_green"],
    }
    genera = c["genus"].unique(maintain_order=True).to_list()
    ypos = {g: i for i, g in enumerate(genera)}
    for row in c.iter_rows(named=True):
        dx = 0.15 if row["holdout"] == "ii" else -0.15
        ax.barh(
            ypos[row["genus"]] + dx,
            row["n_proteins"],
            height=0.28,
            color=fam_colour[row["taxonomic_family"]],
            alpha=1.0 if row["holdout"] == "ii" else 0.45,
        )
        ax.text(
            row["n_proteins"] * 1.15,
            ypos[row["genus"]] + dx,
            str(row["n_proteins"]),
            va="center",
            fontsize=5,
        )
    ax.set_yticks(range(len(genera)), [f"$\\it{{{g}}}$" for g in genera])
    ax.set_xscale("log")
    xt = [1, 10, 100, 1000]
    ax.set_xticks(xt, [str(t) for t in xt])
    ax.set_xlabel("Proteins in the reference")
    hs = [plt.Rectangle((0, 0), 1, 1, color=v) for v in fam_colour.values()]
    hs += [
        plt.Rectangle((0, 0), 1, 1, color="black"),
        plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.45),
    ]
    labs = [
        "Theraphosidae (tarantulas)",
        "Conidae (cone snails)",
        "Metridiidae (sea anemone)",
        "upper bar: genus held out (ii)",
        "lower bar: family held out (iii)",
    ]
    ax.legend(
        hs, labs, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1
    )
    for a, letter in zip(axs, "ab"):
        pf.panel_label(a, letter)
    pf.save(fig, path)
    return fig


def fig_truth(tr: pl.DataFrame, tr_sub: pl.DataFrame, path: str):
    """(a) reads per truth label under the Pfam truth and the UniProt subfamily truth;
    (b) identity of each Pfam-truth family read to its best hit."""
    fig, axs = pf.figure(pf.TWO_COLUMN_MM * 0.7, 55, ncols=2)
    order = ["family", "non_family", "unassigned"]
    names = {"family": "family", "non_family": "non-family", "unassigned": "no hit"}
    ax = axs[0]
    top = 0
    for t, marker, fill, lab in [
        (tr, "o", pf.OKABE_ITO["blue"], f"Pfam truth ({PFAM})"),
        (tr_sub, "o", "none", "UniProt subfamily truth"),
    ]:
        c = t.group_by("label").len()
        n = dict(zip(c["label"], c["len"]))
        top = max(top, max(n.values()))
        for i, l in enumerate(order):
            ax.plot(
                n[l],
                i,
                marker,
                markerfacecolor=fill,
                markeredgecolor=pf.OKABE_ITO["blue"],
                linestyle="",
                label=lab if i == 0 else None,
            )
            ax.annotate(
                f"{n[l]:_}",
                (n[l], i),
                xytext=(0, 5 if fill != "none" else -9),
                textcoords="offset points",
                ha="center",
                fontsize=5,
            )
    ax.set_yticks(range(3), [names[l] for l in order])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0, top * 1.15)
    xt = [0, 50_000, 100_000, 150_000]
    ax.set_xticks(xt, [f"{t:_}" for t in xt])
    ax.set_xlabel("Sampled reads (of 200_000)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax = axs[1]
    f = tr.filter(pl.col("label") == "family")["pident"].to_numpy()
    ax.hist(f, bins=np.arange(30, 90, 2), color=pf.OKABE_ITO["blue"])
    ax.set_xlabel("Identity to best Swiss-Prot hit (%)")
    ax.set_ylabel("Family reads (Pfam truth)")
    for a_, letter in zip(axs, "ab"):
        pf.panel_label(a_, letter)
    pf.save(fig, path)
    return fig


def main_result(
    tab: pl.DataFrame, comp_tab: pl.DataFrame, tr: pl.DataFrame, n: int = 400
):
    """F1 rows (best k-mer setting picked at hold-out ii, fixed; DNA rule; slow bound) at both
    hold-outs, with Wilson intervals and verdict sentences for both hold-outs."""
    best = best_setting(tab, "ii", n)
    k = tab.filter(
        (pl.col("setting") == best["setting"]) & (pl.col("chunk_bases_requested") == n)
    ).with_columns(rule=pl.lit("k-mer rule"))
    c = comp_tab.filter(pl.col("chunk_bases_requested") == n)
    rows = pl.concat([k, c], how="diagonal_relaxed")
    fk = {
        (r["rule"], r["holdout"]): r["false_keep"] for r in rows.iter_rows(named=True)
    }
    rows = rows.with_columns(
        label=pl.struct("rule").map_elements(
            lambda s: (
                f"{s['rule']}"
                + (f", {best['setting']}" if s["rule"] == "k-mer rule" else "")
                + f"; false keep {fk[(s['rule'], 'ii')]:.1%} (genus), {fk[(s['rule'], 'iii')]:.1%} (family)"
            ),
            return_dtype=pl.Utf8,
        )
    )
    ci = [wilson(a, b) for a, b in zip(rows["n_family_kept"], rows["n_family"])]
    rows = rows.with_columns(
        sensitivity_ci_low=pl.Series([x[0] for x in ci]),
        sensitivity_ci_high=pl.Series([x[1] for x in ci]),
    )
    n_fam = tr.filter(pl.col("label") == "family").height
    verdicts = []
    for lvl in ["ii", "iii"]:
        b = rows.filter(
            (pl.col("rule") == "k-mer rule") & (pl.col("holdout") == lvl)
        ).row(0, named=True)
        d = rows.filter(
            (pl.col("rule") == "DNA rule (minimap2)") & (pl.col("holdout") == lvl)
        ).row(0, named=True)
        verdicts.append(verdict(lvl, n, b, d, n_fam))
    return (
        rows.select(
            "rule",
            "label",
            "holdout",
            "n_family_kept",
            "n_family",
            "sensitivity",
            "sensitivity_ci_low",
            "sensitivity_ci_high",
            "false_keep",
            "shuffled_keep",
            "enrichment",
            "pore_time_enrichment",
        ),
        verdicts,
    )


def decision_rule_met(f1_rows: pl.DataFrame) -> tuple[bool, str]:
    """The brief's decision, at the genus hold-out: k-mer sensitivity >= 50% at false keep
    <= 5%, and DNA rule sensitivity < 10%."""
    k = f1_rows.filter(
        (pl.col("rule") == "k-mer rule") & (pl.col("holdout") == "ii")
    ).row(0, named=True)
    d = f1_rows.filter(
        (pl.col("rule") == "DNA rule (minimap2)") & (pl.col("holdout") == "ii")
    ).row(0, named=True)
    ok_k = k["sensitivity"] >= 0.5 and k["false_keep"] <= FALSE_KEEP_MAX
    ok_d = d["sensitivity"] < 0.10
    msg = (
        f"k-mer rule keeps {k['sensitivity']:.1%} of family reads at false keep {k['false_keep']:.1%} "
        f"(needs >= 50% at <= 5%: {'met' if ok_k else 'not met'}); DNA rule keeps {d['sensitivity']:.1%} "
        f"(needs < 10%: {'met' if ok_d else 'not met'})."
    )
    return ok_k and ok_d, msg


def comparator_hits(n: int, rule: str, level: str) -> set[str]:
    """Read ids (real reads only) kept by a comparator."""
    path = f"{DATA}/comparators/N{n}.{'minimap2' if rule.startswith('DNA') else 'diamond'}_{level}.{'paf' if rule.startswith('DNA') else 'tsv'}"
    if os.path.getsize(path) == 0:
        return set()
    q = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        columns=[0],
        new_columns=["qid"],
        truncate_ragged_lines=True,
    )["qid"]
    return {x.split("|")[0] for x in q if x.endswith("|0")}


def diamond_alignment(read_id: str, read: str) -> str:
    """Re-run DIAMOND blastx for one read with the same options as the truth step and return
    DIAMOND's own pairwise alignment text (BLAST pairwise format) for the best hit."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "r.fasta")
        with open(fa, "w") as fo:
            fo.write(f">{read_id}\n{read}\n")
        out = subprocess.run(
            [
                DIAMOND,
                "blastx",
                "-q",
                fa,
                "-d",
                f"{DATA}/ref/sprot_2026_03.dmnd",
                "--evalue",
                "1e-5",
                "-F",
                "15",
                "--sensitive",
                "--max-target-seqs",
                "1",
                "--quiet",
                "--threads",
                "4",
                "--outfmt",
                "0",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    start = out.index(">")
    end = out.find("\n>", start + 1)
    return out[start:] if end < 0 else out[start:end]


def print_hero(best: dict, tr: pl.DataFrame, n: int = 400) -> None:
    """Print the example read: chunk, firing frame, the k-mers that fired with positions,
    the reference residues at those positions, and the DIAMOND alignment of the full read.
    """
    import decide  # scripts/adaptive/decide.py

    fam_ids = set(tr.filter(pl.col("label") == "family")["read_id"])
    for level in ["iii", "ii"]:
        s = (
            scores()
            .filter(
                (pl.col("copy") == 0)
                & (pl.col("holdout") == level)
                & (pl.col("alphabet") == best["alphabet"])
                & (pl.col("k") == best["k"])
            )
            .pipe(with_requested_chunk)
            .filter(pl.col("chunk_bases_requested") == n)
            .filter(pl.col("n_kmers_in_index") >= best["min_kmers"])
            .collect()
        )
        dna = comparator_hits(n, "DNA", level)
        cand = s.filter(pl.col("read_id").is_in(list(fam_ids - dna))).sort(
            ["n_kmers_in_index", "read_id"], descending=[True, False]
        )
        print(
            f"hold-out {level} ({HOLDOUT_LABEL[level]}), {best['setting']}, N = {n}: "
            f"{cand.height} family reads kept by the k-mer rule and not by the DNA rule"
        )
        if cand.height:
            break
    if cand.height == 0:
        print("No read meets the rule at either hold-out.")
        return
    rid = cand["read_id"][0]
    read = read_sequence(rid)
    chunk = read[:n]
    refs = reference_proteins(level)
    index = decide.build_index(refs.values(), best["alphabet"], best["k"])
    frames = dict(decide.six_frames(chunk))
    frame = cand["best_frame"][0]
    prot = frames[frame]
    enc = decide.encode(prot, best["alphabet"])
    fired = [(i, km) for i, km in decide.kmers(enc, best["k"]) if km in index]
    t = tr.filter(pl.col("read_id") == rid).row(0, named=True)
    print(
        f"\nread {rid}, length {t['read_length']} bases, truth: best Swiss-Prot hit {t['best_hit']} "
        f"({t['best_hit_species']}), {t['pident']}% identity, E = {t['evalue']}"
    )
    print(f"\nchunk (first {n} bases, as sequenced):")
    for i in range(0, len(chunk), 80):
        print(f"{i + 1:>5} {chunk[i:i + 80]}")
    off = int(frame[1]) - 1
    print(
        f"\nframe {frame} translation (residue 1 = chunk base {off + 1} {'of the reverse complement' if frame[0] == '-' else ''}):"
    )
    for i in range(0, len(prot), 60):
        print(f"{i + 1:>5} {prot[i:i + 60]}")
    print(
        f"\n{len(fired)} distinct {best['alphabet']} k={best['k']} k-mers in the reference; first ones:"
    )
    # Where each fired k-mer sits in a reference protein (first protein that holds it).
    for i, km in fired[:12]:
        for acc, ps in refs.items():
            e = decide.encode(ps, best["alphabet"])
            j = e.find(km)
            if j >= 0:
                print(
                    f"  read residues {i + 1}-{i + best['k']} {prot[i:i + best['k']]}  =  {acc} residues {j + 1}-{j + best['k']} {ps[j:j + best['k']]}"
                )
                break
    # Ungapped comparison of the read frame with the reference protein that holds the most fired k-mers.
    counts = {}
    for i, km in fired:
        for acc, ps in refs.items():
            j = decide.encode(ps, best["alphabet"]).find(km)
            if j >= 0:
                counts.setdefault((acc, j - i), 0)
                counts[(acc, j - i)] += 1
    (acc, shift), _ = max(counts.items(), key=lambda kv: kv[1])
    ps = refs[acc]
    a0 = max(0, -shift)
    a1 = min(len(prot), len(ps) - shift)
    print(
        f"\nungapped comparison, read frame {frame} against {acc} (offset {shift}), residues:"
    )
    print_block(
        f"read {frame}",
        prot[a0:a1],
        a0 + 1,
        acc,
        ps[a0 + shift : a1 + shift],
        a0 + shift + 1,
    )
    print(f"\nthe same stretch encoded in {best['alphabet']}:")
    print_block(
        f"read {frame}",
        decide.encode(prot[a0:a1], best["alphabet"]),
        a0 + 1,
        acc,
        decide.encode(ps[a0 + shift : a1 + shift], best["alphabet"]),
        a0 + shift + 1,
    )
    print(
        "\nDIAMOND blastx of the full read against Swiss-Prot (the truth step), best hit:"
    )
    print(
        {
            k: t[k]
            for k in [
                "best_hit",
                "pident",
                "aln_length",
                "qstart",
                "qend",
                "sstart",
                "send",
                "evalue",
                "bitscore",
            ]
        }
    )
    print(
        "DIAMOND's own alignment (Query = the read, translated; Sbjct = the Swiss-Prot protein; "
        "'\\' and '/' in the query line mark frameshifts DIAMOND allowed). 'Identities' gives x of y identical:"
    )
    print(diamond_alignment(rid, read))


def shared_kmer_sources(
    level: str = "ii",
    alphabet: str = "protein",
    k: int = 10,
    n: int = 400,
    cache: str | None = None,
) -> pl.DataFrame:
    """For every reference k-mer found in a read's first n bases (any frame): how many family
    and non-family reads contain it, and where it sits in the first reference protein that
    has it (residue range, and the Swiss-Prot feature it falls in)."""
    if cache and os.path.exists(cache):
        return pl.read_csv(cache)
    import collections

    import decide

    tr = truth()
    lab = dict(zip(tr["read_id"], tr["label"]))
    refs = reference_proteins(level)
    index = decide.build_index(refs.values(), alphabet, k)
    cnt = {"family": collections.Counter(), "non_family": collections.Counter()}
    for rid, seq in decide.read_records(
        f"{DATA}/reads/{RUN}_sample200k_seed20261014.fastq"
    ):
        lb = lab.get(rid)
        if lb not in cnt:
            continue
        hits = set()
        for _, p in decide.six_frames(seq[:n]):
            hits |= {
                km
                for _, km in decide.kmers(decide.encode(p, alphabet), k)
                if km in index
            }
        cnt[lb].update(hits)
    feats = pl.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "292_data",
            f"reference_features_{PFAM}_sprot_2026_03.csv",
        )
    )
    rows = []
    for km in set(cnt["family"]) | set(cnt["non_family"]):
        acc, j = next(
            (a, decide.encode(s, alphabet).find(km))
            for a, s in sorted(refs.items())
            if decide.encode(s, alphabet).find(km) >= 0
        )
        start, end = j + 1, j + k
        f = feats.filter(
            (pl.col("accession") == acc)
            & (pl.col("start") <= start)
            & (pl.col("end") >= end)
            & (pl.col("feature") != "CHAIN")
        )
        rows.append(
            {
                "kmer": km,
                "n_family_reads": cnt["family"][km],
                "n_non_family_reads": cnt["non_family"][km],
                "reference": acc,
                "ref_start": start,
                "ref_end": end,
                "ref_feature": (
                    f["feature"][0] if f.height else "mature peptide or chain"
                ),
            }
        )
    out = pl.DataFrame(rows).sort("n_non_family_reads", descending=True)
    if cache:
        out.write_csv(cache)
    return out


def fig_shared_kmers(src: pl.DataFrame, path: str, top: int = 12):
    """Reads (family and non-family) containing each of the most common shared k-mers, with
    where the k-mer sits in the reference protein."""
    d = (
        src.with_columns(total=pl.col("n_family_reads") + pl.col("n_non_family_reads"))
        .sort("total", descending=True)
        .head(top)
    )
    d = d.reverse()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM * 0.75, 75)
    feat_marker = {"SIGNAL": "o", "PROPEP": "s", "mature peptide or chain": "^"}
    ys = range(d.height)
    for y, row in zip(ys, d.iter_rows(named=True)):
        fam_x = max(row["n_family_reads"], 0.5)
        ax.plot(
            [fam_x, row["n_non_family_reads"]], [y, y], color=pf.GREY, linewidth=0.5
        )
        ax.plot(
            fam_x,
            y,
            marker=feat_marker[row["ref_feature"]] if row["n_family_reads"] else "x",
            color=pf.OKABE_ITO["blue"],
            linestyle="",
        )
        ax.plot(
            row["n_non_family_reads"],
            y,
            marker=feat_marker[row["ref_feature"]],
            color=pf.OKABE_ITO["vermillion"],
            linestyle="",
        )
    ax.set_yticks(
        list(ys),
        [
            f"{r['kmer']}  ({r['reference']} {r['ref_start']}-{r['ref_end']})"
            for r in d.iter_rows(named=True)
        ],
        family="monospace",
    )
    ax.set_xscale("log")
    xt = [0.5, 1, 10, 100, 1000, 10_000]
    ax.set_xticks(xt, ["0"] + [f"{t:_}" for t in xt[1:]])
    ax.set_xlabel("Reads with the k-mer in their first 400 bases")
    hs = [
        plt.Line2D(
            [],
            [],
            color=pf.OKABE_ITO["blue"],
            marker="o",
            linestyle="",
            label=f"family reads ({PFAM_NAME})",
        ),
        plt.Line2D(
            [],
            [],
            color=pf.OKABE_ITO["vermillion"],
            marker="o",
            linestyle="",
            label="non-family reads",
        ),
    ]
    hs += [
        plt.Line2D(
            [],
            [],
            color="black",
            marker=m,
            linestyle="",
            label=f"k-mer in the reference's {'signal peptide' if f == 'SIGNAL' else 'propeptide' if f == 'PROPEP' else f}",
        )
        for f, m in feat_marker.items()
    ]
    hs += [
        plt.Line2D(
            [],
            [],
            color=pf.OKABE_ITO["blue"],
            marker="x",
            linestyle="",
            label="no family read (0)",
        )
    ]
    ax.legend(
        handles=hs, loc="lower left", bbox_to_anchor=(-0.6, 1.0), ncol=2, frameon=False
    )
    pf.save(fig, path)
    return fig


def print_signal_peptide_example(
    src: pl.DataFrame,
    level: str = "ii",
    alphabet: str = "protein",
    k: int = 10,
    n: int = 400,
) -> None:
    """One non-family read that carries the most common shared k-mer: its frame, the reference
    residues around the k-mer, an ungapped comparison, and the read's own best hit."""
    import decide

    top = src.sort("n_non_family_reads", descending=True).row(0, named=True)
    tr = truth()
    non = tr.filter(pl.col("label") == "non_family").sort("read_id")
    refs = reference_proteins(level)
    ref = refs[top["reference"]]
    for rid, seq in decide.read_records(
        f"{DATA}/reads/{RUN}_sample200k_seed20261014.fastq"
    ):
        if rid not in set(non["read_id"].head(20_000)):
            continue
        for frame, p in decide.six_frames(seq[:n]):
            j = p.find(top["kmer"])
            if j >= 0:
                t = tr.filter(pl.col("read_id") == rid).row(0, named=True)
                print(
                    f"most common shared k-mer: {top['kmer']} ({top['n_family_reads']:_} family reads, "
                    f"{top['n_non_family_reads']:_} non-family reads); in {top['reference']} residues "
                    f"{top['ref_start']}-{top['ref_end']}, inside its {top['ref_feature']}"
                )
                print(
                    f"example non-family read {rid}: best Swiss-Prot hit {t['best_hit']} ({t['best_hit_species']}), "
                    f"Pfam {t['best_hit_pfam'] or 'none'}, {t['pident']}% identity"
                )
                shift = (top["ref_start"] - 1) - j
                a0, a1 = max(0, -shift), min(len(p), len(ref) - shift)
                a1 = min(a1, a0 + 60)
                print_block(
                    f"read {frame}",
                    p[a0:a1],
                    a0 + 1,
                    top["reference"],
                    ref[a0 + shift : a1 + shift],
                    a0 + shift + 1,
                )
                return


def subfamily_truth(tr: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Second truth: a read is a family read when its best hit belongs to a UniProt toxin
    subfamily ("CC -!- SIMILARITY" line) that one of the tarantula proteins in reference (ii)
    belongs to. The Pfam truth splits such a subfamily: B1P1F1 (U16-theraphotoxin-Cg1a)
    carries PF02950 while its paralog B1P1F2 (Cg1b, same subfamily) has no Pfam line."""
    here = os.path.dirname(os.path.abspath(__file__))
    sim = pl.read_csv(
        os.path.join(
            here,
            "292_data",
            f"uniprot_similarity_best_hits_and_{PFAM}_sprot_2026_03.csv",
        )
    )
    comp = pl.read_parquet(f"{DATA}/ref/sprot_2026_03_table.parquet").select(
        "accession", "lineage"
    )
    ref_ii = set(reference_proteins("ii"))
    tarantula = comp.filter(
        pl.col("accession").is_in(list(ref_ii))
        & pl.col("lineage").str.contains("Theraphosidae")
    )["accession"]
    subfams = sorted(
        set(
            sim.filter(pl.col("accession").is_in(tarantula.to_list()))[
                "similarity"
            ].to_list()
        )
        - {""}
    )
    out = tr.join(
        sim, left_on="best_hit", right_on="accession", how="left"
    ).with_columns(
        label=pl.when(pl.col("best_hit").is_null())
        .then(pl.lit("unassigned"))
        .when(pl.col("similarity").is_in(subfams))
        .then(pl.lit("family"))
        .otherwise(pl.lit("non_family"))
    )
    return out.drop("similarity"), subfams
