#!/usr/bin/env python3
"""PR 160's headline figures: the best-hit rows beside the main run's rows, twice.

    fmax_best_hit_vs_qfo_relatives_midi_plus           x = Fmax
    median_iou_best_hit_vs_qfo_relatives_midi_plus     x = median IoU of correct calls

Both figures have the same rows, colours and shapes. Top to bottom:
  - each QfO relative of the midi-plus run: phmmer, MMseqs2 seq-seq -s 7 and kmerseek
    (the gbmr4 k21 setting on its tool card, and the best of its 15 settings for that
    relative, picked after seeing the result, so an upper bound). Each hit's whole aligned
    region is the call, with duplicate calls kept (filled) and merged (open). Values come
    from PR 148's results.parquet (experiment qfo_region_midi_plus). The IoU figure's
    correct-call counts come from the run's all_domain_metrics.parquet, whose Fmax and IoU
    must equal PR 148's.
  - Swiss-Prot 2026_03 minus Mammalia, one row per PR 160 setting: the best hit, scored with
    the main run's call (whole aligned region) and with PR 160's call (the hit's Pfam
    domains placed through the alignment), from call_rules.parquet.
  - hmmscan against Pfam-A, the ceiling.
All rows: 998 human midi-plus queries, Pfam-A truth, split "all", stratum "all".

    python scripts/annotator_g1_figure.py --pr148 <PR 148 results.parquet> \\
        --qfo-metrics <midi-plus all_domain_metrics.parquet> \\
        --call-rules results/leaderboard/annotator_tools/call_rules.parquet \\
        --rows results/leaderboard/annotator_tools/rows.parquet
"""

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIGDIR = ROOT / "figures/annotator_tools"
KMER_CARD = "gbmr4_k21_lcFalse"
K_CARD = "kmerseek, gbmr4 k21 (the setting on its tool card)"
K_BEST = (
    "kmerseek, best of its 15 settings for each relative on this figure's score\n"
    "(picked after seeing the result: an upper bound)"
)
COLOUR = {
    "phmmer": pf.OKABE_ITO["orange"],
    "MMseqs2": pf.OKABE_ITO["blue"],
    K_CARD: pf.OKABE_ITO["vermillion"],
    # A light tint of the same hue: the same tool, but a pick made after the fact.
    K_BEST: "#F0B48C",
    "DIAMOND": pf.OKABE_ITO["bluish_green"],
    "hmmscan": "#000000",
}
RAW = "every hit, whole aligned region"
DEDUP = "every hit, whole aligned region, duplicates merged"
BEST_WHOLE = "best hit, whole aligned region"
PLACED = "best hit, Pfam domain placed through the alignment"
CEIL = "Pfam-A models on the query (hmmscan)"
RULE = {
    RAW: dict(marker="o", fill=True),
    DEDUP: dict(marker="o", fill=False),
    BEST_WHOLE: dict(marker="^", fill=True),
    PLACED: dict(marker="s", fill=True),
    CEIL: dict(marker="D", fill=True),
}
NAME = {
    "ptroglodytes": "chimpanzee",
    "rnorvegicus": "rat",
    "btaurus": "cow",
    "clupus": "dog",
    "mdomestica": "opossum",
    "xtropicalis": "Xenopus",
    "ciona": "Ciona",
    "fly": "Drosophila",
    "worm": "C. elegans",
    "arabidopsis": "Arabidopsis",
    "ecoli": "E. coli",
}
METRIC = {
    "fmax": dict(
        out="fmax_best_hit_vs_qfo_relatives_midi_plus",
        xlabel="Fmax over Pfam domains on 998 human proteins (higher is better)",
        right="truth instances",
        count="n_truth_instances",
    ),
    "median_iou_tp": dict(
        out="median_iou_best_hit_vs_qfo_relatives_midi_plus",
        xlabel="Median IoU of correct calls: overlap between call and true domain, "
        "divided by their union (higher is better)",
        right="correct calls\n(fewest to most\nover the row's marks)",
        count="n_tp_calls",
    ),
}


NONE = "no correct calls, so no IoU (drawn at 0)"


def mark(x, colour, rule, n):
    return {"x": x, "colour": colour, "rule": rule, "n": n}


def build_rows(
    metric: str,
    pr148: pl.DataFrame,
    qm: pl.DataFrame,
    rules: pl.DataFrame,
    ours: pl.DataFrame,
) -> list[tuple]:
    """(label, group, [marks]) per row; group 0 QfO, 1 Swiss-Prot, 2 ceiling."""
    r = pr148.filter(
        (pl.col("experiment") == "qfo_region_midi_plus")
        & (pl.col("metric") == metric)
        & (pl.col("stratum_axis") == "all")
        & (pl.col("truth_source") == "Pfam-A domain instances")
    )
    q = qm.filter(
        (pl.col("split") == "all")
        & (pl.col("stratum_axis") == "all")
        & (pl.col("truth_set") == "pfam")
    )

    def count(tool, setting, species, calls, value):
        """Correct calls (or truth instances) behind one PR 148 value, from the run's own
        metrics, after checking the run's value is PR 148's."""
        m = q.filter(
            (pl.col("tool") == tool)
            & (pl.col("variant") == setting)
            & (pl.col("species") == species)
            & (pl.col("dedup_transfers") == (calls == "dedup"))
        )
        if m.height != 1:
            raise SystemExit(
                f"{tool} {setting} {species} {calls}: {m.height} metric rows"
            )
        got = m[metric][0]
        if not (got is None and value is None) and abs(got - value) > 1e-12:
            raise SystemExit(
                f"{tool} {setting} {species} {calls}: {got} != PR 148 {value}"
            )
        return m[METRIC[metric]["count"]][0]

    kmer = r.filter(pl.col("tool") == "kmerseek")
    if kmer["setting"].n_unique() != 15:
        raise SystemExit(
            f"expected 15 kmerseek settings, found {kmer['setting'].n_unique()}"
        )
    qfo = r.filter(pl.col("tool").is_in(["hmmer3_phmmer", "mmseqs2_seqseq"]))
    rows = []
    order = (
        qfo.select("target_db", "target_mya_to_human")
        .unique()
        .sort("target_mya_to_human", "target_db")
    )
    for db, mya in order.iter_rows():
        sp = db.removeprefix("QfO ")
        marks = []
        for calls, rule in (("raw", RAW), ("dedup", DEDUP)):
            kc = kmer.filter(
                (pl.col("target_db") == db)
                & (pl.col("dedup_or_raw") == calls)
                & pl.col("value").is_not_null()
            )
            if kc.height:
                b = kc.sort("value", "setting", descending=[True, False]).row(
                    0, named=True
                )
                marks.insert(
                    0,
                    mark(
                        b["value"],
                        COLOUR[K_BEST],
                        rule,
                        count("kmerseek", b["setting"], sp, calls, b["value"]),
                    ),
                )
        for t in qfo.filter(pl.col("target_db") == db).iter_rows(named=True):
            tool = "phmmer" if t["tool"] == "hmmer3_phmmer" else "MMseqs2"
            rule = RAW if t["dedup_or_raw"] == "raw" else DEDUP
            marks.append(
                mark(
                    t["value"],
                    COLOUR[tool],
                    rule,
                    count(t["tool"], t["setting"], sp, t["dedup_or_raw"], t["value"]),
                )
            )
        for t in kmer.filter(
            (pl.col("target_db") == db) & (pl.col("setting") == KMER_CARD)
        ).iter_rows(named=True):
            if t["value"] is None:
                continue
            rule = RAW if t["dedup_or_raw"] == "raw" else DEDUP
            marks.append(
                mark(
                    t["value"],
                    COLOUR[K_CARD],
                    rule,
                    count("kmerseek", KMER_CARD, sp, t["dedup_or_raw"], t["value"]),
                )
            )
        rows.append((f"QfO {NAME.get(sp, sp)} ({mya:g} Mya)", 0, marks))
    labels = {
        ("diamond_besthit", "sensitive"): ("DIAMOND --sensitive", "DIAMOND"),
        ("diamond_besthit", "ultra-sensitive"): (
            "DIAMOND --ultra-sensitive",
            "DIAMOND",
        ),
        ("mmseqs2_besthit", "s7.5"): ("MMseqs2 -s 7.5", "MMseqs2"),
    }
    for (tool, setting), (label, colour_key) in labels.items():
        g = rules.filter((pl.col("tool") == tool) & (pl.col("setting") == setting))
        marks = [
            mark(
                t[metric],
                COLOUR[colour_key],
                BEST_WHOLE if t["call_rule"] == "whole_region" else PLACED,
                t[METRIC[metric]["count"]],
            )
            for t in g.iter_rows(named=True)
        ]
        # The placed value must be the pipeline's own row.
        row = ours.filter(
            (pl.col("tool") == tool)
            & (pl.col("setting") == setting)
            & (pl.col("metric") == metric)
            & (pl.col("stratum_axis") == "all")
        )
        placed = g.filter(pl.col("call_rule") == "placed")[metric][0]
        assert abs(row["value"][0] - placed) < 1e-12, (tool, setting, metric)
        rows.append((f"Swiss-Prot minus Mammalia, {label}", 1, marks))
    h = r.filter(pl.col("tool") == "hmmscan").row(0, named=True)
    rows.append(
        (
            "Pfam-A library (hmmscan, the ceiling)",
            2,
            [
                mark(
                    h["value"],
                    COLOUR["hmmscan"],
                    CEIL,
                    count("hmmscan", h["setting"], "all", "raw", h["value"]),
                )
            ],
        )
    )
    return rows


def draw(metric: str, rows: list[tuple]) -> Path:
    spec = METRIC[metric]
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 140)
    y, ys, last = 0.0, [], 0
    for _, group, _ in rows:
        if group != last:
            y += 0.8
            last = group
        ys.append(y)
        y += 1
    for yy, (label, _, marks) in zip(ys, rows):
        ax.axhline(yy, color="#dddddd", lw=0.3, zorder=0)
        for m in marks:
            if metric == "median_iou_tp" and m["n"] == 0:
                # A median over no correct calls does not exist: a grey cross, not a 0.
                ax.scatter(
                    0,
                    yy,
                    s=14,
                    marker="x",
                    color="#999999",
                    linewidths=0.8,
                    zorder=3,
                    clip_on=False,
                )
                continue
            st = RULE[m["rule"]]
            ax.scatter(
                m["x"],
                yy,
                s=22,
                marker=st["marker"],
                linewidths=0.8,
                facecolors=m["colour"] if st["fill"] else "white",
                edgecolors=m["colour"],
                zorder=3,
            )
        ns = sorted(m["n"] for m in marks if m["n"] is not None and m["n"] > 0)
        text = (
            f"n = {ns[0]:,}"
            if ns and ns[0] == ns[-1]
            else f"n = {ns[0]:,} to {ns[-1]:,}" if ns else ""
        )
        ax.text(
            1.005, yy, text, transform=ax.get_yaxis_transform(), va="center", ha="left"
        )
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    # A little room past 1, so a mark at exactly 1 (an IoU of 1) is not cut by the edge.
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(spec["xlabel"])
    ax.text(
        1.005,
        -0.9,
        spec["right"],
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="left",
    )
    # Two keys above the plot: colour = search tool (drawn as a line, so it is not read
    # as a shape); shape and fill = how the call is made (drawn in grey).
    tools = [Line2D([], [], color=c, lw=3, label=k) for k, c in COLOUR.items()]
    shapes = [
        Line2D(
            [],
            [],
            ls="",
            marker=st["marker"],
            markersize=5,
            markerfacecolor="#555555" if st["fill"] else "white",
            markeredgecolor="#555555",
            label=k,
        )
        for k, st in RULE.items()
    ]
    if metric == "median_iou_tp":
        shapes.append(
            Line2D([], [], ls="", marker="x", markersize=5, color="#999999", label=NONE)
        )
    fig.legend(
        handles=tools,
        title="Colour: search tool",
        loc="outside upper left",
        ncol=1,
        frameon=False,
        alignment="left",
    )
    fig.legend(
        handles=shapes,
        title="Shape: how the call is made",
        loc="outside upper right",
        ncol=1,
        frameon=False,
        alignment="left",
    )
    out = FIGDIR / spec["out"]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pf.save(fig, str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pr148", type=Path, required=True)
    ap.add_argument("--qfo-metrics", type=Path, required=True)
    ap.add_argument("--call-rules", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    a = ap.parse_args()
    pr148 = pl.read_parquet(a.pr148)
    qm = pl.read_parquet(a.qfo_metrics)
    rules = pl.read_parquet(a.call_rules)
    ours = pl.read_parquet(a.rows)
    for metric in METRIC:
        rows = build_rows(metric, pr148, qm, rules, ours)
        print(f"== {metric}")
        for label, _, marks in rows:
            print(
                f"{label:52s} "
                + "  ".join(
                    f"{m['x']:.3f} (n={m['n']}) [{m['rule'][:22]}]" for m in marks
                )
            )
        print(f"wrote {draw(metric, rows)}.png")


if __name__ == "__main__":
    main()
