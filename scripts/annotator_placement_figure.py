#!/usr/bin/env python3
"""Figure for PR body G1: placing a hit protein's domain on the query, exactly through
the alignment versus on the straight line between the hit's ends (PR 121's rule).

Data: the committed CI fixture, not a benchmark result. MMseqs2 -s 7.5 best hit of human
ZNF Q9HCG1 against the 120-protein yeast fixture is P39933; hmmscan --cut_ga puts a C2H2
zinc finger (PF00096) at P39933 163-186. Everything drawn is read from
nextflow-runs/qfo-pfam-region-benchmark/tests/fixtures (annotator/ and annotations/).

    python scripts/annotator_placement_figure.py
"""

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "nextflow-runs/qfo-pfam-region-benchmark/bin"))
import annotator_best_hit as abh  # noqa: E402
import pubfig as pf  # noqa: E402

FIX = ROOT / "nextflow-runs/qfo-pfam-region-benchmark/tests/fixtures"
QUERY, TARGET, FAMILY, TARGET_DOMAIN = "Q9HCG1", "P39933", "PF00096", (163, 186)
OUT = ROOT / "figures/annotator_tools/placement_exact_vs_line_fixture"


def main() -> None:
    hits = abh.read_hits(FIX / "annotator/mmseqs2_besthit.s7.5.hits.tsv.gz")
    best = abh.best_hits(hits).filter(pl.col("query") == QUERY)
    h = best.row(0, named=True)
    assert h["target"] == TARGET, h["target"]
    doms = abh.read_domains(FIX / "annotator/hit_target_domains.tsv.gz")
    tdom = doms.filter((pl.col("target") == TARGET) & (pl.col("pfam_id") == FAMILY))
    calls, _ = abh.place_domains(best, tdom, 0.5)
    call = (
        calls.filter(
            (pl.col("pfam_id") == FAMILY)
            & (pl.col("linear_qstart") != pl.col("qstart"))
        )
        .sort((pl.col("linear_qstart") - pl.col("qstart")).abs(), descending=True)
        .row(0, named=True)
    )
    truth = pl.read_parquet(FIX / "annotations/human_pfam_domains.parquet").filter(
        (pl.col("accession") == QUERY) & (pl.col("pfam_id") == FAMILY)
    )
    qlen = truth["protein_length"][0]
    tlen = len(abh.read_fasta(FIX / "annotator/hit_targets.fasta")[TARGET])
    print(
        f"hit {QUERY} {h['qstart']}-{h['qend']} ~ {TARGET} {h['tstart']}-{h['tend']}, "
        f"{h['pident']}% identity, E {h['evalue']:.1e}; target {FAMILY} "
        f"{TARGET_DOMAIN[0]}-{TARGET_DOMAIN[1]} -> exact {call['qstart']}-{call['qend']}, "
        f"straight line {call['linear_qstart']}-{call['linear_qend']}"
    )

    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 48)
    c_exact, c_line = pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"]
    y_t, y_q, y_e, y_l = 3.0, 2.0, 1.25, 0.75
    box_h = 0.28

    def protein(y, length, label):
        ax.plot([1, length], [y, y], color="black", lw=1)
        ax.text(-12, y, label, ha="right", va="center")

    def box(y, s, e, color, h=box_h, label=None):
        ax.add_patch(
            pf.plt.Rectangle(
                (s, y - h / 2), e - s + 1, h, color=color, lw=0, label=label
            )
        )

    protein(y_t, tlen, f"yeast {TARGET}\n(best hit)")
    protein(y_q, qlen, f"human {QUERY}\n(query)")
    for r in tdom.iter_rows(named=True):
        box(y_t, r["dstart"], r["dend"], pf.GREY)
    for i, r in enumerate(truth.iter_rows(named=True)):
        box(
            y_q,
            r["domain_start"],
            r["domain_end"],
            pf.GREY,
            label="C2H2 zinc finger (PF00096), Pfam annotation" if i == 0 else None,
        )
    # The aligned span of the hit, as a bar under each protein line.
    for y, s, e in ((y_t, h["tstart"], h["tend"]), (y_q, h["qstart"], h["qend"])):
        ax.plot(
            [s, e],
            [y - 0.24, y - 0.24],
            color="black",
            lw=2.5,
            solid_capstyle="butt",
            label="aligned part of the best hit" if y == y_q else None,
        )
    box(
        y_e,
        call["qstart"],
        call["qend"],
        c_exact,
        label=f"call placed through the alignment ({call['qstart']}-{call['qend']})",
    )
    box(
        y_l,
        call["linear_qstart"],
        call["linear_qend"],
        c_line,
        label="call placed on the straight line, PR 121 "
        f"({call['linear_qstart']}-{call['linear_qend']})",
    )
    # Which target domain the two calls come from.
    for x_t, x_q in (
        (TARGET_DOMAIN[0], call["qstart"]),
        (TARGET_DOMAIN[1], call["qend"]),
    ):
        ax.plot([x_t, x_q], [y_t - box_h / 2, y_e + box_h / 2], color=c_exact, lw=0.5)
    ax.set_xlim(-5, max(qlen, tlen) + 5)
    ax.set_ylim(0.4, 3.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Residue position (aa)")
    pf.shared_legend(fig, ncol=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pf.save(fig, str(OUT))


if __name__ == "__main__":
    main()
