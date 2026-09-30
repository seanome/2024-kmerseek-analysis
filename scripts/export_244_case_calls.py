#!/usr/bin/env python3
"""Export the call coordinates behind every case of tables/244_hero_candidates.csv.

Writes, for every case (one case = one CSV row, ``case_id`` = its 0-based row index):

* tables/244_case_calls.csv: one row per (case, tool), nine tools per case, with the
  coordinates notebook 244 draws (1-based, inclusive on both proteins).
* tables/244_case_sequences.fasta: the human and target sequence of every case, read from
  the QfO release FASTAs the midi-plus run searched (params.qfo_dir in the pipeline).
* figures/244_cases/<case_id>_<gene>_<feature_type>_<species>.png (150 dpi) and .pdf: the
  notebook's two-panel case figure.
* figures/244_cases/contact_sheet_<feature_type>.png: the human panels of one feature type,
  two across.

Every coordinate comes from the landing tables (scripts/reduce_swissprot_instance_landing.py)
and the Kyte-Doolittle scan instances, through hero_example_utils.load_cases and
candidate_bars, the same code the notebook draws its top 10 with.

Run with the 2025-kmerseek-analysis env:
    python scripts/export_244_case_calls.py [--no-figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import hero_example_utils as he  # noqa: E402

TAB = ROOT / "tables"
FIG = ROOT / "figures" / "244_cases"


def export_calls(cases: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in cases.iter_rows(named=True):
        tlen = len(he.sequences(r["species"], {r["target"]})[r["target"]])
        rows += he.case_calls(r, tlen)
    return pl.DataFrame(rows, infer_schema_length=None)


def write_fasta(cases: pl.DataFrame, path: Path) -> int:
    """Human then target sequences, each accession once, in case order."""
    seen: set[tuple[str, str]] = set()
    n = 0
    with open(path, "w") as fh:
        for r in cases.iter_rows(named=True):
            for sp, acc in (("human", r["accession"]), (r["species"], r["target"])):
                if (sp, acc) in seen:
                    continue
                seen.add((sp, acc))
                seq = he.sequences(sp, {acc})[acc]
                name = he.protein_name(sp, acc)
                fh.write(f">{acc} species={sp} {name}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i : i + 60] + "\n")
                n += 1
    return n


def contact_sheet(cases: pl.DataFrame, notes: pl.DataFrame, ftype: str, path: Path):
    """Human panels of one feature type, two across, legend above the first row."""
    sub = cases.filter(pl.col("pfam_id") == ftype)
    n = sub.height
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(22, 4.4 * nrow + 1.6), squeeze=False)
    for ax, r in zip(axes.flat, sub.iter_rows(named=True)):
        bars = he.candidate_bars(r, he.case_other(r))
        he.draw_human_panel(ax, r, bars, notes)
        sym = r["hgnc_symbol"] or r["accession"]
        ax.set_title(
            f"{r['case_id']}. human {sym} \"{he.short_note(r['note'])}\" "
            f"({r['feature_length']} aa) and {r['species']} {r['target']}",
            fontsize=9.5,
            loc="left",
        )
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    top = 1 - 1.6 / fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.0)
    he.category_legend(fig, top + 0.002)
    fig.suptitle(
        f"{ftype}: {n} cases, human protein only (case number = row of 244_hero_candidates.csv)",
        y=1 + 0.35 / fig.get_size_inches()[1],
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cases = he.load_cases()
    calls = export_calls(cases)
    calls.write_csv(TAB / "244_case_calls.csv")
    print(
        f"wrote {TAB / '244_case_calls.csv'}: {calls.height} rows, {cases.height} cases"
    )
    n_seq = write_fasta(cases, TAB / "244_case_sequences.fasta")
    print(f"wrote {TAB / '244_case_sequences.fasta'}: {n_seq} sequences")
    if args.no_figures:
        return

    FIG.mkdir(parents=True, exist_ok=True)
    notes = pl.read_parquet(he.EXTRACT / "244_swissprot_feature_notes.parquet")
    for r in cases.iter_rows(named=True):
        fig = he.case_figure(
            r, notes, r["case_id"], FIG / f"{he.case_file_stem(r)}.png"
        )
        plt.close(fig)
    print(f"wrote {cases.height} case figures (PNG + PDF) to {FIG}")
    for ftype in sorted(cases["pfam_id"].unique().to_list()):
        contact_sheet(cases, notes, ftype, FIG / f"contact_sheet_{ftype}.png")
    print(f"wrote {cases['pfam_id'].n_unique()} contact sheets")


if __name__ == "__main__":
    main()
