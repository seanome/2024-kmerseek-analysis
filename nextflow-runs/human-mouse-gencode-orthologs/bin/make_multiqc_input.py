#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
make_multiqc_input.py

Convert the human-mouse GENCODE ortholog k-mer sweep summary (kmer_sweep_summary.json,
produced by aggregateResults in kmerseek_human_mouse_orthologs.nf) into MultiQC
custom-content files, so the full encoding x ksize sweep can be read as one report
instead of grepping through ~90 ortholog_evaluation.*.summary.txt files.

Every value here comes straight out of the JSON — this script makes no claims about
which encoding/ksize combos *should* exist (that depends on the workflow's parameter
channels, which can change run to run). It only reports what actually completed.
A combo counts as "complete" iff its JSON summary block contains an 'mht' key; a
combo present in the sweep file without one failed evaluation (e.g. empty search
results) and is reported as incomplete rather than silently dropped or crashing.

Outputs (all written to <outdir>/):
  completeness_mqc.yaml         — heatmap: 1=complete, 0=incomplete, per encoding x ksize
  encoding_completion_mqc.tsv   — table: n complete / incomplete per encoding
  summary_table_mqc.tsv         — table: full metrics for every complete combo
  bh_recall_vs_ksize_mqc.yaml   — linegraph: BH recall vs ksize, one line per encoding
  bh_precision_vs_ksize_mqc.yaml— linegraph: BH precision vs ksize, one line per encoding
  total_hits_vs_ksize_mqc.yaml  — linegraph: search-space size (total_hits) vs ksize
  multiqc_config.yaml           — section order, titles, colors

Usage:
    make_multiqc_input.py <kmer_sweep_summary.json> <outdir>
"""

import json
import sys
from pathlib import Path

ENCODING_COLORS = {
    "hp": "#9C27B0",
    "hp-lehninger": "#AB47BC",
    "hp-thomas-dill": "#7B1FA2",
    "hp-thomas-dill-no-c": "#CE93D8",
    "hp-kyte-doolittle": "#BA68C8",
    "hp-lehninger-plus-c": "#E1BEE7",
    "hp-pbotc-1st-ed": "#4A148C",
    "dayhoff": "#1E88E5",
    "protein": "#43A047",
}

MHT_METHODS = ["bonferroni", "bh", "by", "two_stage_bh"]


def color_for(encoding: str) -> str:
    return ENCODING_COLORS.get(encoding, "#888888")


def load_results(sweep_json: str) -> list[dict]:
    with open(sweep_json) as f:
        return json.load(f)["results"]


def is_complete(r: dict) -> bool:
    return "mht" in r


def write_completeness_heatmap(path: Path, results: list[dict]) -> None:
    encodings = sorted({r["encoding"] for r in results})
    ksizes = sorted({r["ksize"] for r in results})

    status = {(r["encoding"], r["ksize"]): (1 if is_complete(r) else 0) for r in results}

    lines = [
        f"id: 'sweep_completeness'",
        f"section_name: 'Sweep completeness'",
        f"description: 'Which encoding x ksize combinations produced a valid evaluation "
        f"(had a poisson-test p-value column and non-empty search results). "
        f"1 = complete, 0 = present in the sweep summary but incomplete (e.g. empty search "
        f"results). Blank = no entry at all for that combination.'",
        f"plot_type: 'heatmap'",
        f"pconfig:",
        f"  id: 'sweep_completeness_plot'",
        f"  title: 'Sweep completeness (encoding x ksize)'",
        f"  xlab: 'ksize'",
        f"  ylab: 'encoding'",
        f"  min: 0",
        f"  max: 1",
        f"xcats: [{', '.join(str(k) for k in ksizes)}]",
        f"ycats: [{', '.join(repr(e) for e in encodings)}]",
        f"data:",
    ]
    for enc in encodings:
        row = [status.get((enc, k), "null") for k in ksizes]
        lines.append(f"  - [{', '.join(str(v) for v in row)}]")

    path.write_text("\n".join(lines) + "\n")


def write_encoding_completion_table(path: Path, results: list[dict]) -> None:
    encodings = sorted({r["encoding"] for r in results})
    with open(path, "w") as f:
        f.write("# plot_type: 'table'\n")
        f.write("# section_name: 'Completion by encoding'\n")
        f.write("# description: 'Count of complete vs incomplete evaluations per encoding, "
                "across all ksizes attempted in this sweep.'\n")
        f.write("# pconfig:\n")
        f.write("#   namespace: 'Human-Mouse Ortholog Sweep'\n")
        f.write("Sample\tn_attempted\tn_complete\tn_incomplete\tpct_complete\n")
        for enc in encodings:
            rows = [r for r in results if r["encoding"] == enc]
            n_complete = sum(is_complete(r) for r in rows)
            n_total = len(rows)
            pct = round(100 * n_complete / n_total, 1) if n_total else 0.0
            f.write(f"{enc}\t{n_total}\t{n_complete}\t{n_total - n_complete}\t{pct}\n")


def write_summary_table(path: Path, results: list[dict]) -> None:
    complete = [r for r in results if is_complete(r)]
    complete.sort(key=lambda r: (r["encoding"], r["ksize"]))

    cols = ["ksize", "n_ortholog", "n_non_ortholog", "total_hits"]
    for m in MHT_METHODS:
        cols += [f"{m}_precision", f"{m}_recall"]

    with open(path, "w") as f:
        f.write("# plot_type: 'table'\n")
        f.write("# section_name: 'Sweep metrics (complete combos only)'\n")
        f.write("# description: 'Multiple-hypothesis-testing precision/recall (alpha=0.05) "
                "for every encoding x ksize combination that completed evaluation.'\n")
        f.write("# pconfig:\n")
        f.write("#   namespace: 'Human-Mouse Ortholog Sweep'\n")
        f.write("Sample\t" + "\t".join(cols) + "\n")
        for r in complete:
            sample = f"{r['encoding']}_k{r['ksize']}"
            vals = [r["ksize"], r["n_ortholog"], r["n_non_ortholog"], r["total_hits"]]
            for m in MHT_METHODS:
                s = r["mht"].get(m, {})
                vals += [s.get("precision", ""), s.get("recall", "")]
            f.write(sample + "\t" + "\t".join(str(v) for v in vals) + "\n")


def write_linegraph(
    path: Path, results: list[dict], value_fn, section_name: str, description: str,
    ylab: str, ylog: bool = False,
) -> None:
    """value_fn(record) -> float|None. Only complete records are eligible."""
    complete = [r for r in results if is_complete(r)]
    encodings = sorted({r["encoding"] for r in complete})

    datasets = {}
    for enc in encodings:
        pts = {}
        for r in sorted((x for x in complete if x["encoding"] == enc), key=lambda x: x["ksize"]):
            v = value_fn(r)
            if v is not None:
                pts[r["ksize"]] = v
        if pts:
            datasets[enc] = pts

    lines = [
        f"id: '{path.stem}'",
        f"section_name: '{section_name}'",
        f"description: '{description}'",
        f"plot_type: 'linegraph'",
        f"pconfig:",
        f"  id: '{path.stem}_plot'",
        f"  title: '{section_name}'",
        f"  xlab: 'ksize'",
        f"  ylab: '{ylab}'",
        f"  colors:",
    ]
    lines += [f"    '{enc}': '{color_for(enc)}'" for enc in datasets]
    if ylog:
        lines.append("  ylog: true")
    lines.append("data:")
    for enc, pts in datasets.items():
        lines.append(f"  '{enc}':")
        for x, y in sorted(pts.items()):
            lines.append(f"    {x}: {y}")

    path.write_text("\n".join(lines) + "\n")


def write_multiqc_config(path: Path, encodings: list[str]) -> None:
    content = """\
title: "Human-Mouse GENCODE Ortholog K-mer Sweep"
subtitle: "Kmerseek encoding x ksize sweep — human vs mouse canonical proteins"
intro_text: >
  Sweeps kmerseek alphabet encodings (hp variants, dayhoff, protein) across their
  respective ksize ranges, searching all human GENCODE proteins against a mouse
  GENCODE index. Ground truth: MGI/JAX human-mouse ortholog gene pairs. Evaluated
  with Poisson-test multiple-hypothesis correction (Bonferroni, BH, BY, two-stage BH).

report_header_info:
  - Ground truth: "MGI/JAX HOM_MouseHumanSequence orthologs"
  - Comparison: "human vs mouse, all canonical GENCODE proteins"
  - Correction: "Bonferroni / BH / BY / two-stage BH, alpha=0.05"

custom_data:
  sweep_completeness:
    colors:
      - ['0', '#E53935']
      - ['1', '#43A047']
"""
    path.write_text(content)


def main(sweep_json: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(sweep_json)
    encodings = sorted({r["encoding"] for r in results})
    n_complete = sum(is_complete(r) for r in results)
    print(f"Loaded {len(results)} sweep entries ({n_complete} complete, "
          f"{len(results) - n_complete} incomplete) across {len(encodings)} encodings")

    write_completeness_heatmap(out / "completeness_mqc.yaml", results)
    write_encoding_completion_table(out / "encoding_completion_mqc.tsv", results)
    write_summary_table(out / "summary_table_mqc.tsv", results)

    write_linegraph(
        out / "bh_recall_vs_ksize_mqc.yaml", results,
        value_fn=lambda r: r["mht"].get("bh", {}).get("recall"),
        section_name="BH recall vs ksize",
        description="Recall of BH-corrected significant hits (alpha=0.05) against MGI/JAX orthologs.",
        ylab="BH recall",
    )
    write_linegraph(
        out / "bh_precision_vs_ksize_mqc.yaml", results,
        value_fn=lambda r: r["mht"].get("bh", {}).get("precision"),
        section_name="BH precision vs ksize",
        description="Precision of BH-corrected significant hits (alpha=0.05) against MGI/JAX orthologs.",
        ylab="BH precision",
    )
    write_linegraph(
        out / "total_hits_vs_ksize_mqc.yaml", results,
        value_fn=lambda r: r.get("total_hits"),
        section_name="Search space size vs ksize",
        description="Total human-mouse protein pairs tested (pre-filtered to poisson p<0.001 "
                     "where noted in the underlying summary.txt) — proxy for compute/storage cost.",
        ylab="total_hits",
        ylog=True,
    )

    write_multiqc_config(out / "multiqc_config.yaml", encodings)

    print(f"Wrote MultiQC input files to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
