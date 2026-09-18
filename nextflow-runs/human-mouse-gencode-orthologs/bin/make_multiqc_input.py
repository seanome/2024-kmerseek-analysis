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
    make_multiqc_input.py <kmer_sweep_summary.json> <outdir> \\
        [ortholog_stats.txt] [human.fa] [mouse.fa]

The three optional files feed the overview section's counts (answer-key pairs, query and
target proteins); without them those numbers print as n/a.
"""

import json
import sys
from pathlib import Path

# The data-flow diagram in the overview is drawn by the module every report in this
# repository shares. Under Nextflow it is staged into the task directory and found through
# PYTHONPATH; run by hand from bin/, it is found at ../../shared.
try:
    import flow_diagram as fd
except ImportError:  # pragma: no cover - the by-hand path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import flow_diagram as fd

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



# --- overview: what was done, why, and the data flow -----------------------------------

# The three encoding families, in the colours the curves use for their members. HP has
# six variants and one colour family; the box names the count rather than each one.
FAMILIES = [
    ("hp", "HP alphabets", lambda e: e.startswith("hp"), ENCODING_COLORS["hp"]),
    ("dayhoff", "Dayhoff", lambda e: e == "dayhoff", ENCODING_COLORS["dayhoff"]),
    ("protein", "protein, 20 letters", lambda e: e == "protein", ENCODING_COLORS["protein"]),
]

# Every section id this script writes, in reading order. report_section_order is one
# scale, so every id is listed. Each file here is its own MultiQC module (no parent_id),
# and modules sort with the LARGEST order first, so the first id gets the largest number.
SECTION_ORDER = ["overview", "sweep_completeness", "encoding_completion", "summary_table",
                 "bh_recall_vs_ksize_mqc", "bh_precision_vs_ksize_mqc",
                 "total_hits_vs_ksize_mqc"]


def count_fasta(path: str | None) -> int | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as fh:
        return sum(1 for line in fh if line.startswith(">"))


def read_ortholog_stats(path: str | None) -> dict:
    """The counts parseOrthologMapping wrote, keyed by the words before the colon."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return out


def overview_facts(results: list[dict], stats: dict, n_human, n_mouse) -> dict:
    complete = [r for r in results if is_complete(r)]
    fams = []
    for fid, name, member, colour in FAMILIES:
        rows = [r for r in results if member(r["encoding"])]
        if not rows:
            continue
        done = [r for r in rows if is_complete(r)]
        ks = sorted({r["ksize"] for r in rows})
        encodings = {}
        for enc in sorted({r["encoding"] for r in rows}):
            er = [r for r in rows if r["encoding"] == enc]
            ed = [r for r in er if is_complete(r)]
            def best(metric):
                vals = [(r["mht"]["bh"][metric], r["ksize"]) for r in ed
                        if r["mht"].get("bh", {}).get(metric) is not None]
                return max(vals) if vals else None
            encodings[enc] = {"n_attempted": len(er), "n_complete": len(ed),
                              "best_recall": best("recall"), "best_precision": best("precision")}
        fams.append({"id": fid, "name": name, "colour": colour,
                     "n_encodings": len(encodings), "encodings": encodings,
                     "k_min": ks[0], "k_max": ks[-1],
                     "n_attempted": len(rows), "n_complete": len(done)})
    hits = [r["total_hits"] for r in complete if r.get("total_hits")]
    return {
        "n_human": n_human, "n_mouse": n_mouse,
        "n_attempted": len(results), "n_complete": len(complete), "families": fams,
        "hits_min": min(hits) if hits else None, "hits_max": max(hits) if hits else None,
        "pairs": stats.get("Total human-mouse pairs"),
        "human_with_ortholog": stats.get("Number of human genes with mouse orthologs"),
        "one_to_one": stats.get("Human genes with exactly one mouse ortholog"),
    }


def overview_flow_svg(f: dict) -> str:
    F = fd.Flow()
    y = F.legend([
        [dict(text="sequences or results, with a count"), dict(text="a step", arrow=True),
         dict(text="an encoding family this run did not do", dashed=True)],
        [dict(text=fam["name"], stroke=fam["colour"]) for fam in f["families"]],
        [dict(text="query proteins", icon="genetics"), dict(text="target proteins", icon="database"),
         dict(text="a search", icon="search"), dict(text="hits and labels", icon="table_rows"),
         dict(text="the report", icon="summarize")],
        fd.INTERACTION_LEGEND,
    ])
    half = 350
    lx, rx = 20, 410
    lmid, rmid = lx + half // 2, rx + half // 2
    F.label(lmid, y + 8, ["QUERY: what is searched"], bold=True, size=14)
    F.label(rmid, y + 8, ["TARGET: what is searched against"], bold=True, size=14)
    ya = y + 18
    tw = half - fd.ICON_PX - fd.SVG_PAD
    a_l = F.box(lx, ya, half, ["human GENCODE canonical proteins", f"{fd.num(f['n_human'])} proteins"],
                bold_first=True, icon="genetics", node="q0")
    a_r = F.box(rx, ya, half, ["mouse GENCODE canonical proteins", f"{fd.num(f['n_mouse'])} proteins"]
                + fd.wrap("indexed once per encoding and k", tw), bold_first=True, icon="database", node="t0")
    y_from = max(a_l[1] + a_l[3], a_r[1] + a_r[3])

    n = max(len(f["families"]), 1)
    cols = fd.columns(n)
    mids = [fd.mid(c) for c in cols]
    fam_ids = [fam["id"] for fam in f["families"]]
    yb = y_from + 18 + fd.SVG_LINE_PX * 2 + 26
    F.fan([lmid, rmid], y_from, mids, yb,
          ["kmerseek search, every human protein against the mouse index,",
           f"once per encoding x k: {fd.num(f['n_complete'])} of {fd.num(f['n_attempted'])} combos completed"],
          frm=["q0", "t0"], to=fam_ids)
    boxes = []
    aw = cols[0][1] - fd.ICON_PX - fd.SVG_PAD
    for fam, (x, w) in zip(f["families"], cols):
        lines = [fam["name"]] + fd.wrap(
            f"{fam['n_encodings']} encoding(s), k {fam['k_min']} to {fam['k_max']}: "
            f"{fam['n_complete']} of {fam['n_attempted']} combos completed", aw)
        boxes.append(F.box(x, yb, w, lines, stroke=fam["colour"], stroke_w=2.5, bold_first=True,
                           icon="search", dashed=fam["n_complete"] == 0, node=fam["id"]))
    y_arms = max(b[1] + b[3] for b in boxes)
    for m, fid in zip(mids, fam_ids):
        F.line(m, y_arms, m, y_arms + 26, arrow=True, frm=fid, to="hits")
    h = F.box(20, y_arms + 26, 740,
              ["hits: human protein x mouse protein pairs with at least 2 shared k-mers and Poisson p at most 0.05"]
              + fd.wrap(f"{fd.num(f['hits_min'])} to {fd.num(f['hits_max'])} per combo", 700),
              icon="table_rows", node="hits")

    # The answer key comes in from the side; the hits pass down the middle.
    y_side = h[1] + h[3] + 30
    xm = fd.SVG_W // 2
    sw = 300
    key = F.box(20, y_side, sw, ["answer key: MGI/JAX ortholog pairs"]
                + fd.wrap(f"{fd.num(f['pairs'])} human-mouse gene pairs over "
                          f"{fd.num(f['human_with_ortholog'])} human genes, "
                          f"{fd.num(f['one_to_one'])} of them with exactly one mouse ortholog",
                          sw - fd.ICON_PX - fd.SVG_PAD),
                bold_first=True, icon="table_rows", node="key")
    y_sc = key[1] + key[3] + 30
    F.step(xm + 100, h[1] + h[3], y_sc, ["a hit is an ortholog when", "its gene pair is in the key"],
           frm="hits", to="metrics")
    F.line(20 + sw // 2, key[1] + key[3], 20 + sw // 2, y_sc, arrow=True, frm="key", to="metrics")
    m = F.box(20, y_sc, 740, ["precision and recall after multiple-testing correction, alpha 0.05"]
              + fd.wrap("Bonferroni, BH, BY and two-stage BH over every hit's Poisson p-value; "
                        "precision is the corrected hits that are orthologs, recall is the "
                        "ortholog hits that survive correction over the ortholog hits with "
                        "p below 0.05", 680),
              icon="table_rows", bold_first=True, node="metrics")
    F.line(xm, m[1] + m[3], xm, m[1] + m[3] + 26, arrow=True, frm="metrics", to="report")
    F.box(20, m[1] + m[3] + 26, 740,
          ["this report: which combos completed, the metrics table, and precision, recall "
           "and hit count against k"], icon="summarize", node="report")
    return F.render()


def overview_details(f: dict) -> dict:
    fam_text = {
        "hp": "Two-letter hydrophobic/polar alphabets. Low k under a dense alphabet can run out "
              "of memory, which is why completeness is reported rather than assumed.",
        "dayhoff": "The six-letter Dayhoff alphabet.",
        "protein": "The unreduced 20-letter alphabet: the control for what reduction adds or costs.",
    }
    d = {
        "q0": {"title": "human GENCODE canonical proteins", "count": f"{fd.num(f['n_human'])} proteins",
               "text": "One canonical protein per human protein-coding gene.",
               "facts": [["proteins", fd.num(f["n_human"])]], "links": [["What was done, and why", "overview"]]},
        "t0": {"title": "mouse GENCODE canonical proteins", "count": f"{fd.num(f['n_mouse'])} proteins",
               "text": "One canonical protein per mouse gene, indexed once for every encoding x k "
                       "combination in the sweep.",
               "facts": [["proteins", fd.num(f["n_mouse"])],
                         ["indexes built", f"{fd.num(f['n_attempted'])} (one per combo)"]],
               "links": [["What was done, and why", "overview"]]},
        "hits": {"title": "hits", "count": f"{fd.num(f['hits_min'])} to {fd.num(f['hits_max'])} per combo",
                 "text": "The search's own filters: at least 2 shared k-mers and a Poisson p-value "
                         "of at most 0.05.",
                 "facts": [["hits per combo", f"{fd.num(f['hits_min'])} to {fd.num(f['hits_max'])}"]],
                 "links": [["Search space size vs ksize", "total_hits_vs_ksize_mqc"]]},
        "key": {"title": "answer key: MGI/JAX ortholog pairs", "count": f"{fd.num(f['pairs'])} pairs",
                "text": "Assigned by people from several lines of evidence, so the key does not "
                        "depend on any one similarity search.",
                "facts": [["pairs", fd.num(f["pairs"])], ["human genes covered", fd.num(f["human_with_ortholog"])],
                          ["with exactly one mouse ortholog", fd.num(f["one_to_one"])]],
                "links": [["What was done, and why", "overview"]]},
        "metrics": {"title": "precision and recall after multiple-testing correction", "count": "alpha 0.05",
                    "text": "Each hit's p-value is corrected over everything that combo reported, four "
                            "ways. Precision is the share of corrected hits that are orthologs; recall "
                            "is the share of ortholog hits with p below 0.05 that survive correction. A "
                            "combo that reports more pairs is not rewarded for it.",
                    "facts": [], "links": [["Sweep metrics (complete combos only)", "summary_table"],
                                           ["BH recall vs ksize", "bh_recall_vs_ksize_mqc"],
                                           ["BH precision vs ksize", "bh_precision_vs_ksize_mqc"]]},
        "report": {"title": "this report", "count": "",
                   "text": "Completion first, then the metrics table, then precision, recall and hit "
                           "count against k, one line per encoding.",
                   "facts": [], "links": [["Sweep completeness", "sweep_completeness"]]},
    }
    for fam in f["families"]:
        d[fam["id"]] = {"title": fam["name"],
                        "count": f"{fam['n_encodings']} encoding(s), k {fam['k_min']} to {fam['k_max']}",
                        "text": fam_text[fam["id"]],
                        "facts": [["combos completed", f"{fam['n_complete']} of {fam['n_attempted']}"]],
                        "links": [["Sweep completeness", "sweep_completeness"],
                                  ["Completion by encoding", "encoding_completion"]]}
    return d


def overview_control(f: dict) -> dict | None:
    """The encoding-family control: for the chosen family, each encoding's completed
    combos and its best BH recall and precision over k."""
    if not f["families"]:
        return None
    options = []
    for fam in f["families"]:
        facts = []
        for enc in fam["encodings"]:
            e = fam["encodings"][enc]
            facts.append([f"{enc}: combos completed", f"{e['n_complete']} of {e['n_attempted']}"])
            if e.get("best_recall") is not None:
                facts.append([f"{enc}: best BH recall (k)", f"{e['best_recall'][0]:.3f} (k={e['best_recall'][1]})"])
            if e.get("best_precision") is not None:
                facts.append([f"{enc}: best BH precision (k)", f"{e['best_precision'][0]:.3f} (k={e['best_precision'][1]})"])
        options.append({"id": fam["id"], "label": fam["name"],
                        "facts": {fam["id"]: facts},
                        "links": {fam["id"]: [["Sweep completeness", "sweep_completeness"],
                                              ["Completion by encoding", "encoding_completion"],
                                              ["BH recall vs ksize", "bh_recall_vs_ksize_mqc"],
                                              ["BH precision vs ksize", "bh_precision_vs_ksize_mqc"]]}})
    return {"label": "Encoding family", "options": options}


def write_overview(out: Path, results: list[dict], stats: dict, n_human, n_mouse) -> None:
    f = overview_facts(results, stats, n_human, n_mouse)
    fam_txt = "; ".join(
        f"{fam['name']} ({fam['n_encodings']} encoding(s), k {fam['k_min']} to {fam['k_max']}, "
        f"{fam['n_complete']} of {fam['n_attempted']} combos completed)" for fam in f["families"])
    steps = [
        f"<b>Query: human GENCODE canonical proteins.</b> {fd.num(f['n_human'])} proteins, one "
        f"per protein-coding gene.",
        f"<b>Target: mouse GENCODE canonical proteins.</b> {fd.num(f['n_mouse'])} proteins, "
        f"indexed once per encoding and k.",
        f"<b>Search.</b> kmerseek, every human protein against the mouse index, once per "
        f"encoding x k: {fam_txt}. A hit is a human x mouse protein pair with at least 2 "
        f"shared k-mers and a Poisson p-value of at most 0.05 (the search's own filters); "
        f"{fd.num(f['hits_min'])} to {fd.num(f['hits_max'])} hits per combo.",
        f"<b>Answer key: MGI/JAX ortholog pairs.</b> {fd.num(f['pairs'])} human-mouse gene "
        f"pairs from HOM_MouseHumanSequence.rpt over {fd.num(f['human_with_ortholog'])} human "
        f"genes, {fd.num(f['one_to_one'])} of them with exactly one mouse ortholog. A hit "
        f"is an ortholog when its gene pair is in the key.",
        "<b>Score.</b> Every hit's Poisson p-value is corrected for multiple testing over "
        "all hits of that combo, four ways (Bonferroni, BH, BY, two-stage BH), at alpha "
        "0.05. Precision is the share of corrected hits that are orthologs. Recall is the "
        "share of ortholog hits that survive correction, over the ortholog hits with "
        "p below 0.05: it is recall among what the search reported, not among every MGI "
        "pair.",
        "<b>Report.</b> Which encoding x k combos completed, the metrics table, and "
        "precision, recall and hit count against k, one line per encoding.",
    ]
    why = "".join(f"<li>{w}</li>" for w in [
        "<b>Which alphabet and which k.</b> The same query and target under every encoding "
        "and k is what says where a reduced alphabet helps and where it costs, on one pair "
        "of proteomes close enough that most genes have an ortholog to find.",
        "<b>A curated answer key.</b> MGI/JAX ortholog pairs are assigned by people from "
        "several lines of evidence, so they do not depend on any one similarity search.",
        "<b>Corrected p-values rather than a score cutoff</b>, so a combo that reports more "
        "pairs is not rewarded for reporting more pairs: a hit counts only after correction "
        "over everything that combo reported.",
        "<b>Completeness is reported, not assumed.</b> Low-k searches under the dense HP "
        "alphabets can run out of memory; a combo that did not finish is listed as "
        "incomplete rather than read as zero.",
    ])
    cfg = {
        "id": "overview",
        "section_name": "What was done, and why",
        "description": (
            "<p>Human GENCODE proteins searched against a mouse GENCODE index by kmerseek "
            "under every encoding and k, every hit labelled against MGI/JAX ortholog pairs "
            "and counted after multiple-testing correction. The steps, this run's numbers, "
            "and the flow of data from the inputs to the sections below.</p>"),
        "plot_type": "html",
        "data": (
            "<h4 style='margin-top:0.4em'>What was done</h4><ol>"
            + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
            "<h4>Why</h4><ul>" + why + "</ul>"
            "<h4>Data flow</h4>"
            "<p>Read top to bottom. A box is a set of sequences or results with its count in "
            "this run; an arrow is the step that makes the next one. Encoding-family boxes "
            "take the colour their encodings have in the curves below. Hover a box to see "
            "what feeds it; click it for what it is, its numbers, and the sections that show "
            "it; pick an encoding family to see each of its encodings.</p>"
            + fd.interactive(overview_flow_svg(f), overview_details(f), control=overview_control(f),
                             footnote="Recall is measured among the ortholog hits the search "
                                      "reported with p below 0.05, not among every MGI pair, so "
                                      "it is recall of the correction step, not of the search.",
                             uid="sweep-flow")),
    }
    (out / "overview_mqc.json").write_text(json.dumps(cfg, indent=1))


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
        f"results). Blank = no entry at all for that combination. "
        f"<a href=\"#overview\">&uarr; back to the data flow</a>'",
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
                "across all ksizes attempted in this sweep. "
                "<a href=\"#overview\">&uarr; back to the data flow</a>'\n")
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
                "for every encoding x ksize combination that completed evaluation. "
                "<a href=\"#overview\">&uarr; back to the data flow</a>'\n")
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
        f"description: '{description} <a href=\"#overview\">&uarr; back to the data flow</a>'",
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
  The first section says what was run, with this run's numbers, why, and how the data
  flows to the sections.

report_header_info:
  - Ground truth: "MGI/JAX HOM_MouseHumanSequence orthologs"
  - Comparison: "human vs mouse, all canonical GENCODE proteins"
  - Correction: "Bonferroni / BH / BY / two-stage BH, alpha=0.05"

custom_data:
  sweep_completeness:
    colors:
      - ['0', '#E53935']
      - ['1', '#43A047']

report_section_order:
""" + "\n".join(f"  {sid}: {{ order: {len(SECTION_ORDER) - i} }}"
                for i, sid in enumerate(SECTION_ORDER)) + "\n"
    path.write_text(content)


def main(sweep_json: str, outdir: str, stats_txt: str | None = None,
         human_fa: str | None = None, mouse_fa: str | None = None) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(sweep_json)
    write_overview(out, results, read_ortholog_stats(stats_txt),
                   count_fasta(human_fa), count_fasta(mouse_fa))
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
    if not 3 <= len(sys.argv) <= 6:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
