#!/usr/bin/env python3
"""Turn the dark-set products into MultiQC custom-content sections.

One report per species: how much of the proteome no sequence arm could place, which arm
placed what, and -- if the kmerseek arm ran -- how much of the dark set kmerseek reached
with the low-complexity mask on and off.

Written as `*_mqc.json` rather than YAML on purpose: the scoring container carries polars
but no PyYAML, and these files are machine-written either way. The one hand-edited file,
assets/multiqc_config.yaml, stays YAML.

Three rules run through every section here.

  Both mask settings or neither   A dark protein kmerseek reaches only with the
                                  low-complexity mask OFF is a composition artifact, not a
                                  homology signal -- BHF's flagship matches included
                                  polar-biased low-complexity segments, which is why the
                                  mask is a paired setting in this pipeline and not a sweep
                                  dimension. A combo with only one side measured is named
                                  and dropped, never drawn alone.
  A missing input omits a section A run without the length or disorder arm has no length
                                  or disorder result. It does not have an empty one. An
                                  empty panel reads as "we looked and found nothing",
                                  which is a different claim, so what is absent is listed
                                  by name in a section of its own.
  Reach is not accuracy           A region on a dark protein says kmerseek found something
                                  there, not that the family label is right. That needs the
                                  structural key, which does not exist for these species
                                  yet. Every kmerseek section says so.
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl

# The data-flow diagram in the overview is drawn by the module every report in this
# repository shares. Under Nextflow it is staged into the task directory and found
# through PYTHONPATH; run by hand from bin/, it is found at ../../shared.
try:
    import flow_diagram as fd
    import metric_explainers as mx
except ImportError:  # pragma: no cover - the by-hand path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import flow_diagram as fd
    import metric_explainers as mx


PARENT_ID = "invertebrate_dark_set"
PARENT_NAME = "Invertebrate dark set"

# The species' name in prose (Botryllus schlosseri), from the registry row; the label
# (botryllus) stays in plot titles, table keys and file names. Set once in main().
SPECIES_NAME: str | None = None


def sp_name(species: str) -> str:
    """The species for a sentence: the binomial in italics, or the label."""
    return f"<i>{SPECIES_NAME}</i>" if SPECIES_NAME else species


def registry_row(registry: Path | None, species: str) -> dict:
    if registry is None or not registry.exists():
        return {}
    return dict(json.loads(registry.read_text()).get(species) or {})
# The module's own intro line. Without one MultiQC reuses the first section's description
# as the module description, so the first panel's text appeared twice in a row.
PARENT_DESCRIPTION = ("One report per species: what was done, how much of the proteome no "
                      "sequence search could place, and what the optional arms say about "
                      "that part.")

# One colour per idea, reused across sections so a thing keeps its identity.
#
# Green is "something was found" and grey is "nothing was found", in every panel: a protein
# placed by a sequence arm, the union of the arms, and a kmerseek reach the mask supports
# are all finds and all green; the dark set is grey. Until 2026-09-17 the dark set was
# green because it is the object of the report, which read as the opposite of what it is.
C_FOUND = "#0f9d76"      # a hit exists: placed, any arm, kmerseek reach with the mask ON
C_PLACED = C_FOUND       # placed by a sequence arm
C_UNION = C_FOUND        # placed by any arm at all
C_MASK_ON = C_FOUND      # kmerseek reach with the low-complexity mask ON
C_DARK = "#7f7f7f"       # the dark set: no arm found anything
C_MASK_OFF = "#c9528f"   # mask OFF: a reach that may be composition rather than homology
C_SINGLE = "#2b7bba"     # single-pass search  (the per-arm panel only)
C_ITER = "#c99a00"       # iterative search    (the per-arm panel only)
C_CLADE = "#7b4fb3"      # the query's own clade: the query itself, and what the target drops

# The arms this pipeline runs, and whether each is a single-pass or an iterative search.
# The ordering iterative >= single-pass is the correctness signal for the whole dark set:
# jackhmmer at 3 iterations and mmseqs2 at --num-iterations 3 see everything phmmer sees
# and more, so an arm count that inverts that means an arm did not really run.
ARM_CLASS = {
    "phmmer": "single_pass",
    "jackhmmer": "iterative",
    "mmseqs2": "iterative",
}
ARM_ORDER = ["phmmer", "jackhmmer", "mmseqs2"]

# Column names the optional arms might use for the same quantity. Both scripts are written
# by other hands, so the join is on meaning rather than on a name agreed in advance; a name
# that resolves to nothing omits the section and says which names were looked for.
GROUP_COLS = ["group", "dark_group", "is_dark", "dark"]
LENGTH_COLS = ["length", "protein_length", "aa_length", "n_residues"]
# The first two are what label_dark_disorder.py actually writes; the rest are the tolerant
# fallbacks. This list was originally written from a guess, because the disorder arm did not
# exist yet, and the guess was wrong -- it looked for `mean_disorder` against a parquet
# carrying `mean_disorder_metapredict`, so the panel was dropped. It failed the right way
# (the omission was named in the report rather than drawn as an empty plot), which is how
# the mismatch was found at all. Keep the real names FIRST so an exact match wins.
DISORDER_COLS = ["mean_disorder_metapredict", "disorder_fraction_metapredict",
                 "mean_disorder", "disorder", "disorder_mean", "mean_disorder_score",
                 "fraction_disordered", "disorder_fraction"]

# Protein-length bins. Fixed edges rather than quantiles so the two groups are binned
# identically -- quantile bins computed per group would put a different cut in each and the
# two histograms would not be comparable. The first two edges are the ones that matter:
# a spurious gene model is usually short, so a dark-set excess below 100 aa is the shape
# that would qualify the headline.
LENGTH_EDGES = [0, 50, 100, 200, 400, 800, 1600]

# Mean-disorder bins. metapredict scores a residue 0..1, so a per-protein mean is on the
# same scale and deciles are the natural bins.
DISORDER_EDGES = [round(0.1 * i, 1) for i in range(10)]


def num(value) -> str:
    """Integers with underscore groups: 20_448, never 20,448.

    The project writes them that way in prose and in code, and a comma-grouped integer
    pasted back into python is a tuple rather than a number.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}".replace(",", "_")
    return f"{int(value):,}".replace(",", "_")


def evalue_txt(value) -> str:
    """10.0 prints as 10 and 0.001 as 0.001: the cutoff as a person would write it."""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.{digits}f}%"


def clean(value):
    """JSON has no NaN or Infinity. Plotly reads null as a gap, which is what these are."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def bullets(*items: str) -> str:
    """A section description's points as a list rather than a wall of paragraphs.

    Empty strings are dropped, so a caller can pass a fragment a run may not have.
    """
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items if i) + "</ul>"


def write_section(outdir: Path, section_id: str, cfg: dict) -> None:
    # A way back to the flow from every panel it links to.
    if section_id != "dark_overview" and "description" in cfg:
        cfg["description"] += ('<p style="font-size:12px"><a href="#dark_overview">'
                               '&uarr; back to the data flow</a></p>')
    cfg.setdefault("parent_id", PARENT_ID)
    cfg.setdefault("parent_name", PARENT_NAME)
    cfg.setdefault("parent_description", PARENT_DESCRIPTION)
    (outdir / f"{section_id}_mqc.json").write_text(json.dumps(clean(cfg), indent=1))


def pick(mapping: dict, *names, default=None):
    """First key present, so a summary written by another hand still resolves."""
    if not isinstance(mapping, dict):
        return default
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def pick_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def load_parquet(path: Path | None) -> pl.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except Exception:  # a truncated or half-written file is a missing input, not a crash
        return None


class Omitted:
    """What the run did not produce, named so the report says it out loud.

    A section built from an input that is not there would render as an empty panel, and an
    empty panel reads as a measured null. These are collected instead and written as one
    closing section listing what is absent and why.
    """

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, what: str, why: str) -> None:
        self.items.append((what, why))


# --- 0) overview: what was done, why, and the data flow ------------------------------

def dark_by_length(length_df) -> dict[int, dict]:
    """{min_aa: {kept, dark}} from the length parquet, the same numbers as the
    'Dark fraction by minimum length' table, or {} when the run has no length arm."""
    if length_df is None:
        return {}
    value_col = pick_col(length_df, LENGTH_COLS)
    group_col = pick_col(length_df, GROUP_COLS)
    if value_col is None or group_col is None:
        return {}
    is_dark = length_df[group_col].cast(pl.Utf8).str.to_lowercase().is_in(["dark", "true", "1"])
    lengths = length_df[value_col].cast(pl.Float64)
    out = {}
    for cut in MIN_AA:
        keep = lengths >= cut
        out[cut] = {"kept": int(keep.sum()), "dark": int((keep & is_dark).sum())}
    return out


def dark_flow_spec(species: str, summary: dict, ref: dict | None, clade: str | None,
                   run: dict, arms: dict[str, bool], gain: dict | None, by_len: dict) -> dict:
    """The pipeline as a box-and-arrow spec for the shared renderer: query and target meet
    at a junction and one bus fans out to the three arms, so no line crosses another; a
    box holds a name and one number; everything else opens in the panel."""
    total = pick(summary, "proteins_in_proteome")
    placed = pick(summary, "proteins_placed_by_any_arm")
    dark = pick(summary, "proteins_dark")
    frac = pick(summary, "fraction_dark")
    raw_rows = pick(summary, "raw_hit_rows")
    evalue = evalue_txt(pick(summary, "evalue_call"))
    e_rep = evalue_txt(pick(run, "evalue_report", default=10))
    chunk = pick(run, "query_chunk_size")
    n_chunks = math.ceil(total / chunk) if total and chunk else None
    kept = pick(ref, "entries_kept")
    excluded = pick(ref, "entries_excluded")
    sp_total = kept + excluded if kept is not None and excluded is not None else None
    clade_txt = clade or pick(ref, "excluded_clade") or "the query's own clade"
    pairs = pick(gain, "mask_pairs", default=[]) or []
    combos = pick(gain, "by_combo", default=[]) or []
    frac_txt = pct(excluded / sp_total, 2) if excluded is not None and sp_total else "n/a"

    nodes = {
        "q0": {"x": 20, "y": 66, "w": 340, "h": 52, "icon": "genetics", "kind": "clade",
               "title": f"query: {species} proteome", "sub": f"{num(total)} proteins"},
        "t0": {"x": 400, "y": 66, "w": 340, "h": 52, "icon": "database",
               "title": "target: reviewed Swiss-Prot", "sub": f"{num(sp_total)} entries", "strip": C_CLADE},
        "q1": {"x": 20, "y": 186, "w": 320, "h": 52, "icon": "genetics", "kind": "clade",
               "title": f"query: {num(n_chunks)} chunks of the proteome", "sub": "headers cut to the accession"},
        "t1": {"x": 420, "y": 186, "w": 320, "h": 52, "icon": "database",
               "title": f"target: Swiss-Prot minus {clade_txt}", "sub": f"{num(kept)} entries"},
        "J1": {"junction": True, "x": 380, "y": 212},
        "phmmer": {"x": 20, "y": 318, "w": 220, "h": 48, "icon": "search", "title": "phmmer", "sub": "one pass",
                   "samples": ["phmmer"]},
        "jackhmmer": {"x": 270, "y": 318, "w": 220, "h": 48, "icon": "search", "title": "jackhmmer",
                      "sub": f"{pick(run, 'jackhmmer_iterations', default=3)} iterations", "samples": ["jackhmmer"]},
        "mmseqs2": {"x": 520, "y": 318, "w": 220, "h": 48, "icon": "search", "title": "mmseqs2",
                    "sub": f"sensitivity {pick(run, 'mmseqs2_sensitivity', default=7)}, 3 iterations",
                    "samples": ["mmseqs2"]},
        "hits": {"x": 20, "y": 430, "w": 720, "h": 48, "icon": "table_rows",
                 "title": f"hits, all three arms, kept at E ≤ {e_rep}", "sub": f"{num(raw_rows)} rows"},
        "placed": {"x": 20, "y": 550, "w": 320, "h": 60, "icon": "task_alt", "kind": "placed",
                   "title": "placed: a hit from at least one arm",
                   "sub": f"{num(placed)} proteins ({pct(1 - frac) if frac is not None else 'n/a'})", "bar": True},
        "dark": {"x": 420, "y": 550, "w": 320, "h": 60, "icon": "search_off", "kind": "dark",
                 "title": "dark: no hit from any arm", "sub": f"{num(dark)} proteins ({pct(frac)})", "bar": True},
        "J2": {"junction": True, "x": 380, "y": 580},
        "length": {"x": 20, "y": 690, "w": 220, "h": 56, "icon": "straighten", "title": "protein length",
                   "sub": "from the FASTA", "dashed": not arms["length"]},
        "disorder": {"x": 270, "y": 690, "w": 220, "h": 56, "icon": "waves", "title": "predicted disorder",
                     "sub": "metapredict, mean per protein", "dashed": not arms["disorder"]},
        "kmerseek": {"x": 520, "y": 690, "w": 220, "h": 56, "icon": "search", "title": "kmerseek regions",
                     "sub": (f"{len(pairs)} alphabet x k pair(s), mask on / off" if pairs
                             else f"{len(combos)} setting(s)" if combos else "did not run"),
                     "dashed": not arms["kmerseek"],
                     "samples": ["kmerseek"] + [f"{r.get('alphabet')}_k{r.get('ksize')}" for r in pairs]},
        "report": {"x": 20, "y": 820, "w": 720, "h": 44, "icon": "summarize",
                   "title": "this report: one panel per step, in this order"},
    }
    edges = [
        {"from": "q0", "to": "q1", "label": f"split into chunks of {num(chunk)}, headers cut to the accession"},
        {"from": "t0", "to": "t1", "label": f"remove {clade_txt}: {num(excluded)} of {num(sp_total)} entries ({frac_txt})"},
        {"from": "q1", "to": "J1"}, {"from": "t1", "to": "J1"},
        {"from": "J1", "to": "phmmer", "bus": 60, "label": ["every chunk against the target:",
                f"{num(n_chunks)} chunks x 3 arms = {num(3 * n_chunks) if n_chunks else 'n/a'} searches"],
         "lx": 392, "ly": 266, "anchor": "start"},
        {"from": "J1", "to": "jackhmmer", "bus": 60}, {"from": "J1", "to": "mmseqs2", "bus": 60},
        {"from": "phmmer", "to": "hits"}, {"from": "jackhmmer", "to": "hits"}, {"from": "mmseqs2", "to": "hits"},
        {"from": "hits", "to": "placed", "label": f"placed: some arm has a hit at E ≤ {evalue}"},
        {"from": "hits", "to": "dark", "label": "dark: no arm has one"},
        {"from": "placed", "to": "J2"}, {"from": "dark", "to": "J2"},
        {"from": "J2", "to": "length", "bus": 50, "label": "dark set against placed set", "lx": 392, "ly": 626, "anchor": "start"},
        {"from": "J2", "to": "disorder", "bus": 50}, {"from": "J2", "to": "kmerseek", "bus": 50},
        {"from": "length", "to": "report"}, {"from": "disorder", "to": "report"}, {"from": "kmerseek", "to": "report"},
    ]
    return {
        "height": 900,
        "kinds": {"clade": {"color": C_CLADE, "label": f"the query's own clade, {clade_txt}: the query, "
                                                        f"and the strip on Swiss-Prot for what the target drops"},
                  "placed": {"color": C_PLACED, "fill": True, "label": "placed: some arm found it in the target"},
                  "dark": {"color": C_DARK, "fill": True, "label": "dark: no arm found anything"}},
        "barLegend": "bar along the bottom of placed / dark: its share of the proteins counted (full width = all of them)",
        "lanes": [[30, 130, "inputs"], [160, 260, "prepared"], [290, 390, "search"], [410, 500, "hits"],
                  [520, 630, "the call"], [660, 780, "read-outs"], [800, 880, "report"]],
        "headers": [[190, 52, "QUERY: what is searched"], [570, 52, "TARGET: what is searched against"]],
        "nodes": nodes, "edges": edges,
    }


def flow_details(species: str, summary: dict, ref: dict | None, clade: str | None, run: dict,
                 gain: dict | None, length_summary: dict | None, disorder_summary: dict | None,
                 arms: dict[str, bool], by_len: dict) -> dict:
    """What each box opens with: a sentence or two, the numbers, and the report sections
    that show it. Every number here is one the panels below also print."""
    total = pick(summary, "proteins_in_proteome")
    placed = pick(summary, "proteins_placed_by_any_arm")
    dark = pick(summary, "proteins_dark")
    raw_rows = pick(summary, "raw_hit_rows")
    evalue = evalue_txt(pick(summary, "evalue_call"))
    chunk = pick(run, "query_chunk_size")
    n_chunks = math.ceil(total / chunk) if total and chunk else None
    kept = pick(ref, "entries_kept")
    excluded = pick(ref, "entries_excluded")
    sp_total = kept + excluded if kept is not None and excluded is not None else None
    clade_txt = clade or pick(ref, "excluded_clade") or "the query's own clade"
    per_arm = pick(summary, "proteins_placed_per_arm", default={}) or {}
    e_rep = evalue_txt(pick(run, "evalue_report", default=10))

    def stat(s, side, key="median", digits=0):
        v = pick(pick(s, side, f"{side}_stats", default={}) or {}, key)
        return f"{float(v):.{digits}f}" if v is not None else None

    def iqr(s, side, digits=0):
        lo, hi = stat(s, side, "p25", digits), stat(s, side, "p75", digits)
        return f" (IQR {lo}-{hi})" if lo and hi else ""

    d = {}
    d["q0"] = {
        "title": f"query: {species} proteome", "count": f"{num(total)} proteins",
        "text": f"The proteome as annotated: every protein in the {species} FASTA. The "
                f"direction is the one a new genome faces: the proteome asks, Swiss-Prot answers.",
        "facts": [["proteins", num(total)]]
                 + ([["median length, dark / placed",
                      f"{stat(length_summary, 'dark')} aa / {stat(length_summary, 'placed')} aa"]]
                    if stat(length_summary, "dark") else []),
        "links": [["How much of the proteome is dark", "dark_headline"]]}
    d["t0"] = {
        "title": "target: reviewed Swiss-Prot", "count": f"{num(sp_total)} entries",
        "text": "Reviewed Swiss-Prot before anything is removed. Every species this pipeline "
                "runs is searched against the same construction, which is what makes the dark "
                "fractions comparable.",
        "facts": [["entries", num(sp_total)]],
        "links": [["What was done, and why", "dark_overview"]]}
    d["q1"] = {
        "title": f"query: {num(n_chunks)} chunks of the proteome", "count": "",
        "text": f"The proteome split into chunks of {num(chunk)} so each search is one job. "
                f"Headers are cut to the bare accession because the three searches and kmerseek "
                f"each report a UniProt header differently and the dark call needs one key.",
        "facts": [["chunks", num(n_chunks)],
                  ["searches (chunks x 3 arms)", num(3 * n_chunks) if n_chunks else "n/a"]],
        "links": [["What was done, and why", "dark_overview"]]}
    d["t1"] = {
        "title": f"target: Swiss-Prot minus {clade_txt}", "count": f"{num(kept)} entries",
        "text": f"Swiss-Prot with every {clade_txt} entry removed, so a hit has to come from "
                f"outside the query's own clade. Without this a well-curated species would place "
                f"its proteome on its own entries and the dark fraction would measure curation "
                f"depth, not how far sequence search reaches.",
        "facts": [["entries kept", num(kept)],
                  [f"entries dropped ({clade_txt})",
                   f"{num(excluded)} ({pct(excluded / sp_total, 2) if excluded is not None and sp_total else 'n/a'})"]],
        "links": [["What was done, and why", "dark_overview"]]}
    arm_text = {
        "phmmer": "Single-pass profile search (HMMER). The cheapest arm; the two iterative arms "
                  "must place at least as many proteins as this one, which is the check the "
                  "per-arm panel makes.",
        "jackhmmer": f"Iterative profile search: {pick(run, 'jackhmmer_iterations', default=3)} "
                     f"rounds, the profile rebuilt from the hits after each.",
        "mmseqs2": f"Iterative k-mer prefilter plus alignment at sensitivity "
                   f"{pick(run, 'mmseqs2_sensitivity', default=7)}, 3 rounds.",
    }
    arm_kind = {"phmmer": "single pass",
                "jackhmmer": f"iterative, {pick(run, 'jackhmmer_iterations', default=3)} rounds",
                "mmseqs2": f"iterative, sensitivity {pick(run, 'mmseqs2_sensitivity', default=7)}, 3 rounds"}
    for arm in ["phmmer", "jackhmmer", "mmseqs2"]:
        n_arm = per_arm.get(arm)
        d[arm] = {
            "title": arm, "count": arm_kind[arm],
            "text": arm_text[arm],
            "facts": ([[f"proteins placed at E <= {evalue}",
                        f"{num(n_arm)} ({pct(n_arm / total)} of proteome)" if n_arm is not None and total else num(n_arm)]]
                      if n_arm is not None else []) + [["kind", arm_kind[arm]]],
            "links": ([["What each arm placed", "dark_per_arm"],
                       ["Placed per arm, as numbers", "dark_per_arm_table"]] if per_arm else [])}
    d["hits"] = {
        "title": "hits of query proteins in the target", "count": f"{num(raw_rows)} rows",
        "text": f"Every hit any arm reported, kept at the permissive cutoff E <= {e_rep}. Keeping "
                f"this table is what lets the call cutoff be moved without re-running a search.",
        "facts": [["rows", num(raw_rows)], ["report cutoff", f"E <= {e_rep}"], ["call cutoff", f"E <= {evalue}"]],
        "links": [["Dark set in numbers", "dark_headline_table"]]}
    len_links = [["Dark fraction by minimum protein length", "dark_headline_by_length"]] if by_len else []
    d["placed"] = {
        "title": "placed: a target hit from at least one arm", "count": "",
        "text": f"A protein is placed when any arm has a hit at E \u2264 {evalue}. Placed is the union "
                f"of the three arms, so it is larger than any single arm.",
        "facts": [["placed by any arm, no length cut", num(placed)]],
        "links": [["How much of the proteome is dark", "dark_headline"]] + len_links}
    d["dark"] = {
        "title": "dark: no target hit from any arm", "count": "",
        "text": f"A protein is dark when no arm has a hit at E \u2264 {evalue}. This set is the "
                f"denominator for every claim about kmerseek adding annotation: it can only add "
                f"something where phmmer, jackhmmer and mmseqs2 all found nothing. Dark is not the "
                f"same as annotatable: a dark protein may be a real protein that sequence search "
                f"cannot reach, or a spurious gene model.",
        "facts": [["dark, no length cut", num(dark)]],
        "links": [["How much of the proteome is dark", "dark_headline"]] + len_links}
    d["length"] = {
        "title": "protein length", "count": "read from the FASTA",
        "text": "Is the dark set just short gene models? Length is read from the FASTA and "
                "compared dark against placed. If dark proteins skew sharply shorter, part of "
                "the headline is annotation noise rather than hard homology.",
        "facts": ([["median length, dark", f"{stat(length_summary, 'dark')} aa{iqr(length_summary, 'dark')}"],
                   ["median length, placed", f"{stat(length_summary, 'placed')} aa{iqr(length_summary, 'placed')}"]]
                  if stat(length_summary, "dark") else []),
        "links": [["Protein length, dark vs placed", "dark_length"]] if arms["length"] else []}
    d["disorder"] = {
        "title": "predicted disorder", "count": "metapredict, mean per protein",
        "text": "Is the dark set unfolded? Mean predicted disorder per protein from metapredict "
                "(each residue scored 0 to 1), dark against placed. A protein with no folded "
                "core has little for a profile or a k-mer to hold on to.",
        "facts": ([["median disorder, dark", stat(disorder_summary, "dark", digits=3)],
                   ["median disorder, placed", stat(disorder_summary, "placed", digits=3)]]
                  if stat(disorder_summary, "dark", digits=3) else []),
        "links": [["Predicted disorder, dark vs placed", "dark_disorder"]] if arms["disorder"] else []}
    pairs = pick(gain, "mask_pairs", default=[]) or []
    d["kmerseek"] = {
        "title": "kmerseek regions",
        "count": f"{len(pairs)} alphabet x k pair(s), mask on / off" if pairs else "",
        "text": f"The target is indexed once per alphabet, k and low-complexity mask setting; "
                f"every query chunk is searched against it. A protein counts as reached when "
                f"kmerseek reports at least one region on it (region score >= "
                f"{pick(run, 'min_region_score', default=1.3)}, query p <= "
                f"{pick(run, 'max_query_pvalue', default=0.05)}, at least "
                f"{pick(run, 'min_shared_kmers', default=2)} shared k-mers). Reached is counted "
                f"inside the dark set and, as a control, inside the placed set.",
        "facts": [[f"{r.get('alphabet')} k{r.get('ksize')}, mask on / off",
                   f"{num(r.get('dark_reached_mask_on'))} / {num(r.get('dark_reached_mask_off'))} of {num(dark)} dark"]
                  for r in pairs],
        "links": ([["kmerseek reach inside the dark set", "dark_kmerseek_mask"],
                   ["Mask pair, as numbers", "dark_kmerseek_mask_table"]] if pairs else [])}
    d["report"] = {
        "title": "this report", "count": "",
        "text": "One panel per step, in the order of this drawing. What a run did not do is "
                "listed by name at the end.",
        "facts": [], "links": [["What was done, and why", "dark_overview"]]}
    return d


def length_control(by_len: dict) -> dict | None:
    """The length cut: the placed / dark counts, shares and facts over proteins of at
    least 0, 50, 100 and 200 aa, from the same table the report prints."""
    if not by_len:
        return None
    options = []
    for cut in MIN_AA:
        if cut not in by_len:
            continue
        d = by_len[cut]
        kept, dark = d["kept"], d["dark"]
        placed = kept - dark
        cut_txt = f"at least {cut} aa" if cut else "no length cut"
        options.append({
            "id": str(cut), "label": "all" if cut == 0 else f"\u2265 {cut} aa",
            "subs": {"placed": f"{num(placed)} proteins ({pct(placed / kept) if kept else 'n/a'})",
                     "dark": f"{num(dark)} proteins ({pct(dark / kept) if kept else 'n/a'})"},
            "bars": {"placed": placed / kept if kept else 0, "dark": dark / kept if kept else 0},
            "facts": {"placed": [[f"proteins counted ({cut_txt})", num(kept)], ["placed", num(placed)],
                                 ["fraction placed", pct(placed / kept) if kept else "n/a"]],
                      "dark": [[f"proteins counted ({cut_txt})", num(kept)], ["dark", num(dark)],
                               ["fraction dark", pct(dark / kept) if kept else "n/a"]]},
        })
    return {"label": "Count only proteins at least this long", "options": options}


# --- the first screen -----------------------------------------------------------------

def best_pair(gain: dict | None) -> dict | None:
    pairs = pick(gain, "mask_pairs", default=[]) or []
    return max(pairs, key=lambda r: r.get("dark_reached_mask_on") or 0) if pairs else None


def combo_row(gain: dict | None, alphabet: str, ksize: int, mask: bool) -> dict | None:
    for r in pick(gain, "by_combo", default=[]) or []:
        if (r.get("alphabet"), r.get("ksize"), bool(r.get("low_complexity_mask"))) == (alphabet, ksize, mask):
            return r
    return None


def alphabet_words(gain: dict | None) -> str:
    """'hp_thomas_dill2: 2 classes, hydrophobic and polar; protein20: no grouping'."""
    names = sorted({r["alphabet"] for r in pick(gain, "by_combo", default=[]) or []
                    if r.get("alphabet")})
    parts = []
    for a in names:
        n = int(a.rstrip("0123456789") != a and a[len(a.rstrip("0123456789")):] or 0)
        if n >= 20:
            parts.append(f"<code>{a}</code>: no grouping, all 20 amino acids")
        elif a.startswith("hp_") and n:
            parts.append(f"<code>{a}</code>: {n} classes, hydrophobic and polar")
        elif n:
            parts.append(f"<code>{a}</code>: {n} classes")
        else:
            parts.append(f"<code>{a}</code>")
    return "; ".join(parts)


def first_screen_html(species: str, summary: dict, ref: dict | None, clade_txt: str,
                      run: dict, gain: dict | None, length_summary: dict | None,
                      disorder_summary: dict | None, by_len: dict,
                      registry: dict | None) -> str:
    """What kmerseek is, the question, the answer with its numbers, what kmerseek reaches
    and the caveat, why the clade is removed, the focus protein, and a reading order."""
    name = sp_name(species)
    total = pick(summary, "proteins_in_proteome")
    dark = pick(summary, "proteins_dark")
    frac = pick(summary, "fraction_dark")
    evalue = pick(summary, "evalue_call")
    kept = pick(ref, "entries_kept")
    excluded = pick(ref, "entries_excluded")
    own = pick(registry or {}, "swissprot_reviewed")

    b100 = by_len.get(100) if by_len else None
    at100 = (b100["dark"] / b100["kept"]) if b100 and b100.get("kept") else None
    answer = f"{pct(frac)} of the proteome is dark: {num(dark)} of {num(total)} proteins."
    if at100 is not None:
        answer += (f" Counting only proteins of at least 100 residues it is {pct(at100)}; "
                   f"the difference is short gene models, which are dark because there is "
                   f"little to align.")
    medians = []
    if length_summary:
        d, q = pick(length_summary, "dark", default={}), pick(length_summary, "placed", default={})
        if pick(d, "median") is not None and pick(q, "median") is not None:
            word = "shorter" if float(pick(d, "median")) < float(pick(q, "median")) else "longer"
            medians.append(f"{word} than placed ones (median {float(pick(d, 'median')):.0f} "
                           f"against {float(pick(q, 'median')):.0f} residues)")
    if disorder_summary:
        d, q = pick(disorder_summary, "dark", default={}), pick(disorder_summary, "placed", default={})
        if pick(d, "median") is not None and pick(q, "median") is not None:
            word = "more" if float(pick(d, "median")) > float(pick(q, "median")) else "less"
            medians.append(f"{word} disordered (median metapredict score "
                           f"{float(pick(d, 'median')):.3f} against "
                           f"{float(pick(q, 'median')):.3f}, on a 0 to 1 scale)")
    if medians:
        answer += " Dark proteins are " + " and ".join(medians) + "."

    reach = ""
    best = best_pair(gain)
    if best:
        dark_n = pick(gain, "dark_proteins") or dark
        on, off = best.get("dark_reached_mask_on"), best.get("dark_reached_mask_off")
        alpha = alphabet_words(gain)
        row_on = combo_row(gain, best["alphabet"], best["ksize"], True)
        placed_n = pick(gain, "placed_proteins")
        reach = (f"With <code>{best['alphabet']}</code> at k={best['ksize']}, kmerseek "
                 f"reports a region on {num(on)} of the {num(dark_n)} dark proteins "
                 f"({pct(on / dark_n if dark_n else None)}) with the low-complexity mask "
                 f"on, {num(off)} with it off. That is reach, not accuracy: it says the "
                 f"search returns something for a protein, not that the family label "
                 f"would be right.")
        if dark_n and on is not None and on >= 0.99 * dark_n:
            reach += (" At 100% it also says this setting is too permissive to be "
                      "informative on its own.")
        if row_on and row_on.get("placed_reached") is not None and placed_n:
            reach += (f" The same setting reaches {pct(row_on['placed_reached'] / placed_n)} "
                      f"of the placed proteins, the control for whether reach separates "
                      f"dark from placed at all.")
        else:
            reach += " The control, the same count inside the placed set, is not in this run."
        if not pick(gain, "has_shuffled_control"):
            reach += (" The shuffled-sequence control (the same search on the dark proteins "
                      "with their residues shuffled) is not in this run.")
        else:
            sh = row_on.get("shuffled_dark_reached") if row_on else None
            if sh is not None and dark_n:
                reach += (f" On the same proteins with their residues shuffled it reaches "
                          f"{pct(sh / dark_n)}, which is what composition alone gives.")
        others = [r for r in pick(gain, "mask_pairs", default=[]) or []
                  if (r["alphabet"], r["ksize"]) != (best["alphabet"], best["ksize"])]
        if others:
            o = others[0]
            reach += (f" <code>{o['alphabet']}</code> at k={o['ksize']} reaches "
                      f"{num(o.get('dark_reached_mask_on'))} dark proteins "
                      f"({pct((o.get('dark_reached_mask_on') or 0) / dark_n if dark_n else None)}), "
                      f"{num(o.get('dark_reached_mask_off'))} with the mask off.")
        reach = (f"<p><b>What kmerseek reaches inside the dark set, and the caveat.</b> "
                 f"{reach}</p>")
        alphabets = f" ({alpha})" if alpha else ""
    else:
        alphabets = ""

    own_txt = (f" {name}'s own {num(own)} reviewed entries are among them, so a protein "
               f"cannot be placed by hitting itself." if own else
               f" {name} itself has no reviewed Swiss-Prot entry, so the entries removed "
               f"are its relatives'.")
    n_all = (num(kept + excluded) if kept is not None and excluded is not None
             else "the reviewed")
    clade = (f"<p><b>Why the target has {clade_txt} removed.</b> All {num(excluded)} "
             f"reviewed Swiss-Prot entries from the query's own clade are taken out of the "
             f"{n_all} entries, so every hit has to come from outside that clade.{own_txt} "
             f"The same is done for every species this pipeline runs, which is what makes "
             f"their dark fractions comparable.</p>")

    focus = ""
    for acc, f in (pick(summary, "focus_proteins", default={}) or {}).items():
        what = f.get("what") or acc
        if not f.get("in_proteome", True):
            focus += f"<p><b>{what}</b> (<code>{acc}</code>) is not in this proteome.</p>"
            continue
        arms = f.get("per_arm") or {}
        placed_by = [a for a, v in arms.items() if v.get("placed")]
        call = (f"placed by {', '.join(placed_by)}" if placed_by else
                "dark: none of phmmer, jackhmmer and mmseqs2 places it")
        g = (pick(gain, "focus_proteins", default={}) or {}).get(acc, {})
        hits = [(k, v) for k, v in (g.get("per_combo") or {}).items()
                if v.get("reached") and v.get("low_complexity_mask")]
        km = ""
        if g:
            if hits:
                k, v = max(hits, key=lambda kv: kv[1].get("best_region_score") or 0)
                where = (f" at residues {v['best_region'][0]}-{v['best_region'][1]}"
                         if v.get("best_region") else "")
                km = (f"; kmerseek reaches it with the mask on under {len(hits)} of "
                      f"{sum(1 for v2 in g['per_combo'].values() if v2.get('low_complexity_mask'))} "
                      f"settings, best <code>{k.rsplit(' lc', 1)[0]}</code>{where}")
            else:
                km = "; no kmerseek setting with the mask on reaches it"
        focus += (f"<p><b>{what}</b> (<code>{acc}</code>) is {call}{km}. It is followed "
                  f"through every panel in <a href='#dark_focus'>One protein followed "
                  f"through</a>.</p>")

    return (
        f"<p><b>What kmerseek is.</b> A search tool that first groups the 20 amino acids "
        f"into a few classes{alphabets} and then looks for runs of <i>k</i> consecutive "
        f"classes that two proteins share, instead of aligning residues. The idea is that "
        f"a domain keeps its pattern of hydrophobic and polar positions long after the "
        f"residues themselves have changed.</p>"
        f"<p><b>The question this report answers.</b> How much of the {name} proteome can "
        f"no sequence search place into a curated database, and what does kmerseek see "
        f"inside that part? A protein is <b>placed</b> when phmmer, jackhmmer or mmseqs2 "
        f"finds it a hit at E &le; {evalue_txt(evalue)} in reviewed Swiss-Prot; it is "
        f"<b>dark</b> when none of the three does. That call needs no answer key, so it "
        f"can be made for a species with no curated entries of its own.</p>"
        f"<p><b>The answer.</b> {answer}</p>"
        + reach + clade + focus
        + "<p><b>Read in this order:</b> <a href='#dark_headline'>How much of the proteome "
          "is dark</a>, <a href='#dark_headline_by_length'>by minimum length</a>, "
          "<a href='#dark_length'>length</a> and <a href='#dark_disorder'>disorder</a>, "
          "then <a href='#dark_kmerseek_mask'>kmerseek reach</a>. The low-complexity mask "
          "is always shown as a pair, on and off, because a protein reached only with the "
          "mask off was found through a low-complexity stretch, which is composition "
          "rather than homology.</p>")


def section_overview(out: Path, species: str, summary: dict, ref: dict | None,
                     clade: str | None, run: dict, gain: dict | None,
                     length_df, disorder_df, length_summary: dict | None = None,
                     disorder_summary: dict | None = None,
                     registry: dict | None = None) -> None:
    total = pick(summary, "proteins_in_proteome")
    placed = pick(summary, "proteins_placed_by_any_arm")
    dark = pick(summary, "proteins_dark")
    frac = pick(summary, "fraction_dark")
    raw_rows = pick(summary, "raw_hit_rows")
    evalue = pick(summary, "evalue_call")
    chunk = pick(run, "query_chunk_size")
    n_chunks = math.ceil(total / chunk) if total and chunk else None
    kept = pick(ref, "entries_kept")
    excluded = pick(ref, "entries_excluded")
    sp_total = kept + excluded if kept is not None and excluded is not None else None
    clade_txt = clade or pick(ref, "excluded_clade") or "the query's own clade"

    combos = pick(gain, "by_combo", default=[]) or []
    pairs = pick(gain, "mask_pairs", default=[]) or []
    if pairs:
        kmerseek_line = (f"{len(pairs)} alphabet x k pair(s), low-complexity mask on and off; "
                         f"target indexed once per setting, every query chunk searched")
    elif combos:
        kmerseek_line = (f"{len(combos)} alphabet x k x mask setting(s); target indexed once "
                         f"per setting, every query chunk searched")
    else:
        kmerseek_line = "target indexed per alphabet, k and mask setting; every query chunk searched"
    arms = {"length": length_df is not None, "disorder": disorder_df is not None,
            "kmerseek": bool(combos)}
    by_len = dark_by_length(length_df)

    steps = [
        f"<b>Query: the {sp_name(species)} proteome.</b> {num(total)} proteins, split into "
        f"{num(n_chunks)} chunks of {num(chunk)} so each search is one job. Headers are "
        f"cut to the bare accession first, because the three searches and kmerseek each "
        f"report a UniProt header differently and the dark call needs one key. The "
        f"direction is the one a new genome faces: the proteome asks, Swiss-Prot answers. "
        f"(The other pipelines in this repository run the reverse, human as the query "
        f"against a proteome as the target.)",
        f"<b>Target database: reviewed Swiss-Prot.</b> {num(sp_total)} entries, with every "
        f"entry from the query's own clade removed ({clade_txt}: {num(excluded)} "
        f"entries), leaving {num(kept)}. Every species this pipeline runs gets the same "
        f"construction. The pipeline's files call this the reference.",
        f"<b>Three sequence searches</b>, every chunk against the reference: phmmer (one "
        f"pass), jackhmmer ({pick(run, 'jackhmmer_iterations', default=3)} iterations, the "
        f"profile rebuilt from the hits after each round) and mmseqs2 (sensitivity "
        f"{pick(run, 'mmseqs2_sensitivity', default=7)}, 3 iterations). Hits are kept "
        f"at E &le; {evalue_txt(pick(run, 'evalue_report', default=10))}: {num(raw_rows)} rows.",
        f"<b>The call.</b> A protein is <i>placed</i> when any arm has a hit at "
        f"E &le; {evalue_txt(evalue)}: {num(placed)} proteins. It is <i>dark</i> when no arm has one: "
        f"{num(dark)} proteins, {pct(frac)} of the proteome.",
    ]
    if arms["length"]:
        steps.append("<b>Length.</b> Every protein's length, read from the FASTA, dark "
                     "against placed.")
    if arms["disorder"]:
        steps.append("<b>Disorder.</b> Every protein's mean predicted disorder from "
                     "metapredict, dark against placed.")
    if arms["kmerseek"]:
        steps.append(
            f"<b>kmerseek.</b> The target indexed once per alphabet, k and "
            f"low-complexity mask setting, every query chunk searched against it. A protein "
            f"counts as <i>reached</i> when kmerseek reports at least one region on it "
            f"(region score &ge; {pick(run, 'min_region_score', default=1.3)}, query "
            f"p &le; {pick(run, 'max_query_pvalue', default=0.05)}, at least "
            f"{pick(run, 'min_shared_kmers', default=2)} shared k-mers). Reached is "
            f"counted inside the dark set and, as a control, inside the placed set.")
    steps.append("<b>This report.</b> One panel per step, in the order above; what a run "
                 "did not do is listed by name at the end.")

    why = bullets(
        "<b>The dark set is the denominator.</b> kmerseek can only add something where "
        "phmmer, jackhmmer and mmseqs2 all found nothing, so any claim that it annotates "
        "more of a proteome is a claim about this set of proteins and no other.",
        f"<b>It needs no answer key.</b> Whether a family label is right cannot be scored "
        f"for a species with no curated entries, but \"no arm hit this protein\" is a "
        f"property of the searches alone. That is why this number exists for "
        f"{sp_name(species)} before any structural key does.",
        "<b>The clade is removed so a hit has to come from outside it.</b> A species with "
        "deep Swiss-Prot coverage would otherwise place most of its proteome on its own "
        "entries, and the dark fraction would measure curation depth rather than how far "
        "sequence search reaches. Doing the same to every species is what makes their "
        "fractions comparable.",
        "<b>Dark is not the same as annotatable.</b> A dark protein may be a real protein "
        "whose homologs sequence search cannot reach, or a spurious gene model with nothing "
        "to find. Length and disorder are the cheap ways to tell those apart.",
    )

    # The key-value block under the report title: the run in five lines. Written as a
    # second config file that darkMultiqcReport passes after the main one.
    per_arm = pick(summary, "proteins_placed_per_arm", default={}) or {}
    combos_txt = (f"; kmerseek: {len(pairs)} alphabet x k pair(s), mask on and off" if pairs
                  else f"; kmerseek: {len(combos)} setting(s)" if combos else "")
    # The title and the first screen come from the builder too: the title names the
    # species (one report per species, and the config's title cannot know which), and the
    # first screen carries numbers, which a hand-written config line cannot keep current.
    name = SPECIES_NAME or species
    (out / "report_header.yaml").write_text(
        fd.yaml_html("title", f"{name} dark set")
        + fd.yaml_html("subtitle", f"The <i>{name}</i> proteins that phmmer, jackhmmer and "
                                   f"mmseqs2 all fail to place, and what kmerseek reaches "
                                   f"inside them")
        + fd.yaml_html("intro_text", first_screen_html(
            species, summary, ref, clade_txt, run, gain, length_summary, disorder_summary,
            by_len, registry))
        + fd.header_yaml([
        ("Query", f"{name} proteome, {num(total)} proteins"),
        ("Target", f"reviewed Swiss-Prot minus {clade_txt}: {num(kept)} entries kept, {num(excluded)} removed"),
        ("Arms", "phmmer (one pass), jackhmmer (3 iterations), mmseqs2 (sensitivity 7, 3 iterations)"
                 + combos_txt),
        ("The call", f"placed when some arm has a hit at E <= {evalue_txt(evalue)}; dark when none has"),
        ("Dark set", f"{num(dark)} of {num(total)} proteins ({pct(frac)})"),
    ]))
    write_section(out, "dark_overview", {
        "id": "dark_overview",
        "section_name": "What was done, and why",
        "description": (
            f"<p>The dark set of {sp_name(species)}: the proteins that three sequence "
            f"searches all fail to place into reviewed Swiss-Prot with {clade_txt} removed, "
            f"and what the optional arms say about them. The steps, this run's numbers, and "
            f"the flow of data from the two inputs to the panels below.</p>"),
        "plot_type": "html",
        "data": (
            "<h4>What was done</h4><ol>" + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
            "<h4>Why</h4>" + why +
            "<h4>Data flow</h4>"
            "<p>Read top to bottom; the label on the left says what kind of thing each row "
            "holds. A box holds a name and one number; an arrow is the step that makes the "
            "next box. Hover a box to see what feeds it; click it for what it is, its "
            "numbers, and the panels that show it. Switch the length cut and the bars in "
            "the placed / dark boxes move.</p>"
            + fd.block(dict(
                dark_flow_spec(species, summary, ref, clade, run, arms, gain, by_len),
                title=f"{species} dark set",
                details=flow_details(species, summary, ref, clade, run, gain, length_summary,
                                     disorder_summary, arms, by_len),
                control=length_control(by_len),
                footnote=("The searches were run on the whole proteome; the length cut only "
                          "changes which proteins are counted in the placed / dark row, the "
                          "same numbers as the \"Dark fraction by minimum length\" table. "
                          "Per-arm and kmerseek numbers do not change with the cut.")
                         if by_len else None),
                uid="dark-flow")),
    })


def section_metric_explainers(out: Path, species: str) -> None:
    """How to read the numbers: the dark fraction and the mask pair, as a schematic."""
    write_section(out, "dark_metric_explainers", {
        "id": "dark_metric_explainers",
        "section_name": "How to read the numbers",
        "description": (
            f"<p>The two ideas the panels below rest on. The bar widths are a schematic, not "
            f"{sp_name(species)}'s numbers; those are in the panels.</p>"),
        "plot_type": "html",
        "data": mx.bundle(mx.fraction()),
    })


# --- a) headline ---------------------------------------------------------------------

def section_headline(out: Path, species: str, summary: dict) -> None:
    total = pick(summary, "proteins_in_proteome")
    placed = pick(summary, "proteins_placed_by_any_arm")
    dark = pick(summary, "proteins_dark")
    frac = pick(summary, "fraction_dark")
    evalue = pick(summary, "evalue_call")
    raw_rows = pick(summary, "raw_hit_rows")

    if frac is None and total:
        frac = dark / total

    write_section(out, "dark_headline", {
        "id": "dark_headline",
        "section_name": "How much of the proteome is dark",
        "description": (
            f"<p>Every protein in {sp_name(species)} that phmmer, jackhmmer and mmseqs2 "
            f"<i>all</i> failed to place into reviewed Swiss-Prot with the query's own "
            f"clade removed.</p>"
            + bullets(
                f"<b>{num(dark)} of {num(total)} proteins ({pct(frac)}) are dark</b> at "
                f"E&le;{evalue}.",
                f"<b>{num(placed)} were placed</b> by at least one arm.",
                f"<b>{num(raw_rows)} raw hit rows</b> were kept at the permissive report "
                f"cutoff, so the call cutoff can be moved without re-running a search."
                if raw_rows is not None else "",
                "<b>This needs no answer key.</b> \"No arm hit this protein\" is a property "
                "of the searches alone, which is why it can be measured on a species with "
                "no curated entries at all.",
                "<b>Dark is not the same as annotatable.</b> A dark protein may be a real "
                "protein whose homologs sequence search cannot reach, or a spurious gene "
                "model with nothing to find. The length and disorder panels below are the "
                "cheap discriminators; the annotation evidence is what settles it.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "dark_headline_plot",
                    "title": f"{species}: placed vs dark",
                    "ylab": "proteins",
                    "cpswitch_c_active": False,
                    "sort_samples": False},
        "categories": {
            "placed": {"name": "placed by a sequence arm", "color": C_PLACED},
            "dark": {"name": "dark to every arm", "color": C_DARK},
        },
        "data": {species: {"placed": placed, "dark": dark}},
    })

    row = {
        "proteins_in_proteome": total,
        "placed_by_any_arm": placed,
        "dark": dark,
        "fraction_dark": frac,
        "evalue_call": evalue,
    }
    if raw_rows is not None:
        row["raw_hit_rows"] = raw_rows
    write_section(out, "dark_headline_table", {
        "id": "dark_headline_table",
        "section_name": "Dark set in numbers",
        "description": "<p>The panel above as numbers.</p>",
        "plot_type": "table",
        "pconfig": {"id": "dark_headline_table_plot",
                    "title": f"{species}: dark set",
                    "col1_header": "species", "sort_rows": False},
        "headers": {
            "proteins_in_proteome": {"title": "proteome", "format": "{:,.0f}"},
            "placed_by_any_arm": {"title": "placed", "format": "{:,.0f}"},
            "dark": {"title": "dark", "format": "{:,.0f}"},
            "fraction_dark": {"title": "fraction dark", "format": "{:,.4f}",
                              "min": 0, "max": 1},
            "evalue_call": {"title": "E-value call cutoff", "format": "{:,.1e}"},
            "raw_hit_rows": {"title": "raw hit rows", "format": "{:,.0f}"},
        },
        "data": {species: row},
    })


# --- b) placed per arm ---------------------------------------------------------------

def section_per_arm(out: Path, species: str, summary: dict, omitted: Omitted) -> None:
    per_arm = pick(summary, "proteins_placed_per_arm", default={}) or {}
    if not per_arm:
        omitted.add("Placed per arm",
                    "the dark summary carries no <code>proteins_placed_per_arm</code> "
                    "block, so no per-arm count could be read.")
        return

    union = pick(summary, "proteins_placed_by_any_arm")
    total = pick(summary, "proteins_in_proteome")

    # Ascending, so the ordering IS the figure. Iterative search sees everything a
    # single-pass search sees, so phmmer must sit below jackhmmer and mmseqs2, and the
    # union must sit above all three. A bar out of place is an arm that did not run.
    entries = [(arm, int(n)) for arm, n in per_arm.items()]
    if union is not None:
        entries.append(("any arm", int(union)))
    entries.sort(key=lambda kv: kv[1])

    data = {}
    for arm, n in entries:
        cls = "union" if arm == "any arm" else ARM_CLASS.get(arm, "single_pass")
        data[arm] = {cls: n}

    single = max((n for a, n in entries if ARM_CLASS.get(a) == "single_pass"), default=None)
    iterative = [(a, n) for a, n in entries if ARM_CLASS.get(a) == "iterative"]
    below = [a for a, n in iterative if single is not None and n < single]
    if single is None or not iterative:
        verdict = ("<b>The single-pass / iterative check could not be run</b>: this run "
                   "does not carry both kinds of arm.")
    elif below:
        verdict = ("<b>An iterative arm placed FEWER proteins than the single-pass one</b> "
                   f"({', '.join(below)} below phmmer). Iterative search sees everything "
                   "phmmer sees, so this is an arm that did not really run, not a result. "
                   "The dark fraction above is inflated until it is fixed.")
    else:
        verdict = ("<b>Both iterative arms placed at least as many proteins as the "
                   "single-pass one</b>, which is the ordering that has to hold. It is a "
                   "check that every arm ran, not a comparison between the arms.")

    unions_ok = ""
    if union is not None:
        worst = max((n for a, n in entries if a != "any arm"), default=0)
        if union < worst:
            unions_ok = ("<b>The union is smaller than an individual arm</b>, which cannot "
                         "happen if the arms were combined correctly. Read nothing else on "
                         "this page until that is explained.")

    write_section(out, "dark_per_arm", {
        "id": "dark_per_arm",
        "section_name": "What each arm placed",
        "description": (
            f"<p>Proteins each search arm placed into the reference at the call cutoff, "
            f"smallest first, out of {num(total)}.</p>"
            + bullets(
                verdict,
                unions_ok,
                "<b>Colour is the kind of search</b>: single-pass, iterative, or the union "
                "of all three. That is the axis the check is about.",
                "<b>The dark set is defined against the union of all three arms, not the "
                "cheapest one.</b> A protein any arm reaches, iterative search included, is "
                "not dark; counting it as dark would inflate the headline in the direction "
                "the claim wants.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "dark_per_arm_plot",
                    "title": f"{species}: proteins placed, by arm",
                    "ylab": "proteins placed",
                    "cpswitch": False, "stacking": "group", "sort_samples": False},
        "categories": {
            "single_pass": {"name": "single-pass search", "color": C_SINGLE},
            "iterative": {"name": "iterative search", "color": C_ITER},
            "union": {"name": "any arm", "color": C_UNION},
        },
        "data": data,
    })

    table = {}
    for arm, n in entries:
        table[arm] = {
            "kind": "any arm" if arm == "any arm" else ARM_CLASS.get(arm, "unknown"),
            "placed": n,
            "fraction_of_proteome": (n / total) if total else None,
        }
    write_section(out, "dark_per_arm_table", {
        "id": "dark_per_arm_table",
        "section_name": "Placed per arm, as numbers",
        "description": "<p>The panel above as numbers, smallest first.</p>",
        "plot_type": "table",
        "pconfig": {"id": "dark_per_arm_table_plot",
                    "title": f"{species}: placed per arm",
                    "col1_header": "arm", "sort_rows": False},
        "headers": {
            "kind": {"title": "kind of search"},
            "placed": {"title": "proteins placed", "format": "{:,.0f}"},
            "fraction_of_proteome": {"title": "fraction of proteome", "format": "{:,.4f}",
                                     "min": 0, "max": 1},
        },
        "data": table,
    })


# --- c) kmerseek reach inside the dark set, mask ON beside mask OFF --------------------

def section_kmerseek(out: Path, species: str, gain: dict | None, omitted: Omitted) -> None:
    if gain is None:
        omitted.add("kmerseek reach in the dark set",
                    "this run did not include the kmerseek arm "
                    "(<code>--with_kmerseek true</code>), so no "
                    "<code>_kmerseek_dark_gain.json</code> was produced.")
        return

    dark_n = pick(gain, "dark_proteins")
    pairs = pick(gain, "mask_pairs", default=[]) or []
    combos = pick(gain, "by_combo", default=[]) or []

    section_kmerseek_sweep(out, species, combos, dark_n, omitted)

    # Every combo that has only one mask setting measured. Drawn alone, a mask-OFF bar
    # would present a composition artifact as a result, so these are named and dropped.
    seen: dict[tuple, set] = {}
    for row in combos:
        key = (row.get("alphabet"), row.get("ksize"))
        seen.setdefault(key, set()).add(bool(row.get("low_complexity_mask")))
    unpaired = {k: v for k, v in seen.items() if len(v) < 2}

    unpaired_note = ""
    if unpaired:
        listed = "; ".join(
            f"<code>{a} k{k}</code> (only mask "
            f"{'ON' if True in v else 'OFF'} was measured)"
            for (a, k), v in sorted(unpaired.items(), key=lambda kv: str(kv[0])))
        unpaired_note = (
            f"<b>Dropped for having only one mask setting:</b> {listed}. A rescue count "
            f"from one side of the pair alone cannot be told apart from a composition "
            f"artifact, so it is not drawn.")

    if not pairs:
        omitted.add("kmerseek mask pair",
                    "the kmerseek arm ran but no alphabet/ksize combo has BOTH mask "
                    "settings, so the paired panel is not drawn: a single mask setting "
                    "alone cannot separate a rescue from a composition artifact. The "
                    "reach-by-k panels above still show what ran. "
                    + (f"Combos with one side only: "
                       f"{', '.join(f'{a} k{k}' for a, k in sorted(unpaired, key=str))}."
                       if unpaired else "No combos were scored at all."))
        return

    placed_n = pick(gain, "placed_proteins")
    thresholds = [str(t) for t in (pick(gain, "thresholds", default=[]) or [])]
    has_shuffled = bool(pick(gain, "has_shuffled_control"))

    # Three datasets on one panel, each a fraction so the bars are comparable: the dark
    # set, the placed set (the control), and the shuffled dark proteins (the null) when
    # the run has them. Colour is the mask setting in every dataset.
    dark_data, placed_data, shuffled_data = {}, {}, {}
    table = {}
    for row in pairs:
        label = f"{row['alphabet']} k{row['ksize']}"
        on = row.get("dark_reached_mask_on")
        off = row.get("dark_reached_mask_off")
        lost = row.get("lost_to_masking")
        if lost is None and on is not None and off is not None:
            lost = off - on
        r_on = combo_row(gain, row["alphabet"], row["ksize"], True) or {}
        r_off = combo_row(gain, row["alphabet"], row["ksize"], False) or {}
        frac = lambda v, n: (v / n) if (v is not None and n) else None
        dark_data[label] = {"mask_on": frac(on, dark_n), "mask_off": frac(off, dark_n)}
        placed_data[label] = {"mask_on": frac(r_on.get("placed_reached"), placed_n),
                              "mask_off": frac(r_off.get("placed_reached"), placed_n)}
        shuffled_data[label] = {"mask_on": frac(r_on.get("shuffled_dark_reached"), dark_n),
                                "mask_off": frac(r_off.get("shuffled_dark_reached"), dark_n)}
        cell = {
            "mask_on": on, "mask_off": off, "lost_to_masking": lost,
            "fraction_dark_mask_on": frac(on, dark_n),
            "fraction_dark_mask_off": frac(off, dark_n),
            "fraction_placed_mask_on": frac(r_on.get("placed_reached"), placed_n),
            "fraction_placed_mask_off": frac(r_off.get("placed_reached"), placed_n),
        }
        for t in thresholds:
            cell[f"fraction_dark_mask_on_at_{t}"] = frac((r_on.get("dark_reached_at") or {}).get(t), dark_n)
            cell[f"fraction_placed_mask_on_at_{t}"] = frac((r_on.get("placed_reached_at") or {}).get(t), placed_n)
        if has_shuffled:
            cell["fraction_shuffled_mask_on"] = frac(r_on.get("shuffled_dark_reached"), dark_n)
            cell["fraction_shuffled_mask_off"] = frac(r_off.get("shuffled_dark_reached"), dark_n)
        table[label] = cell

    best = max(pairs, key=lambda r: r.get("dark_reached_mask_on") or 0)
    best_on = best.get("dark_reached_mask_on")
    worst_loss = max(pairs, key=lambda r: r.get("lost_to_masking") or 0)

    # An arm that reaches every protein, placed and dark alike, has a region cutoff that
    # lets everything through; its reach then says nothing about the dark set and must
    # not be read as a rescue. Computed from by_combo, which carries the placed side.
    saturated = sorted({
        f"{r['alphabet']} k{r['ksize']}" for r in combos
        if dark_n and placed_n and (r.get("dark_reached") or 0) >= 0.99 * dark_n
        and (r.get("placed_reached") or 0) >= 0.99 * placed_n})
    saturated_note = ""
    if saturated:
        saturated_note = (
            "<b>Not a rescue: </b>" + ", ".join(f"<code>{a}</code>" for a in saturated)
            + (" reaches" if len(saturated) == 1 else " reach")
            + " at least 99% of the placed proteins and 99% of the dark ones alike. At "
              "this run's region cutoff the arm puts a region on nearly every protein in "
              "the proteome. A reach that does not separate dark from placed says nothing "
              "about the dark set; read that arm's bar as a cutoff that needs tightening, "
              "not as a result."
            + (" The stricter cutoffs in the table below are that tightening." if thresholds
               else ""))

    best_placed = combo_row(gain, best["alphabet"], best["ksize"], True) or {}
    best_frac = (best_on / dark_n) if dark_n and best_on is not None else None
    best_txt = (f"<b>Best arm with the mask ON: <code>{best['alphabet']} "
                f"k{best['ksize']}</code></b>, reaching {num(best_on)} of {num(dark_n)} "
                f"dark proteins ({pct(best_frac)})")
    if best_placed.get("placed_reached") is not None and placed_n:
        best_txt += (f", and {pct(best_placed['placed_reached'] / placed_n)} of the placed "
                     f"proteins")
    best_txt += "."
    if best_frac is not None and best_frac >= 0.99:
        best_txt += (" A setting that reaches every dark protein is reaching everything; "
                     "on its own this row says the setting is permissive, not that the "
                     "dark set is annotatable. What tells the two apart is the same count "
                     "inside the placed set (the column beside it) and on shuffled "
                     "sequences"
                     + (" (the third tab of the panel)." if has_shuffled else
                        ", which is not in this run."))

    control_note = (
        "<b>The placed set is the control.</b> An arm that reaches 95% of the placed "
        "proteins and 10% of the dark ones is discriminating; one that reaches 100% of "
        "both is saturated, and its dark number means nothing. The two are side by side "
        "as fractions, one tab each, so the bars are comparable.")
    shuffled_note = (
        "<b>The shuffled dark proteins are the null.</b> Each dark protein with its "
        "residues shuffled keeps its length and composition and loses everything else, so "
        "what kmerseek reaches on those is what it reaches by composition alone. A dark "
        "reach the shuffled copies match is not a homology signal."
        if has_shuffled else
        "<b>No shuffled-sequence control in this run.</b> The same search on the dark "
        "proteins with their residues shuffled would say what composition alone reaches; "
        "run with <code>--with_shuffle_control true</code> to add it.")
    threshold_note = (
        f"<b>Reach at stricter cutoffs</b> is in the table below: the run's own region "
        f"score cutoff and {', '.join(thresholds[1:])} (region p &le; "
        f"{', '.join(f'1e-{int(float(t))}' for t in thresholds[1:])}). A reach of 100% "
        f"at the run cutoff can then be read against what survives a tighter one."
        if len(thresholds) > 1 else
        "<b>Reach at a stricter cutoff is not in this run</b>; it needs the per-query "
        "scores a run before 2026-09-20 did not keep.")

    datasets = [dark_data, placed_data] + ([shuffled_data] if has_shuffled else [])
    data_labels = [{"name": "dark set", "ylab": "fraction of dark proteins reached"},
                   {"name": "placed set (control)", "ylab": "fraction of placed proteins reached"}]
    if has_shuffled:
        data_labels.append({"name": "shuffled dark proteins (null)",
                            "ylab": "fraction of shuffled dark proteins reached"})
    write_section(out, "dark_kmerseek_mask", {
        "id": "dark_kmerseek_mask",
        "section_name": "kmerseek reach inside the dark set",
        "description": (
            f"<p>The fraction of the {num(dark_n)} dark proteins kmerseek put a region on, "
            f"with the low-complexity mask ON and OFF side by side. The second tab is the "
            f"same fraction inside the {num(placed_n)} placed proteins, the control"
            + ("; the third is the shuffled dark proteins, the null" if has_shuffled else "")
            + ". The two bars in a group are the same search with the filter switched.</p>"
            + bullets(
                "<b>Read the pair, never one bar.</b> A rescue present with the mask off "
                "and gone with it on is a property of amino-acid composition, not of "
                "homology. BHF's flagship matches included polar-biased low-complexity "
                "segments, which is why this pipeline runs the mask as a paired setting "
                "rather than as a sweep dimension.",
                control_note,
                shuffled_note,
                saturated_note,
                best_txt,
                f"<b>Largest loss to masking: <code>{worst_loss['alphabet']} "
                f"k{worst_loss['ksize']}</code></b> at "
                f"{num(worst_loss.get('lost_to_masking'))} proteins, which is the part of "
                f"that arm's mask-off number that the filter does not support.",
                threshold_note,
                unpaired_note,
                "<b>This is reach, not accuracy.</b> A region on a dark protein says "
                "kmerseek found something there, not that the family label is right. "
                "Checking the label needs a structural answer key, which does not exist "
                "for this species yet. The number here is necessary for any claim about "
                "annotating the dark set, and nowhere near sufficient.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "dark_kmerseek_mask_plot",
                    "title": f"{species}: fraction reached, mask ON vs OFF",
                    "ylab": "fraction of proteins reached", "ymax": 1,
                    # Grouped, never stacked. The two bars are the same proteins counted
                    # twice under different settings, so stacking them would draw a total
                    # that does not exist.
                    "cpswitch": False, "stacking": "group", "sort_samples": False,
                    "data_labels": data_labels},
        "categories": {
            "mask_on": {"name": "low-complexity mask ON", "color": C_MASK_ON},
            "mask_off": {"name": "mask OFF", "color": C_MASK_OFF},
        },
        "data": datasets,
    })

    headers = {
        "mask_on": {"title": "dark reached, mask ON", "format": "{:,.0f}"},
        "mask_off": {"title": "dark reached, mask OFF", "format": "{:,.0f}"},
        "lost_to_masking": {"title": "lost to masking", "format": "{:,.0f}"},
        "fraction_dark_mask_on": {"title": "fraction of dark, mask ON",
                                  "format": "{:,.4f}", "min": 0, "max": 1},
        "fraction_dark_mask_off": {"title": "fraction of dark, mask OFF",
                                   "format": "{:,.4f}", "min": 0, "max": 1},
        "fraction_placed_mask_on": {"title": "fraction of placed, mask ON (control)",
                                    "format": "{:,.4f}", "min": 0, "max": 1},
        "fraction_placed_mask_off": {"title": "fraction of placed, mask OFF (control)",
                                     "format": "{:,.4f}", "min": 0, "max": 1},
    }
    for t in thresholds:
        headers[f"fraction_dark_mask_on_at_{t}"] = {
            "title": f"fraction of dark, mask ON, score \u2265 {t}",
            "format": "{:,.4f}", "min": 0, "max": 1}
        headers[f"fraction_placed_mask_on_at_{t}"] = {
            "title": f"fraction of placed, mask ON, score \u2265 {t}",
            "format": "{:,.4f}", "min": 0, "max": 1}
    if has_shuffled:
        headers["fraction_shuffled_mask_on"] = {
            "title": "fraction of shuffled dark, mask ON (null)",
            "format": "{:,.4f}", "min": 0, "max": 1}
        headers["fraction_shuffled_mask_off"] = {
            "title": "fraction of shuffled dark, mask OFF (null)",
            "format": "{:,.4f}", "min": 0, "max": 1}
    write_section(out, "dark_kmerseek_mask_table", {
        "id": "dark_kmerseek_mask_table",
        "section_name": "Mask pair, as numbers",
        "description": (
            "<p>The panel above as numbers, with what masking costs each arm, the placed "
            "set beside the dark one, and reach at stricter region-score cutoffs where the "
            "run kept the scores.</p>"
            + bullets(
                "<b>lost_to_masking</b> is mask-off minus mask-on: dark proteins reached "
                "only when the low-complexity filter is off.",
                "<b>The fractions are of the set named in the column</b>: the dark set, "
                "the placed set, or the shuffled dark proteins.",
                "<b>A score cutoff column</b> counts the proteins whose best region scores "
                "at or above that cutoff; the run's own cutoff is the first of them.")),
        "plot_type": "table",
        "pconfig": {"id": "dark_kmerseek_mask_table_plot",
                    "title": f"{species}: dark reach by alphabet and k",
                    "col1_header": "alphabet, ksize", "sort_rows": False},
        "headers": headers,
        "data": table,
    })


def section_kmerseek_sweep(out: Path, species: str, combos: list[dict], dark_n,
                           omitted: Omitted) -> None:
    """Reach against ksize, one line per alphabet, dark and placed side by side.

    Drawn only when some alphabet was run at more than one ksize: the two-combo default
    run has nothing to put on a k axis, and the mask-pair panel already shows it.

    The placed panel is what keeps the dark panel honest. "Reached" saturates: on
    Botryllus hp_thomas_dill2 k23 put a region on every one of the 45_339 proteins, dark
    and placed alike, which is an arm reporting something for everything rather than a
    rescue. Read the two panels together -- the k where the placed fraction begins to
    fall away from 1.0 is the k from which the dark fraction starts to mean anything, and
    an alphabet whose dark line sits well under its placed line is discriminating.
    """
    by_lc: dict[bool, dict[str, dict[int, dict]]] = {}
    for row in combos:
        a, k = row.get("alphabet"), row.get("ksize")
        if a is None or k is None:
            continue
        by_lc.setdefault(bool(row.get("low_complexity_mask")), {}) \
             .setdefault(a, {})[int(k)] = row
    swept = any(len(ks) > 1 for arms in by_lc.values() for ks in arms.values())
    if not swept:
        return

    table = {}
    for lc in sorted(by_lc, reverse=True):
        arms = by_lc[lc]
        suffix = "on" if lc else "off"
        dark_lines = {a: {k: r.get("fraction_dark_reached") for k, r in sorted(ks.items())}
                      for a, ks in sorted(arms.items())}
        placed_lines = {a: {k: r.get("fraction_placed_reached") for k, r in sorted(ks.items())}
                        for a, ks in sorted(arms.items())}
        for a, ks in sorted(arms.items()):
            for k, r in sorted(ks.items()):
                table[f"{a} k{k} mask {suffix.upper()}"] = {
                    "dark_reached": r.get("dark_reached"),
                    "fraction_dark_reached": r.get("fraction_dark_reached"),
                    "placed_reached": r.get("placed_reached"),
                    "fraction_placed_reached": r.get("fraction_placed_reached"),
                }
        has_placed = any(v is not None for line in placed_lines.values() for v in line.values())

        write_section(out, f"dark_kmerseek_sweep_{suffix}", {
            "id": f"dark_kmerseek_sweep_{suffix}",
            "section_name": f"kmerseek reach by alphabet and k, mask {suffix.upper()}",
            "description": (
                f"<p>Fraction of the {num(dark_n)} dark proteins with any kmerseek region, "
                f"against ksize, one line per alphabet, low-complexity mask {suffix.upper()}."
                f"</p>"
                + bullets(
                    "<b>A line at 1.0 is saturation, not rescue.</b> An arm that reports a "
                    "region for every dark protein reports one for every placed protein "
                    "too; see the placed panel that follows. The k where the placed line "
                    "starts to fall away from 1.0 is the k from which this line means "
                    "anything.",
                    "<b>Reach is not accuracy.</b> A region on a dark protein says kmerseek "
                    "found something there, not that the family label is right; that "
                    "needs the structural key.")),
            "plot_type": "linegraph",
            "pconfig": {"id": f"dark_kmerseek_sweep_{suffix}_plot",
                        "title": f"{species}: dark proteins reached vs k, mask {suffix.upper()}",
                        "xlab": "ksize", "ylab": "fraction of dark proteins reached",
                        "ymin": 0, "ymax": 1},
            "data": dark_lines,
        })
        if has_placed:
            write_section(out, f"dark_kmerseek_sweep_placed_{suffix}", {
                "id": f"dark_kmerseek_sweep_placed_{suffix}",
                "section_name": f"the same on the PLACED proteins, mask {suffix.upper()}",
                "description": (
                    "<p>Fraction of the proteins some sequence arm DID place that kmerseek "
                    "also put a region on, against ksize. These are proteins with a known "
                    "homolog in the reference, so this line is the closest thing to "
                    "sensitivity a run without an answer key has.</p>"
                    + bullets(
                        "<b>Read it against the dark panel above.</b> An alphabet whose dark "
                        "line sits well under its placed line is discriminating; one where "
                        "the two coincide is reporting on composition, not homology.")),
                "plot_type": "linegraph",
                "pconfig": {"id": f"dark_kmerseek_sweep_placed_{suffix}_plot",
                            "title": f"{species}: placed proteins reached vs k, mask {suffix.upper()}",
                            "xlab": "ksize", "ylab": "fraction of placed proteins reached",
                            "ymin": 0, "ymax": 1},
                "data": placed_lines,
            })

    write_section(out, "dark_kmerseek_sweep_table", {
        "id": "dark_kmerseek_sweep_table",
        "section_name": "Reach by alphabet and k, as numbers",
        "description": "<p>Every combo of the sweep, dark and placed reach side by side.</p>",
        "plot_type": "table",
        "pconfig": {"id": "dark_kmerseek_sweep_table_plot",
                    "title": f"{species}: reach per combo",
                    "col1_header": "alphabet, ksize, mask", "sort_rows": False},
        "headers": {
            "dark_reached": {"title": "dark reached", "format": "{:,.0f}"},
            "fraction_dark_reached": {"title": "fraction of dark", "format": "{:,.4f}",
                                      "min": 0, "max": 1},
            "placed_reached": {"title": "placed reached", "format": "{:,.0f}"},
            "fraction_placed_reached": {"title": "fraction of placed", "format": "{:,.4f}",
                                        "min": 0, "max": 1},
        },
        "data": table,
    })


# --- d) and e) the two dark-vs-placed covariates --------------------------------------

def grouped_histogram(df: pl.DataFrame, value_col: str, group_col: str,
                      edges: list[float]) -> tuple[dict, dict, int]:
    """Percent of each group falling in each bin, on bins shared by both groups.

    Shared edges, not per-group quantiles: quantile bins computed inside each group put a
    different cut in each one and the two histograms stop being comparable.

    Rows whose group or value is null are dropped and counted, never folded into a group.
    A null here means a protein the upstream join did not resolve, and silently calling it
    'placed' is the failure mode this project has already been bitten by.
    """
    usable = df.filter(pl.col(value_col).is_not_null() & pl.col(group_col).is_not_null())
    dropped = df.height - usable.height

    labels = []
    for i, low in enumerate(edges):
        if i + 1 < len(edges):
            labels.append((low, edges[i + 1]))
        else:
            labels.append((low, None))

    def label_of(low, high) -> str:
        if high is None:
            return f"≥{num(low)}"
        if isinstance(low, float) or isinstance(high, float):
            return f"{low:g}-{high:g}"
        return f"{num(low)}-{num(high - 1)}"

    data = {label_of(lo, hi): {} for lo, hi in labels}
    counts = {}
    for group in sorted(usable[group_col].unique().to_list()):
        sub = usable.filter(pl.col(group_col) == group)
        counts[group] = sub.height
        for lo, hi in labels:
            expr = pl.col(value_col) >= lo
            if hi is not None:
                expr = expr & (pl.col(value_col) < hi)
            n = int(sub.filter(expr).height)
            data[label_of(lo, hi)][group] = (100 * n / sub.height) if sub.height else 0.0
    return data, counts, dropped


def normalise_groups(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """'dark'/'placed' whatever the upstream arm called the column.

    A boolean is_dark column and a string group column both arrive here; mapping them to
    one vocabulary is what lets the section text say 'dark' unconditionally. An unmappable
    value is left as-is rather than guessed at, so it shows up in the panel as its own
    series instead of being silently merged into one of the two.
    """
    if df[group_col].dtype == pl.Boolean:
        return df.with_columns(
            pl.when(pl.col(group_col).is_null()).then(None)
              .when(pl.col(group_col)).then(pl.lit("dark"))
              .otherwise(pl.lit("placed")).alias("_group"))
    return df.with_columns(
        pl.when(pl.col(group_col).is_null()).then(None)
          .otherwise(pl.col(group_col).cast(pl.Utf8).str.to_lowercase()).alias("_group"))


def mw_bullet(summary: dict | None, higher_word: str = "higher",
              lower_word: str = "lower") -> str:
    """The Mann-Whitney line, effect size first. `higher_word` / `lower_word` are the
    axis's own words for the two directions: longer / shorter, more / less disordered."""
    if not summary:
        return ""
    mw = pick(summary, "mann_whitney_u", "mann_whitney", "mannwhitneyu", "mannwhitney",
              default=None)
    if not isinstance(mw, dict):
        mw = summary
    p = pick(mw, "p_value", "pvalue", "p", "mannwhitneyu_p")
    cles = pick(mw, "common_language_effect_size", "cles", "effect_size")
    rbc = pick(mw, "rank_biserial_correlation", "rank_biserial")
    if p is None and cles is None and rbc is None:
        return ""
    # Both scripts count U for the dark group, so the common-language effect size is
    # P(a random dark protein is HIGHER than a random placed one) and rank-biserial is
    # 2 x that - 1. Until 2026-09-20 this bullet printed that probability with the word
    # "lower", so a dark set that was shorter (P(higher) = 0.195) read as "lower ... with
    # probability 0.195", the direction reversed. The probability is now stated as the
    # chance of the direction the data shows, and the words come from the axis.
    p_higher = float(cles) if cles is not None else (float(rbc) + 1) / 2
    parts = []
    if p_higher > 0.5:
        parts.append(f"The chance that a randomly drawn dark protein is {higher_word} than "
                     f"a randomly drawn placed one is {p_higher:.3f} (0.5 would be no "
                     f"difference)")
    elif p_higher < 0.5:
        parts.append(f"The chance that a randomly drawn dark protein is {lower_word} than "
                     f"a randomly drawn placed one is {1 - p_higher:.3f} (0.5 would be no "
                     f"difference)")
    else:
        parts.append("A randomly drawn dark protein is as likely to be above as below a "
                     "randomly drawn placed one (0.5, no difference)")
    parts.append(f"rank-biserial correlation, the same comparison on a -1 to 1 scale, is "
                 f"{2 * p_higher - 1:+.3f}")
    if p is not None:
        p = float(p)
        # Below about 1e-300 the software prints 0; say that rather than "p=0".
        parts.append("the Mann-Whitney p-value is below the smallest number the software "
                     "prints (1e-300)" if p < 1e-300 else f"Mann-Whitney p={p:.3g}")
    return ("<b>Read the effect size, not the p-value.</b> " + "; ".join(parts)
            + ". At tens of thousands of proteins the test is significant at effect sizes "
              "far too small to matter, which is why the p-value is not the number to "
              "read.")


def stat_bullet(summary: dict | None, unit: str, digits: int = 0) -> str:
    if not summary:
        return ""
    dark = pick(summary, "dark", "dark_stats", default={})
    placed = pick(summary, "placed", "placed_stats", default={})
    d_med = pick(dark, "median")
    p_med = pick(placed, "median")
    if d_med is None or p_med is None:
        return ""

    def fmt(v):
        return f"{float(v):.{digits}f}"

    d_iqr = (f" (IQR {fmt(pick(dark, 'p25'))}–{fmt(pick(dark, 'p75'))})"
             if pick(dark, "p25") is not None else "")
    p_iqr = (f" (IQR {fmt(pick(placed, 'p25'))}–{fmt(pick(placed, 'p75'))})"
             if pick(placed, "p25") is not None else "")
    return (f"<b>Median {unit}</b>: dark {fmt(d_med)}{d_iqr}, "
            f"placed {fmt(p_med)}{p_iqr}, over {num(pick(dark, 'n'))} dark and "
            f"{num(pick(placed, 'n'))} placed proteins.")


def covariate_section(out: Path, species: str, section_id: str, name: str,
                      df: pl.DataFrame | None, summary: dict | None,
                      value_candidates: list[str], edges: list[float],
                      axis_label: str, unit: str, digits: int,
                      preamble: str, extra_bullets: list[str],
                      omitted: Omitted, what: str,
                      higher_word: str = "higher", lower_word: str = "lower") -> None:
    """One dark-vs-placed distribution panel, or a named omission if it cannot be built."""
    if df is None and summary is None:
        omitted.add(name, f"this run produced no {what} products, so the comparison was "
                          f"not built.")
        return

    if df is None:
        omitted.add(name, f"the {what} summary is present but its parquet is not, so the "
                          f"distribution could not be drawn. Its summary statistics are "
                          f"still in the published JSON.")
        return

    value_col = pick_col(df, value_candidates)
    group_col = pick_col(df, GROUP_COLS)
    if value_col is None or group_col is None:
        looked = ", ".join(f"<code>{c}</code>" for c in
                           (value_candidates if value_col is None else GROUP_COLS))
        found = ", ".join(f"<code>{c}</code>" for c in df.columns)
        omitted.add(name,
                    f"the {what} parquet has no column this report recognises as the "
                    f"{'value' if value_col is None else 'dark/placed group'}. Looked for "
                    f"{looked}; found {found}.")
        return

    marked = normalise_groups(df, group_col)
    data, counts, dropped = grouped_histogram(marked, value_col, "_group", edges)
    if not counts:
        omitted.add(name, f"every row in the {what} parquet has a null group or a null "
                          f"value, so there is nothing to compare.")
        return

    unexpected = sorted(set(counts) - {"dark", "placed"})
    series = {}
    for group in sorted(counts):
        colour = C_DARK if group == "dark" else C_PLACED if group == "placed" else "#8c564b"
        series[group] = {"name": f"{group} (n={num(counts[group])})", "color": colour}

    write_section(out, section_id, {
        "id": section_id,
        "section_name": name,
        "description": (
            f"<p>{preamble}</p>"
            + bullets(
                stat_bullet(summary, unit, digits),
                mw_bullet(summary, higher_word, lower_word),
                *extra_bullets,
                "<b>Each series is a percentage of its own group</b>, because the two "
                "groups are nowhere near the same size and raw counts would show only "
                "that.",
                (f"<b>{num(dropped)} rows were dropped</b> for a null group or a null "
                 f"value. They are not folded into either group: a null here is a protein "
                 f"the upstream join did not resolve, and calling it placed would be an "
                 f"invented answer." if dropped else ""),
                (f"<b>Unexpected group label(s)</b>: {', '.join(unexpected)}. Drawn as "
                 f"their own series rather than merged into dark or placed."
                 if unexpected else ""))),
        "plot_type": "bargraph",
        "pconfig": {"id": f"{section_id}_plot",
                    "title": f"{species}: {axis_label}, dark vs placed",
                    "xlab": axis_label, "ylab": "% of group",
                    "cpswitch": False, "stacking": "group", "sort_samples": False,
                    "ymax": 100},
        "categories": series,
        "data": data,
    })


MIN_AA = (0, 50, 100, 200)


def section_dark_by_length(out: Path, species: str, df: pl.DataFrame | None,
                           omitted: Omitted) -> None:
    """The headline fraction recomputed over proteins of at least 50, 100 and 200 aa.

    Computed from the length parquet here rather than read from the length summary, so a
    report re-rendered over a run that predates dark_fraction_by_min_length still gets
    the panel. The raw number counts a 60-aa gene model that nothing can align the same as
    a 400-aa protein nothing can place; on the QfO ladder 48% of ciona's dark set was under
    100 aa against 16-18% everywhere else, and its fraction moved from 29% to 18% at
    >= 100 aa while no other species moved more than four points.
    """
    if df is None:
        omitted.add("dark fraction by minimum length",
                    "this run produced no length parquet, so the fraction could not be "
                    "recomputed over proteins of at least 50, 100 and 200 aa.")
        return
    value_col = pick_col(df, LENGTH_COLS)
    group_col = pick_col(df, GROUP_COLS)
    if value_col is None or group_col is None:
        omitted.add("dark fraction by minimum length",
                    f"the length parquet has no column named any of {LENGTH_COLS} or "
                    f"{GROUP_COLS}; found {list(df.columns)}.")
        return
    is_dark = (df[group_col].cast(pl.Utf8).str.to_lowercase().is_in(["dark", "true", "1"]))
    lengths = df[value_col].cast(pl.Float64)
    rows = {}
    for cut in MIN_AA:
        keep = lengths >= cut
        n = int(keep.sum())
        n_dark = int((keep & is_dark).sum())
        rows[f">= {cut} aa" if cut else "all proteins"] = {
            "proteins": n, "dark": n_dark,
            "fraction_dark": (n_dark / n) if n else None,
        }
    raw = rows["all proteins"]["fraction_dark"]
    at100 = rows[">= 100 aa"]["fraction_dark"]
    write_section(out, "dark_headline_by_length", {
        "id": "dark_headline_by_length",
        "section_name": "Dark fraction by minimum protein length",
        "description": (
            "<p>The headline fraction again, over proteins of at least 50, 100 and 200 "
            "residues.</p>"
            + bullets(
                f"<b>All proteins {pct(raw)}, at least 100 aa {pct(at100)}.</b> The gap "
                f"between the two is the part of the dark set that is short gene models, "
                f"which are dark because there is little to align, not because homology "
                f"detection failed.",
                "<b>Compare species at the same cut.</b> Annotations differ in how many "
                "short gene models they carry, so the raw fraction is not comparable "
                "across proteomes; the &ge; 100 aa one is closer to comparable.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "dark_headline_by_length_plot",
                    "title": f"{species}: dark fraction by minimum length",
                    "ylab": "fraction dark", "ymax": 1, "cpswitch": False,
                    "sort_samples": False},
        "categories": {"fraction_dark": {"name": "fraction dark", "color": C_DARK}},
        "data": {k: {"fraction_dark": v["fraction_dark"]} for k, v in rows.items()},
    })
    write_section(out, "dark_headline_by_length_table", {
        "id": "dark_headline_by_length_table",
        "section_name": "Dark fraction by minimum length, as numbers",
        "description": "<p>The panel above as counts.</p>",
        "plot_type": "table",
        "pconfig": {"id": "dark_headline_by_length_table_plot",
                    "title": f"{species}: dark fraction by minimum length",
                    "col1_header": "proteins kept", "sort_rows": False},
        "headers": {
            "proteins": {"title": "proteins", "format": "{:,.0f}"},
            "dark": {"title": "dark", "format": "{:,.0f}"},
            "fraction_dark": {"title": "fraction dark", "format": "{:,.4f}", "min": 0, "max": 1},
        },
        "data": rows,
    })


def section_length(out: Path, species: str, df, summary, omitted: Omitted) -> None:
    extras = []
    if summary:
        dark = pick(summary, "dark", default={})
        placed = pick(summary, "placed", default={})
        for cut in (50, 100):
            key = f"fraction_under_{cut}aa"
            d, q = pick(dark, key), pick(placed, key)
            # A proteome with no protein under the cut has nothing to say at that cut.
            if d is not None and q is not None and (float(d) or float(q)):
                extras.append(f"<b>Under {cut} aa</b>: {pct(d)} of dark against "
                              f"{pct(q)} of placed.")
        ratio = pick(summary, "median_ratio_dark_over_placed")
        if ratio is not None:
            extras.append(f"<b>Median length ratio dark/placed</b>: {float(ratio):.3f}.")
    extras.append(
        "<b>Length is a proxy and only a proxy.</b> A short protein is also harder for "
        "sequence search, so a shortness skew is consistent with spurious gene "
        "models but does not prove them. Separating the two needs the annotation evidence, "
        "not this panel.")
    covariate_section(
        out, species, "dark_length", "Protein length, dark vs placed",
        df, summary, LENGTH_COLS, LENGTH_EDGES, "protein length (aa)", "length (aa)", 0,
        "Is the dark set just short gene models? A new annotation carries a tail of "
        "fragments and mispredictions, and nothing places a spurious model into Swiss-Prot "
        "because there is nothing to place. If dark proteins skew sharply shorter, part of "
        "the headline is annotation noise rather than hard homology.",
        extras, omitted, "length", higher_word="longer", lower_word="shorter")


def section_disorder(out: Path, species: str, df, summary, omitted: Omitted) -> None:
    extras = [
        "<b>Disorder is not where this method gains ground.</b> On the QfO region "
        "benchmark's disorder axis the coarse-alphabet arms sit below the profile "
        "baselines in every disorder bin. No arm of any kind does well on the most "
        "disordered queries. So a dark set that is mostly disordered is a limit on what "
        "any of this can reach, not an opportunity.",
        "<b>metapredict scores a residue 0 to 1</b>; the value binned here is the mean "
        "over each protein, so a protein with one long disordered loop and a folded domain "
        "lands mid-scale rather than at either end.",
    ]
    covariate_section(
        out, species, "dark_disorder", "Predicted disorder, dark vs placed",
        df, summary, DISORDER_COLS, DISORDER_EDGES, "mean predicted disorder",
        "disorder", 3,
        "Is the dark set disordered? A protein with no folded core has little for a "
        "profile or a k-mer to hold on to, so a dark set concentrated at high disorder is "
        "a different explanation of the headline than divergence is.",
        extras, omitted, "disorder", higher_word="more disordered",
        lower_word="less disordered")


# --- one protein followed through -----------------------------------------------------

def protein_svg(length: int, region: tuple[int, int] | None, label: str) -> str:
    """The protein as a line, the kmerseek region as a box on it. One mark per meaning:
    the grey line is the protein, the green box is where the best region with the mask
    on lands, and the legend sits above the drawing."""
    w, pad = 640, 30
    scale = (w - 2 * pad) / max(length, 1)
    y = 56
    box = ""
    if region:
        a, b = region
        x0, x1 = pad + (a - 1) * scale, pad + b * scale
        box = (f"<rect x='{x0:.1f}' y='{y - 9}' width='{max(x1 - x0, 2):.1f}' height='18' "
               f"fill='{C_MASK_ON}' fill-opacity='0.85'/>"
               f"<text x='{x0:.1f}' y='{y + 26}' font-size='11' fill='currentColor'>{a}</text>"
               f"<text x='{x1:.1f}' y='{y + 26}' font-size='11' fill='currentColor' "
               f"text-anchor='end'>{b}</text>")
    legend = (f"<rect x='{pad}' y='8' width='22' height='4' fill='#7f7f7f'/>"
              f"<text x='{pad + 28}' y='13' font-size='11' fill='currentColor'>the protein, "
              f"{length} residues</text>"
              + (f"<rect x='{pad + 200}' y='4' width='14' height='12' fill='{C_MASK_ON}' "
                 f"fill-opacity='0.85'/><text x='{pad + 220}' y='13' font-size='11' "
                 f"fill='currentColor'>best kmerseek region with the mask on ({label})</text>"
                 if region else
                 f"<text x='{pad + 200}' y='13' font-size='11' fill='currentColor'>no "
                 f"kmerseek region with the mask on</text>"))
    return (f"<svg viewBox='0 0 {w} 100' width='100%' style='max-width:{w}px;display:block'>"
            f"{legend}<line x1='{pad}' y1='{y}' x2='{w - pad}' y2='{y}' stroke='#7f7f7f' "
            f"stroke-width='4'/><text x='{pad}' y='{y + 26}' font-size='11' "
            f"fill='currentColor'>{'' if region and region[0] <= 3 else 1}</text>"
            f"<text x='{w - pad}' y='{y + 26}' font-size='11' fill='currentColor' "
            f"text-anchor='end'>{'' if region and region[1] >= length - 3 else length}</text>"
            f"{box}</svg>")


def section_focus(out: Path, species: str, summary: dict, gain: dict | None,
                  length_df, disorder_df, run: dict) -> None:
    """One protein (BHF for Botryllus) followed through every panel: its call per arm
    with the best E-value, its length and disorder against the two distributions, and
    its kmerseek reach per setting with the best region drawn on the protein."""
    focus = pick(summary, "focus_proteins", default={}) or {}
    if not focus:
        return
    evalue = pick(summary, "evalue_call")
    gfocus = pick(gain, "focus_proteins", default={}) or {}
    blocks = []
    for acc, f in focus.items():
        what = f.get("what") or acc
        if not f.get("in_proteome", True):
            blocks.append(f"<h4>{what}</h4><p><code>{acc}</code> is not in this proteome.</p>")
            continue
        arms = f.get("per_arm") or {}
        rows = ""
        for arm, v in sorted(arms.items(),
                             key=lambda kv: ARM_ORDER.index(kv[0]) if kv[0] in ARM_ORDER else 9):
            e = v.get("best_evalue")
            rows += (f"<tr><td>{arm}</td><td>{evalue_txt(e) if e is not None else 'no hit'}</td>"
                     f"<td>{'placed' if v.get('placed') else 'not placed'}</td></tr>")
        call = "dark" if not f.get("placed") else "placed"
        per_arm = (f"<p><b>{call.capitalize()}.</b> Each arm's best E-value against the "
                   f"protein; placed means at or below the call cutoff, E &le; "
                   f"{evalue_txt(evalue)}.</p>"
                   f"<table class='table table-sm' style='max-width:420px;font-size:12px'>"
                   f"<thead><tr><th>arm</th><th>best E-value</th><th>call</th></tr></thead>"
                   f"<tbody>{rows}</tbody></table>")
        length = None
        covar = []
        if length_df is not None:
            vc, gc = pick_col(length_df, LENGTH_COLS), pick_col(length_df, GROUP_COLS)
            if vc:
                hit = length_df.filter(pl.col("accession") == acc) if "accession" in length_df.columns else length_df.head(0)
                if hit.height:
                    length = int(hit[vc][0])
                    lo = int((length_df[vc] <= length).sum())
                    covar.append(f"<b>Length</b> {length} residues, longer than "
                                 f"{pct(lo / length_df.height)} of the proteome, in the "
                                 f"<code>{bin_label(length, LENGTH_EDGES)}</code> bin of "
                                 f"the length panel.")
        if disorder_df is not None:
            vc = pick_col(disorder_df, DISORDER_COLS)
            if vc and "accession" in disorder_df.columns:
                hit = disorder_df.filter(pl.col("accession") == acc)
                if hit.height and hit[vc][0] is not None:
                    d = float(hit[vc][0])
                    lo = int((disorder_df[vc] <= d).sum())
                    covar.append(f"<b>Mean predicted disorder</b> {d:.3f}, more disordered "
                                 f"than {pct(lo / disorder_df.height)} of the proteome, in "
                                 f"the <code>{bin_label(d, DISORDER_EDGES)}</code> bin of "
                                 f"the disorder panel.")
        km = ""
        g = gfocus.get(acc)
        if g:
            per = g.get("per_combo") or {}
            trs = ""
            for _k, v in sorted(per.items(), key=lambda kv: (kv[1]["alphabet"], kv[1]["ksize"],
                                                              not kv[1]["low_complexity_mask"])):
                score = v.get("best_region_score")
                region = v.get("best_region")
                sh = v.get("shuffled_reached")
                trs += ("<tr><td>{a} k{k}</td><td>{m}</td><td>{r}</td><td>{s}</td><td>{g}</td>"
                        "<td>{h}</td></tr>").format(
                    a=v["alphabet"], k=v["ksize"],
                    m="on" if v.get("low_complexity_mask") else "off",
                    r="yes" if v.get("reached") else "no",
                    s="-" if score is None else f"{score:.2f}",
                    g="-" if not region else f"{region[0]}-{region[1]}",
                    h="-" if sh is None else ("yes" if sh else "no"))
            best_on = [(k, v) for k, v in per.items() if v.get("reached") and v.get("low_complexity_mask")]
            pic = ""
            if length:
                if best_on:
                    k, v = max(best_on, key=lambda kv: kv[1].get("best_region_score") or 0)
                    pic = protein_svg(length, tuple(v["best_region"]) if v.get("best_region") else None,
                                      k.rsplit(" lc", 1)[0])
                else:
                    pic = protein_svg(length, None, "")
            km = (f"<p><b>kmerseek, per setting.</b> Reached means at least one region at "
                  f"the run's cutoff (score &ge; {pick(run, 'min_region_score', default=1.3)}); "
                  f"the best region is the highest-scoring one. The shuffled column is the "
                  f"same search on this protein with its residues shuffled, where the run "
                  f"has that control.</p>"
                  f"<table class='table table-sm' style='max-width:640px;font-size:12px'>"
                  f"<thead><tr><th>setting</th><th>mask</th><th>reached</th>"
                  f"<th>best region score</th><th>best region (residues)</th>"
                  f"<th>shuffled copy reached</th></tr></thead><tbody>{trs}</tbody></table>"
                  + pic)
        blocks.append(f"<h4>{what}</h4><p><code>{acc}</code></p>" + per_arm
                      + (bullets(*covar) if covar else "") + km)
    write_section(out, "dark_focus", {
        "id": "dark_focus",
        "section_name": "One protein followed through",
        "description": (
            f"<p>The same numbers as every panel above, for one protein at a time: its "
            f"call per arm, where it sits on the length and disorder distributions, and "
            f"what each kmerseek setting reports on it. The proteins are named in the "
            f"species registry.</p>"),
        "plot_type": "html",
        "data": "".join(blocks),
    })


def bin_label(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}"
    return f"{edges[-1]:g}+" if value >= edges[-1] else f"<{edges[0]:g}"


# --- what the run did not produce ------------------------------------------------------

def section_omitted(out: Path, omitted: Omitted) -> None:
    if not omitted.items:
        return
    rows = "".join(f"<li><b>{what}</b> — {why}</li>" for what, why in omitted.items)
    write_section(out, "dark_missing", {
        "id": "dark_missing",
        "section_name": "Not in this report",
        "description": (
            "<p>Panels this run could not build, and why. They are listed rather than "
            "drawn empty: an empty panel reads as a measured null, which is a different "
            "claim from not having looked.</p>"),
        "plot_type": "html",
        "data": f"<ul>{rows}</ul>",
    })


# --- input resolution -------------------------------------------------------------------

# Suffix -> the argument it fills, for --extra-dir. Files arrive from Nextflow staged into
# one directory with their published names, so the suffix is what identifies them; nothing
# here depends on the species prefix, which differs per run.
EXTRA_SUFFIXES = {
    "gain_json": "_kmerseek_dark_gain.json",
    "length_summary": "_length_summary.json",
    "length_parquet": "_length_comparison.parquet",
    "disorder_summary": "_disorder_summary.json",
    "disorder_parquet": "_disorder.parquet",
}


def resolve_extras(args, extra_dir: Path | None) -> dict[str, Path | None]:
    """Explicit flags win; anything not given is looked for by suffix in --extra-dir."""
    resolved = {key: getattr(args, key) for key in EXTRA_SUFFIXES}
    if extra_dir is None or not extra_dir.is_dir():
        return resolved
    for key, suffix in EXTRA_SUFFIXES.items():
        if resolved[key] is not None:
            continue
        found = sorted(p for p in extra_dir.iterdir() if p.name.endswith(suffix))
        if found:
            resolved[key] = found[0]
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--dark-summary", type=Path, required=True,
                    help="<species>_dark_summary.json from computeDarkSet")
    ap.add_argument("--clade", default=None,
                    help="the clade removed from the reference, for the overview")
    ap.add_argument("--reference-summary", type=Path, default=None,
                    help="minus_<clade>/summary.json from buildReference: how many "
                         "Swiss-Prot entries were kept and removed")
    ap.add_argument("--run-params", type=Path, default=None,
                    help="JSON of the search settings the run used (chunk size, report "
                         "E-value, iterations, kmerseek thresholds), for the overview")
    ap.add_argument("--gain-json", type=Path, default=None,
                    help="optional <species>_kmerseek_dark_gain.json")
    ap.add_argument("--length-summary", type=Path, default=None)
    ap.add_argument("--length-parquet", type=Path, default=None)
    ap.add_argument("--disorder-summary", type=Path, default=None)
    ap.add_argument("--disorder-parquet", type=Path, default=None)
    ap.add_argument("--extra-dir", type=Path, default=None,
                    help="directory of optional products, matched by published suffix. "
                         "Anything named by an explicit flag is not looked for here.")
    ap.add_argument("--registry", type=Path, default=None,
                    help="species_metadata.json: the species' name for the title and the "
                         "prose, and its focus proteins")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    global SPECIES_NAME
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    reg_row = registry_row(args.registry, args.species)
    SPECIES_NAME = reg_row.get("name") or None

    summary = load_json(args.dark_summary)
    if summary is None:
        raise SystemExit(
            f"no readable dark summary at {args.dark_summary}.\n"
            f"That file is the report, not an optional input: without it there is no "
            f"proteome size, no dark count and nothing for the other sections to be read "
            f"against. Run -entry darkSet first."
        )

    extras = resolve_extras(args, args.extra_dir)
    omitted = Omitted()

    gain = load_json(extras["gain_json"])
    length_df = load_parquet(extras["length_parquet"])
    disorder_df = load_parquet(extras["disorder_parquet"])

    run_params = load_json(args.run_params) or {}
    section_overview(out, args.species, summary, load_json(args.reference_summary),
                     args.clade, run_params, gain,
                     length_df, disorder_df,
                     load_json(extras["length_summary"]), load_json(extras["disorder_summary"]),
                     registry=reg_row)
    section_metric_explainers(out, args.species)
    section_headline(out, args.species, summary)
    section_per_arm(out, args.species, summary, omitted)
    section_kmerseek(out, args.species, gain, omitted)
    section_dark_by_length(out, args.species, length_df, omitted)
    section_length(out, args.species, length_df,
                   load_json(extras["length_summary"]), omitted)
    section_disorder(out, args.species, disorder_df,
                     load_json(extras["disorder_summary"]), omitted)
    section_focus(out, args.species, summary, gain, length_df, disorder_df, run_params)
    section_omitted(out, omitted)

    written = sorted(p.name for p in out.glob("*_mqc.json"))
    print(f"{args.species}: wrote {len(written)} section files to {out}")
    for name in written:
        print(f"  {name}")
    if omitted.items:
        print("\nomitted, and said so in the report:")
        for what, why in omitted.items:
            print(f"  {what}: {why[:120]}")


if __name__ == "__main__":
    main()
