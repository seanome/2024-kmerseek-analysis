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

PARENT_ID = "invertebrate_dark_set"
PARENT_NAME = "Invertebrate dark set"

# One colour per idea, reused across sections so a thing keeps its identity.
C_DARK = "#0f9d76"       # the dark set, and the mask-ON arm: what survives scrutiny
C_PLACED = "#7f7f7f"     # placed by a sequence arm: the part that is not the question
C_MASK_OFF = "#c9528f"   # mask OFF: the number that may be composition rather than homology
C_SINGLE = "#2b7bba"     # single-pass search
C_ITER = "#c99a00"       # iterative search
C_UNION = "#0f9d76"      # any arm at all

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
    cfg.setdefault("parent_id", PARENT_ID)
    cfg.setdefault("parent_name", PARENT_NAME)
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
            f"<p>Every protein in <b>{species}</b> that phmmer, jackhmmer and mmseqs2 "
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
                "<b>The dark set is defined against the strongest arm, not the cheapest.</b> "
                "A protein iterative search reaches is not dark, and counting it as dark "
                "would inflate the headline in the direction the claim wants.")),
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

    data = {}
    table = {}
    for row in pairs:
        label = f"{row['alphabet']} k{row['ksize']}"
        on = row.get("dark_reached_mask_on")
        off = row.get("dark_reached_mask_off")
        lost = row.get("lost_to_masking")
        if lost is None and on is not None and off is not None:
            lost = off - on
        data[label] = {"mask_on": on, "mask_off": off}
        table[label] = {
            "mask_on": on,
            "mask_off": off,
            "lost_to_masking": lost,
            "fraction_dark_mask_on": (on / dark_n) if dark_n else None,
            "fraction_dark_mask_off": (off / dark_n) if dark_n else None,
        }

    best = max(pairs, key=lambda r: r.get("dark_reached_mask_on") or 0)
    best_on = best.get("dark_reached_mask_on")
    worst_loss = max(pairs, key=lambda r: r.get("lost_to_masking") or 0)

    write_section(out, "dark_kmerseek_mask", {
        "id": "dark_kmerseek_mask",
        "section_name": "kmerseek reach inside the dark set",
        "description": (
            f"<p>Dark proteins kmerseek put a region on, out of {num(dark_n)}, with the "
            f"low-complexity mask ON and OFF side by side. The two bars in a group are the "
            f"same search with the filter switched.</p>"
            + bullets(
                "<b>Read the pair, never one bar.</b> A rescue present with the mask off "
                "and gone with it on is a property of amino-acid composition, not of "
                "homology. BHF's flagship matches included polar-biased low-complexity "
                "segments, which is why this pipeline runs the mask as a paired setting "
                "rather than as a sweep dimension.",
                f"<b>Best arm with the mask ON: <code>{best['alphabet']} "
                f"k{best['ksize']}</code></b>, reaching {num(best_on)} of {num(dark_n)} "
                f"dark proteins ({pct((best_on / dark_n) if dark_n else None)}).",
                f"<b>Largest loss to masking: <code>{worst_loss['alphabet']} "
                f"k{worst_loss['ksize']}</code></b> at "
                f"{num(worst_loss.get('lost_to_masking'))} proteins, which is the part of "
                f"that arm's mask-off number that the filter does not support.",
                unpaired_note,
                "<b>This is reach, not accuracy.</b> A region on a dark protein says "
                "kmerseek found something there, not that the family label is right. That "
                "needs the structural key, which does not exist for this species yet: the "
                "number here is necessary for the claim and nowhere near sufficient.")),
        "plot_type": "bargraph",
        "pconfig": {"id": "dark_kmerseek_mask_plot",
                    "title": f"{species}: dark proteins reached, mask ON vs OFF",
                    "ylab": "dark proteins reached",
                    # Grouped, never stacked. The two bars are the same proteins counted
                    # twice under different settings, so stacking them would draw a total
                    # that does not exist.
                    "cpswitch": False, "stacking": "group", "sort_samples": False},
        "categories": {
            "mask_on": {"name": "low-complexity mask ON", "color": C_DARK},
            "mask_off": {"name": "mask OFF", "color": C_MASK_OFF},
        },
        "data": data,
    })

    write_section(out, "dark_kmerseek_mask_table", {
        "id": "dark_kmerseek_mask_table",
        "section_name": "Mask pair, as numbers",
        "description": (
            "<p>The panel above as numbers, with what masking costs each arm.</p>"
            + bullets(
                "<b>lost_to_masking</b> is mask-off minus mask-on: dark proteins reached "
                "only when the low-complexity filter is off.",
                "<b>The fractions are of the dark set</b>, not of the proteome.")),
        "plot_type": "table",
        "pconfig": {"id": "dark_kmerseek_mask_table_plot",
                    "title": f"{species}: dark reach by alphabet and k",
                    "col1_header": "alphabet, ksize", "sort_rows": False},
        "headers": {
            "mask_on": {"title": "reached, mask ON", "format": "{:,.0f}"},
            "mask_off": {"title": "reached, mask OFF", "format": "{:,.0f}"},
            "lost_to_masking": {"title": "lost to masking", "format": "{:,.0f}"},
            "fraction_dark_mask_on": {"title": "fraction of dark, mask ON",
                                      "format": "{:,.4f}", "min": 0, "max": 1},
            "fraction_dark_mask_off": {"title": "fraction of dark, mask OFF",
                                       "format": "{:,.4f}", "min": 0, "max": 1},
        },
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


def mw_bullet(summary: dict | None) -> str:
    """The Mann-Whitney line, effect size first."""
    if not summary:
        return ""
    mw = pick(summary, "mann_whitney_u", "mann_whitney", "mannwhitneyu", "mannwhitney",
              default=None)
    if not isinstance(mw, dict):
        mw = summary
    p = pick(mw, "p_value", "pvalue", "p")
    cles = pick(mw, "common_language_effect_size", "cles", "effect_size")
    rbc = pick(mw, "rank_biserial_correlation", "rank_biserial")
    if p is None and cles is None:
        return ""
    parts = []
    if cles is not None:
        direction = ("lower" if float(cles) < 0.5
                     else "higher" if float(cles) > 0.5 else "the same as")
        parts.append(
            f"a randomly drawn dark protein is {direction} than a randomly drawn placed "
            f"one with probability {float(cles):.3f} (0.5 is no difference)")
    if rbc is not None:
        parts.append(f"rank-biserial {float(rbc):+.3f}")
    if p is not None:
        parts.append(f"Mann-Whitney p={float(p):.3g}")
    return ("<b>Read the effect size, not the p-value</b> — " + ", ".join(parts)
            + ". At tens of thousands of proteins the test is significant at effect sizes "
              "far too small to matter.")


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
                      omitted: Omitted, what: str) -> None:
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
                mw_bullet(summary),
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
                "short models they carry (ciona: 48% of its dark set under 100 aa; mouse, "
                "worm, Botryllus: 16-18%), so the raw fraction is not comparable across "
                "proteomes and the >= 100 aa one is closer to it.")),
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
            if pick(dark, key) is not None and pick(placed, key) is not None:
                extras.append(f"<b>Under {cut} aa</b>: {pct(pick(dark, key))} of dark "
                              f"against {pct(pick(placed, key))} of placed.")
        ratio = pick(summary, "median_ratio_dark_over_placed")
        if ratio is not None:
            extras.append(f"<b>Median length ratio dark/placed</b>: {float(ratio):.3f}.")
    extras.append(
        "<b>Length is a proxy and only a proxy.</b> A short protein is also genuinely "
        "harder for sequence search, so a shortness skew is consistent with spurious gene "
        "models but does not prove them. Separating the two needs the annotation evidence, "
        "not this panel.")
    covariate_section(
        out, species, "dark_length", "Protein length, dark vs placed",
        df, summary, LENGTH_COLS, LENGTH_EDGES, "protein length (aa)", "length (aa)", 0,
        "Is the dark set just short gene models? A new annotation carries a tail of "
        "fragments and mispredictions, and nothing places a spurious model into Swiss-Prot "
        "because there is nothing to place. If dark proteins skew sharply shorter, part of "
        "the headline is annotation noise rather than hard homology.",
        extras, omitted, "length")


def section_disorder(out: Path, species: str, df, summary, omitted: Omitted) -> None:
    extras = [
        "<b>Disorder is where this method does not win.</b> On the benchmark's disorder "
        "axes the coarse-alphabet arms fall off faster than the profile and structure "
        "baselines do, so a dark set that is mostly disordered is a limit on what any of "
        "this can reach, not an opportunity.",
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
        extras, omitted, "disorder")


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
    ap.add_argument("--gain-json", type=Path, default=None,
                    help="optional <species>_kmerseek_dark_gain.json")
    ap.add_argument("--length-summary", type=Path, default=None)
    ap.add_argument("--length-parquet", type=Path, default=None)
    ap.add_argument("--disorder-summary", type=Path, default=None)
    ap.add_argument("--disorder-parquet", type=Path, default=None)
    ap.add_argument("--extra-dir", type=Path, default=None,
                    help="directory of optional products, matched by published suffix. "
                         "Anything named by an explicit flag is not looked for here.")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

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

    section_headline(out, args.species, summary)
    section_per_arm(out, args.species, summary, omitted)
    section_kmerseek(out, args.species, load_json(extras["gain_json"]), omitted)
    length_df = load_parquet(extras["length_parquet"])
    section_dark_by_length(out, args.species, length_df, omitted)
    section_length(out, args.species, length_df,
                   load_json(extras["length_summary"]), omitted)
    section_disorder(out, args.species,
                     load_parquet(extras["disorder_parquet"]),
                     load_json(extras["disorder_summary"]), omitted)
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
