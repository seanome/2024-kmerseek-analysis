#!/usr/bin/env python3
"""Generate notebooks/356_exon_landing_midi_plus_chicken_zebrafish.ipynb.

The notebook reads the scorer's tables (scripts/score_356_exon_landing.py) and draws the
pre-registered figures. --stage pilot reads tables/356/pilot (Stop 0, 200 pairs); --stage
full reads tables/356 (Stop 1). The results text is in --results-md, written after the
numbers are seen, so the builder never states a result it did not read.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"md-{len(cells):02d}",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        }
    )


def code(source):
    cells.append(
        {
            "cell_type": "code",
            "id": f"code-{len(cells):02d}",
            "execution_count": None,
            "metadata": {"jupyter": {"source_hidden": True}},
            "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True),
        }
    )


def build(stage: str, results_md: str | None) -> dict:
    tables = "../tables/356/pilot" if stage == "pilot" else "../tables/356"
    figs = "../figures/356/pilot" if stage == "pilot" else "../figures/356"
    md(r"""
# 356. Does a search call land on one exon's residues? (midi-plus, chicken and zebrafish)

**Question.** A human query protein is searched against chicken or zebrafish. When a tool
calls a region of the target protein, does that call sit on the residues one exon codes,
or does it spread over the whole protein? Notebook
[244](https://github.com/seanome/2024-kmerseek-analysis/pull/49) asked this with short
Swiss-Prot features as the truth: kmerseek's call landed on 176 of 182 features, Foldseek
on 2 and ProstT5 on 3. Here the truth is the target protein's exons.

**What decision this informs.** If kmerseek lands on short exons and structure tools do
not, the paper can say that kmerseek places a match on a single exon, which structure
search cannot. If it fails, the exon claim stays out of the paper, and nb 244's result is
about Swiss-Prot features only.

**Words.** A call *lands* on an exon when at least 80% of the call is inside the exon and
the call covers at least 30% of the exon (nb 244's rule). It *spills* when less than half
of the call is inside the exon. *Partial*: it overlaps the exon with at least half of
itself inside, but does not land. *No call*: no call between the two proteins touches the
exon. IoU is the overlap of the call and the exon divided by their union.

**Decision rule, written before the run**
([preregistered.md](356_data/preregistered.md), with its 2026-10-20 amendments). On
internal coding exons of 60 residues or fewer, on pairs where both proteins have an
AlphaFold model:

1. kmerseek (best of all settings) lands on at least 50% of them;
2. every structure tool (Foldseek, Reseek, ProstT5) lands on under 15%;
3. kmerseek's landing rate, counting only calls whose placement p-value is below 0.05,
   is at least twice the Kyte-Doolittle scan's.

Pass needs all three. Stop 0 also stops the work if kmerseek lands on under 10% of short
exons, or if the Kyte-Doolittle scan lands as often as kmerseek (then the result is about
amino-acid composition, not search).

**Data.** The midi-plus run (1-8 Sept 2026): 998 human queries (chromosome 6, plus B2M,
CD1A-E, MR1, KIR and LILR) searched against the nine QfO 2020_04 proteomes, image
`olgabot/kmerseek:2026-08-24-reduced-alphabets`. Query = human; target database = the
chicken (UP000000539) or zebrafish (UP000000437) proteome. Only calls between a query and
its 1:1 OMA ortholog are scored. kmerseek calls are kept at Bonferroni p < 0.05, as in
nb 244. Exons come from UniProt's exon map for the same protein sequences (the QfO
release's `_gcoord.xml`).
""")
    code(f"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path.cwd()))
import exon_landing_356 as el

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
STAGE = {stage!r}
TABLES = Path({tables!r})
FIG = Path({figs!r})
d = el.load(TABLES)
v = d["verdict"]
inputs = json.loads(Path("356_data/exon_inputs.json").read_text())
et = d["exon_tool"]
one_tool = et.filter(pl.col("tool") == "kmerseek_best_of_all")
print(f"stage: {{v['stage']}}; pairs scored: {{v['n_pairs']}}")
print(one_tool.group_by("species").agg(
    pairs=pl.struct("query_acc", "target_acc").n_unique(), exons=pl.len(),
    internal=pl.col("internal").sum(), internal_le60=pl.col("short_internal").sum(),
    both_models=pl.col("both_models").sum(), verdict_exons=pl.col("verdict_set").sum()))
""")
    md(r"""
## 1. Verdict against the pre-registered rules

Each row is one rule with the count behind it. Higher landing is better for kmerseek; for
structure tools the rule asks for low landing, because nb 244 found that they cover the
whole protein.
""")
    code(r"""
rows = [
    ("1: kmerseek best of all lands >= 50%", *v["rule_1"]["kmerseek_best_of_all_lands"], v["rule_1"]["pass"]),
    *[(f"2: {t} lands < 15%", *x, x[2] is not None and x[2] < 0.15) for t, x in v["rule_2"]["structure_tools_land"].items()],
    ("3: kmerseek lands with p_place < 0.05", *v["rule_3"]["kmerseek_lands_with_p_place_lt_0.05"], None),
    ("3: Kyte-Doolittle scan lands", *v["rule_3"]["kd_scan_lands"], None),
]
verdict_tab = pl.DataFrame(rows, schema=["rule", "n_land", "n_exons", "rate", "pass"], orient="row")
print(verdict_tab.with_columns(pl.col("rate").round(3)))
print(f"rule 3 ratio (kmerseek / scan): {v['rule_3']['ratio']}; pass: {v['rule_3']['pass']}")
print(f"random-protein null of kmerseek's best of all: median {v['random_protein_null']['median']:.3f}, "
      f"99.9th percentile {v['random_protein_null']['quantile_0.999']:.3f}; "
      f"observed above it: {v['random_protein_null']['observed_above_0.999']}")
print(f"Stop 0 empty question (kmerseek < 10%): {v['empty_question_kmerseek_below_0.10']}")
print(f"Stop 0 kill condition (scan at kmerseek's rate): {v['kill_condition_kd_at_kmerseek_rate']}")
print(f"OVERALL PASS: {v['pass']}")
vfig = el.verdict_figure(d, FIG / "356_verdict_rules_midi_plus_chicken_zebrafish")
""")
    md(r"""
## 2. Landing rate by exon length (Figure 1)

How to read it: each line is one tool; a point is the share of exons in that length bin
that some call of the tool lands on. The grey band is the exons the rules use. A tool
that places calls on single exons sits high on the left; a tool whose calls cover whole
proteins sits near zero everywhere except, perhaps, the longest exons.
""")
    code(r"""
tools_line = ("TOOLS: kmerseek (all alphabets, k and masks in midi-plus; best of all, and best single setting), "
              "Foldseek, Reseek, ProstT5, phmmer, MMseqs2, MMseqs2 iterative, jackhmmer, HHblits; "
              "Kyte-Doolittle scan (window 19, threshold 1.6). Query: human; target: chicken or zebrafish 1:1 ortholog.")
fig1 = el.landing_by_length(d, FIG / "356_fig1_landing_by_exon_length_midi_plus_chicken_zebrafish", tools_line)
print(fig1.pivot(on="length_bin", index="tool", values="rate").select("tool", *[b for b in el.BIN_ORDER if b in fig1["length_bin"].unique()]))
""")
    md(r"""
## 3. What each tool's call does on the short exons (SF2)

For each tool, the verdict exons split by outcome: lands, partial, spills, no call.
""")
    code(r"""
sf2 = el.outcome_shares(d, FIG / "356_sf2_outcome_shares_short_internal_exons_midi_plus")
print(sf2)
""")
    md(r"""
## 4. How tight the landing calls are (SF1)

Median IoU of the landing call, by exon-length bin, over the exons where the tool lands.
Higher is a tighter fit to the exon.
""")
    code(r"""
sf1 = el.iou_by_length(d, FIG / "356_sf1_median_iou_by_exon_length_midi_plus")
print(sf1.with_columns(pl.col("median_iou").round(3)))
""")
    md(r"""
## 5. Ordered against disordered exons (SF3)

Exons split by the mean AlphaFold pLDDT of their residues on the target (ordered >= 70,
disordered < 70). Structure tools are expected to do worse on disordered exons.
""")
    code(r"""
sf3 = el.landing_by_disorder(d, FIG / "356_sf3_landing_by_exon_disorder_midi_plus")
print(sf3.with_columns(pl.col("rate").round(3)))
""")
    md(r"""
## 6. kmerseek's best of all settings against chance

(a) Every kmerseek setting's landing rate on the verdict exons, with the best of all
settings as a line. (b) The same best-of-all rule scored on decoys: each exon's interval
moved, at the same coordinates, onto another pair's target protein. If the real rate sits
inside the null, landing on an exon is something kmerseek's short matches do anywhere.
""")
    code(r"""
null_tab = el.settings_and_null(d, FIG / "356_kmerseek_settings_and_random_protein_null_midi_plus")
print(null_tab.with_columns(pl.col(pl.Float64).round(3)))
print(d["settings"].head(10))
""")
    if stage == "full":
        md(r"""
## 7. Are the landing calls at the aligned position? (added after Stop 1)

Not in the pre-registration. For every verdict exon kmerseek lands on, each landing call
is checked against a global alignment of the human query and its ortholog (Biopython,
BLOSUM62, gap open -11, extend -1). A call is *on position* when at least half of its
residue pairs are pairs the global alignment also makes. This catches a call that lands
on the exon but pairs it with the wrong stretch of the human protein, as in a repeat.
""")
        code(r"""
onpos = pl.read_parquet(TABLES / "landing_on_position.parquet")
hero = json.loads((TABLES / "hero.json").read_text())
print(f"verdict exons kmerseek lands on: {hero['n_verdict_exons_landed']}")
print(f"  with at least one landing call on position: {hero['n_landed_with_a_call_on_position']}")
print(el.on_position_figure(onpos, FIG / "356_landing_on_position_global_alignment_midi_plus"))
""")
        md(r"""
## 8. Hero exon

Rule, written before looking: a verdict exon that is disordered (mean pLDDT < 70), where
kmerseek's best of all lands with p_place < 0.05 and Foldseek, Reseek, ProstT5, phmmer,
MMseqs2 and MMseqs2 iterative all spill; highest landing IoU, then smallest p_place.
Its first pick is shown with both kmerseek calls that land on it: the one with IoU 1.0
pairs the exon with the wrong copy of the collagen repeat, and a second setting pairs it
with the aligned residues. (a) the whole chicken protein, every tool's call on the pair;
(b) the same around the exon; (c) the residues of both kmerseek calls, classes underneath.
""")
        code(r"""
exons = pl.read_parquet("356_data/exons.parquet")
el.hero_figure(hero, et, exons, FIG / "356_hero_col9a1_chicken_exon14_midi_plus")
print(pl.DataFrame([{k: v for k, v in hero[c].items() if not k.endswith("residues")}
                    for c in ("hero_on_position", "rule_first_pick")]))
""")
    md(
        results_md
        or "## Summary\n\nNot written yet: the numbers above have not been read."
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--results-md", type=Path, default=None)
    args = ap.parse_args()
    nb = build(args.stage, args.results_md.read_text() if args.results_md else None)
    out = REPO / "notebooks" / "356_exon_landing_midi_plus_chicken_zebrafish.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
