#!/usr/bin/env python3
"""Generate notebooks/359_isoform_blank_exons_WTC11_kmerseek_vs_IsoAnnotLite.ipynb.

The notebook reads the Sherlock run's tables from notebooks/359_data/run/ (score.sbatch
gathers them in for_repo/) and draws SF-A, SF-B and SF-C. Before the run the folder does
not exist and every section says so; nothing is drawn from data that is not there. The
results text is passed in --results-md, written after the numbers are seen, so the
builder never states a result it did not read.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "359_isoform_blank_exons_WTC11_kmerseek_vs_IsoAnnotLite.ipynb"
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})  # fmt: skip


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})  # fmt: skip


def build(results_md: str | None) -> dict:
    md(r"""
# 359. kmerseek on alternative exons that IsoAnnotLite leaves blank (WTC11 long reads, Swiss-Prot regions index)

**Question.** Long-read sequencing finds alternative isoforms. When an isoform holds an
alternative exon in a stretch of protein with no Pfam domain, IsoAnnotLite (the annotation
step of SQANTI3 and tappAS) copies nothing onto it, because it copies features from a
precomputed database by genomic position. Does kmerseek land a named Swiss-Prot region on
those exons more often than on the same exons with their residues shuffled, where a
hydropathy scan does not?

**What decision this informs.** PASS: Seanome has a tool that says something where
SQANTI3 + IsoAnnotLite + tappAS say nothing, and the next step is the same test on
*Botryllus*. KILL: the idea stops here, and this notebook records why.

**Data.** WTC11 PacBio cDNA transcripts from ENCODE experiment
[ENCSR507JOF](https://www.encodeproject.org/experiments/ENCSR507JOF/) (the LRGASP WTC11
set), run through SQANTI3 v6.0.1 with IsoAnnotLite and tappAS's GENCODE v39 human GFF3;
InterProScan 5.78-109.0 (Pfam only). Queries: the isoforms holding a blank exon, and one
decoy copy per blank exon. Target database: the Swiss-Prot regions index (Swiss-Prot
2026_03 with every mammal removed, annotated regions with 21-residue flanks, clustered;
[PR 114](https://github.com/seanome/2024-kmerseek-analysis/pull/114)). kmerseek 0.4.1-rc.1,
six H/P settings: hp_lehninger2 k17 and k25, hp_lehninger_c_nonpolar2 k19,
hp_pbotc_1st_ed2 k19, k21 and k23.

**Words.** An *alternative exon* here is a cassette exon: one isoform of the gene has it,
and another isoform that spans it skips it. A *blank* exon has no IsoAnnotLite feature
and no InterProScan Pfam hit on its protein residues. A kmerseek call *lands* on an exon
when it covers at least half of the exon's protein residues. The *decoy* is the same
isoform with only that exon's residues shuffled.

**Rule, fixed before the run** ([preregistered.md](359_data/preregistered.md)). At the
region E-value where kmerseek lands on at most 1% of the blank set in decoys, PASS needs
all three: (1) real landings minus decoy landings are at least 10% of the blank set;
(2) at least half of the landings hit a named Swiss-Prot feature; (3) the Kyte-Doolittle
scan, also at 1% decoys, picks fewer than half of the same exons. KILL otherwise. Before
that, a sanity gate: InterProScan finds again at least 90% of IsoAnnotLite's Pfam features.

```mermaid
flowchart LR
  A["WTC11 transcripts"] --> B["SQANTI3 + IsoAnnotLite"]
  B --> C["InterProScan Pfam"]
  B --> D["exons to protein residues"]
  C --> E["blank cassette exons"]
  D --> E
  E --> F["decoy: exon shuffled"]
  E --> K["kmerseek x 6 settings"]
  F --> K
  E --> S["Kyte-Doolittle scan"]
  F --> S
  K --> V["1% decoy threshold, rules 1-3"]
  S --> V
```
""")
    code(r"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, ".")
import isoform_blank_exons_359 as ib

RUN = Path("359_data/run")
FIGS = Path("../figures/359")
FIGS.mkdir(parents=True, exist_ok=True)
RAN = (RUN / "verdict.json").exists()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
print("Run tables present:" if RAN else "Not run yet: notebooks/359_data/run/ holds no verdict.json.", RUN.resolve())
""")
    md(r"""
## 1. Inputs and the blank set

Every input file with its md5 and download date, and the count at each step from
SQANTI3's isoforms to the blank set.
""")
    code(r"""
if RAN:
    print(pl.read_csv(RUN / "manifest.tsv", separator="\t"))
    print((RUN / "header.txt").read_text())
    summary = json.loads((RUN / "build_summary.json").read_text())
    for k, v in summary.items():
        print(f"{k}: {v}")
else:
    print("Not run yet.")
""")
    md(r"""
## 2. Sanity gate (SF-A)

How to read it: one bar, the share of IsoAnnotLite's Pfam features (on coding isoforms
that passed the SQANTI3 filter) that InterProScan finds again, covering at least half of
the feature, on the same isoform. At or above the dashed 90% line, the coordinates from
SQANTI3, IsoAnnotLite and InterProScan agree and the rest can be scored. Below it, the run
stopped.
""")
    code(r"""
if RAN:
    gate = json.loads((RUN / "gate.json").read_text())
    print(pl.DataFrame([gate]))
    ib.fig_gate(gate, str(FIGS / "359_sfa_sanity_gate_interproscan_vs_isoannotlite_pfam_WTC11"))
else:
    print("Not run yet.")
""")
    md(r"""
## 3. Landing rates on the blank set (SF-B)

How to read it: for each kmerseek setting and the union of the six, the share of blank
exons landed on, at that setting's 1% decoy threshold. Blue is kmerseek on the real
exons, grey is kmerseek on the same exons shuffled, orange is the Kyte-Doolittle scan on
the real exons (one scan, so the same orange bar in each group). "net" is blue minus
grey; rule 1 needs net at or above the dashed 10% line. Panel b splits the union by
whether the exon is in a GENCODE basic isoform with a CCDS tag, a stand-in for evidence
that the isoform is made into protein. The table under the figure has the counts behind
every bar and the three rules for each row.
""")
    code(r"""
if RAN:
    summary_tbl = pl.read_csv(RUN / "summary.csv")
    ib.fig_landing(summary_tbl, str(FIGS / "359_sfb_landing_rate_real_decoy_scan_by_setting_WTC11_swissprot_regions"))
    print(summary_tbl.select(
        "setting", "stratum", "n_blank", "real_landings", "decoy_landings", "net_landings", "net_fraction",
        "named", "named_fraction", "scan_landings", "scan_overlap", "scan_overlap_fraction", "passes",
    ))
else:
    print("Not run yet.")
""")
    md(r"""
## 4. Every surviving landing, residues shown (SF-C)

Every blank exon the union lands on, with the isoform's residues over the Swiss-Prot
region's residues and a bar where they are the same. The figure shows five picked at
random (seed 359), not the five that look best.
""")
    code(r"""
if RAN:
    survivors = pl.read_parquet(RUN / "survivors.parquet")
    picks = json.loads((RUN / "examples.json").read_text())["exon_ids"]
    ib.fig_examples(survivors, picks, str(FIGS / "359_sfc_five_random_landings_residues_WTC11_swissprot_regions"))
    for r in survivors.iter_rows(named=True):
        print(f"{r['exon_id']} {r['gene']} {r['rep_isoform']} exon residues {r['exon_p_start'] + 1}-{r['exon_p_end']}; "
              f"{r['setting']} E={r['evalue']:.2g}; {r['target']} ({r['feature_type']}: {r['description']}); "
              f"named={r['named']}; scan also lands={r['scan_also_lands']}; {r['n_identical']} of {r['n_aligned']} identical")
        print(f"  isoform    {r['qstart'] + 1:>5} {r['qseq']}")
        print(f"                   {r['match']}")
        print(f"  Swiss-Prot {r['tstart'] + 1:>5} {r['tseq']}")
else:
    print("Not run yet.")
""")
    md(results_md or r"""
## 5. Verdict

Not run yet. The verdict is written here from `verdict.json` after the Sherlock run.
""")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-md", help="markdown file with the verdict section, written after the run")
    a = p.parse_args()
    nb = build(Path(a.results_md).read_text() if a.results_md else None)
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(NB)
