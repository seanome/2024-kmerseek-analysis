#!/usr/bin/env python3
"""Generate notebooks/290_dark_set_kmerseek_reach_vs_shuffled_copies.ipynb.

Stop 1 (2026-10-13): the question, the data, the decision rule and the predictions only,
written before any shuffled search was run or read. The analysis cells are added after the
pilot (Stop 2) and the full run (Stop 3) without changing this first cell.
"""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


md(r"""
# 290. Dark proteins kmerseek reaches, against shuffled copies of the same proteins

**Question.** Notebook 279 ([PR 120](https://github.com/seanome/2024-kmerseek-analysis/pull/120))
found that kmerseek puts a region with E-value below 1 on 14-32% of *dark* proteins (no
phmmer, jackhmmer or MMseqs2 hit in Swiss-Prot with the query's own clade removed), against
43-79% of *placed* proteins (every protein that is not dark). How many of those dark-protein
regions would kmerseek also give to shuffled copies? We search copies of every protein
that keep its composition while changing residue order, and count how often they get a
region. Short matches can survive shuffling, especially within 10-residue windows.

**Decision.** Whether the dark-protein hint list can be handed to genome annotators (the
Earth BioGenome Project use case), or whether at E < 1 it is what chance alone gives.

**What is searched.** Every query chunk of the dark-set run, each protein replaced by a
shuffled copy, in two ways (both made before any search, seeds fixed):

* *Dipeptide shuffle*: every adjacent residue pair keeps its count (Altschul and Erickson
  1985), over the whole protein.
* *10-residue window shuffle*: residues shuffled only within non-overlapping 10-residue
  windows. A hydrophobic helix or a low-complexity run stays in place. This is the stricter
  copy for the low-complexity and helix filter, because a whole-protein dipeptide shuffle
  breaks up transmembrane stretches.

A kmerseek *setting* is one alphabet, k-size, scaled value, extension penalty and mask (the
pipeline code calls it an "arm"). The six settings are those notebook 279 sections 7 and 9
read, all at scaled 1 with the mask off, each extended at its alphabet's own mismatch
penalty C with X-drop 4C:

| alphabet | k | C | X-drop |
|---|---|---|---|
| hp_lehninger2 | 17, 25 | 1.51 | 6.04 |
| hp_lehninger_c_nonpolar2 | 19 | 1.52 | 6.08 |
| hp_pbotc_1st_ed2 | 19, 21, 23 | 1.59 | 6.36 |

**Data.** The invertebrate dark-set midi run (`runs/ladder-0.4-midi`,
[PR 57](https://github.com/seanome/2024-kmerseek-analysis/pull/57), finished 2026-10-12),
image `olgabot/kmerseek:0.4.0-rc5`. Queries: human (UP000005640), *C. elegans*
(UP000001940), *S. cerevisiae* S288C (UP000002311), *B. burgdorferi* B31 (UP000001807) and
*Botryllus schlosseri* gene models, in the run's own 49 chunks of up to 2,000 proteins. Target:
Swiss-Prot minus the query's clade (Mammalia, Chromadorea, Saccharomycetes, Spirochaetia,
Ascidiacea), the run's stored kmerseek indexes. Each shuffled chunk is the real chunk with
every protein replaced by its copy, so kmerseek sees the same chunk-level k-mer statistics.
The search command for every setting and chunk was checked to be the run's own
(`scripts/290_shuffle_control/tasks.py`: all 294 match).

**Measures**, per species and setting, with dark proteins defined at tool cutoffs E < 0.001
(the pipeline's dark set), E < 1 and E < 10:

* *R*, real reach: share of dark proteins with a region at E < 1, from the run's region files.
* *S*, shuffled reach: share of their shuffled copies with a region at E < 1 (and at E < 0.01
  and E < 0.0001), separately for the dipeptide and the window copies.
* Both before and after notebook 279 section 7's filter: a region is dropped when at least
  half of its residues are low complexity (SEG first pass) or a predicted helix
  (Kyte-Doolittle) on the query or on the target.
* *R - S* with a 95% interval, and the same for placed proteins.
* All of it split by structure-model confidence (mean AlphaFold pLDDT below 50, 50-70, 70 or
  more, and no model; *B. schlosseri* has no AlphaFold models, so metapredict disorder
  quartiles instead).

## Decision rule and predictions, written 2026-10-13 before any search output was read

**Rule.** A setting *beats chance* in a species at E < 1 when, after the filter,
R >= 2 x max(S_dipeptide, S_window) **and** R - max(S) >= 5 percentage points with the lower
end of its 95% interval above zero. Higher R - S is better; R - S near 0 means kmerseek finds
dark-protein regions no more often than it finds them on these shuffled copies.

The dark set has hints worth handing to annotators if some setting beats chance in **both
*B. schlosseri* and *C. elegans*** (the two large invertebrate dark sets). If not, the
dark-set hint list has not met the selection rule at E < 1: it is written up as a negative
result with the shuffled reach numbers, and the experiment stops. Failure to pass does
not establish equal reach in real proteins and controls. No setting, cutoff or filter changes after
the numbers are read.

**Olga's expectation**, from the shuffled copies of BHF: S under hp_lehninger2 k17 is high,
within a factor of 2 of R. (36 of 100 dipeptide copies of BHF get a region with E < 1 under
hp_lehninger2 k17, notebook 279 section 6c; 25.5% of *B. schlosseri* dark proteins do, 3,274
of 12,845, section 7.)

**Claude's expectation**, written before reading:

* hp_lehninger2 k17 fails the 2x test in both species: S_dipeptide 10-20% and S_window
  15-25% before the filter, against R of 25.5% (*B. schlosseri*) and 27.7% (*C. elegans*).
* The window copies get more regions than the dipeptide copies before the filter (they keep
  helices and low-complexity runs in place); after the filter the two come closer.
* The higher-k settings (hp_pbotc_1st_ed2 k21 and k23, hp_lehninger2 k25) have S near 0,
  but their R is at most 6% before the filter (0.3-6.0% in these two species), so they
  cannot clear the 5-point margin after it.
* hp_lehninger_c_nonpolar2 k19 and hp_pbotc_1st_ed2 k19 are the only candidates (R 7-21%
  before the filter). I expect at most one species to pass, so the overall verdict I expect
  is the negative one (about 70% sure).
* Placed proteins: R - S stays large (R 39-60% under the k17 and k19 settings in these two species), so the shuffled copies
  of placed proteins should confirm that kmerseek does find homologs where the other tools
  do.

**pLDDT split (reported, not part of the verdict).** Expected: S is highest in the
pLDDT < 50 bin (and in the top disorder quartile for *B. schlosseri*), since disordered,
low-complexity sequence gives chance k-mer matches. If a setting beats chance overall, we
report whether it still does inside pLDDT < 50: that row is the one the Earth BioGenome pitch
needs, because it is where low-confidence structure limits a structural interpretation.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "290_dark_set_kmerseek_reach_vs_shuffled_copies.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")
