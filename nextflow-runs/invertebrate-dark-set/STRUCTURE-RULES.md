# Foldseek and ProstT5 on the dark set: rules and predictions

Written 2026-10-16, before the run (`-entry darkStructure`) was launched. No Foldseek or
ProstT5 result on the dark set existed when this was written. Changing a rule after the run
means writing the new rule below the old one with the date and the reason, and showing both
results.

## The question

The dark set (PR 57) is every protein of a query proteome that phmmer, jackhmmer and MMseqs2
cannot place in reviewed Swiss-Prot minus the query's own clade. How many dark proteins does
a structure search place, how often is its best hit right, and what does it cost? This is
"is tool B worth running after tool A", with A the three sequence tools and B Foldseek (or
ProstT5).

## What is searched

| | Foldseek | ProstT5 |
|---|---|---|
| query | AlphaFold DB model of the protein | 3Di letters predicted from the protein's sequence by ProstT5, on a GPU |
| species | human, worm, yeast, Borrelia B31 | the same four and Botryllus |
| target | Foldseek's AlphaFold DB Swiss-Prot database, cut to the accessions of the clade-excluded reference the sequence tools searched | the same cut database |
| not run | a protein with no AlphaFold DB model | a protein over 6,000 residues |

Botryllus's 45,339 gene models have no UniProt accession, so they have no AlphaFold DB model
and Foldseek is not run on them. A protein that is not run is counted as "not run", never
as a protein the tool searched and missed. The share of queries with a model and the share
of reference proteins with a model are printed for every species.

## Rules

1. **Placed.** Each tool also searches a decoy for every protein it runs on. The decoy for
   ProstT5 is the protein's sequence reordered with the dipeptide shuffle in
   `nextflow-runs/shared/decoys.py` (same first and last residue, same count of every
   adjacent residue pair), passed through ProstT5. A shuffled sequence has no structure, so
   the decoy for Foldseek reorders the model's residues as (amino acid, 3Di letter) pairs
   with the same dipeptide shuffle: the decoy keeps the model's amino-acid and 3Di
   composition and has no C-alpha coordinates, which Foldseek's default alignment and its
   E-value do not use. The tool's cutoff for a species is the largest E-value at which at
   most 5% of that species' decoys have a hit. A protein is placed by the tool when its best
   hit has an E-value at or below the cutoff.
2. **Dark.** A protein is dark when no phmmer, jackhmmer or MMseqs2 hit reaches E < 1
   (`dark_E1`, the set notebook 291 scored); `dark_E10` (none below E < 10) is reported too.
   Sequence-tool E-values are the per-protein export of the midi run after the queries.tsv
   repair (job 47668732).
3. **Right.** A placed protein's best hit (smallest E-value) is right when the target's
   Swiss-Prot entry lists a Pfam family the query has (Pfam's annotation of UniProt), the
   rule of PR 133's accuracy panel. A shared clan with no shared family is reported apart.
   Judged = right + clan only + wrong; a target with no Pfam family, or a query with none, is
   not judged. If the stricter judgement of the weak-hit accuracy work is pushed before
   scoring, it is used and both are shown.
4. **Domain metrics.** Fmax, family Fmax, IoU and reachable recall come from
   `evaluate_domain_calls.py` with notebook 291's settings (minimum overlap 0.5, strict IoU
   0.8, transfer dedup off and on), on `dark_E1`, `dark_E10` and the 1,000 random placed
   proteins per species of notebook 291, with each truth set cut to the proteins the tool ran
   on.
5. **Small numbers.** A species or set with fewer than 30 dark proteins run, or fewer than
   30 judged best hits, is shown grey and decides nothing.
6. **No tuning on truth.** Nothing is fitted to Pfam. The only fitted value, each cutoff,
   comes from the decoys. So there is no tune/test split to make.

## Predictions

**P1.** Foldseek places under 10% of the dark proteins that have a model, in each of human,
worm, yeast and Borrelia.
Overturned if, in any species with at least 30 dark proteins that have a model, 10% or more
are placed. The share of all dark proteins (with or without a model) is printed beside it.

**P2.** Foldseek's best hits on dark proteins are right in under 20% of judged cases (the
four species together).
Overturned if 20% or more of at least 30 judged best hits are right. With fewer than 30
judged, P2 is undecided. Each species is printed too.

ProstT5 gets no prediction; its numbers are reported beside Foldseek's.

## What was seen before these rules were written

- Notebook 291 (PR 57): kmerseek found 1 known Pfam domain on dark proteins over 230
  settings; phmmer and jackhmmer found 2 to 53 domains per species from hits at E 1 to 10.
- PR 133: on dark proteins, the sequence tools' hits at E 1 to 10 were right 1 to 4% of the
  time, and kmerseek's best hits were wrong in 726 of 733 judged cases.
- The sizes of the dark sets (dark_E1: human 1,696, worm 4,667, yeast 725, Borrelia 292).
- Foldseek's AlphaFold DB Swiss-Prot database on Sherlock (590,183 entries, all model v6,
  all fragment F1) and the clade-excluded references (505,077 to 572,850 entries); how many
  reference proteins have a model was not counted.
- Run times and memory of Foldseek and ProstT5 in the QfO region benchmark and the
  convergence-afp run (`assets/structure_tools_measured.tsv`).

## A premise that did not hold

The task said placement should use "the same decoy-matched cutoff rule the pipeline uses
for sequence tools". This pipeline has none: a sequence tool places a protein at a fixed
E-value (`--evalue_call`, 0.001 in the run; E < 1 in notebook 291). The decoy rule above is
applied to Foldseek and ProstT5 only, and the sequence tools keep their E-value cutoffs.

## Known risk, found before the run (2026-10-16)

The Foldseek decoy keeps runs of adjacent (amino acid, 3Di) pairs, so it keeps some real
signal: in a five-structure test, a decoy of 1fmc chain A hit 1fmc chains A and B at E 1e-4
and 3e-3. A decoy that hits a distant relative of its own protein makes the 5% cutoff
stricter than a chance-hit rate would, which lowers Foldseek's placed count and makes P1
easier to hold. After the run, the decoys' best hits are judged against Pfam like the real
ones; if more than 1 in 5 judged decoy hits share a family with the protein, P1 is reported
as unreliable rather than as holding.

## Strict judgement wired in (2026-10-16, still before the run)

Rule 3 said the stricter judgement of the weak-hit accuracy work would be used beside PR
133's if pushed before scoring. It was pushed (PR 163, commit 13749b87), so scoring now also
runs its `dark-set-accuracy score-bands` on each tool's best hits, with the hit's query
stretch: loose right = a shared Pfam family or clan; strict right = a shared family and the
hit's query stretch overlaps a query domain of that family. P2 is still decided on the
family call written above; the loose and strict shares are printed beside it. score-bands'
dark set is "best sequence-tool E > 1", which differs from dark_E1 (E >= 1) only for a
protein whose best sequence hit is exactly 1.
