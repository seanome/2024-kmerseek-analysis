# Adhesive class: truth set and the mussel foot protein 1 case study

Underwater glues rich in tyrosine and lysine, where the tyrosine becomes DOPA after
translation. Mussels (B1) and sandcastle worms (B2) arrived at them separately. Barnacle
cement (B3) does the same job with a different chemistry and is the control. The truth
set has 10 entries: 6 mussel, 2 sandcastle worm, 2 barnacle
(`scripts/convergence_classes_truth.py`, UniProt 2026_03; queries in
`truth_query_log.tsv`, dropped entries in `dropped_entries.tsv`).

## Q25460 as a target in the ice-binding run

`q25460_as_target_in_ice_binding_run.tsv`: every kmerseek hit to Q25460 (mussel foot
protein 1, *Mytilus edulis*, 875 residues) in the ice-binding run, whose queries were 6
ice-binding proteins and 20 null A and 20 null B draws of each. Made by
`scripts/convergence_q25460_as_target.py` from
`$SCRATCH/2024-kmerseek-convergence-afp/nextflow-runs/convergence-afp/results/hits/kmerseek`
(164 settings with a ranked list) on 2026-10-19.

| What | Count |
|---|---|
| hits to Q25460 | 40, in 18 of 164 settings |
| from null A queries (residues shuffled) | 31 |
| from null B queries (repeat period kept, word changed) | 7 |
| from real ice-binding queries | 2, both from H7FWB6 (bacterial), E = 151_672 and E = inf |
| smallest E-value | 5.4, from a null A draw of the spruce budworm query Q9GSA6 |

No hit has an E-value below 1. Nine hits have E = inf: the hit was kept, but its regions
have no E-value (bits 0). Q25460 is reached by scrambled queries 38 times and by real
ice-binding queries twice, so these hits come from its composition, not from a shared
pattern.

## Q25460 as the query

`make adhesive_q25460` in `nextflow-runs/convergence-afp` runs the ice-binding pipeline
with `--truth_dir` set to this folder. The query is Q25460 with 20 null A and 20 null B
draws (41 queries). Null B uses a 10-residue unit, the HP autocorrelation peak
(r = 0.739), which matches the 10-residue REPEAT features of its UniProt entry (from
residue 75). The searched database is
Swiss-Prot 2026_03 plus the 3 truth entries that are only in TrEMBL, so every truth entry
can be found. Every method of the ice-binding run is run, kmerseek at all 230 settings.

Files for that run, written by `scripts/convergence_adhesive_q25460_inputs.py`:

- `queries.tsv`: Q25460, with the columns of `data/convergence_afp/queries.tsv`.
- `extra_reference.fasta`: Q7M3Q2 and Q7M3Q3 (sandcastle worm, *Phragmatopoma
  californica*) and Q9TWD7 (acorn barnacle, *Megabalanus rosa*), sequences from `truth_set.tsv`.
- `structures/`: the AlphaFold DB model of Q25460 (v6) for the Foldseek structure query,
  with its URL and MD5 in `structures.tsv`.
