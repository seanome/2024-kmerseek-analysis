# Full run results, 2026-10-16

The comparison tables from the full 975-query run on Sherlock, copied from
`$SCRATCH/regions-index-pilot/results/` on 2026-10-20. The run (launch folder `runs/full`,
head job 48130096) ended with status 0 on 2026-10-16 at 11:20.

## What is here

- `comparison/*.tsv`: the tables `bin/compare_indexes.py` writes (see its docstring).
  `feature_calls.parquet` (11 MB) is not copied; it stays on Sherlock.
- `comparison/feature_identity.tsv`: as the run wrote it.
- `scored_thresholds/*.threshold.json`: one per label, from `scoreCalls`.
- `pfam_index/`: the Pfam index's cluster summary, filter counts and decoy logs.
- `pfam_index_recall.{pdf,svg,png}`: made by `tools/pfam_index_figure.py`.

## The comparison tables were rebuilt here

The run's own `compareIndexes` task got no "pfamshared" label (the Pfam index searched
with the whole-protein index's fit). `main.nf` matched each regions or Pfam search to the
whole-protein fit with `join`, which pairs a key once; both indexes share each key (decoy
window, setting), so only the regions index got the shared fit in that run. The 37
pfamshared labels had been searched and scored by an earlier launch of the same run
(2026-10-16, 08:09) and sit in `results/scored/`.

So `compare_indexes.py` was run again on the Mac on all 474 files in `results/scored/`
with the run's truth table and `feature_identity.tsv`. Every row of the run's own
`composition.tsv`, `identity_bins.tsv`, `recall_by_kind.tsv` and `mcnemar.tsv` is in the
new tables unchanged; the new tables only add pfamshared rows. `main.nf` now uses
`combine(..., by: 0)`, so a new run searches both.

## Pfam index, headline reading

Aligned-piece landing rule, decoys shuffled in 10-residue windows, all evidence, 5% decoy
error.

- **Folded domains (907):** against whole proteins, the Pfam index raises kmerseek's
  recall for protein20 k5 (+7.5 points, McNemar p = 1.6e-7), mmseqs12 k6 (+4.0, p = 0.005)
  and sdm12 k7 (+3.3, p = 0.015): 3 of the 17 settings with a fit. It lowers it for 11
  (p < 0.05, -3.5 to -8.8 points), the five two-letter H/P alphabets with a fit among
  them. Best kmerseek recall: 44.0% with Pfam (mmseqs12 k6)
  against 41.5% with whole proteins (wwmj5 k10). MMseqs2 drops from 75.9% to 64.2%.
- **Disordered (45) and short (5_460) features:** the Pfam index finds no disordered
  feature with any setting, and finds fewer short features than whole proteins with every
  setting. That fits the expectation in `PREDICTIONS.md`: Pfam holds few families for these.
- **All features (7_174):** below whole proteins for every setting and for MMseqs2.
- **Shared fit:** "pfamshared" is within 0.9 points of "pfam" in every setting and
  feature kind.
- **No fit:** gbmr7 k16 (both windows) and hp_kyte_doolittle2 k22 (10-residue window) got
  no E-value fit on the Pfam index; they have no Pfam dot in the figure.
- With 20-residue decoy windows the same three settings gain on folded domains (+11.0,
  +7.9, +4.3), and hsdm17 k6 and uniprot18 k5 gain too (+2.6, +4.1).

Caveat from `PREDICTIONS.md`: 834 of the 1_028 query DOMAIN features were annotated from
sequence models like Pfam's, so a Pfam index can match how they were drawn.

## Two columns that look wrong and are not

- **`found_name` is 0 on every Pfam row.** A name match needs the index entry's
  description to equal the query feature's. Pfam entries are named by family
  (`PF01266/DAO`, all 445_238 of them start with "PF"), UniProt features by UniProt's notes
  ("TIR", "Disordered"). No query feature description equals any Pfam description, so a
  name match cannot happen. Recall does not use the name; it is a secondary column.
- **The composition classifier's `features_found` of 3_239 in `thresholds.tsv`** counts all
  54_301 of its calls with no decoy limit (its threshold is empty, `n_target` 0): a
  window-shuffled decoy has the same composition as its source, so the classifier cannot
  be held to the decoy rule. It is not comparable with the searches. The comparison
  `PREDICTIONS.md` set (Olga, 2026-10-13) is `composition.tsv`, at kmerseek's own call
  count on the regions index. There, on disordered features, no kmerseek setting beats it:
  the best are protein20 k5 (kmerseek 11 of 45, composition 16, p = 0.36) and uniprot18 k5
  (10 against 9, p = 1.0).
