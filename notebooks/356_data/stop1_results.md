## Summary and conclusions (Stop 1, all pairs, 2026-10-20)

**Data.** All 546 1:1 ortholog pairs; 541 have exon truth (203 chicken, 338 zebrafish).
They hold 3,178 internal exons of 60 residues or fewer, and 1,669 of those are on pairs
where both proteins have an AlphaFold model (the verdict set). The Stop 0 pilot (200
pairs, 614 verdict exons) is in `tables/356/pilot/` and `figures/356/pilot/`.

**Under the pre-registered rule, all three parts pass.**

1. kmerseek's best of all 404 settings lands on 1,129 of 1,669 short exons (68%; rule:
   at least 50%).
2. Foldseek, Reseek and ProstT5 each land on 0 of 1,669 (rule: under 15%). They spill
   on 1,615, 1,666 and 1,605 of them: one call over most of the protein.
3. Counting only landing calls with a placement p-value below 0.05, kmerseek lands on 675
   (40%), 8.2 times the Kyte-Doolittle scan's 82 (5%; rule: at least 2 times).

**Against the random-protein null, kmerseek beats chance, but by a small margin.** Each
exon's interval was moved, at the same coordinates, onto the target protein of another
pair of the same species, and scored against that pair's kmerseek calls (1,000 draws).

| kmerseek number | real exons | null median | null max | draws at or above real | exons above the null median |
|---|---|---|---|---|---|
| best of all settings (pre-registered null) | 67.6% | 62.0% | 65.3% | 0 of 1,000 | 95 |
| best of all, p_place < 0.05 (added after Stop 0) | 40.4% | 37.6% | 41.0% | 11 of 1,000 | 48 |
| best single setting, sdm12 k 6, mask on (added after Stop 0) | 30.3% | 27.1% | 30.4% | 1 of 1,000 | 53 |

The pre-registered null for the best of all settings is beaten at p < 0.001 (no draw
reaches the real rate). The excess is about 95 of 1,669 exons; the other ~1,034 landings
happen at the same rate on an interval that is not the exon. The placement-filtered rate
is inside its null. The best single setting was chosen on these same exons, so its null
comparison favours it; it is shown for scale, not as a test.

**Why most landings are chance.** kmerseek's calls are short gapless runs along both
proteins. A short run inside any 20-60 residue stretch passes the landing rule (80% of
the call inside, 30% of the stretch covered), whether or not the stretch is an exon. The
rule's other controls (structure tools, aligners, the scan) make long calls or almost
none, so none of them measures this; only the null does.

**Conclusion.** On 1,669 short internal exons of chicken and zebrafish orthologs of the
midi-plus queries, kmerseek's calls land on the exon about 6 percentage points more often
than on the same coordinates of an unrelated ortholog pair (67.6% against a null median
of 62.0%, p < 0.001 over 1,000 draws); structure tools and aligners never land, because
they cover the whole protein. The pre-registered rule passes, but its 50% threshold is
met mostly by chance landings. The defensible claim is the excess over the null, not the
68%.

**Smaller points.** kmerseek lands more often on ordered exons (977 of 1,373, 71%) than
on disordered ones (152 of 296, 51%); the null was not split by disorder, so this does
not say where the excess comes from. HHblits lands on 68 of 1,669; phmmer, MMseqs2,
MMseqs2 iterative and jackhmmer on none. Reseek target positions on 4 F-digit TrEMBL
accessions ran past the protein end and were corrected (the known fragment-offset bug).
The pre-registration planned Stop 1 as a Sherlock job; it ran on the laptop in 11 s
because the Sherlock job had already cut the calls to the pairs.

**Not done.** Stop 2: the second-agent check and the hero case. A tighter null that
moves the interval along the same protein, or keeps only decoys whose ends fall on real
exon boundaries of the decoy protein, would say whether the 95-exon excess is about exon
boundaries or about which part of the protein the exon sits in.
