## 7. Conclusions

The question was whether kmerseek keeps placing a functional label correctly at lower
identity than the sequence tools. On these three families it does not. It stops at higher
identity than phmmer and MMseqs2, and Foldseek reaches the bottom of every family.

Lowest global identity at which each tool placed the label correctly, and how many of the
pairs that carry the label it got right:

| family | Foldseek | phmmer | MMseqs2 | kmerseek, best of 151 alphabet-k combinations |
|---|---|---|---|---|
| globins (56) | 12.7%, 56 of 56 | 17.6%, 45 of 56 | 19.9%, 15 of 56 | 23.9% |
| cystatins (24) | 12.5%, 24 of 24 | 12.5%, 18 of 24 | 14.0%, 13 of 24 | 14.0% |
| lysozymes and alpha-lactalbumins, same-function pairs (4) | 56.8%, 4 of 4 | 56.8%, 4 of 4 | 56.8%, 4 of 4 | 56.8% |

The kmerseek column is the best single combination for each family, picked after seeing
the results, so it overstates what one fixed setting would do.

* Globins. Below 84% identity, kmerseek placed the heme iron histidine on only two pairs.
  Human hemoglobin alpha and beta (43.6%) were right with dayhoff6 k7, mmseqs12 k5 and
  sdm12 k5. Human myoglobin and hemoglobin beta (23.9%) were right with gbmr4 k10 only.
  On the other 50 globin pairs (12.7% to 35.2%, including soybean leghemoglobin and
  barley hemoglobin at 35.2%) no combination placed it.
* Cystatins. At 12.5% to 15.5% (the stefins against chicken cystatin and cystatin C),
  kmerseek placed the label correctly with at most 4 of 151 combinations. At 41.8% (chicken cystatin and
  cystatin C) it was 13% to 32% of combinations, and at 53.1% (stefin A and B) 26% to 62%.
* Lysozyme and alpha-lactalbumin. phmmer, MMseqs2 and Foldseek each aligned all 8
  lysozyme-to-lactalbumin pairs (33% to 36%). They carried the catalytic residues onto
  alpha-lactalbumin, which has none, and the calcium-binding residues onto lysozyme, which
  Swiss-Prot does not annotate with them. An alignment of two homologs covers the same
  positions whether or not the function was kept, so none of these tools can tell the two
  functions apart this way. kmerseek had no call on most of these pairs. Where it did have
  one (264 combinations and labels), it carried the label wrongly in 211.

Foldseek's result is what this control was built to show. Every protein here has a
confident AlphaFold model (mean pLDDT 88 to 98), and structure search is expected to be
strong on such proteins. These families are not the regime kmerseek is meant for: short
features in regions with no confident structure.

Two limits of the kmerseek side of this run:

* 85 of the 151 combinations were searched without extension. Their index could not fit
  the Karlin-Altschul constants, and an extended search refuses to run without them. On
  those combinations every region is an exact run and `region_evalue` is
  `region_run_evalue`. All 8 protein20 combinations are among them. gbmr7 k8 has no
  results: it was killed for memory at 96 GB during that fit.
* The placement check says nothing about a call that spans the whole target, because such
  a call has only one position it can take. That is 376 of kmerseek's 1_078 correct calls.
  Of the rest, 354 have $\Pr(\text{correct by placement}) < 0.05$.
