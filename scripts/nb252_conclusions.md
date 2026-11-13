## 7. Conclusions

The question was whether kmerseek keeps placing a functional label correctly at lower
identity than the sequence tools. On these three families it does not. It stops at higher
identity than phmmer and MMseqs2, and Foldseek reaches the bottom of every family.

Lowest global identity at which each tool placed the label correctly, and how many of the
pairs that carry the label it got right:

| family | Foldseek | phmmer | MMseqs2 | kmerseek, best of 150 alphabet-k combinations |
|---|---|---|---|---|
| globins (56) | 12.7%, 56 of 56 | 17.6%, 45 of 56 | 19.9%, 15 of 56 | 23.9% |
| cystatins (24) | 12.5%, 24 of 24 | 12.5%, 18 of 24 | 14.0%, 13 of 24 | 14.0% |
| lysozymes and alpha-lactalbumins, same-function pairs (4) | 56.8%, 4 of 4 | 56.8%, 4 of 4 | 56.8%, 4 of 4 | 56.8% |

The kmerseek column is the best single combination for each family, picked after seeing
the results, so it overstates what one fixed setting would do.

* Globins. Below 84% identity, kmerseek placed the heme iron histidine on only two pairs.
  Human hemoglobin alpha and beta (43.6%) were right with dayhoff6 k7 and k9, mmseqs12
  k5 and sdm12 k5. Human myoglobin and hemoglobin beta (23.9%) were right with gbmr4 k10 only, at
  E = 154 and 162: an unrelated query would find that many regions as good in this
  database, so that call is at chance level.
  On the other 50 globin pairs (12.7% to 35.2%, including soybean leghemoglobin and
  barley hemoglobin at 35.2%) no combination placed it.
* Cystatins. At 12.5% to 15.5% (the stefins against chicken cystatin and cystatin C),
  kmerseek placed the label correctly with at most 4 of 150 combinations. At 41.8% (chicken cystatin and
  cystatin C) it was 25% to 36% of combinations, and at 53.1% (stefin A and B) 41% to 63%.
* Lysozyme and alpha-lactalbumin. phmmer, MMseqs2 and Foldseek each aligned all 8
  lysozyme-to-lactalbumin pairs (33% to 36%). They carried the catalytic residues onto
  alpha-lactalbumin, which has none, and the calcium-binding residues onto lysozyme, which
  Swiss-Prot does not annotate with them. An alignment of two homologs covers the same
  positions whether or not the function was kept, so none of these tools can tell the two
  functions apart this way. kmerseek had no call on most of these pairs. Where it did have
  one (266 combinations and labels), it carried the label wrongly in 247.

Foldseek's result is what this control was built to show. Every protein here has a
confident AlphaFold model (mean pLDDT 88 to 98), and structure search is expected to be
strong on such proteins. These families are not the regime kmerseek is meant for: short
features in regions with no confident structure.

Every kmerseek combination was searched with ungapped extension (kmerseek PRs #88 and #89).
110 of the 150 have a Karlin-Altschul fit on this database; the other 40, all 8 protein20
combinations among them, have no `region_ka_evalue` and their `region_evalue` is
`region_run_evalue`. gbmr7 k8 and k10 are left out of the ladder (commit c0be2ef). An
earlier run of this notebook, where only the 65 combinations with a fit were extended and
the rest searched exact, gave the same lowest identity for every family.

The placement check says nothing about a call that spans the whole target, because such a
call has only one position it can take. Extension makes that common: 674 of kmerseek's
1_214 correct calls. Of the rest, 224 have $\Pr(\text{correct by placement}) < 0.05$.
