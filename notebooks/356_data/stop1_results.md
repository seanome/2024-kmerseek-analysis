## Summary and conclusions (Stops 1 and 2, all pairs, 2026-10-20)

**Data.** All 546 1:1 ortholog pairs; 541 have exon truth (203 chicken, 338 zebrafish).
They hold 3,178 internal exons of 60 residues or fewer, and 1,669 of those, on 261 pairs,
have both proteins with an AlphaFold model (the verdict set). The Stop 0 pilot (200 pairs,
614 verdict exons) is in `tables/356/pilot/` and `figures/356/pilot/`.

**Under the pre-registered rule, all three parts pass.**

1. kmerseek's best of all 404 settings lands on 1,129 of 1,669 short exons (68%; rule:
   at least 50%).
2. Foldseek, Reseek and ProstT5 each land on 0 of 1,669 (rule: under 15%).
3. Counting only landing calls with a placement p-value below 0.05, kmerseek lands on 675
   (40%), 8.2 times the Kyte-Doolittle scan's 82 (5%; rule: at least 2 times).

**Two parts of that pass are not evidence.**

* *Rule 2 could not have failed.* Every Foldseek, Reseek and ProstT5 call on the verdict
  pairs is at least 84 residues long (median 388 to 396). A call lands on an exon of 60
  residues or fewer only if 80% of it fits inside, so only if it is 75 residues or
  shorter. These tools report one alignment over most of the protein, and that alignment
  covers at least 80% of 96 to 99% of the verdict exons. The 0 says the tools report
  whole-protein alignments, not that they miss short exons. Testing them fairly would need
  each tool's residue-by-residue alignment, which the call tables do not hold.
* *Rule 1's 50% is met mostly by chance landings.* kmerseek's calls are short gapless
  runs along both proteins, and a short run inside any 20-60 residue stretch passes the
  landing rule whether or not the stretch is an exon.

**What is left: a small excess over chance.** Each exon's interval was moved to a decoy
position and scored against the same kind of kmerseek calls (1,000 draws each):

| null (where the decoy interval goes) | real exons | null median | excess exons | 95% interval, resampling whole pairs |
|---|---|---|---|---|
| another pair of the same species, same coordinates (pre-registered) | 67.6% | 62.0% | 95 | 40 to 151 |
| as above, decoy protein within 20% of the length and calls per residue within 0.8-1.25x | 68.9% | 65.4% | 55 | 7 to 108 |
| the same protein, a same-length window not touching the exon | 67.6% | 64.3% | 55 | 22 to 89 |

The last two rows are from the second-agent check (`356_data/adversarial_check/`).
Against the matched nulls, about 50 to 55 of the 1,129 landings (3 percentage points of
1,669 exons) are above chance; the rest happen at the same rate on an interval that is
not the exon. The placement-filtered rate (40.4% against a null median of 37.6%; 11 of
1,000 draws reach it) and the best single setting (sdm12, k 6, mask on: 30.3% against
27.1%, chosen on these same exons) are not above chance at the 0.001 level.

**Most landing calls are at the homologous position (added after Stop 1).** Of the 1,129
landed exons, 1,097 have a landing call whose residue pairs a global alignment of the two
orthologs also makes. The other 32 are landed only by calls pairing the exon with the
wrong stretch of the human protein. The hero shows one exon with both kinds: in COL9A1, a
polarity4 k 13 call covers chicken exon 14 exactly (IoU 1.0) but pairs it with human
residues the alignment puts at chicken 297-314, another copy of the Gly-X-Y collagen
repeat (9 of 18 identical). A dayhoff6 k 11 call pairs the same exon with its aligned
human residues (11 of 14 identical, 14 of 14 in the same dayhoff6 class). Every other
search tool's call covers 426 to 920 residues of the 920-residue protein; the
Kyte-Doolittle scan makes no call there.

**Conclusion.** On 1,669 short internal exons of chicken and zebrafish orthologs of the
midi-plus queries, kmerseek's calls land on the exon about 3 percentage points more often
than on a matched decoy interval (about 55 exons; 95% interval 22 to 89 when the decoy
window moves along the same protein). The pre-registered rule passes, but its 50% bar is
met mostly by chance landings, and its structure-tool bar could not have failed. The
evidence does not support "kmerseek places calls on single exons where structure search
cannot"; it supports, at most, a small exon-specific excess for kmerseek, with the
structure tools untested.

**Smaller points.** kmerseek lands more often on ordered exons (977 of 1,373, 71%) than on
disordered ones (152 of 296, 51%); the null was not split by disorder. HHblits lands on 68
of 1,669; phmmer, MMseqs2, MMseqs2 iterative and jackhmmer on none, for the same reason as
the structure tools. Reseek target positions on 4 F-digit TrEMBL accessions ran past the
protein end and were corrected (the known fragment-offset bug). 408 kmerseek settings are
on disk; 4 `hp_thomas_dill2_ext2` settings made no call, and 3 of the 404 have only a
chicken table. Stop 1 was pre-registered as a Sherlock job; it ran on the laptop in 11 s
because the Sherlock job had already cut the calls to the pairs.

**Not done.** A test of the structure tools that can come out either way (map the exon
through each tool's residue-level alignment). A null that keeps only decoys whose ends
fall on real exon boundaries of the decoy protein.
