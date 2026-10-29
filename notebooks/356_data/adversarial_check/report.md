# Notebook 356, second-agent check (2026-10-20)

An agent that had not seen the notebook, its write-ups or `score_356_exon_landing.py`
recomputed every number with its own code (the scripts and outputs in this folder), then
compared. Paths in `common.py` point at the nb 356 worktree and at
`~/data/356-exon-landing/out` (the cut calls from Sherlock job 48688997).

| Claim | Verdict | What was found |
|---|---|---|
| 1,669 verdict exons; kmerseek best of all lands on 1,129; Foldseek, Reseek, ProstT5 on 0 | HELD | Same counts, exon by exon; best p_place per exon differs by 0.0 (`compare_output.txt`) |
| Exon-to-protein coordinates | HELD | All 4,881 rows rebuilt from the XML exon lengths; 4,844 exons translated from their own codons match the protein; 20 hand-picked exons (phase 0, 1, 2, both species, first and last exons) match residue by residue. 37 exons on targets whose CDS does not translate to the UniProt sequence were checked by length only |
| No exon counted twice | HELD | 1,669 distinct exon IDs and intervals; one transcript per target. One human query can appear for both species, so the 261 pairs are not 261 independent proteins |
| p_place per exon and per call | HELD | Closed form equals a brute-force loop on 2,010 calls; 675 exons with a landing call at p < 0.05 |
| Pre-registered random-protein null | HELD | Median 62.0%, max 65.6%, 0 of 1,000 draws reach 67.6% |
| "About 95 exons above chance" | PARTLY | 95 under the pre-registered null; 55 when decoys are matched on protein length and calls per residue (null F, pair-resampled 95% interval 7 to 108); 49 to 55 when the window moves along the same protein (nulls G and H, interval 22 to 89 for H) |
| "Structure tools never land" | FAILED as evidence | Forced by the rule. Every Foldseek, Reseek and ProstT5 call on the verdict pairs is at least 84 residues long (median 388 to 396); a call lands on an exon of 60 residues or fewer only if it is at most 75 residues. These tools cover at least 80% of 96 to 99% of the verdict exons (kmerseek: 40.5%) |
| Rule 3 and the best single setting above chance | WEAK | 11 of 1,000 null draws reach the p_place-filtered rate; 1 of 1,000 reaches the best single setting, which was chosen on these exons |
| 404 kmerseek settings | Clarified | 408 on disk; 4 `hp_thomas_dill2_ext2` settings are empty; 3 of the 404 have only a chicken table (`gbmr4_k12_lcTrue`, `gbmr7_k11_lcFalse`, `gbmr7_k10_lcTrue`) |

Nothing was found wrong in the coordinate systems, the pair joins, or which proteins
structure tools were scored on.

**Changed in the notebook after this check:** the conclusion now reports the excess over
the matched nulls (about 50 exons, 3 percentage points) with the pair-resampled interval,
and says that rule 2 could not have failed, so "structure tools never land" is not
evidence. Not changed: rule 2 itself (the pre-registration says the rule is not changed
afterwards). A test that can come out either way would map the exon through each tool's
residue-level alignment; the call tables do not hold those alignments.
