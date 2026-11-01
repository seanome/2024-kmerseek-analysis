## Second reader: held / failed

A fresh agent session, given only the branch at cdf6415d and the Sherlock run folder, tried
to disprove each claim with its own scripts (read-only). It recomputed every headline number
from the per-run `scores.tsv` files, `calls.tsv` and `db_full.fasta.gz`. Its result: the FAIL
holds, and nothing it found changes the verdict or the explanation.

| Claim | Result | What it found |
|---|---|---|
| 1. Rule committed before any method ran | partly held | Rule commit 3743f18a at 14:35; dev job started 14:37; sdm12 k=4 stop committed 15:20, before the first dev table (15:25). But a smoke test (job 48680816, 14:27) ran kmerseek on 10 dev calls of an earlier freeze, none of them among the final 500, and computed no score. Fair wording: "before any method ran on the frozen calls". |
| 2. 500 calls, 250/250, 50 absent, 59 removed, 20,734 proteins, no shared protein | held | All counts reproduced. |
| 3. PEAKS-to-PD join on the precursor | held, as far as testable | All 500 mass-consistent (433 equal; the rest differ by +57, +16 or +114 Da of modifications). 7 pairs share 3 or fewer residues; wrong pairing or wrong de novo read cannot be told apart. |
| 4. massspec18 k=8 chosen correctly | held | Unique maximum 0.70; hsdm17 and sdm12 tie at k=8, so not sent to test. |
| 5. Recovery 140/200 (K), 128/200 (C) | held | |
| 6. False assignment 4/50 (K), 6/50 (C) | held | Absent runs used the reduced database: no removed accession in their output. |
| 7. Median list 1 and 1; sent to check 1 and 221 | held | |
| 8. Rule 4a: C's 53.5% at <= 8%, +16.5 points | held | C reaches it at threshold 74. |
| 9. Rule 4b: 125 s vs 27 s, 4.6x, one thread, fresh builds | held | Single timing run each, so 4.6x against 5x is close. |
| 10. Lead is a threshold effect | held | Whenever both send the source to the check they give it the same score. |
| 11. massspec18 8-mer = full-alphabet 8-mer with I=L, Q=K | held | Brute-force normalised 8-mer search gives kmerseek's candidate sets. |
| 12. 566/1,250 adjacent-pair shuffles identical; 33.6% / 35.6%; 1.04% / 1.52% | held | |
| 13. Threshold rule | held | Brute-force scan matches every setting. |
| 14. Compute 8.2 / 3.1 CPU-h, 1.2 h, 14:07-15:36 | held | Login-node and laptop work before 14:07 is not counted. |
| 15. Same check and normalisation for both | held | |
| 16. Printed alignments | held | All 739 rows of `alignments.tsv` agree with the database. |

Other problems it raised, none changing the numbers:

1. The accepted-source set is Proteome Discoverer's 2015 protein list, matched by accession.
   In 51 of 500 calls another protein in the frozen database also holds the database-search
   peptide but does not count as correct. Counting those too leaves recovery at 140 and 128.
2. kmerseek reports only one accession for proteins with identical sequences (20,650 targets
   at k=8 for 20,734 proteins). No recovery was lost here.
3. 22 of the 200 source-present test calls are shorter than 8 residues, so kmerseek at k=8
   cannot find them; the pre-registration counts them as abstentions.
4. Rule 4b rests on one timing run per workflow; the verdict does not depend on it.
