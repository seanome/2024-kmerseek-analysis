## Summary and conclusions (Stop 0, 200-pair pilot, 2026-10-20)

**Data.** 200 of the 546 1:1 ortholog pairs, picked by the SHA-1 of the pair name. 198
of them have exon truth (72 chicken, 126 zebrafish). They hold 1,041 internal exons of
60 residues or fewer, and 614 of those are on pairs where both proteins have an AlphaFold
model (the verdict set).

**Under the pre-registered rule, the pilot passes all three parts.**

1. kmerseek's best of all 404 settings lands on 416 of 614 short exons (68%; rule: at
   least 50%).
2. Foldseek, Reseek and ProstT5 each land on 0 of 614 (rule: under 15%). Their calls
   cover most of the protein: Foldseek spills on 595 of 614 exons, Reseek on 612.
3. Counting only landing calls with a placement p-value below 0.05, kmerseek lands on
   237 of 614 (39%), 6.8 times the Kyte-Doolittle scan's 35 of 614 (6%; rule: at least
   2 times).

The Stop 0 checks do not stop the work: kmerseek lands on well over 10% of short exons,
and the scan does not reach kmerseek's rate.

**The random-protein null shows the rule does not test what it was written to test.**
When each exon's interval is moved, at the same coordinates, onto the target protein of
another pair, kmerseek still lands on it at almost the same rate (1,000 draws):

| kmerseek number | real exons | null median | null 99.9th percentile |
|---|---|---|---|
| best of all settings | 67.8% | 63.7% | 69.7% |
| best of all, p_place < 0.05 | 38.6% | 36.6% | 42.6% |
| best single setting (sdm12, k 6, no mask) | 31.8% | 28.0% | 33.9% |

All three real rates are inside their nulls. kmerseek's calls are short gapless runs
spread along both proteins. A short run that falls inside any 20-60 residue stretch
passes the landing rule (80% of the call inside, 30% of the stretch covered), whether or
not that stretch is an exon. Structure tools and aligners never land because they make
one call over most of the protein, so they are not a fair control for this; only the
null is. The placement p-value does not fix it either: it asks whether a call of that
length would land at a random position on the same protein, which a short call often
does not, so it passes calls that the null shows are not specific to exons. The null
was pre-registered for the best of all settings only; the two other rows were added
after Stop 0 and are labelled as such in `verdict.json`.

**Conclusion.** On the pilot, kmerseek's calls land on short exons as often as they
land on the same coordinates of an unrelated protein. The pilot gives no evidence that
kmerseek places a call on a single exon. The pre-registered rule passes, and that pass
should not be quoted: the rule's controls (structure tools, aligners, the scan) all make
long calls or almost no calls, so none of them measures how often a short call lands by
chance.

**Smaller points.** kmerseek lands more often on ordered exons (377 of 522, 72%) than
on disordered ones (39 of 92, 42%); the null was not split by disorder. HHblits lands on
28 of 614 and phmmer, MMseqs2 and jackhmmer on none. Reseek target positions on 4
F-digit TrEMBL accessions were past the protein end and were corrected (the known
fragment-offset bug).

**Decision for Olga.** Stop 1 (all 546 pairs) costs seconds on the laptop, since the
calls are already cut. It would tighten the numbers but cannot change the reading
unless the null and the real rates separate with more exons.
