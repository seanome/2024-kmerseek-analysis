## Pre-registered rule (written 2026-10-18, before any word was counted)

This cell is the first cell of notebook 343 and was committed before the pipeline counted
any word. It does not change after the run.

**Question.** We hold some proteomes from a clade. Would sequencing one more add new local
sequence, or would it mostly repeat sequence we have already seen?

**Panel.** The Quest for Orthologs (QfO) reference proteomes, release 2020_04: 78 species
from UniProt, one canonical proteome per species. One species is one independent sample.
A hidden sample is a whole proteome, so none of its sequences is seen while fitting.
Clades are NCBI taxa, fixed in `scripts/nb343_build_proteome_table.py`: Archaea (7
proteomes), Bacteria (23), Eukaryota (48), Vertebrata (13), Mammalia (8), Fungi (12). A
clade needs at least six proteomes. Viridiplantae (5) and Arthropoda (4) have too few.
The clades overlap: Mammalia sits inside Vertebrata, which sits inside Eukaryota.

**Words.** A word is a window of k residues of one protein, written in an alphabet. A
window that holds a residue other than the 20 standard amino acids is skipped. A word
*recurs* in a set of proteomes when at least two of them contain it. A word found in one
proteome only is never counted, so a one-off error or a species-private sequence does not
look like new content.

**Quantities.**

- $n$: the number of proteomes in hand; $N$: the number in the clade.
- $s$: the FracMinHash scaled value. A word is kept when its 64-bit hash is below
  $2^{64}/s$. Here $s = 1$, which keeps every word, as in every kmerseek pipeline in this repo.
- $U(n) = \frac{1}{s} \times$ (number of kept words that recur among $n$ proteomes),
  averaged over every ordering of the clade's proteomes. The average is computed exactly
  from how many proteomes hold each word, not from random orderings.
- $U_\text{null}(n)$: the same count for synthetic proteomes. Each real proteome gets one
  synthetic copy with the same protein lengths. Its residues come from a first-order Markov
  chain: each residue is drawn given only the residue before it, with the pair frequencies
  counted in that real proteome. The copy keeps the proteome's amino-acid composition and
  nothing longer than two residues.
- $U(n)/U_\text{null}(n)$: how many times more recurring words the real proteomes share
  than composition alone produces. A closed repertoire gives a flat $U(n)$ while
  $U_\text{null}(n)$ keeps growing, so the ratio falls.
- Heaps' law $U(n) = \kappa\, n^{\gamma}$. $\kappa$ (kappa) is the curve's scale, in words;
  $\gamma$ (gamma) is how fast the count grows with each added proteome. $\gamma$ near 0
  means adding proteomes adds almost nothing. Both are fitted by least squares of
  $\log U$ on $\log n$, for $n = 2, \dots, N$ ($U(1)$ is 0 by definition).

**Settings.** Two alphabets, three word budgets. Budgets are matched in bits of
information per word (`ou.bits_per_kmer`, real amino-acid frequencies):

| budget | reduced alphabet | 20-letter alphabet |
|---|---|---|
| about 21 bits | `hp_pbotc_1st_ed` k = 21 (20.9 bits) | `protein` k = 5 (20.9 bits) |
| **about 30 bits (primary)** | **`hp_pbotc_1st_ed` k = 30 (29.8 bits)** | **`protein` k = 7 (29.2 bits)** |
| about 42 bits | `hp_pbotc_1st_ed` k = 42 (41.7 bits) | `protein` k = 10 (41.8 bits) |

Only the 30-bit pair enters the verdict. The other two are reported.

**Baseline: MMseqs2 family rarefaction.** For each clade, all its proteins are clustered
with `mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0` (MMseqs2 release 18,
BioContainers image pinned by digest in `nextflow-runs/proteome-novelty/main.nf`). A family
is one cluster. A family recurs when it holds proteins from at least two proteomes. $F(n)$
counts recurring families the way $U(n)$ counts recurring words. Clustering runs once per
clade on all its proteomes, hidden ones included. That can only make the MMseqs2
prediction better than it would be from the observed proteomes alone.

**Primary test: predict the next sample.**

1. In a clade of $N$ proteomes, hide the final $\lceil 0.2N \rceil$ of a random order and
   keep $n_0 = N - \lceil 0.2N \rceil$. The next sample is the first hidden proteome.
2. Draw 200 distinct (kept set, next sample) splits with a seed fixed by the clade's name,
   or take every distinct split when there are fewer than 200 (Archaea has 42, Mammalia
   56). The same splits are used for every setting and for MMseqs2.
3. Fit Heaps' law on the kept proteomes only ($n = 2, \dots, n_0$). Predicted added words
   $= \hat\kappa\,\big((n_0+1)^{\hat\gamma} - n_0^{\hat\gamma}\big)$.
4. True added words = words held by exactly one kept proteome and by the next one.
5. Relative error $= |\text{predicted} - \text{true}| / \text{true}$. A split whose true value
   is 0 is dropped and counted.
6. The statistic is the median relative error over all splits, with each clade weighted
   equally: a split weighs $1/(\text{splits in its clade})$.

**PASS** only if, at the 30-bit budget, the `hp_pbotc_1st_ed` k = 30 median is at most 0.15
**and** at most 0.75 times the MMseqs2 median (at least 25% below it). Otherwise FAIL.

**20-letter check.** If the `protein` k = 7 median is at or below the `hp_pbotc_1st_ed`
k = 30 median, the reduced alphabet adds nothing to the prediction. The application then
stands or falls on the 20-letter curve against the same two thresholds, and the case for
reduction fails.

Per-clade medians are reported and do not enter the verdict.

**Secondary test (structure).** Run only if AlphaFold models are on hand for every
proteome of the clades. This pipeline does not fetch models for the 78 proteomes, so the
secondary test is not run in notebook 343. The rule, for a later run: across clades, the
Spearman correlation of $\gamma$ (30-bit `hp_pbotc_1st_ed`) with the $\gamma$ of Foldseek
structure-cluster rarefaction passes at $\rho \ge 0.70$, and only if it beats the same
correlation for a composition baseline (the variance, across the clade's proteomes, of
their amino-acid frequencies).

**Sequencing decision, per clade.** At the 30-bit `hp_pbotc_1st_ed` setting, another
proteome is worth sequencing when the words it is predicted to add (step 3 with
$n_0 = N$) are at least 1% of $U(N)$ **and** at least twice what the synthetic proteomes
predict for their next copy. Each clade's paragraph reports $\gamma$, the predicted added
words, the null's predicted added words and $U(N)/U_\text{null}(N)$.

**Example stretch.** For the clade with the lowest $\gamma$ and the clade with the highest
$\gamma$ (30-bit `hp_pbotc_1st_ed`), name the proteome that adds the most newly recurring
words when it is added last, and print its longest stretch of such words: both sequences,
coordinates, amino-acid match line and encoded match line.
