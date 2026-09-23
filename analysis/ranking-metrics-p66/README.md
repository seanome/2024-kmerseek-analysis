# Which kmerseek value ranks a matched region best, and does P66 match anything human

Two questions, one benchmark and one case study.

## What ranks a region

`build_labeled_benchmark.py` searches the 998 human proteins that carry a Pfam
domain in the midi-plus truth set against each other, then labels every matched
region. `score_ranking_metrics.py` scores kmerseek's values against those labels.

Two corrections matter more than the metric comparison itself.

**Every match is reported twice.** An all-against-all search emits A→B and B→A,
and all 11_298 region rows have an exact mirror, so there are 5_649 independent
matches, not 11_298. The two copies disagree: mean IDF differs in 4_780 pairs and
the E-value in 3_547, because both are computed query-side, so the same physical
match scores differently depending on which protein you call the query. Region
length always agrees. The script keeps the orientation with the better E-value.

**Region length is a strong ranker on its own**, so it is carried through every
table as the control. Under a permissive rule — the region need only touch the
right domain by one residue — length ties the best value, because a long region
touches some domain almost automatically. Requiring the region to overlap the
domain by at least 20% of its own length is what separates them:

| correctness rule | correct | E-value | length | sum IDF | mean IDF |
|---|---|---|---|---|---|
| any overlap at all | 808 | 0.8711 | 0.8777 | 0.8778 | 0.3549 |
| overlap ≥ 20% of the region | 687 | **0.8198** | 0.7293 | 0.7205 | 0.2190 |
| overlap ≥ 20% of the domain | 765 | 0.8703 | 0.8848 | 0.8828 | 0.3535 |
| intersection over union ≥ 0.20 | 653 | **0.8376** | 0.7509 | 0.7406 | 0.2154 |

Average precision. The 20%-of-domain variant behaves like "any overlap" for the
same reason: clearing 20% of a domain is easy for a long region.

What each value turns out to be:

* **E-value** — the only one that beats length once the region has to sit on the
  domain, and the only one that holds up within a single region length.
* **sum of IDF** — region length rescaled. It correlates with length at Spearman
  0.967, and inside one length its AUC falls to 0.50 at 24–29 aa.
* **mean IDF** — better than the E-value below about 29 aa and worse above it,
  because the direction reverses: short correct regions hold rarer k-mers than
  short wrong ones, long correct regions hold commoner k-mers than long wrong
  ones. The crossover is near 35 aa, but see the caveat below.
* **enrichment** — ranks backwards overall (AUC 0.30) purely because it
  anti-correlates with length; inside one length it is positively predictive.

**The short-region result is not usable as it stands.** Only 10 of the 687
correct matches are shorter than 30 aa, so every bootstrap interval below 40 aa
spans 0.3 or more and no threshold can be located. Pfam domains are long; a truth
set built from short families or linear motifs is what would settle it.

## Does P66 match anything human

`p66_alphabet_ladder.sh` searches the mature P66 chain of *Borreliella
burgdorferi* against human GENCODE v49 across alphabets and k-mer sizes, and
pairs it directly against its documented receptor family.

| alphabet | k | regions | genes hit | % of proteome | best E |
|---|---|---|---|---|---|
| protein20 | 13, 10 | 0 | 0 | 0.0% | — |
| funcgroups8 | 15 | 0 | 0 | 0.0% | — |
| funcgroups8 | 12 | 15 | 15 | 0.1% | — |
| funcgroups8 | 8 | 5_149 | 2_526 | 13.0% | 20.48 |
| hp_lehninger2 | 24 | 220 | 193 | 1.0% | 1.09 |
| polarity4 | 8 | 759_871 | 16_549 | 85.2% | 10.80 |

No setting puts a single P66 region below E = 1. Half of P66's regions at
hp_lehninger2 k=24 have no E-value at all: their own two spans match at or past
the composition boundary, which is what a β-barrel's alternating hydrophobic and
polar strands do.

P66 shares **no k-mer at all** with any integrin in 20 pair tests across four
alphabet and k combinations, while a P66-against-itself control returns 574.
Integrins appear only at `polarity4` k=8, which matches 85% of the proteome —
saturation, not detection, and they rank worse there than the average hit.

CD47 behaves the same way. It is found only at k ≤ 20 in hp_lehninger2, k ≤ 12 in
polarity4 and k ≤ 8 in funcgroups8, never in protein20. Against 300 random human
proteins of similar length, CD47 sits at the 91st percentile of shared k-mers at
k=18 and the 92nd at k=20, with 26 and 24 random proteins matching or beating it.
It is an ordinary protein by this measure, not a hit.

## Running it

```bash
KS=$HOME/code/kmerseek-ka-lambda-region/target/release/kmerseek
python3 build_labeled_benchmark.py --kmerseek $KS --outdir out
python3 score_ranking_metrics.py --pairs out/labeled_pairs.parquet
bash p66_alphabet_ladder.sh $KS out
```

Use the Python in `2025-kmerseek-analysis`. The kmerseek build is the
`olgabot/ka-lambda-per-region` branch, which solves the Karlin-Altschul lambda
from each region's own composition; earlier builds give every region of a pair
the same lambda and score compositionally biased stretches as evidence.
