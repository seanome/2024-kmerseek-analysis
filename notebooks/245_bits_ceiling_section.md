# Why an exact reduced-alphabet seed cannot rank a remote homolog above chance

*Draft section for Botvinnik et al., in preparation. Every number is computed or read back in
notebook 245, which says for each one whether it was measured, is a table value, or comes
from a model. Figures are `figures/245_*.png`.*

Since the 1990s, papers have reported that the hydrophobic/polar (HP) pattern of a protein
is conserved between remote homologs, and several have proposed searching for homologs by
that pattern. The claims stayed anecdotal. Here we give the reason in one inequality. On
average, a remote homolog does not carry enough information for an exact seed in any
reduced alphabet to stand out in a search of a whole proteome. The limit is on the
information, so no choice of alphabet and no ranking metric removes it. It also says what
a method has to do instead.

## The pair we follow

We use one pair throughout: human BCL2 against *C. elegans* CED-9, two apoptosis
regulators in the same SCOP superfamily with under 30% identity. The BH1 region aligns
without gaps, BCL2 130–166 against CED-9 154–190 (Figure 4):

```
BCL2      130 FATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREM 166
                              ||      |||              5 of 37 identical
CED-9     154 RTVGNAQTDQCPMSYGRLIGLISFGGFVAAKMMESVE 190

hp_lehninger2 classes (h = AFGILMPVWY, p = the other residues)
BCL2      130 hhphhpphhpphhphhphhhhhphhhhhphpphppph 166
                 |  |  ||||||||||||||||||| || |||     26 of 37 same class
CED-9     154 pphhphppppphhphhphhhhhphhhhhhhphhpphp 190
                       ^^^^^^^^^^^^^^^^^^^            longest class-identical run: 19,
                                                      BCL2 139–157, CED-9 163–181

hp_thomas_dill2 classes (h = ACFILMVWY, p = the other residues)
BCL2      130 hhphhpphhppphphpphhhhhphpphhhhpphppph 166
                    |  | |||||||| ||||||||||| |||     24 of 37 same class
CED-9     154 pphpphpppphphphpphhphhphpphhhhphhpphp 190
                                  ^^^^^^^^^^^         longest class-identical run: 11,
                                                      BCL2 150–160, CED-9 174–184
```

Five of 37 residues are identical, but 26 of 37 positions have the same class in the
Lehninger HP alphabet, including a run of 19 in a row. A structure alignment of the two
AlphaFold models (USalign) puts 25 of these 37 positions on the same diagonal, BCL2 140–164.
If HP conservation were enough to find homologs, this pair would be found.

## Step 1. Bits per aligned position, and why merging letters can only lose them

The information one aligned position carries about relatedness is the mutual information
between the class in the query and the class in the target. For an alphabet $A$, in bits
per aligned position:

$$I_A = \sum_{a,\,b} \Pr(a, b)\,\log_2 \frac{\Pr(a, b)}{\Pr(a)\,\Pr(b)} \qquad (1)$$

- $\Pr(a, b)$: the chance that a true homologous pair has class $a$ in the query and class $b$ in the target at one aligned position
- $\Pr(a)$, $\Pr(b)$: the class shares of the query and of the target

$I_A$ is the expected log-likelihood ratio, per position, between "these residues descend
from one ancestral residue" and "these residues are unrelated". So it is the most any
scoring scheme can collect from one position on average; a log-odds matrix reaches it
(Altschul 1991).

A reduced alphabet is a fixed map $\phi$ from the 20 amino acids to classes. For aligned
residues $X$ and $Y$,

$$I_A = I\bigl(\phi(X);\,\phi(Y)\bigr) \;\le\; I(X;\,Y) = I_{20} \qquad (2)$$

This is the data-processing inequality (Cover and Thomas 2006). It needs no model of
evolution and holds for every reduced alphabet at once. Merging letters cannot create
information about relatedness. It can only discard it.

We take $I_{20} = 0.38$ bits per position, the relative entropy of BLOSUM45 (Henikoff and
Henikoff 1992), the matrix built for this range of identity. This is a table value, not a
measurement on our pairs. For two letters we also compute a model value. Part 2 of the
kmerseek documentation describes a true pair by its copy rate $\kappa$ (Cohen's κ). At each
position the target copies the query's class with probability $\kappa$. Otherwise it draws
a class from the class shares. For hp_thomas_dill2, κ = 0.463 was measured on 37,085 Pfam
seed pairs at 20–30% identity, and the class shares on the human proteome are $h = 0.39$ and
$p = 0.61$. With these, Equation (1) gives $I_\text{HP} = 0.157$ bits per position.

## Step 2. Bits in a conserved block

A block of $L$ aligned positions carries, on average,

$$\text{bits}_\text{block} = L \cdot I_A \;\le\; L \cdot I_{20} \qquad (3)$$

For the 37-residue BH1 block, $37 \times 0.38 = 14.1$ bits at the 20-letter ceiling.
Two letters under the copy-rate model give $37 \times 0.157 = 5.8$ bits.

## Step 3. Bits a hit needs

Karlin and Altschul (1990) showed that between unrelated sequences, the expected number of
ungapped regions scoring at least $S$ is $E = K\,m\,n\,e^{-\lambda S}$. With the score in bits,
$S_\text{bits} = \lambda S/\ln 2$, a region reaches $E$ only if

$$\text{bits}_\text{needed}(E) = \log_2\frac{K\,m\,n}{E} = \log_2 K + \log_2 m + \log_2 n - \log_2 E \qquad (4)$$

- $K$: the share of the $m \cdot n$ query-by-target position pairs that can start an independent region
- $m$: query length in residues (CED-9, 280)
- $n$: residues in the database

Each doubling of the database, or of the query, adds one bit to the cost. On a proteome the
size of the human one ($n \approx 1.1 \times 10^7$ residues, $K = 0.018$), $E = 1$ costs
$\log_2(0.018 \times 280 \times 1.1 \times 10^7) = 25.7$ bits. It costs 24.0 bits on SCOPe40,
29.9 on Swiss-Prot and 35.3 on UniRef50 (Figure 2).

An exact seed faces a second threshold. An exact class match of $k$ positions between
unrelated sequences occurs at one query-target position pair with probability
$\Pr(\text{match} \mid \text{unrelated})^k$, if positions are independent. There are $m \cdot n$
such pairs, so a seed is expected less than once by chance only if

$$-k\,\log_2 \Pr(\text{match} \mid \text{unrelated}) \;\ge\; \log_2(m\,n) \qquad (5)$$

For CED-9 against the human proteome, $\log_2(m\,n) = 31.5$ bits. $K$ does not appear in
Equation (5), because every position pair is a place a seed could sit.

## Step 4. The gap

The BH1 block carries at most 14.1 bits; a region needs 25.7 (Figure 1a). The ceiling
bounds the average over true pairs, not every pair. A single pair can beat its expected
value, and BCL2/CED-9 does: its 19-residue exact run in hp_lehninger2 is worth
$19 \times -\log_2 0.501 = 18.9$ bits of rarity. That is still 12.6 bits short of the 31.5 an
exact seed needs (Figure 1b). About 6,100 chance matches that rare are expected in one
search of the human proteome.

The search agrees. Notebook 241 searched CED-9 against the 19,732 canonical human proteins
with all 19 alphabets kmerseek supports, at seed lengths chosen to carry 16 to 44 bits.
BCL2's best rank over every alphabet, seed length and ranking metric was 213 of the 18,064
proteins hit (polarity4, mean IDF, k = 9). Its best E-value was 1,955, 10.9 bits short of E = 1.
A human protein picked at random reaches a best rank at least as good in 97% of 20,000
draws over the same hit lists. The proposed partner of *Borrelia* P66, CD47, fares no
better: best rank 166 of 18,775, best E-value 707, beaten by a random protein in 90% of
draws.

Combining alphabets does not help. If every alphabet ranks the same proteins above the
partner, any combination that rises with each alphabet's score ranks them above it too.
That is what notebook 242 found. Under mean IDF, about 1,200 proteins beat BCL2 in at
least half of the 13 alphabets that see it, 7.9 times what independent alphabets would give.
Combining all 19 alphabets moved BCL2 from rank 213 to between 7,148 and 13,343.

## Step 5. Why exact seeds lose the signal: conservation is not reachability

The HP pattern is conserved. At 20–30% identity, hp_thomas_dill2 has κ = 0.463 on Pfam seed
pairs and 0.433 on SCOPe pairs from different families of one superfamily. The 20-letter
alphabet on the same Pfam pairs has κ = 0.199.

Conservation is a per-position quantity. An exact seed needs something else: $k$
consecutive same-class positions. If positions were independent, a given window of $k$
positions would be all same-class with probability $\Pr(\text{same})^k$, which falls
exponentially with $k$ while κ does not move. On the same 37,085 Pfam pairs the mean
longest exact class-identical run is 13.0 residues (6.3 with the target shuffled). Only 3.8%
of pairs share an exact 23-mer, 1.7% a 26-mer and 0.6% a 30-mer (Figure 3). Shorter seeds
reach more pairs but carry fewer bits, and Equation (5) then fails by a wider margin.

## Step 6. What the inequality does not forbid

The inequality is about one exact seed and one ungapped block, on average, against a whole
proteome. Three things lie outside it.

- **High identity.** At 50–90% identity a short block carries many more bits per position.
  Notebook 244 found 87 cases where kmerseek places a human Swiss-Prot feature under 60
  residues in another species and no structure tool does. Identity of 50–90% was one of the
  selection criteria. An example is human COL9A1 "Collagen-like 7" in chicken, 25 of 32
  residues identical in the matched region.
- **Seeds that tolerate mismatches.** Notebook 232 starts from a 10-residue HP seed and extends
  it without gaps, allowing class changes. At a false-positive rate of 0.001 per comparison it
  reaches 35.5% of Pfam pairs at 20–30% identity and 63.5% at 30–40%. That rate is per pair,
  against a shuffled copy of the pair. The rate per query against a whole database has not
  been measured.
- **Smaller search spaces.** Equation (4) falls by one bit per halving of $n$, so the
  same block can suffice against one family or one proteome.

None of these adds bits to a 20–30% identity block. They change which blocks a method
looks at, how much of each block it scores, and how large the search space is.

## Step 7. Caveats about the E-value itself

- **Many arms had no fit.** 81 of 152 arms in notebook 241 ended without a Karlin–Altschul
  fit (an arm is one alphabet at one seed length). Above about 30 bits per seed, too few
  chance regions remain to fit.
- **The best-matching regions lose their E-value first.** kmerseek solves λ from each region's
  own two sequences. A region whose own identity is above $C/(1+C)$ gets λ = 0, and so no
  E-value. With the κ-derived penalties, that removes 35% of hp_lehninger2 regions and 88–98%
  of regions in the 12- to 20-class alphabets.
- **Two alphabets have no λ at any k, and a third is at the edge.** λ exists only if one
  position between unrelated sequences scores below zero on average, that is, when
  $\Pr(\text{match} \mid \text{unrelated}) - C \cdot (1 - \Pr(\text{match} \mid \text{unrelated})) < 0$.
  This average is +0.20 points for gbmr7 and +0.03 for hp_lehninger_hpc3, so neither can
  give an E-value. gbmr4 sits at −0.0003, just below zero, so its λ exists but is close to 0
  (Figure 5). All three come from a mismatch penalty derived for equal class shares. On the
  human proteome the classes are far from equal.

## What this contributes

For thirty years, HP-pattern homology has rested on striking single cases like BH1. Here
we show why such cases did not become a search method. Merging amino acids into classes
can only lose information. And an exact seed, the unit a k-mer index is built on, reaches
fewer remote homologs the longer it gets. A method that wants remote homologs at proteome
scale needs more information per position than any sequence alphabet has. It can also
look at longer stretches than an exact seed reaches, or search a smaller space. This fits
the field's current view that more structural information buys more sensitivity (Sahakyan
et al. 2026). It also fits the finding that a three-letter secondary-structure alphabet can
match the 3Di structure alphabet in remote homology detection (El-Hendi et al. 2026). Both
alphabets carry structure, not sequence.

## Figures

- **Figure 1** (`245_bits_have_vs_need.png`). Blue: bits the BCL2/CED-9 pair has, or can
  have at most. Red: bits the search needs. (a) The BH1 block at the 20-letter ceiling and
  under the two-letter copy-rate model, against $E = 1$ for a scored region. (b) The rarity
  of the longest exact hp_lehninger2 run, against one chance seed per search.
- **Figure 2** (`245_bits_needed_vs_database_size.png`). Bits needed against database size.
  Red dots: $E = 1$ for a scored region, $\log_2(K \cdot m \cdot n)$, one per database with its
  own $K$. Red dashed line: one chance exact seed per search, $\log_2(m \cdot n)$. Blue lines:
  the BH1 block.
- **Figure 3** (`245_conservation_vs_reach.png`). Black: share of Pfam pairs at 20–30%
  identity whose longest exact class-identical run is at least $k$. Grey dashed: κ for the
  same pairs, which does not depend on $k$.
- **Figure 4** (`245_bh1_alignment.png`). The BH1 residues, identity line and class strings
  in two HP alphabets, with the longest class-identical run boxed.
- **Figure 5** (`245_chance_score_by_alphabet.png`). Expected score of one position between
  unrelated sequences, per alphabet. Hatched black: at or above zero, no λ at any $k$.

## Numbers this draft does not yet have

- $I_{20}$ measured on the notebook 230 pairs, in place of BLOSUM45's table value.
- $K$ fitted on the human index, in place of the $K$ copied from Swiss-Prot. It is in notebook
  241's `arms.csv`, which is not in the repository.
- The residue count of the 19,732 GENCODE v49 proteins, in place of the docs' approximate
  $1.1 \times 10^7$.
- El-Hendi et al.: the title and DOI were read from bioRxiv; the full author list could not
  be retrieved, so the entry below is incomplete.

## References

Karlin, S., and S. F. Altschul. "Methods for Assessing the Statistical Significance of
Molecular Sequence Features by Using General Scoring Schemes." *Proceedings of the National
Academy of Sciences of the United States of America* 87, no. 6 (1990): 2264–68.
https://doi.org/10.1073/pnas.87.6.2264.
> **Note:** The source of $E = K\,m\,n\,e^{-\lambda S}$ and of the condition that λ exists only when the expected chance score is negative (Equations 4 and 5, Step 7).

Altschul, S. F. "Amino Acid Substitution Matrices from an Information Theoretic
Perspective." *Journal of Molecular Biology* 219, no. 3 (1991): 555–65.
https://doi.org/10.1016/0022-2836(91)90193-a.
> **Note:** Any substitution matrix is a log-odds matrix; its expected score per position on true pairs is the relative entropy, the most a score can collect (Step 1).

Henikoff, S., and J. G. Henikoff. "Amino Acid Substitution Matrices from Protein Blocks."
*Proceedings of the National Academy of Sciences of the United States of America* 89, no. 22
(1992): 10915–19. https://doi.org/10.1073/pnas.89.22.10915.
> **Note:** The BLOSUM matrices; the 0.38 bits per position used here for $I_{20}$ is BLOSUM45's relative entropy.

Cover, T. M., and J. A. Thomas. *Elements of Information Theory*. 2nd ed. Hoboken, NJ:
Wiley-Interscience, 2006. https://doi.org/10.1002/047174882X.
> **Note:** The data-processing inequality, Equation (2).

Sahakyan, H., P. Mutz, V. Tobiasson, and E. V. Koonin. "Exploring the Protein Universe with
Distant Similarity Detection Methods." *Protein Science* 35, no. 1 (2026): e70397.
https://doi.org/10.1002/pro.70397.
> **Note:** Review of distant-homology methods, cited for the view that sensitivity rises with the structural information a method uses.

El-Hendi, [authors to be completed]. "Structural Alphabets Approach Performance of Structural
Alignment in Remote Homology Detection." *bioRxiv* (2026).
https://doi.org/10.64898/2026.01.16.699908.
> **Note:** Reports that the Q3 and Q8 secondary-structure alphabets approach 3Di in remote homology detection, and on the redundant CATH set outperform it. The author list could not be verified.
