"""Build notebooks/287_sum_statistics_BH4_BH1_chains.ipynb from the cells below.

Run: /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 notebooks/287_build_notebook.py
then execute it with nbconvert (see the PR). The question and decision rule in the first
cell were written on 2026-10-13 before any chain was scored.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "287_sum_statistics_BH4_BH1_chains.ipynb"

INTRO = r"""# 287: does combining kmerseek calls with BLAST's sum statistics rescue Ced9 → BCL2?

**Question.** kmerseek scores each ungapped call (a stretch matched without gaps) alone. On
BCL2, Ced9 has two calls that sit on UniProt's BH4 and BH1 motifs, in the same order on both
proteins ([notebook 284's table](284_ced9_bcl2_family_hits.csv), [PR 125](https://github.com/seanome/2024-kmerseek-analysis/pull/125)):

| motif | Ced9 (1-based) | BCL2 (1-based) | setting | E-value |
|---|---|---|---|---|
| BH4 | 78-99 | 9-30 | wwmj5, k = 9 | 1_955 |
| BH1 | 163-188 | 139-164 | hp_lehninger2, k = 19 | 11_173 |

Neither call alone is better than chance, and they sit on different diagonals (target start
minus query start is -69 for BH4 and -24 for BH1) because BCL2 has 45 more residues than
Ced9 between the two motifs (109 against 64). BLAST combines several such hits into one E-value with sum
statistics. If kmerseek did the same, where would BCL2 rank among human proteins for the
query Ced9, and CD47 for the query P66, and would shuffled copies of the queries score as
well?

**Decision this informs.** Whether to ask kmerseek to combine calls across alphabets (an
issue in seanome/kmerseek), and whether the paper can show Ced9 → BCL2 as a pair kmerseek
finds once its calls are combined.

**Data.** Notebook 241's search ([PR 46](https://github.com/seanome/2024-kmerseek-analysis/pull/46)):
queries Ced9 (280 aa), P66 (597 aa) and BHF (252 aa) against the GENCODE v49 canonical human
proteome (one protein per gene), kmerseek 982a055, 152 alphabet-k settings, every region
kept (`/Users/olga/data/botryllus/alphabet-ranking-three-cases/regions.parquet`). 69 of the
152 settings have a fitted E-value; the other 83 report E = infinity and cannot join a chain.
Coordinates are kmerseek's: 0-based, end-exclusive; tables below print 1-based.

**The formula** (Karlin & Altschul 1993, equation 4; Altschul & Gish 1996 for the order rule;
as NCBI BLAST implements it in `blast_stat.c`, `BLAST_LargeGapSumE`). Each call $i$ gets a
score in natural-log units for this one protein pair, from its own $\lambda_i$ and $K$:

$$y_i = \lambda_i S_i - \ln K - \ln(m\,n_t) = \ln\frac{N}{n_t\,E_i}$$

with $S_i$ its raw score, $m$ the query length, $n_t$ the target length, $N$ the number of
k-mers in the human index ("total k-mers found" when it was built) and $E_i$ kmerseek's
E-value. For $r$ calls in the same order on both proteins,

$$s = \sum_{i=1}^{r} y_i + \ln r!, \qquad P = \Pr(r, s) \approx \frac{e^{-s}\, s^{r-1}}{r!\,(r-1)!}$$

($\Pr(r,s)$ computed from Karlin & Altschul's exact double integral; the right-hand form
holds for large $s$), and the chain's E-value over the whole proteome is

$$E_{\text{chain}} = -\ln(1-P)\;\frac{N}{n_t}\;2^{\,r-1}.$$

$\ln r!$ is the order rule: of the $r!$ orders in which $r$ chance hits can fall, only one is
the same on both proteins. $2^{r-1}$ is BLAST's price for picking a set of $r$ hits out of
all possible sets (gap-decay rate 0.5, BLAST's default for ungapped search). A chain of one
call gets back kmerseek's own E-value.

**Rules, fixed before running (2026-10-13).**

1. *Which calls may join.* Calls with a fitted E-value and a positive score for this pair
   ($y_i > 0$, that is $E_i < N/n_t$, about 44_000 for BCL2). BLAST links only hits above a
   score cutoff; kmerseek's own `chain_regions` takes only positive ones.
2. *Order and gap.* Each call starts after the previous one ends, on both proteins: no
   overlap. The next call starts at most **200 residues** after the previous one ends, on
   each protein. This cap was set after reading that BH4 to BH1 is 64 residues on Ced9 and
   109 on BCL2, so it is not blind to them; 50, 100 and 400 are run too. The formula above
   does not use the cap: it counts each extra call as free to start anywhere on the pair.
   The diagonal may shift by any amount.
3. *Different alphabets in one chain: yes.* Raw scores from two alphabets are in different
   units, but $y_i$ uses each call's own $\lambda_i$ and $K$, so every $y_i$ has the same
   chance distribution. A chain may also be all from one setting; that version (what
   kmerseek's existing `--chain-max-gap` does inside one search, with BLAST's order rule
   here) is reported beside it.
4. *Search space.* Each call is the best of 69 searches (one per setting with an E-value).
   So every call's E-value is multiplied by 69 before it joins a chain ($y_i - \ln 69$), and
   the lowest single E-value is multiplied by 69 too. This is the **main score**. The same
   numbers without the factor of 69 are reported beside it.
5. *A protein's score* is its lowest chain E-value; a chain may be one call.

**Decision rule.** Ced9 → BCL2 is rescued if, with the main score:
(a) BCL2 is in the top 20 of the human proteome and its chain E-value is below 1, and
(b) on dipeptide-shuffled copies of Ced9 (each keeps Ced9's length and the count of every
adjacent residue pair, so it keeps composition and short repeats but has no homology),
BCL2 gets a chain E-value below 1 in at most 5% of copies, and
(c) the main score is calibrated: across shuffled copies, the mean number of human proteins
with chain E-value below 1 is between 0.3 and 3 (a calibrated E-value gives 1).
The same rule for P66 → CD47. Until the shuffled searches exist, (c) is checked on the real
searches: proteins outside the known family (for Ced9: HGNC gene groups 1057, BCL2 family,
and 1277, BH3-only; for P66: CD47; for BHF: none) with chain E-value below 1 should number
about 1 per search and at most 3.

On the Pfam 998 all-against-all, chaining helps if the chain score's average precision is at
least 0.01 higher than the lowest single E-value's.

Lower E-value is better; E-value 1 means one hit this good is expected by chance in a search
of the whole proteome."""

CELLS = [("md", INTRO)]


def build() -> None:
    nb = nbf.v4.new_notebook()
    for kind, src in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            c = nbf.v4.new_code_cell(src)
            c.metadata["jupyter"] = {"source_hidden": True}
            nb.cells.append(c)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbf.write(nb, OUT)


if __name__ == "__main__":
    build()
