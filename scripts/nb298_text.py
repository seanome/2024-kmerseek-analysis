"""Markdown cells of notebook 298 (read by scripts/make_nb298.py)."""

PR141 = "https://github.com/seanome/2024-kmerseek-analysis/pull/141"
PR143 = "https://github.com/seanome/2024-kmerseek-analysis/pull/143"
NB295 = "295_convergence_search_ice_binding_proteins_swissprot.ipynb"
PRIOR_ART = (
    "https://github.com/seanome/2024-kmerseek-analysis/blob/"
    "olgabot/linked-pairs-kill-test/notebooks/294_prior_art.md"
)

VERDICT = """**Verdict: KILL at stage 1. No search was run.** The setting chosen on four lineages (the
20 amino acids, 4-letter words, every pair, spacing // 3 in the key) reaches 4 of 55
ice-binding proteins from the held-out AFGP query, fewer than the best exact k-mer
(6 of 55), and 0 of 40 from the held-out fish type III query, the same as the best exact
k-mer. The rule needed 10 points more on both. Under the 1% random-share bar, no
hydrophobic-polar or threonine key shares 2 or more keys with more than 1 ice-binding
protein of another lineage, from any query. The matches that do exist are proteins that share an ancestor
(the bacterial query and five fungal proteins of UniProt's "Ice-binding protein family")
or alanine and threonine runs (AFGP against fish type I)."""

INTRO = f"""
# 298. Could linked word pairs reach ice-binding proteins of other lineages where k-mers could not?

{VERDICT}

**The question.** [Notebook 295]({NB295}) section 9 ([PR 141]({PR141}), figure SF6)
counted, for each ice-binding query, how many ice-binding proteins of other lineages share
at least 2 exact k-mers with it, against the share of 5_000 random Swiss-Prot entries that
do. In the hydrophobic-polar alphabets the two shares were the same at every k from 3 to
40. kmerseek reports nothing below 2 shared k-mers (`--min-shared-kmers 2`), so that count
is a ceiling on any kmerseek setting.

A linked word pair is two short exact words plus the number of residues between them,
hashed as one search key. DNA tools call it a strobemer (Sahlin 2021). It changes what
counts as a shared key, so it is the one change that could move that ceiling. kmerseek has
no index for it, but no index is needed: the same direct count runs on pair keys.

**The decision this informs.** Whether a pair-key index in kmerseek (Rust, about a week of
work) is worth building for proteins that share a pattern without sharing an ancestor.
This notebook builds nothing in kmerseek and runs no search unless stage 1 passes.

**Prediction, written before the run.** (a) With the spacing stored in the key, the key
carries the query's repeat period. The published repeat lengths differ by lineage (3
residues for AFGP, 11 for fish type I, 15 for spruce budworm; `known_motifs.tsv` in the
convergence-afp pipeline), so a stored spacing will not carry across lineages. (b) With
the spacing left out, the key is two exact words, and section 9 found that exact words of
3 to 40 letters carry no cross-lineage signal. Expected: KILL at stage 1 for every key
family.

**Decision rule, fixed before the run.** One setting (alphabet, word length, key family)
is chosen on the queries of lineages 1, 3, 5 and 7 (fish type I, fish type II, insects,
bacteria): the highest mean reach at 2 shared keys among settings whose random share is
at most 1% on each of those four queries. Reach is the share of ice-binding proteins of
other lineages that share at least 2 keys with the query; random share is the same for
the 5_000 random Swiss-Prot entries. The 1% bar is read off SF6. The chosen setting is
then reported on lineages 2 and 4 (fish AFGP, fish type III), which played no part in
choosing it. Stage 2, a ranked search of Swiss-Prot, runs only if on both of those:

* (i) the chosen setting reaches at least 10 percentage points more than the best exact
  k-mer setting with random share at most 1%, read from section 9's table
  (`kmer_reach_ceiling.tsv`); and
* (ii) the real query reaches at least 10 points more than the median of its 20 null C
  draws, and more than the medians of its null A and null B draws.

Otherwise the verdict is KILL and no search runs. Higher reach is better; reach equal to
the random share means the key finds ice-binding proteins no more often than any protein.

[Notebook 294]({PR143}) tested the same keys for homology on Pfam seed pairs (+6.7 points
of recall at 20-30% identity, short of its 10-point bar, explained by a second extended
segment). Nothing here revisits that. This notebook asks a different question, a shared
pattern without shared ancestry, on a different truth set. It also gives PR 143 the number
it listed as untested: how often keys match by chance across a protein database (SF2).
Prior work on pair keys is reviewed in [notebook 294's prior-art note]({PRIOR_ART}) and is
not repeated here.

**Data.** Queries: the 6 real queries of the convergence-afp run (one per lineage), its
own 20 null A and 20 null B draws per query (`results/inputs/draws.tsv` of the run on
Sherlock, 14 Oct 2026), and 20 new null C draws per query: 366 queries. Targets: the 59
reviewed ice-binding proteins of the truth set (a query's own lineage is never counted)
and the same 5_000 random Swiss-Prot entries section 9 used. Swiss-Prot release 2026_03,
575_748 entries, file `~/data/uniprot/uniprot_sprot.2026_03.fasta.gz`, md5
`bc9d398533e6df582b563c6c03093bd0`. No kmerseek call, no index, no search: every number
is a count of distinct keys (`scripts/convergence_strobemer_reach.py`).

**Keys.** Two words of `a` letters in one alphabet, with `g` residues between the end of
the first and the start of the second. Five ways to build them:

| key family | which pairs | key |
|---|---|---|
| every pair, spacing in key (`pair_gap`) | every pair with g = 1 to 40 | word 1, g, word 2 |
| every pair, spacing // 3 in key (`pair_gapbin3`) | the same | word 1, g // 3, word 2 |
| every pair, no spacing (`pair_nogap`) | every pair with g = 0 to 32 | word 1, word 2 (notebook 294's key when a = 6) |
| one pair per position, spacing in key (`randstrobe_gap`) | one second word per position, picked by notebook 294's hash rule among g = 0 to 32 | word 1, g, word 2 |
| one pair per position, no spacing (`randstrobe_nogap`) | the same | word 1, word 2 (notebook 294's randstrobe) |

`pair_nogap` and the two `randstrobe` families keep notebook 294's spacing window, 0 to 32,
so that its keys are rebuilt as they were; `pair_gap` and `pair_gapbin3` use 1 to 40 as the
plan set out. Word codes and the randstrobe choice come from notebook 294's
`linked_pairs_utils.py`, copied unchanged from PR 143 (commit dea19e6).

Alphabets: the eight section 9 used, read by name from
`scripts/convergence_kmer_reach_ceiling.py` (the 20 amino acids, Dayhoff, and six
hydrophobic-polar alphabets; the plan and notebook 295's text say seven, but the script
has six), plus `thr2`, threonine against every other residue. The three published repeats
in `known_motifs.tsv` are all threonine-spaced (`(?:AAT){{3,}}`, `T.{{10}}T.{{10}}T`,
`[TVIR].T(?:.{{12}}[TVIR].T){{2,}}`), and ice-binding sites are flat, extensive and
relatively hydrophobic on folds that evolved separately (Davies 2014). Word lengths: 3, 4,
6, 8 and 10 for the two-letter alphabets, 3 to 6 for Dayhoff, 2 to 4 for the 20 amino
acids. That is 42 alphabet and word-length pairs and 210 settings (the plan counted 47 and
235 from seven hydrophobic-polar alphabets).
"""

NULL_C = """
## 1. Null C: shuffling residues within each phase of the repeat

Null C keeps the letter mix at every phase of the repeat: residues are shuffled only among
positions with the same remainder after dividing by the period p. A key whose two words sit
one period apart survives this by construction. What null C tests is whether a key reaches
targets through anything other than the period.

p is the repeat length the convergence-afp pipeline used for null B (`unit_length` in
`draws.tsv`): the published repeat length for AFGP (3), fish type I (11) and spruce
budworm (15), and for the other three the lag with the largest hydrophobic-polar
autocorrelation, from 2 to 30, as `bin/prep_inputs.py` computes it. The plan named
`scans.py`'s period scan; the period that defined null B comes from `prep_inputs.py`, so
that one is reused. Draws: seed 20261018, saved as `298_inputs/null_c_draws.fasta` beside
null A and B.

The AFGP query, P02732, is 31 residues of a perfect Ala-Ala-Thr repeat. With p = 3
every phase holds one letter, so all 20 null C draws are the query itself. For AFGP, part
(ii) of the rule cannot pass whatever the key: the rule is kept as written and this is
reported, not repaired.
"""

SELECTION = """
## 2. Choosing one setting on four lineages

The rule from the first cell, applied to the real queries of lineages 1, 3, 5 and 7 at 2
shared keys. The table below ranks every setting: eligible ones (random share at most 1%
on each of the four queries) first, then by mean reach. Below it, the full grid: every
setting against every real query, written also to `298_grid_real_queries_2_shared_keys.tsv`.

11 of the 210 settings pass the 1% bar on all four queries: six with the 20 amino acids,
one with Dayhoff and four hydrophobic-polar ones, all of those with 10-letter words, one
pair per position and the spacing in the key. Every two-letter key with shorter words
fails the bar on at least one of the four queries. In Dayhoff, alanine and threonine are one class, so the AFGP query has a
single distinct key in 12 Dayhoff settings and can never share 2: those rows read 0 of 55.
"""

F1 = """
## 3. F1: reach against the random share, every setting

One panel per real query. Each point is one setting at 2 shared keys: across, the share of
5_000 random Swiss-Prot entries that share at least 2 keys with the query; up, the share of
other-lineage ice-binding proteins that do. Grey points are section 9's exact k-mers. A
useful key sits high and to the left of the dashed line at 1%. A point on the dotted line
finds ice-binding proteins no more often than any protein. The ring marks the setting the
rule chose. Lineages 2 and 4 are held out: they played no part in the choice.
"""

DECISION = """
## 4. The decision

The rule's numbers for the two held-out lineages. The best exact k-mer is read from
section 9's table with the same 1% bar. AFGP: the chosen key reaches 4 of 55 (7.3%), the
best exact k-mer (20 amino acids, k = 4) 6 of 55 (10.9%), a margin of -3.6 points; the
residue-shuffled copies (null A) reach a median of 8 of 55 (14.5%), more than the real
query. Fish type III: 0 of 40 for the chosen key and for every exact k-mer under the bar.
Both parts of the rule fail on both lineages.

The chosen setting's selection score, a mean reach of 3.3%, comes almost entirely from the
bacterial query (6 of 55; type I adds 1 of 47, the other two 0). Five of those six are
fungal ice-binding proteins of the bacterial query's own UniProt family, which the truth
set labels "8 other" and so counts as another lineage (section 7). The rule therefore chose
a setting by how well it finds relatives of the query, not a shared pattern. This does
not change the verdict, but a future truth set should not count proteins of the query's
own family as another lineage.

Choosing a setting by looking at the held-out queries, which the rule forbids, would not
change this: the best setting under 1% on AFGP reaches 10 of 55 (the 20 amino acids,
3-letter words, every pair, no spacing), 7.3 points above the best exact k-mer, still short
of 10, and on fish type III no setting under 1% reaches any target.
"""

SF1 = """
## 5. SF1: the chosen setting against its own null draws

For the chosen setting, each real query's reach (black cross) beside its 20 null A draws
(residues shuffled), 20 null B draws (the repeat period kept, the word inside each repeat
reordered) and 20 null C draws (residues shuffled within each repeat phase). A real query
level with its draws means the setting's reach does not depend on the query's order.
"""

SF2 = """
## 6. SF2: how often keys match by chance, and what that means for a database

Over the 6 real queries: the mean number of keys a random Swiss-Prot entry shares with the
query (top) and the share of random entries sharing at least 2 (bottom), by word length,
one line per alphabet and one panel per key family. Multiplied by Swiss-Prot's 575_748
entries, the bottom row is how many entries a search would have to rank before any filter
on score. This is the chance-match number [PR 143](https://github.com/seanome/2024-kmerseek-analysis/pull/143)
listed as untested; the table is `data/convergence_afp/298_strobemer_chance_cost.tsv`, and
notebook 294's key (hp_thomas_dill2, a = 6, every pair, no spacing) is ringed.
"""

RESIDUES = """
## 7. The residues behind the shared keys

The chosen setting reaches 11 pairs of a query and an ice-binding protein of another
lineage, from three of the six queries. All are printed below; the reading is:

* Bacterial query H7FWB6 against five fungal proteins (B8XC03, B8XC04, C7F6X3, Q76CE6,
  Q76CE8). UniProt puts the query and all five in its "Ice-binding protein family". Over
  the stretch from the first word to the second, where the spacing is equal in both, the
  median identity per pair is 54 to 66% (for example 25 of 42 residues with B8XC03).
  This is ancestry, which notebook 295 found too (its F3), not a pattern reached without
  it.
* Bacterial query H7FWB6 against P05140 (fish type II): 2 shared keys, `KILK` and `ETTA`
  about 31 residues apart in both, in otherwise unrelated sequence. One chance match at
  the bar.
* AFGP query P02732 against four fish type I proteins (B1P0S1, P02734, P07835, P23699),
  and the type I query P04002 against the AFGP polyprotein P24856: every shared word is
  made of alanine and threonine (`AATA`, `ATAA`, `TAAT`), plus proline in the type I
  pair (`AAPA`, `APAA`). Both lineages are alanine-rich with threonine at intervals. The
  AFGP query's residue-shuffled copies (null A) reach more ice-binding proteins of other
  lineages than the real query does (SF1), so the order of the residues adds nothing to
  these matches.

Each key is printed as the stretch from the first word's start to the second word's end,
in both sequences, with coordinates (1-based, inclusive), the alphabet's classes
underneath, and `^` under the two words. Only the words must match; the residues between
them are free. Up to 6 keys per pair are shown.
"""

CHECKS = """
## 8. Checks after the verdict

(1) Could a spacing above 40 have passed? A key with words of `a` letters stores a repeat
of period P only if P - a is within the spacing range, so periods up to 50 residues can be
stored with 10-letter words and up to 44 with the chosen 4-letter words. The published
periods (3, 11 and 15 residues) and the periods used for the six queries' null B (3 to 29)
are all inside that. When the period search is allowed to run to half the protein's length
instead of 30, 17 of the 59 ice-binding proteins peak above 40, two of the queries among
them (A0ZT93 at 54, H7FWB6 at 105). These long-lag peaks are weak (autocorrelation 0.14 to
0.40) and rest on fewer overlapping positions. A spacing above 40 was not tested.

(2) Length is not what limits reach: in the worst setting at most 3 of a query's
other-lineage targets are too short to hold 2 distinct keys, and in the best setting none
are.

(3) thr2, threonine against everything else: no thr2 setting keeps the random share under
1% on all six queries (0 of 25). Under the bar, the best any thr2 setting does on any one
query is 1 of 55 targets (AFGP, 8-letter words, spacing // 3, random share 0.3%).
"""

CONCLUSIONS = """## Summary and conclusions

**Claims the data supports.**

1. Linked word pairs do not reach ice-binding proteins of other lineages more often than
   exact k-mers do on the held-out lineages: 4 of 55 against 6 of 55 for AFGP, 0 of 40
   against 0 of 40 for fish type III, both at a random share of 1% or less (section 4).
2. Under the 1% random-share bar, no hydrophobic-polar or threonine key shares 2 or more
   keys with more than 1 ice-binding protein of another lineage, from any of the six
   queries (F1, section 8). At 1 shared key, one such setting reaches 3 (AFGP query).
   Prediction (b) held. Prediction (a), that a stored spacing would not carry across
   lineages, is consistent with the data: the spacing-in-key families reach no more than
   the families without it.
3. The few cross-lineage matches the chosen key makes are shared ancestry (the bacterial
   query and five fungal proteins of one UniProt family, a median 54 to 66% identical
   between the two words) or alanine and threonine runs, where residue-shuffled copies
   of the query reach more (section 7, SF1).
4. Notebook 294's key (hydrophobic-polar, `hp_thomas_dill2`, 6-letter words, every pair,
   no spacing) is shared 2 or more times by 95% of random Swiss-Prot entries, a mean of
   981 shared keys per random entry, so about 546_000 of Swiss-Prot's 575_748 entries for
   a query like these (SF2, `298_strobemer_chance_cost.tsv`). Used as a filter, it removes
   almost nothing.

**Claims the data does not support.**

1. That pair keys cannot carry any pattern across lineages. Only ice-binding proteins
   were tested, with 6 queries, 40 to 56 targets each, and the AFGP query is a 31-residue
   repeat that null C cannot shuffle.
2. That the chance cost in SF2 is the same for every query. It is a mean over 6 short,
   repetitive ice-binding queries (31 to 276 residues); a typical Swiss-Prot query will
   give other numbers.
3. Anything about spacings above 40 residues, or about keys of three or more words.
4. Anything about notebook 294's homology result, which this notebook does not revisit.

**The figure.** F1, `figures/298_F1_linked_pair_reach_vs_random_swissprot_2026_03.png`:
on every query the 210 linked-pair settings sit on the same curve as the exact k-mers of
notebook 295, and none sits high and left of the 1% line on a held-out query.

**Next smallest experiment.** Not another key on this truth set: the 1% bar leaves at most
one target for any hydrophobic-polar key, and what the 20-letter keys reach is ancestry.
The next one is kmerseek's own `--chain-max-shift` on the notebook 232 Pfam seed pairs,
the comparison notebook 294 left untested. It asks the homology question on data that
already exists, with a flag kmerseek already has (`src/rust/main.rs` on `main`), so it
needs no new index and no new truth set. The venom cysteine-framework test asks a pattern
question again, and its first run (PR 145, notebook 297) is already under way.

## References

Davies, P. L. "Ice-Binding Proteins: A Remarkable Diversity of Structures for Stopping and
Starting Ice Growth." *Trends in Biochemical Sciences* 39, no. 11 (2014): 548–55.
https://doi.org/10.1016/j.tibs.2014.09.005.

**Note:** checked against PubMed (PMID 25440715) on 18 Oct 2026. Cited for one sentence:
ice-binding sites are flat, extensive and relatively hydrophobic, on folds that evolved
separately.

Sahlin, K. "Effective Sequence Similarity Detection with Strobemers." *Genome Research* 31,
no. 11 (2021): 2080–94. https://doi.org/10.1101/gr.275648.121.

**Note:** the reference as resolved in notebook 294's prior-art note. Defines the linked
word pairs (strobemers) used here.
"""
