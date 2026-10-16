# Can a tool's weak hits be trusted?

When a tool's best hit on a protein is weak (E between 1 and 10), how often is it right?
[PR 133](https://github.com/seanome/2024-kmerseek-analysis/pull/133) answered this for
phmmer, jackhmmer, MMseqs2 and one kmerseek setting per species. This folder extends it to
every kmerseek setting of the dark-set midi run (`runs/ladder-0.4-midi`), splits every best
hit into E-value bands, and adds a stricter check that looks at positions on the query.

## Rules, written 2026-10-16 before any new result was read

What had been seen before these rules were written: PR 133's table (dark proteins at
E <= 1: human 1,071 with a known Pfam family, phmmer 24 right / 716 wrong at E 1-10,
kmerseek hp_lehninger2 k=17 0 right / 270 wrong at region E <= 1; worm, yeast and
*Borrelia* rows; placed proteins 69-99% right), its perilipin-4 spot check, and the file
counts and sizes of the region files. No count from the stricter check, no count for any
other kmerseek setting, and no count split by band had been seen.

**Species.** Human, *C. elegans* (worm), *S. cerevisiae* (yeast), *B. burgdorferi*
(*Borrelia*). *Botryllus* is left out: it has no Pfam annotation.

**Sources.** phmmer, jackhmmer, MMseqs2 (the run's own hits files), and every kmerseek
setting the run fitted (listed in `by_combo` of `<species>_kmerseek_dark_gain.json`).
A setting the run could not fit has no E-values and is counted as not run.

**Best hit.** For each protein and source, the hit with the smallest E-value (for
kmerseek, the region with the smallest `region_evalue`). Ties keep the first row, as in
PR 133.

**Bands** of the best hit's E: at most 0.001; above 0.001 up to 1; above 1 up to 10.
A protein whose best hit is above 10 (or that has no hit) is in no band.

**Dark and placed.** Dark at T: no phmmer, jackhmmer or MMseqs2 hit with E <= T. Placed at
T: the rest. T = 1 is the headline; 0.001 and 10 are also written. Only proteins with a
known Pfam family count.

**Two calls of a best hit.** Both use the target's Pfam families (DR lines of its
Swiss-Prot entry) and the query's (Pfam's annotation of UniProt; for *Borrelia*, UniProt's
Pfam cross-references, with positions from the InterPro API).

* Family or clan (PR 133's call): right when the target shares a Pfam family or a Pfam
  clan with the query.
* Same family and same place (the strict call, the headline on the cards): right when
  the target shares a Pfam family with the query AND the stretch of the query that the
  hit covers overlaps, by at least one residue, a Pfam domain of that shared family on the
  query.
* Both have one denominator: best hits whose target has any Pfam family ("judged").
  Targets without Pfam families are counted apart.

**Where the query stretch comes from.** kmerseek writes it (`region_start`, `region_end`,
0-based; converted to 1-based). The sequence tools' hits files keep only target positions,
so each sequence tool's best target is aligned to its query again with phmmer (the run's
own HMMER 3.4 image, `--max`, all domains reported), and the domain whose target stretch
overlaps the tool's target stretch most gives the query stretch. For phmmer this repeats
the same alignment. For jackhmmer and MMseqs2, which align with a profile built over
iterations, it is the stretch that a single-sequence alignment of the same two proteins
gives. When no realigned domain overlaps, the strict call counts the hit as not right,
and the number of such hits is in the column `strict_query_stretch_unknown`.

**Precision** = right / judged, with a 95% Wilson interval, and n (judged) beside it.
A value with fewer than 30 judged hits is shown grey.

**Choosing the kmerseek setting for a card.** Proteins are split once into tune and test
halves by the md5 of the UniProt accession (even = tune). Per species, the setting with
the highest strict precision at E 1-10 on dark tune proteins, among settings with at
least 30 judged hits there, is the card's setting; ties go to more judged hits, then to
the setting name. Its precision is reported on the test half. The sequence tools are not
chosen from anything, but the leaderboard column also uses the test half for them so that
every tool is scored on the same proteins.

**Findable.** A protein is findable when some protein in the clade-excluded reference
carries one of its Pfam families. A protein that is not findable cannot be right. The
count of findable dark proteins is reported per species.

## Predictions, written with the rules

1. The strict call is right less often than the family-or-clan call for every source.
2. On dark proteins at E 1-10, phmmer, jackhmmer and MMseqs2 are right (strict) in under
   5% of judged best hits in every species.
3. The chosen kmerseek setting is right (strict) in under 10% of judged dark best hits at
   E 1-10 on the test half, in every species.
4. On placed proteins at E <= 0.001, MMseqs2 is right (strict) in over 90% of judged
   best hits in every species.

## Steps

See `tools/dark-set-accuracy` (docstring) and `tools/dark-set-accuracy-sbatch`.
