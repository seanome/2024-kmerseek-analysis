# Comparing the three tools without a threshold

Design, 2026-09-23. Every flag below was read off the binaries this pipeline runs
(HMMER 3.4, MMseqs2 18.8cc5c, kmerseek 0.4.0), not from memory, and the two measurements
are from the 0.4 midi run's own output.

## The problem

Each tool is clipped somewhere different today, and the clip points are not comparable.

| tool | what the pipeline passes | where it actually stops |
| --- | --- | --- |
| phmmer, jackhmmer | `-E 10.0` | the three acceleration filters (MSV, Viterbi, Forward) and the composition bias filter are all on, and they drop candidates before any E-value is computed |
| mmseqs2 | `-s 7 --num-iterations 3 -e 10.0` | `--max-seqs` is left at its default of 300, so at most 300 of the 572 700 reference sequences survive the prefilter for each query |
| kmerseek | `--threshold 0 --min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3` | reports a match when the whole-query p-value is at most 0.05 **or** the best region scores at least 1.3; the two are an OR, so either one opens the gate |

The HMMER filters are not a formality. Running phmmer on a 47-residue query against a
two-sequence database, the low-complexity target is not reported at all with the defaults;
with `--max` it comes back at E = 0.8. The hit exists, is scored, and is thrown away before
reporting.

So the current setup is not tilted toward kmerseek, which is the point that matters: it is
tilted three different ways at once, and no ranked comparison across the three means
anything until the clip points are matched.

## Three thresholds, not one

The pipeline already separates two of them, and the third is what this design adds.

- **report** — what a tool writes out. Today `--evalue_report 10.0` for the sequence arms.
  Matched across tools by list length, below.
- **call** — what counts as "this protein was found", `--evalue_call 1e-3`. This defines
  the dark set. **It does not change.** Making the reporting deeper cannot shrink or grow
  the dark set, because the dark-set script re-thresholds the stored hits at the call
  cutoff.
- **rank** — the threshold-free comparison: at the same list length, where does each tool
  put the true partner? Nothing is cut; the lists are compared as ranked lists.

Keeping report and call apart is what makes the deeper reporting safe. The headline number
(what fraction of a proteome no sequence search can place) is computed at 1e-3 either way.

## What each tool needs

### phmmer and jackhmmer

Add `--max` and raise both reporting thresholds:

```
phmmer --max -E 1000 --domE 1000 --domtblout /dev/stdout --tblout /dev/null -o /dev/stderr --noali
```

`--max` turns every heuristic filter off. The binary itself is the evidence that the bias
filter is included: it refuses `--nobias` and `--F1/--F2/--F3` alongside `--max` as
incompatible, which it would not do if they were separate switches.

`-E 1000` reports roughly the top thousand chance hits per query, since an E-value of 1000
means about a thousand hits this good are expected by chance. That is the same depth as
the cap set on the other two tools, arrived at from the other direction.

### mmseqs2

```
mmseqs search qdb tdb res tmp -s 7.5 --num-iterations 3 -e 1000 \
  --max-seqs 1000 --min-ungapped-score 0 --mask 0 -c 0
```

- `--max-seqs 1000` is the important one: the default of 300 is a hard truncation of the
  list, not a score threshold.
- `--min-ungapped-score 0` (default 15) and `--mask 0` (default 1, tantan low-complexity
  masking) are the other two places the prefilter drops candidates. Masking off also makes
  the comparison with kmerseek's own low-complexity axis honest, since that axis is run
  both ways here.
- `-s 7.5` is the maximum sensitivity; the pipeline currently uses 7.
- **Not** `--prefilter-mode 2`. That removes the prefilter altogether, which means a full
  Smith-Waterman against all 572 700 targets for every query. It is the only truly
  filter-free setting and it is unaffordable at this scale; `--max-seqs 1000` with the
  score floor at 0 is the affordable approximation, and the design should say so rather
  than claim no filtering.

### kmerseek

Nothing changes, and the measurement says why.

One arm of the midi run — human, hp_pbotc_1st_ed2, k = 19, scaled 1, exact regions, one
chunk of 2 000 queries — reports **79 746 regions per query on average**, 558 at the
thinnest, 8 929 966 for the deepest query, and only 1 query of the 2 000 comes in under a
thousand. kmerseek's filters do not bind anywhere near a list of length 1 000. Opening them
further would only add rows below the depth anything is compared at, and the output is
already 6.0 TB for human, 4.2 TB for worm and 2.0 TB for yeast, with single arm files
reaching 91 GB compressed.

So `--min-shared-kmers 2` stays as well. One shared k-mer is not a weak region, it is not a
region: two is what the match is defined as, not a threshold on it.

### The matched quantity

Every tool is compared at **list length L ≤ 1000 per query**, ranked by that tool's own
score, with the truncation applied in the scoring step rather than by the tools. That is
one number to state in the methods, and each tool's flags are set so that nothing above
depth 1000 is missing for reasons other than the tool's own ranking.

## Which kmerseek score does the ranking

Not the E-value alone. On the human Pfam region benchmark the scores cross by region
length, measured as area under the ROC curve within each length bin:

| region length (aa) | E-value | mean IDF | regions |
| --- | --- | --- | --- |
| 24–29 | 0.58 | **0.67** | 3 392 |
| 30–39 | **0.76** | 0.36 | 2 692 |
| 40–59 | **0.72** | 0.47 | 2 444 |
| 60–119 | **0.89** | 0.40 | 1 590 |
| 120+ | **0.85** | 0.18 | 1 180 |

Below 30 amino acids, how rare the matched k-mers are ranks better than the E-value, and
that bin holds 3 392 of the 11 298 regions. This is the bin the short-epitope cases live
in, so a comparison that ranks on the E-value alone reports the worst available ranker
exactly where the claim is most interesting.

There is a second reason not to rank on the E-value alone: a region whose own composition
sits past the boundary carries no E-value at all (4 010 of 11 298 in that benchmark), so
E-value ranking silently drops a third of the regions.

The comparison to run, on the region files already on disk, is four scores against each
other, stratified by region length:

- `region_evalue`
- `region_tfidf` — the total rarity of the k-mers in the region
- `region_mean_idf` — the same divided by the number of shared k-mers, so length does not
  carry it
- `region_enrichment`

reported as area under the ROC curve and recall at list length L, per length bin, with the
number of regions in each bin printed next to it.

**Where the labels come from.** The dark set has no truth by construction — it is the
proteins nothing places. So the ranking comparison runs on the QfO Pfam region benchmark,
where a region is correct when it lands on the same Pfam family in both proteins, and the
dark set uses the matched-permissive flags for the cross-tool question it can answer: at
the same list length, does any tool reach the landmark pair at all.

## What this costs and what to run

The re-run is the sequence arms only. kmerseek's cached searches stay valid, which is the
whole saving: no index rebuilds, no 12 TB rewritten.

Unknown until measured: `--max` on jackhmmer with three iterations. The HMMER documentation
says "less speed, more power" without a number, and three iterations multiply it. Measure
before committing:

```bash
cd /scratch/users/olgabot/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set && make time-permissive-chunk
```

(That target does not exist yet; it runs one 2 000-query chunk of yeast through all three
tools at both the current and the proposed flags and prints wall time, CPU time, and rows
written per query.)

Then, if jackhmmer at `--max` is affordable, the change is three flags in `main.nf` and a
re-run of `phmmerSearch`, `jackhmmerSearch` and `mmseqs2Search` only.

## Decisions still open

1. **L = 1000.** Deep enough that mmseqs's 300 is not the binding constraint, shallow
   enough that HMMER's output stays bounded. Any value works as long as it is one value.
2. **jackhmmer `--max`**, pending the timing above. If it is 50x, the honest fallback is
   `--max` on phmmer only, with jackhmmer left at its defaults and the asymmetry stated.
3. **`--num-iterations 3` for mmseqs.** It stays, because the dark set is defined against
   the strongest sequence search available, not the cheapest. Worth stating in the methods
   that the iterative arm is included.
