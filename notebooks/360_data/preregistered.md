# Notebook 360: do differentially expressed k-mers find the changed domains without assembly?

Frozen 2026-10-21, before any data was downloaded, simulated or searched. Nothing below
changes once a job has run. If a choice turns out wrong, the notebook says so and still
reports the answer under this text.

## Claim under test

Landing differentially expressed k-mers on protein regions with kmerseek, in a reduced
alphabet and with the region score kept, recovers the domains that actually changed at
least as well as assemble, then BLAST, then borrowed annotation, and does so without
assembly.

## What kills it

On a read set where the changed domains are known in advance, the k-mer path recovers a
smaller fraction of them than the assembly path, or it lands on changed domains no more
often on real labels than on label-swapped data.

## Choices Olga made on 2026-10-21, before anything ran

1. **Label swaps: all 9.** With 3 vs 3 samples there are 10 ways to split 6 samples into
   two groups of 3; one is the real split, so only 9 relabellings exist, not 10. Each one
   puts 2 samples of one condition with 1 of the other, so about a third of the real fold
   change is still there in every relabelling. Every "median of 10 swaps" in the request
   is the median of these 9.
2. **Second half of the rule tests landings on key domains,** not landings outside the
   key. The request said "landings outside the key are at least 5 times the median on the
   swaps". In this design every landing outside the key is a wrong call, so that wording
   would reward false positives. Olga chose: key-domain landings on real labels at least
   5x the median key-domain landings on the 9 swaps.
3. **Both paths search the same database:** Swiss-Prot proteins that have at least one
   Pfam 38.2 row, minus every protein of the organism's own genus (Drosophila, Danio, or Ciona).
4. **One k per alphabet, with the bits of protein20 at k = 7.** Changed by Olga on
   2026-10-21, after the answer key was built and reads were simulated but before any
   kmerseek index or search had run, so no landing had been seen. Her words: "let's use 1
   ksize per alphabet, use the equivalent of protein k=7 bits for each alphabet". A
   protein20 7-mer carries 7 x log2(20) = 30.25 bits; each alphabet gets
   k = round(30.25 / log2(letters)), so every setting has 28.6 to 31.0 bits per k-mer. This
   replaces the first choice, the dark set's best k per alphabet (kept below for the record).
   The mismatch penalty C stays the dark set's per-alphabet value.

## Organisms

**Changed on 2026-11-01: Drosophila melanogaster replaces Ciona intestinalis as the organism
that decides the verdict.** Olga's words: "Use some other species instead of [Ciona] because
[Ciona] isn't annotated well". She picked Drosophila from a list (Ciona savignyi, sea urchin,
Drosophila) on 2026-11-01. What had been seen before the change, so the reader can judge it:
the full run for Ciona and zebrafish had finished, and a Codex session (2026-10-22 and 23,
see `species_change_2026-10-22.md` and `adversarial_review.md`) had already computed both
verdicts under this rule: FAIL in both. Ciona R = mmseqs12 k8, 61 of 100 recovered against
Trinity + DIAMOND's 87; zebrafish R = hsdm17 k7, 77 against 98. That session had made
zebrafish the replacement organism; this change makes Drosophila the deciding organism
instead, with zebrafish back to the second organism. The rule is unchanged. Ciona's and
zebrafish's results stay in the notebook as reported results.

- **Drosophila melanogaster** (insect), UniProt reference proteome UP000000803, Ensembl 116
  (BDGP6.54). 18,179 of 21,947 proteins (82%) have a Pfam cross-reference (UniProt REST,
  2026-11-01). Decides the verdict. Its genus leaves the search database: 6,225 Drosophila
  Swiss-Prot entries in UniProt (fewer of them have Pfam rows). Fly proteins are common in
  Pfam seed alignments, so these genes are expected to sit closer to Pfam than Ciona's;
  section 1 of the notebook measures it.
- **Danio rerio** (zebrafish, vertebrate), UP000000437: 65,026 of 69,241 (94%). Same design,
  run as the second organism to measure how distance from Pfam changes the result. It does
  not enter the verdict.
- **Ciona intestinalis** (tunicate), UP000008144: 11,276 of 17,311 (65%). The first main
  organism (2026-10-21 to 2026-11-01); its run is complete and reported, not in the verdict.

## Answer key (built and committed before any read is simulated)

Per organism, from Ensembl (current release, recorded in `key/<organism>/provenance.txt`)
cDNA and peptide files and Pfam 38.2 `Pfam-A.regions.tsv.gz`:

1. Keep a gene if its Ensembl peptide has the same MD5 as a Pfam 38.2 `pfamseq` sequence,
   so Pfam's domain positions apply to it exactly. One transcript per gene: the one whose
   peptide matches, longest cDNA if several do.
2. Keep it only if all its Pfam rows are one family (repeats of that family allowed),
   its cDNA is 500 to 5,000 nt, and that family occurs at least once in the search
   database (Swiss-Prot with Pfam rows, minus the genus). A family absent from the
   database cannot be recovered by either path.
3. Keep at most one gene per Pfam clan (a family with no clan is its own clan), so no two
   selected genes share a family or a clan.
4. Random seed 360: draw 200 genes, then mark 50 "up", 50 "down", 100 "unchanged".
   Fold change 4 for up, 1/4 for down, 1 for unchanged.

The 100 answer-key domains are the Pfam families of the 50 up and 50 down genes. The
key is `key/<organism>/answer_key.tsv`, committed before the simulation job is submitted.

## Reads

polyester 1.38.0 (BioContainers `bioconductor-polyester:1.38.0--r43hdfd78af_0`; neither
polyester nor ART is installed on Sherlock, checked 2026-10-21). `simulate_experiment`,
3 samples per condition, paired-end 2 x 100 nt, fragment length mean 250, sd 25, uniform
error rate 0.005, seed 360. Baseline read pairs per transcript = round(cDNA length / 5),
about 40x coverage; condition B gets the fold change.

## Labellings

The real split (A1 A2 A3 vs B1 B2 B3) and the 9 other balanced splits of the same 6
samples. Every count in the score is computed for all 10.

## Standard path

1. Trinity 2.15.2 (BioContainers) on all 6 samples pooled, default settings, contigs
   >= 200 nt. (rnaSPAdes is not installed on Sherlock.)
2. salmon 1.10.3 quant of each sample against the Trinity transcripts.
3. DESeq2 1.42.0 (BioContainers; not installed in Sherlock's R) on rounded salmon read
   counts per Trinity transcript, `~ condition`, default settings, one fit per labelling.
4. DIAMOND 2.2.8 (BioContainers image already in the Sherlock cache; corrected from 2.1.23,
   the Sherlock module, on 2026-10-21 before DIAMOND ran on anything) blastx of every Trinity transcript against the search database,
   `--sensitive`, E <= 1e-5, best hit by bit score.
5. Changed domains = every Pfam family of the best hit of each transcript with
   padj < 0.05.

## K-mer path

1. jellyfish 2.2.10 `count -C -m 31` per sample; k-mers seen at least 10 times summed over
   the 6 samples form the count table.
2. DESeq2 1.42.0 on that table, same model and settings as the standard path (DE-kupl is
   not installed on Sherlock, so this is the "plain negative-binomial test on counts"
   fallback), one fit per labelling. Changed k-mers: padj < 0.05.
3. Changed k-mers with the same sign of fold change are joined where they overlap by 30
   nt, into maximal unbranched runs (what DE-kupl calls contigs). This joins only k-mers
   that were each called changed; it is not a transcriptome assembly.
4. Each run is translated in 6 frames and cut at stop codons; pieces of at least 5 amino
   acids are the queries (5 was the smallest k of the first k choice; every k now is 7 or
   more, and kmerseek finds nothing in a piece shorter than its k).
5. kmerseek, image `docker.io/olgabot/kmerseek:0.4.0-pr141-3c353d4` (the 0.4.0 that ran the
   dark set, plus the fix for wrong target coordinates, kmerseek PR 141), one index per
   kmerseek setting and organism, scaled 1, low-complexity k-mers kept, with the dark
   set's search flags: `--threshold 0 --min-shared-kmers 2 --max-query-pvalue 0.05
   --min-region-score 1.3 --extend-mismatch-penalty C --extend-xdrop 4C --ka-queries 0
   --chain-max-gap 30 --chain-max-shift 10`, the Karlin-Altschul fit made at index time as
   the dark set made it.
6. A region **lands** on a Pfam domain of its target protein when at least half of the
   region's target interval `[target_start, target_end)` lies inside the domain, and the
   region's `region_evalue` is at most 1. The region's `region_evalue`, `region_ka_bits`
   and `region_poisson_score` are kept in every landing table.
7. Changed domains = the Pfam families landed on.

### kmerseek settings (alphabet, k, mismatch penalty C), in force from 2026-10-21

| alphabet | letters | k | bits per k-mer | C |
|---|---|---|---|---|
| protein20 (full alphabet) | 20 | 7 | 30.3 | 0.14 |
| uniprot18 | 18 | 7 | 29.2 | 0.15 |
| hsdm17 | 17 | 7 | 28.6 | 0.18 |
| wass14 | 14 | 8 | 30.5 | 0.17 |
| sdm12 | 12 | 8 | 28.7 | 0.24 |
| mmseqs12 | 12 | 8 | 28.7 | 0.22 |
| funcgroups8 | 8 | 10 | 30.0 | 0.29 |
| gbmr7 | 7 | 11 | 30.9 | 0.32 |
| dayhoff6 | 6 | 12 | 31.0 | 0.41 |
| wwmj5 | 5 | 13 | 30.2 | 0.47 |
| gbmr4 | 4 | 15 | 30.0 | 0.67 |
| polarity4 | 4 | 15 | 30.0 | 0.54 |
| hp_lehninger_hpc3 | 3 | 19 | 30.1 | 0.87 |
| hp_pbotc_1st_ed2 | 2 | 30 | 30.0 | 1.59 |
| hp_thomas_dill2 | 2 | 30 | 30.0 | 1.63 |
| hp_thomas_dill_no_c2 | 2 | 30 | 30.0 | 1.62 |
| hp_lehninger2 | 2 | 30 | 30.0 | 1.51 |
| hp_lehninger_c_nonpolar2 | 2 | 30 | 30.0 | 1.52 |
| hp_kyte_doolittle2 | 2 | 30 | 30.0 | 1.57 |

A peptide shorter than an alphabet's k cannot be searched in it. At k = 30, the 2-letter
alphabets need a stop-free piece of at least 30 amino acids, which means a run of changed
k-mers of at least 90 nt; that is a property of this k choice, written down here before
any search.

### Replaced on 2026-10-21, never run: the dark set's best k

dayhoff6 k9, sdm12 k7, hsdm17 k6, mmseqs12 k8, wwmj5 k9, protein20 k5, uniprot18 k5,
hp_pbotc_1st_ed2 k19, hp_thomas_dill2 k20, hp_thomas_dill_no_c2 k20, wass14 k6,
hp_lehninger_c_nonpolar2 k19, hp_lehninger2 k17, hp_kyte_doolittle2 k29, funcgroups8 k9,
polarity4 k13, gbmr4 k16, hp_lehninger_hpc3 k17, gbmr7 k17 (the k with the highest median
Fmax over 4 species on placed proteins in notebook 291's table).

## Score

For each path, each kmerseek setting, and each of the 10 labellings:

- **recovered** = number of the 100 key families among the changed domains; recovery
  fraction = recovered / 100;
- **outside** = number of changed domains that are not key families.

## Pass/fail rule

Let R be the reduced-alphabet setting (any setting except protein20) with the highest
recovery on the real labels in Drosophila (Ciona until 2026-11-01); ties go to the one with fewer outside landings,
then to the alphabet name first in alphabetical order.

**PASS** if both hold, else **FAIL**:

1. recovery(R, real) >= recovery(standard path, real);
2. recovered(R, real) >= 5 x the median of recovered(R, swap) over the 9 swaps.

Known weaknesses, written down now so they are not found later and explained away:

- R is the best of 18 settings on the same data, so part 1 favours the k-mer path.
- Every swap keeps about a third of the real signal (choice 1), so part 2 is stricter
  than a no-signal control would be.
- If the standard path recovers 0 of 100, part 1 passes by default; the notebook then
  reports INSUFFICIENT_EVIDENCE beside the verdict.

## Second control: distance

Run in both organisms. For each key gene, the distance to Pfam is measured, not assumed:
the highest percent identity between the gene's protein and any Pfam 38.2 seed sequence
(`Pfam-A.seed.gz`) of its family, by DIAMOND 2.2.8 blastp `--ultra-sensitive`. Recovery is
reported per distance bin for each path. Zebrafish's and Ciona's results are reported beside Drosophila's
and does not change the verdict.

## Adversarial check

A separate agent that has not seen the analysis recomputes, from the raw count tables
(salmon and jellyfish counts) and the landing tables, the recovery fractions and swap
counts, and reports which claims held and which failed.
