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
   Pfam 38.2 row, minus every protein of the organism's own genus (Ciona, or Danio).
4. **k per alphabet is the dark set's best k,** not one k for all. Taken from notebook 291
   (`notebooks/data/291_dark_fmax/dark_fmax_metrics.csv.gz` on branch
   `olgabot/dark-set-kmerseek-0.4`): for each alphabet, the k (scaled 1) with the highest
   median Fmax over the 4 species on placed proteins, with that row's mismatch penalty.
   Table below.

## Organisms

- **Ciona intestinalis** (tunicate), UniProt reference proteome UP000008144. Pfam coverage
  checked on 2026-10-21 from the UniProt REST API: 11,276 of 17,311 proteins (65%) have a
  Pfam cross-reference. That is enough to pick 200 genes, so Ciona is the main organism and
  decides the verdict.
- **Danio rerio** (zebrafish, vertebrate), UP000000437: 65,026 of 69,241 (94%). Same design,
  run as the second organism to measure how distance from Pfam changes the result. It does
  not enter the verdict.

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
   acids (the smallest k below) are the queries.
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

### kmerseek settings (alphabet, k, mismatch penalty C)

| alphabet | k | C | dark-set median Fmax |
|---|---|---|---|
| dayhoff6 | 9 | 0.41 | 0.390 |
| sdm12 | 7 | 0.24 | 0.378 |
| hsdm17 | 6 | 0.18 | 0.368 |
| mmseqs12 | 8 | 0.22 | 0.367 |
| wwmj5 | 9 | 0.47 | 0.364 |
| protein20 (full alphabet) | 5 | 0.14 | 0.355 |
| uniprot18 | 5 | 0.15 | 0.347 |
| hp_pbotc_1st_ed2 | 19 | 1.59 | 0.342 |
| hp_thomas_dill2 | 20 | 1.63 | 0.331 |
| hp_thomas_dill_no_c2 | 20 | 1.62 | 0.326 |
| wass14 | 6 | 0.17 | 0.321 |
| hp_lehninger_c_nonpolar2 | 19 | 1.52 | 0.313 |
| hp_lehninger2 | 17 | 1.51 | 0.310 |
| hp_kyte_doolittle2 | 29 | 1.57 | 0.256 |
| funcgroups8 | 9 | 0.29 | 0.241 |
| polarity4 | 13 | 0.54 | 0.212 |
| gbmr4 | 16 | 0.67 | 0.097 |
| hp_lehninger_hpc3 | 17 | 0.87 | 0.014 |
| gbmr7 | 17 | 0.32 | 0.000 |

gbmr7 scored 0 at every k in the dark set, so its "best" k17 is the smallest k among ties
and means nothing. It runs anyway, because the request asks for every alphabet.

## Score

For each path, each kmerseek setting, and each of the 10 labellings:

- **recovered** = number of the 100 key families among the changed domains; recovery
  fraction = recovered / 100;
- **outside** = number of changed domains that are not key families.

## Pass/fail rule

Let R be the reduced-alphabet setting (any setting except protein20) with the highest
recovery on the real labels in Ciona; ties go to the one with fewer outside landings,
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
reported per distance bin for each path. Zebrafish's result is reported beside Ciona's
and does not change the verdict.

## Adversarial check

A separate agent that has not seen the analysis recomputes, from the raw count tables
(salmon and jellyfish counts) and the landing tables, the recovery fractions and swap
counts, and reports which claims held and which failed.
