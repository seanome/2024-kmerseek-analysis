# Dark-set summary page

One interactive page for the kmerseek 0.4 dark-set midi run (`runs/ladder-0.4-midi`,
human, *C. elegans*, *S. cerevisiae*, *B. burgdorferi*, *B. schlosseri*): how big the dark
set is, how much of it each alphabet and k-size reaches, and which gene groups it reaches,
at E-value cutoffs the reader switches.

Published as a claude.ai artifact: https://claude.ai/artifact/CJf2yEHb5B7xquUNB9nYcx

## How it was made (2026-10-13)

All heavy steps ran on Sherlock in jobs; only `dark_summary_data.json` came to the laptop.

1. Per-protein E-values from the repaired queries.tsv files (the repairQueriesTsv run
   finished 2026-10-12 23:40), job 47668732, about 5 minutes:

       python3 tools/export-protein-evalues \
         /scratch/users/olgabot/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set/runs/ladder-0.4-midi \
         $SCRATCH/invertebrate-dark-set/results-0.4/protein-evalues-repaired

   It read the trace written by head job 47663224 (2026-10-13 09:56-09:59, every one of
   19_465 tasks cached), so it saw the same work folders as the 23:40 run.
   The older export in `results-0.4/protein-evalues/` predates the repair and undercounts
   kmerseek; do not use it.

2. Gene groups, on a login node (downloads only):

       python3 tools/fetch-gene-groups $SCRATCH/invertebrate-dark-set/gene-groups

   `gene_groups_DOWNLOADED_ON` here is that run's record (URLs, sizes, md5).

3. The page data, job 47670986, 75 s, 1 GB:

       python3 tools/dark-summary-data \
         $SCRATCH/invertebrate-dark-set/results-0.4/protein-evalues-repaired \
         $SCRATCH/invertebrate-dark-set/gene-groups dark_summary_data.json

   Steps 2 and 3 ran from `git archive 1d185a9` of this branch (Sherlock's git 1.8 has no
   worktrees), so the JSON's `commit` is empty and the page shows `1d185a9`.

4. The page, on the laptop:

       python3 summary-page/build.py summary-page/dark_summary_data.json 1d185a9

   writes `dark_set_summary.html` (not committed: it is this JSON plus `template.html`
   and `fonts.css`, 5 MB).

## Panel 4: are the hits right (added 2026-10-13)

Each source's best hit (smallest E-value) per protein is checked against known Pfam
families: the query's from Pfam's annotation of UniProt
(`qfo-pfam-region-benchmark/data/annotations/<species>_pfam_domains.parquet`; for
*Borrelia*, UniProt's Pfam cross-references from
`tools/dark-set-accuracy query-pfam-uniprot UP000001807`), the target's from the DR Pfam
lines of its Swiss-Prot entry (`reference/minus_<clade>/reference_pfam.parquet`). Right =
shared family or Pfam clan (`Pfam-A.clans.tsv.gz`); wrong = target has Pfam families, none
in the query's clan; targets without Pfam are counted apart. *Botryllus* has no Pfam
annotation and is left out.

1. kmerseek best target per protein, one array task per region file:

       TOOLS_DIR=<tools dir> sbatch --export=ALL --array=1-N --time=01:15:00 --mem=2G \
         tools/dark-set-accuracy-sbatch best-targets <file list>

   1.36 GB of region file took 40 s and 14 MB (job 47672502). Job 47672974 ran the best
   setting per species (26 files, 75 GB).

2. Scoring, all sources and species: `sbatch tools/dark-set-accuracy-sbatch score`
   (job 47672617 for the sequence tools). Output `accuracy/<species>_accuracy.json`, copied
   to `summary-page/accuracy/`; `build.py` puts them into the page.

## Checks

* Every count the pipeline's own gain tables hold (`<species>_kmerseek_dark_gain.parquet`,
  dark at E <= 0.001, kmerseek region E <= 0.01 and 0.0001, scaled 1 and 10) matches the
  JSON: 1_686 of 1_686 rows, five species, no setting missing on either side.
* Human hp_lehninger2 k = 17: the smallest region E-value of all 2_277 dark proteins with a
  region at E <= 10 equals the one `tools/dark-regions min-evalue` read from the region
  files (`results-0.4/protein-min-evalue/`).

## Definitions

* Dark at T: no phmmer, jackhmmer or MMseqs2 hit with E <= T (T = 0.001, 1, 10).
* Reached at U: a kmerseek region with E <= U (U = 0.0001, 0.001, 0.01, 1, 10).
* A setting tells dark from placed when its placed share reached is at least twice its dark
  share reached.
* Gene groups: HGNC gene groups (human); the HGNC group of the human gene a *Botryllus*
  protein is named after (`FUN000002_NLRC5`, matched on symbol, then alias and previous
  symbols); UniProt keywords without the "Technical term" category (worm, yeast,
  *Borrelia*).
