# QfO Pfam region benchmark on Sherlock

Domain finding, not orthology. Every tool searches human query proteins against a QfO
target proteome; each aligned region is turned into a Pfam domain call by transferring
the families annotated on the overlapped target interval; the call is scored against the
human protein's real domain instances. Right family in the wrong place is a false
positive, which is the whole reason this scores regions rather than protein pairs.

## What runs

| arm | tool | variants |
|---|---|---|
| search | kmerseek, region-scoped Poisson | 113 alphabet x ksize combos |
| baseline | HMMER3 phmmer | 1 |
| baseline | HMMER3 jackhmmer | 3 iterations |
| baseline | MMseqs2 | seq-seq, iterative (3) |
| baseline | HHblits | profile-profile |
| baseline | Foldseek | AlphaFold models |
| ceiling | HMMER3 hmmscan vs Pfam-A | 1 |

hmmscan is not a competitor. It is handed the Pfam library the others are trying to
reconstruct, so it marks what direct annotation achieves and everything else is read
against it.

The kmerseek matrix is protein k5-15, dayhoff k10-20, and seven HP-family alphabets
k18-30: **113 combos x 9 species = 1017 searches**.

The HP floor is 18, not the 15 used by the kmer-spectra sweep. A 2-letter alphabet at
k=15 has 32768 possible k-mers against ~20k x 20k proteins, and measured output at k=18
was already 838 MB compressed for ecoli, the *smallest* species. k=15-17 is a separate
deliberate experiment; run it with `--hp_kmin 15` on a species subset first.

## Prerequisites

### 1. Push the container

The image carries the region-scoring kmerseek binary (`worktree-region-scoring`,
`0302511`) plus python/polars, so the search and scoring arms share one pull.

```bash
make push-image
```

Baseline tools come from pinned biocontainers and are pulled by Apptainer directly.

### 2. AlphaFold structures

As of 2026-08-18 the flat cache at `~/data/alphafold_structures` holds 54,339 of the
126,734 annotated proteins. **72,828 structures / ~36 GB are missing.**

```bash
make fetch-structures
```

Species with an AFDB proteome archive (human, mouse, zebrafish, fly, worm, yeast, ecoli,
arabidopsis) are fetched as one tarball each. Chicken and ciona are not in AFDB's
model-organism set, so their ~8.4k structures are fetched individually. The script is
resumable and sets a speed floor on every transfer -- a bare curl on multi-GB EBI files
stalls silently and then hangs for hours.

AlphaFold has no model for every accession. Missing ones stay missing; the Foldseek arm
reports its coverage rather than hiding the gap.

### 3. Pfam-A for the hmmscan ceiling

```bash
wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
gunzip Pfam-A.hmm.gz && hmmpress Pfam-A.hmm
```

Point `--pfam_hmm` at it. Absent, the ceiling arm is skipped and everything else runs.

### 4. HHblits background database (optional, but read this)

Without `--hhblits_db`, HHblits builds single-sequence profiles and lands near phmmer.
The run is still valid; it just is not measuring what HHblits is known for. UniRef30 is
~50 GB:

```bash
wget https://wwwuser.gwdg.de/~compbiol/uniclust/current_release/UniRef30_2023_02_hhsuite.tar.gz
```

## Running

```bash
ssh sherlock
cd $SCRATCH/qfo-pfam-region-benchmark
make run              # everything
make run-kmerseek     # the 1017-search sweep alone
make run-baselines    # the 9-species baselines alone
```

Each launches a detached tmux session, so a dropped ssh does not kill the run.
`make status` shows the SLURM queue; `tmux attach -t qfo-region` shows live output.

Jobs go to the `hns` school-condo partition billed to `--account=ayeletv`. Check
`sh_part` before a big submission -- hns is usually far shorter than the public `normal`
partition, but that has flipped before.

Split the arms when debugging. The baselines are 9-14 jobs and finish in hours; the
kmerseek sweep is 1017 jobs and will sit in the queue much longer.

## Memory sizing

HP-family alphabets at k<=20 get 128 GB, everything else 32 GB, doubling on retry. This
is set in `main.nf`, not `nextflow.config`, because it needs the task's own alphabet and
ksize. **Do not add a `memory` directive for `kmerseekIndexAndSearch` to the config** --
config directives always beat script-declared ones in Nextflow, so a config-level
"default" silently overrides the per-combo sizing and the HP jobs OOM.

The reason low ksize needs *more* memory, not less, is that a handful of k-mers absorb a
large share of the proteome, and the inverted index scales with the most-degenerate
k-mer's occurrence count -- the opposite of the usual "more distinct k-mers, more RAM"
intuition.

## Disk

Indexing and search are fused into one task. The RocksDB index is built in the task work
dir, used once, and deleted before the task exits; it is never a declared output, so
Nextflow never copies or stores it. Steady-state disk is bounded by (maxForks x one
index), not by all 1017 combos. An earlier all-vs-all run died with "No space left on
device" doing this the other way.

Results go straight to zstd parquet with the sequence columns dropped. `-resume` after a
crash re-indexes an incomplete combo rather than reusing a saved index; that is the
accepted cost of not persisting indices.

Everything lives under `$SCRATCH`, never `$HOME` -- the home quota is small.

## Output

```
results/
  truth/     human_domain_truth.parquet, <species>_domain_map.parquet, truth_summary.json
  regions/   per-tool raw aligned regions
  kmerseek/  per-combo region parquets (storeDir: survives -resume)
  calls/     per-(tool, variant, species) scored domain calls, one row per call
  metrics/   per-(tool, variant, species) metrics row
  all_domain_metrics.parquet / .csv
```

`make pull-results` brings back metrics and truth only. The `calls/` and `kmerseek/`
trees are the bulk of the output -- pull those selectively.

## Reading the metrics

`recall` and `recall_reachable` differ, and the gap matters. A human Pfam family absent
from a target proteome's annotations cannot be transferred by any search, so raw recall
understates every tool by a species-specific amount. Ecoli has 971 of human's 8909
families; mouse has 8805. **Compare tools on `recall_reachable`; compare species on
neither without saying which denominator you used.**

`median_iou_tp` is where region scoring earns or loses its keep: it says how precisely a
correct call is placed, not just whether the family was right.
