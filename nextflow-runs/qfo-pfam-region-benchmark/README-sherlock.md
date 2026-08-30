# QfO Pfam region benchmark on Sherlock

Domain finding, not orthology. Every tool searches human query proteins against a QfO
target proteome; each aligned region is turned into a Pfam domain call by transferring
the families annotated on the overlapped target interval; the call is scored against the
human protein's real domain instances. Right family in the wrong place is a false
positive, which is the whole reason this scores regions rather than protein pairs.

## How the comparisons are structured

Human is always the query. It is never a target and never appears in the species list.

Every search takes the whole human proteome as queries and searches it against one target
proteome, producing a result named `human_vs_<target>`. So `--target_species yeast,ecoli`
is two targets and therefore two searches. `human_vs_yeast` and `human_vs_ecoli`. not
a yeast-versus-ecoli comparison. The nine targets span 100 MYA (mouse) to 2000 MYA
(ecoli), which is what makes divergence an axis rather than a caveat.

The parameter used to be `--species`, which read as "which species are in this run" when
it means "which proteomes is human searched against". `--species` still works as an alias.
The run banner now prints the query, the targets, and the search count so the asymmetry is
visible before anything is submitted:

```
  query   : human (UP000005640_9606) -- always, and never listed as a target
  targets : yeast, ecoli
  searches: 2 alphabet x ksize combos x 2 targets = 4
            each named human_vs_<target>, e.g. human_vs_yeast
```

## kmerseek version: build from PR #43

This pipeline needs region scoring, the reduced alphabets, and `--remove-low-complexity`.
Checked 2026-08-20:

| branch | region scoring | 18 alphabets | `--remove-low-complexity` |
|---|:---:|:---:|:---:|
| `origin/main` | yes | no | yes |
| `olgabot/bump-version-0.4.0` (PR #34) | **no** | no | yes |
| `worktree-reduced-alphabets` (PR #43) | **yes** | **yes** | **yes** |

Build the image from **PR #43**. Region scoring reached `main` after PR #34 branched, so
#34's head still lacks it -- rebasing it on main before cutting v0.4.0 is what makes the
release usable here.

### Every alphabet was renamed

PR #43 renames each moltype to state how many classes it collapses the 20 residues into:

    protein -> protein20            dayhoff -> dayhoff6
    hp_lehninger -> hp_lehninger2   hp_lehninger_plus_c -> hp_lehninger_c_nonpolar2
    hp_shuffled_control -> hp_random_control2

Older result files will not join these labels. The CLI name and the moltype in the CSV are
the same string, so there is one name to track rather than two.

### K ranges: bit-matched floor, 12 ksizes for HP and 10 for the rest

An alphabet with n classes carries log2(n) bits per position only if every class is equally
likely. They are not, so log2(n) overstates every coarse alphabet. The bits/symbol below
come from amino-acid background frequencies grouped as kmerseek groups them
(`notebooks/ortholog_analysis_utils.entropy_per_symbol`).

HP carries 0.994 bits/symbol, so its k18 floor is 17.9 bits. Every `kmin` is
`round(17.9 / bits)`. The HP family gets twelve consecutive ksizes from there and every
other alphabet gets ten, since HP is what the paper is testing and its k optimum is the
least constrained.

| alphabet | classes | bits/symbol | k range |
|---|:---:|:---:|---|
| `protein20` | 20 | 4.176 | 4-13 |
| `uniprot18` | 18 | 3.951 | 5-14 |
| `hsdm17` | 17 | 3.742 | 5-14 |
| `wass14` | 14 | 3.626 | 5-14 |
| `mmseqs12` | 12 | 3.293 | 5-14 |
| `sdm12` | 12 | 3.127 | 6-15 |
| `dayhoff6` | 6 | 2.278 | 8-17 |
| `wwmj5` | 5 | 2.197 | 8-17 |
| `gbmr7` | 7 | 1.976 | 9-18 |
| `gbmr4` | 4 | 1.522 | 12-21 |
| `hp_lehninger_hpc3` | 3 | 1.128 | 16-27 |
| `hp_lehninger2` | 2 | 1.000 | 18-29 |
| `hp_lehninger_c_nonpolar2` | 2 | 0.999 | 18-29 |
| `hp_pbotc_1st_ed2` | 2 | 0.994 | 18-29 |
| `hp_thomas_dill2` | 2 | 0.966 | 19-30 |
| `hp_thomas_dill_no_c2` | 2 | 0.951 | 19-30 |
| `hp_kyte_doolittle2` | 2 | 0.937 | 19-30 |

Alphabets of similar coarseness overlap, so they can be compared at fixed k within a class.
protein20 and the HP alphabets do not overlap and are not compared at fixed k; at matched
information content their k ranges are 4-13 and 18-30.

Two entries contradict class count, which is why entropy is measured rather than assumed.
`hp_lehninger_hpc3` has three classes but 1.128 bits/symbol against HP's 0.994, because
cysteine is ~1.4% of residues. `gbmr7` carries less information than `wwmj5`, 1.976 against
2.197, despite two more classes, because its classes are unbalanced.

### Scale

368 combos x 9 targets = 3312 searches, up from 1017. Two multipliers: the eight new
reduced alphabets, and `--remove-low-complexity` swept as a toggle rather than fixed.
Whether dropping low-complexity k-mers helps depends on the alphabet, so it is measured.
The setting is carried in the variant label, so the two arms never pool.

One k-mer spectrum is written per combo to `results/spectra` for plotting. The
with/without low-complexity pair is only interpretable if both spectra exist, which is why
they are a first-class output rather than a debug artefact.

## What runs

| arm | tool | variants |
|---|---|---|
| search | kmerseek, region-scoped Poisson | 113 alphabet x ksize combos |
| baseline | HMMER3 phmmer | 1 |
| baseline | HMMER3 jackhmmer | 3 iterations |
| baseline | MMseqs2 | seq-seq, iterative (3) |
| baseline | HHblits | profile-profile |
| baseline | Foldseek | AlphaFold models |
| baseline | Folddisco | discontinuous structural motifs, AlphaFold models |
| baseline | Reseek | structure search over a ~8.6e10-state alphabet |
| baseline | ProstT5 + Foldseek | 3Di predicted from sequence, **no structures** |
| ceiling | HMMER3 hmmscan vs Pfam-A | 1 |

hmmscan is not a competitor. It is handed the Pfam library the others are trying to
reconstruct, so it marks what direct annotation achieves and everything else is read
against it.

The kmerseek matrix is protein k5-15, dayhoff k10-20, and seven HP-family alphabets
k18-30: 113 combos x 9 species = 1017 searches.

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
126,734 annotated proteins. 72,828 structures / ~36 GB are missing.

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

## Which targets run where

Half of these targets push data from the Mac to the cluster and half run on the cluster.
Each one now refuses to run on the wrong side and prints the command to use instead, so a
mistake costs a sentence rather than a confusing rsync failure.

| run on your Mac | run on Sherlock |
|---|---|
| `bootstrap-sherlock`, `push-image`, `build-image-local` | `mini-testset-sherlock` |
| `structure-lists`, `fetch-structures` | `run-mini` |
| `mini-testset`, `mhc-testset` | `run`, `run-kmerseek`, `run-baselines` |
| `sync-pipeline`, `sync-data`, `sync-structures` | `status` |
| `pull-results`, `pull-mini` | |
| `run-mini-local`, `run-mhc-local` | |

The rule is simple: anything that *moves* data, or talks to Docker Hub, runs on the Mac.
Anything that submits SLURM jobs runs on Sherlock. Detection is on `$SCRATCH`, which
Sherlock sets and your Mac does not.

## Mini smoke test — do this first

Before spending queue time on 1017 searches, run the same code path on a few hundred
proteins.

On Sherlock (after `make sync-data`, which stages the inputs it is cut from):

```bash
make run-mini
```

`run-mini` regenerates the set there first. The mini data is **not** rsynced: it is derived
from annotations and proteomes Sherlock already has, so shipping it would duplicate state
that can just be rebuilt. The generator is code and travels by git. Selection is
deterministic — same annotations in, same 200 queries out, on either machine — so a mini
run on your Mac and one on the cluster are comparable.

Structures are the exception, handled by `make sync-structures`: they are real input data,
not derived, and nothing on the cluster can synthesise them.

It takes minutes, uses two species and two kmerseek combos, and exercises every stage:
truth building, the family-grouped split, covariates, search, transfer, scoring, the CAFA
metrics and aggregation. `-profile sherlock,mini` layers on top of the real profile and
overrides only what scale requires, so nothing about the code path differs from `make run`.

The targets are not a random subset. They are chosen *because* they share Pfam
families with the queries — 42 shared families for yeast, 19 for ecoli at the defaults —
so real true positives exist. An equal number of decoy targets share no family with any
query, so precision can fall below 1.0. A random subset of two proteomes would share
almost nothing, every metric would read 0.0, and a broken scoring path would look
identical to a working one.

Queries lean toward multi-domain proteins (~60/40), because a single-domain-only set
leaves `domain_count_mcc` undefined and never tests whether a tool splits a protein
correctly.

Structures are symlinked from whatever the flat AlphaFold cache already holds, so **the
mini test needs none of the 36 GB download** — enough to exercise the Foldseek and
Folddisco arms without waiting on `make fetch-structures`.

Tune with `MINI_SPECIES` and `MINI_COMBOS`, or `--n-queries` / `--n-targets` on
`make mini-testset`.

### What the mini set does and does not prove

Verified locally end to end: 4 searches, 48 metric rows, 47 columns. Yeast at
`hp-pbotc-1st-ed` k19 recovers 21.7% of reachable domains against 2.6% for `protein` k10,
and ecoli sits near zero -- 2000 MYA is beyond what a 200-protein set recovers. So the set
discriminates.

It does not give the held-out split signal. 241 query families split in half leaves
~120 held out, of which only ~21 are shared with yeast, and no true positive lands there.
The split code path runs and emits its rows, but `heldout` reads 0.0 throughout. That is
the set being small, not the split being broken -- do not read a mini-run leaderboard as a
result. Raise `--n-queries` / `--n-targets` if you want the held-out half to carry signal.

### Structures are a separate sync, and the structure arms need them

`make sync-data` ships annotations, proteomes, HGNC/omega and Swiss-Prot. It does **not**
ship AlphaFold structures -- those are ~36 GB and have their own target:

On Sherlock, inside your own tmux:

```bash
make fetch-structures
```

This submits one SLURM job per species rather than downloading on the login node. The
login-node version was SIGKILLed mid-transfer (exit 137): Sherlock enforces limits there
and a sustained multi-GB download trips them. Per-species jobs also mean a kill costs one
proteome rather than the whole set, and the script is resumable so a retry continues from
where it stopped.

If your compute nodes have no outbound internet, the jobs fail immediately with a
message saying so rather than stalling. Fall back to the login node, one species at a time
and with low parallelism to stay under its limits:

```bash
make fetch-structures-login SPECIES_ONE=human
```

Download them where they will be used. Pulling ~60 GB to a laptop and pushing it back over
ssh is strictly worse, and the cluster's connection is much faster. It is resumable, so
re-run after an interruption rather than starting over; it needs outbound internet, which
on Sherlock means a login node, so it is not submitted through SLURM -- it is I/O, not
compute.

The Mac's flat AlphaFold cache saves almost nothing here anyway: 8 of the 10 species come
from whole-proteome tarballs that download in full regardless, and only chicken and ciona
(~8.4k structures) are fetched per accession.

`make fetch-structures-mac` still exists if you want them locally as well, and
`make sync-structures` pushes an existing local set up.

Without them the Foldseek and Folddisco arms are skipped, with a warning naming both
commands. Every sequence-based arm -- kmerseek, phmmer, jackhmmer, MMseqs2, HHblits,
hmmscan -- runs regardless, so a structure-free run is a perfectly valid run of everything
else. That is the recommended way to get first results while the download proceeds.

The skip is a guard rather than a convenience: `folddisco index` on an empty directory
exits 1 printing nothing, and Nextflow then reports only its own unrelated "Command 'ps'
required by nextflow to collect task metrics cannot be found" warning. The cause is
checked in the workflow, where it can be named.

## Compute nodes have no internet — two things must be fetched on a login node

Established 2026-08-20: Sherlock's batch partition cannot reach out. Anything that
downloads therefore runs on a login node, and the pipeline's preflight checks fail in
about 30 seconds with the cause rather than stalling.

ProstT5 weights (~1-2 GB, once):

```bash
make prostt5-weights
```

The run targets pass `--prostt5_weights ../data/prostt5/weights`. If it is missing the
pipeline stops immediately and points here rather than attempting a download that cannot
succeed. `--skip_prostt5 true` leaves the arm out.

**Structures** are the other one; see the fetch section above. `make fetch-structures`
submits SLURM jobs and will hit the same wall, so use the login-node form:

```bash
make fetch-structures-login SPECIES_ONE=human
```

## Containers: prefetch before the first run

```bash
make prefetch-images
```

Nextflow launches all its image pulls concurrently, and **Apptainer's OCI blob cache is
not concurrency-safe**. Simultaneous pulls into one cacheDir race and leave a half-written
blob, which surfaces as `FATAL: ... unexpected end of JSON input` -- an error that says
nothing about the real cause. That is what killed `hhblitsBuildDB` on the first cluster
run while `foldseek`, pulling at the same instant, happened to win.

`prefetch-images` pulls each container once, serially, into the cache under the exact
filename Nextflow looks for, so the pipeline finds them and never pulls concurrently. The
run targets depend on it, so you get this for free; it is a no-op once the cache is warm.

If a pull has already failed, the partial blob is sticky and the next attempt reuses it:

```bash
make clean-image-cache
```

## Running the mini set on your Mac

```bash
make build-image-local
make run-mini-local
```

`build-image-local` is a separate target for a real reason: the pushed image is
`linux/amd64` for Sherlock, and it cannot run the python steps on Apple Silicon. polars
requires AVX2, Rosetta/QEMU emulation does not provide it, and the processes die with
SIGSEGV (exit 139) rather than a readable error. Sherlock nodes are real amd64, so only
the amd64 image is pushed.

## Structures are a separate sync, and the structure arms need them

`make sync-data` ships annotations, proteomes, HGNC/omega and Swiss-Prot. It does **not**
ship AlphaFold structures -- those are ~36 GB and have their own target:

On Sherlock, inside your own tmux:

```bash
make fetch-structures
```

This submits one SLURM job per species rather than downloading on the login node. The
login-node version was SIGKILLed mid-transfer (exit 137): Sherlock enforces limits there
and a sustained multi-GB download trips them. Per-species jobs also mean a kill costs one
proteome rather than the whole set, and the script is resumable so a retry continues from
where it stopped.

If your compute nodes have no outbound internet, the jobs fail immediately with a
message saying so rather than stalling. Fall back to the login node, one species at a time
and with low parallelism to stay under its limits:

```bash
make fetch-structures-login SPECIES_ONE=human
```

Download them where they will be used. Pulling ~60 GB to a laptop and pushing it back over
ssh is strictly worse, and the cluster's connection is much faster. It is resumable, so
re-run after an interruption rather than starting over; it needs outbound internet, which
on Sherlock means a login node, so it is not submitted through SLURM -- it is I/O, not
compute.

The Mac's flat AlphaFold cache saves almost nothing here anyway: 8 of the 10 species come
from whole-proteome tarballs that download in full regardless, and only chicken and ciona
(~8.4k structures) are fetched per accession.

`make fetch-structures-mac` still exists if you want them locally as well, and
`make sync-structures` pushes an existing local set up.

Without them the Foldseek and Folddisco arms are skipped, with a warning naming both
commands. Every sequence-based arm -- kmerseek, phmmer, jackhmmer, MMseqs2, HHblits,
hmmscan -- runs regardless, so a structure-free run is a perfectly valid run of everything
else. That is the recommended way to get first results while the download proceeds.

The skip is a guard rather than a convenience: `folddisco index` on an empty directory
exits 1 printing nothing, and Nextflow then reports only its own unrelated "Command 'ps'
required by nextflow to collect task metrics cannot be found" warning. The cause is
checked in the workflow, where it can be named.

## Compute nodes have no internet — two things must be fetched on a login node

Established 2026-08-20: Sherlock's batch partition cannot reach out. Anything that
downloads therefore runs on a login node, and the pipeline's preflight checks fail in
about 30 seconds with the cause rather than stalling.

ProstT5 weights (~1-2 GB, once):

```bash
make prostt5-weights
```

The run targets pass `--prostt5_weights ../data/prostt5/weights`. If it is missing the
pipeline stops immediately and points here rather than attempting a download that cannot
succeed. `--skip_prostt5 true` leaves the arm out.

**Structures** are the other one; see the fetch section above. `make fetch-structures`
submits SLURM jobs and will hit the same wall, so use the login-node form:

```bash
make fetch-structures-login SPECIES_ONE=human
```

## Containers: prefetch before the first run

```bash
make prefetch-images
```

Nextflow launches all its image pulls concurrently, and **Apptainer's OCI blob cache is
not concurrency-safe**. Simultaneous pulls into one cacheDir race and leave a half-written
blob, which surfaces as `FATAL: ... unexpected end of JSON input` -- an error that says
nothing about the real cause. That is what killed `hhblitsBuildDB` on the first cluster
run while `foldseek`, pulling at the same instant, happened to win.

`prefetch-images` pulls each container once, serially, into the cache under the exact
filename Nextflow looks for, so the pipeline finds them and never pulls concurrently. The
run targets depend on it, so you get this for free; it is a no-op once the cache is warm.

If a pull has already failed, the partial blob is sticky and the next attempt reuses it:

```bash
make clean-image-cache
```

## Running

```bash
ssh sherlock
cd $SCRATCH/qfo-pfam-region-benchmark
make run              # everything
make run-kmerseek     # the 1017-search sweep alone
make run-baselines    # the 9-species baselines alone
```

Every run target already passes `-resume`, so `make run` picks up where the last one
stopped. Do not write `make run -resume`: make consumes the flag itself and exits with
`invalid option -- 'u'` before nextflow is ever reached. Extra nextflow flags go through
`NF_ARGS`:

```bash
make run NF_ARGS="--skip_folddisco true --skip_reseek true"
```

Every run target blocks in the foreground and streams live output, so start your own tmux
session first and run them inside it. They deliberately do not start tmux themselves.
`make status` shows the SLURM queue from another pane.

Run `run-kmerseek` and `run-baselines` in separate panes to get both arms going at once.

Jobs go to the `hns` school-condo partition billed to `--account=ayeletv`. Check
`sh_part` before a big submission -- hns is usually far shorter than the public `normal`
partition, but that has flipped before.

Split the arms when debugging. The baselines are 9-14 jobs and finish in hours; the
kmerseek sweep is 1017 jobs and will sit in the queue much longer.

## Memory sizing

A saturated k-mer keyspace gets 128 GB and everything else 32 GB, scaled by the target
proteome's FASTA size against the largest one (zebrafish, 16.7 MB) and floored at 8 GB,
doubling on retry. This is set in `main.nf`, not `nextflow.config`, because it needs the
task's own alphabet, ksize and target. Do not add a `memory` directive for
`kmerseekIndex` or `kmerseekSearch` to the config -- config directives always beat
script-declared ones in Nextflow, so a config-level "default" silently overrides the
per-combo sizing and the HP jobs OOM.

The proteome-size term applies to the 128 GB (saturated) branch only, and was added on
2026-08-25. Without it every ecoli and yeast task asked for the full 128 GB and peaked at
1.0-4.1 GB, and the resulting standing reservation is what SLURM answered by queueing the
sweep rather than running it.

The 32 GB branch stays flat on purpose. `isSaturated` is a hard threshold, so the
hungriest tasks in the sweep are the ones just outside it -- `fly_dayhoff_k10` is
unsaturated and peaked at 28.1 GB against its 32 GB. Scaling that branch by proteome size
was measured to push it and `fly_dayhoff_k11` into OOM.

The reason low ksize needs *more* memory, not less, is that a handful of k-mers absorb a
large share of the proteome, and the inverted index scales with the most-degenerate
k-mer's occurrence count -- the opposite of the usual "more distinct k-mers, more RAM"
intuition.

## What the midi and full runs share

Both pass the same `--db_cache ../results`, deliberately: midi and the full run differ ONLY
in the human query set (964 chr6 proteins against 19_696), so everything keyed on the
TARGET proteome is identical for both and is cached there -- `kmerseek_index/` (3294 target
indexes), `hhblits_db/` (9 target DBs at ~36 min each), `foldseek_db/`, `mmseqs_db/`,
`prostt5_db/`, `reseek_db/`, `folddisco_index/` and `prostt5_weights/`.

The HUMAN entry in each of those is NOT shareable, and used to be. Every arm wrote a
`human_*` entry keyed on the bare label, so whichever run built it first won and the other
silently searched with the wrong query set -- kmerseek reading the correct FASTA while
every database-backed baseline read the other run's. The human label now carries an md5 of
the query FASTA (`human-74bf5689_mmdb`), so each query set gets its own entry while the
target databases stay fully shared. Pre-digest `human_*` entries are orphaned; `make
index-disk` lists them.

**Do not run `make run` and `make run-midi` at the same time.** They share one `storeDir`,
and Nextflow checks whether a store entry exists when it SCHEDULES a task, not when it
writes one. Two runs both see it absent, both build it, and the slower one's `mv` dies with
`Directory not empty`, taking the whole run down. That is the `foldseekDb (ecoli)` failure
on 2026-08-26.

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

## The MultiQC report

One self-contained HTML for the whole run: accuracy, the alphabet x ksize sweep, and the
trace's time, CPU and memory. It is built at the end of a normal `make run`, and can be
rebuilt from published results at any time without re-running a single search:

```
make multiqc                                          # newest trace under run/
make multiqc TRACE=run/qfo_pfam_region.2026-08-24.trace.txt
make pull-report                                      # on the Mac, fetch it back
```

Published as:

```
results/
  qfo_pfam_region_multiqc.html         the report
  qfo_pfam_region_multiqc_plots/       every figure as png, svg and pdf
  qfo_pfam_region_multiqc_data/        the numbers behind each plot, as tsv
  multiqc/multiqc_in/                  the *_mqc.json section files it was built from
```

The svg exports are the ones that go into a paper. Nothing needs re-plotting in a
notebook to get a publication figure out of a run.

### Rebuilding it is a separate launch, and that is not incidental

Nextflow **truncates its trace file as a run starts**, and overwrites the execution report
and timeline with it. A report run launched in `run/` would therefore destroy the numbers
it exists to read.

Two params keep those roles apart:

| param | role |
|---|---|
| `--trace_file` | where THIS run writes its trace. Leave it alone. |
| `--report_trace` | the earlier run's trace the report READS. Passing it switches this run's own trace observer off. |

`-entry report` refuses to run without `--report_trace`, because there is no way to read a
trace that the same run is writing. `make multiqc` fills it in and launches from
`run-report/`. Pass `--report_trace none` to build the accuracy sections alone.

### What is in it

The frontier figure and the PR/ROC sections are cut against **one** truth set, defaulting
to Swiss-Prot when it exists -- Pfam is circular with the profile baselines, and a number
averaged across the two has no interpretation. Override with
`--multiqc_primary_truth pfam`. Leaderboards are emitted per truth set regardless.

| section | what it answers |
|---|---|
| The frontier | sensitivity in the &lt;40% identity zone against measured throughput, with the best and fastest incumbent drawn in |
| What each tool needs | 3D structures? alignment-free? accuracy and CPU-hours beside them |
| Leaderboards | best variant per tool, one board per truth set, with interval Fmax and family Fmax side by side |
| Truth sets and circularity | provenance, what each set is circular with, instances scored |
| CAFA-style / Threshold / Boundary | the full metric tables, defined in the Metrics section below |
| Twilight zone | Fmax by percent-identity bin |
| Divergence | Fmax and reachable recall against divergence time |
| Recall ceiling per species | how many human families the target proteome even has |
| Alphabet x ksize | Fmax heatmap per low-complexity arm, plus a per-alphabet bar view of the toggle |
| Reduced-alphabet information ceiling | best F1 against feature length / k per HP alphabet; alphabet x Swiss-Prot feature type, with its coverage twin; family Fmax against interval Fmax and the gap between them; BPE token boundaries against domain boundaries |
| Gray-zone accounting | true / false / unscoreable calls per tool |
| Run totals and resources | CPU-hours, task run times, peak RSS against requested, efficiency, I/O, task outcomes |

Accuracy sections use the **held-out** half of the Pfam families: the sweep picks its best
alphabet and ksize on the other half, and reporting the winner on the data that chose it
is optimistically biased.

Sections are written by `bin/build_multiqc_inputs.py` as one `*_mqc.json` each. Report
title, section order and plot limits are the only hand-edited part, in
`assets/multiqc_config.yaml`. To add a section, write another function that calls
`write_section()` and add its id to that file's `report_section_order`.

#### Reduced-alphabet information ceiling

One parent, rebuildable by `make multiqc` from published results with no search re-run.

1. **Feature length against k.** `best_f1` against `median_feature_length / ksize` on a
   log2 grid, one line per HP alphabet, with coverage as a switchable second dataset so no
   number is read without knowing what share of its calls could be judged. 1.0 on the x
   axis is a feature exactly one k-mer long. The point-feature bin is **not** on this
   curve -- see Point features above.
2. **Feature length against k, per k.** The same axis split by k instead of averaged over
   it, one dataset per alphabet. This is the panel that decides what panel 1 means, and it
   is the reason panel 1 alone is not enough:

   - if the per-k curves **collapse** onto each other, `feature_length / k` is the
     sufficient statistic and k trades against feature length one for one -- there is no k
     floor to read here, only a statement about how many k-mers a feature must hold;
   - if they **separate**, there is an absolute-k effect on top of the ratio, and the k at
     which the curves stop improving is a k floor measured on annotated domains rather than
     derived from keyspace arithmetic.

   Averaging over k inside a ratio bin is exactly the operation that hides the difference,
   so panel 1 cannot distinguish those two readings and this one can.
3. **Feature type**, in two figures that never share a colour scale: the placement-scored
   types (with a coverage twin) and the containment-scored point types. Rows run coarsest
   alphabet at the top, columns shortest median feature on the left. Swiss-Prot only: the
   other truth sets carry no feature types, so the axis is null there.
4. **Recognition against delineation.** Interval `fmax` and `family_fmax` side by side per
   alphabet, the gap between them on its own, and coverage on the same bars — then the same
   gap as an alphabet x ksize heatmap so averaging over k cannot hide whether a wide gap
   belongs to the alphabet or to the window length it was run at. The gap is what boundary
   placement costs: a coarse alphabet with a high `family_fmax` and a wide gap recognises
   families it cannot delineate, one with a low `family_fmax` has lost the family signal
   itself, and `fmax` alone reads both as the same failure. Every alphabet is drawn,
   `protein20` included, because the gap needs a reference to be large or small against.
   The heatmap scale is symmetric around zero — see the family-Fmax notes under Metrics for
   why the gap is not guaranteed positive.
5. **BPE token boundaries.** How often a ProtBERTa_2 token boundary lands exactly on a Pfam
   domain boundary, over the same rate on length- and composition-matched shuffled
   sequences. Written by `bin/hp_bpe_boundary_diagnostic.py`, which is run by hand -- see
   below -- so the panel is absent, not empty, when it has not been.

Why these exist: Rannon & Burstein (bioRxiv 2026.02.08.701987v2, doi
10.64898/2026.02.08.701987) trained pLMs on reduced alphabets and found their 2-letter
model worst on signal peptides (ROC-AUC 0.75, PR-AUC 0.47), nearly lossless on solubility
(relative F1 ~0.97) and strong on enzyme detection (~0.90). Signal peptides are ~20
residues; solubility and enzyme class are whole-protein properties. If that is one
feature-length gradient rather than three unrelated task results, their negative result is
the low-k arm of this sweep measured independently by another lab. These panels put both
gradients in domain units so that comparison is a measurement rather than an analogy. No
expected ordering is encoded anywhere in the code.

#### The BPE boundary diagnostic

```
make bpe-tokenizer        # 186 KB from Zenodo doi 10.5281/zenodo.18256943, checksum-verified
make hp-bpe-diagnostic    # ~30 s of CPU; writes data/protberta/hp_bpe_boundary.{json,png}
```

Once the tarball is on disk, every Sherlock run target passes `--bpe_tokenizer`
automatically (`BPE_NF_ARG` in the Makefile), so `make bpe-tokenizer` is the whole opt-in
and there is no flag to remember on the run that matters.

The BPE application is hand-written -- the merge table is a plain `vocab.json` /
`merges.txt` pair, and pulling in `tokenizers` would mean a new container for a script that
otherwise needs nothing. `--self-test` checks it against a textbook reference
implementation on 509 sequences including homopolymer runs, which is where a heap-ordered
merge and a left-to-right sweep can disagree:

```
python3 bin/hp_bpe_boundary_diagnostic.py --tokenizer data/protberta/ProtBERTa_tokenizers.tar.gz --self-test
```

Runs on the Mac or a login node, never on a compute node -- the tokenizer is a download and
compute nodes have no outbound internet. To fold the result into a pipeline run instead,
pass `--bpe_tokenizer ../data/protberta/ProtBERTa_tokenizers.tar.gz`; the `hpBpeBoundary`
process publishes the same JSON to `<outdir>/diagnostics/` and the `report` entry picks it
up from there on any later rebuild.

Their split, read out of `burstein-lab/BioTokenizers`
`data_processing/get_encoded_dataset.py` (`HYDROPHILIC_PHOBIC`), is
`S T N K E Q H D R` hydrophilic (plus the ambiguity codes `Z` and `B`) against
`A G I L M V P F W C Y` hydrophobic (plus `J`). Over the 20 canonical residues that is not
merely close to `hp_lehninger_c_nonpolar2` -- it is **identical** to it, C and G both
hydrophobic. The only difference is the ambiguity codes, which they map and none of our
alphabets define. The diagnostic prints the per-residue disagreement against every one of
our alphabets at run time rather than trusting this paragraph.

What it measures is segmentation agreement, not end-to-end performance. A tokenizer whose
boundaries never coincide with domain boundaries can still support a model that finds
domains, and one whose boundaries agree perfectly can still be beaten by a k-mer method.
`hp_random_control2` -- a random 10/10 split with the same class balance -- is in the
figure for that reason: a bar not clearly above it is measuring the autocorrelation of any
two-letter string rather than hydrophobicity.

Throughput is measured from the trace: query proteins divided by each search task's wall
time, median over target species. Indexing is inside the measurement for every arm,
because each task builds what it needs and searches once -- a tool that would amortise an
index over many searches is undersold, which is the honest reading of a benchmark that
searches each target proteome once.

## Tests

`make test` runs everything downstream of search against a committed 60-protein fixture, in
about 15 seconds, with no containers, no QfO download and no cluster. CI runs the same thing
on every push touching this directory (`.github/workflows/qfo-pfam-region-benchmark.yml`).

Search itself is not tested and cannot be on a hosted runner -- the baselines need multi-GB
profile databases and the structure arms need AlphaFold models. Every arm's output is a
region table, though, so the scoring path is driven with a **synthetic one whose right
answer is known in advance**: for each truth instance, a region placed exactly on it whose
target side covers a same-family domain. A correct scorer must return `recall_reachable`
of exactly 1.0. That catches more than replaying a real tool would, because a real tool's
numbers only tell you they moved, not which direction is right.

The fixture lives in `tests/fixtures` (60 human proteins, 778 instances over 28 families,
120 yeast targets, 180 real Swiss-Prot entries covering all 12 parsed FT types).
Regenerate it with `tests/make_fixture.py` when the shape of the inputs changes.

Each guard corresponds to a bug that actually happened, and each was checked by
reintroducing the bug and confirming the test fails:

| test | the bug it guards |
|---|---|
| `no_rate_metric_exceeds_one_on_any_stratum` | instance-level strata counted TPs from outside the cut; `recall_reachable` reached 2.11 |
| `no_metric_is_nan_when_the_top_block_is_all_gray` | an all-gray threshold block gave precision 0/0, and polars sorts NaN largest, so it won `best_f1` |
| `point_features_are_scoreable_by_containment_not_iou` | IoU against a 1-residue interval is unsatisfiable, so every point stratum scored 0 by construction |
| `scoring_is_deterministic_under_score_ties` | the greedy one-to-one match sorted on a non-unique key; five identical runs gave 169, 169, 169, 169, 168 |
| `perfect_tools_only_errors_are_nested_transfers` | asserts the *reason* precision is below 1.0 (nested Pfam domains), not a threshold that would pass for the wrong reason |


## Metrics

Everything is computed in the pipeline. `all_domain_metrics.csv` is the whole table;
nothing here needs recomputing in a notebook.

### CAFA-style

| column | meaning |
|---|---|
| `fmax` | CAFA's headline metric: max protein-centric F over thresholds, **interval-aware** |
| `wfmax` | same, weighted by family information content |
| `smin` | min sqrt(remaining uncertainty^2 + misinformation^2), in bits. **Lower is better** |
| `smin_ru` / `smin_mi` | the two error terms at that threshold: information missed, information invented |
| `family_fmax` | the same curve on the **set of families** called per protein, placement ignored |
| `family_fmax_precision` / `_recall` | the operating point family Fmax is reached at |
| `family_wfmax`, `family_smin`, `family_smin_ru` / `_mi`, `family_fmax_threshold` | the family-level twin of every scalar above |
| `n_family_truth` / `n_family_calls` / `n_family_found` | distinct (protein, family) pairs in the answer key, predicted, and correct |

`fmax` is *macro-averaged over proteins* -- precision over proteins that predicted
something, recall over every protein with a true domain. `best_f1` below is
micro-averaged over calls. They are different numbers on purpose; a few domain-dense
proteins can move `best_f1` but not `fmax`.

#### Family Fmax, beside interval Fmax and never instead of it

`fmax` gates on `is_tp`, which interval matching sets. A tool that names the right Pfam
family on the right protein but draws the boundary in the wrong place scores **zero**,
identically to a tool that never recognised the family at all. Those are two different
failures and one number cannot separate them.

`family_fmax` is the CAFA-classic reading of the same machinery: per query protein, the
SET of Pfam families called against the set truly present, interval placement ignored
entirely. `cafa_metrics.protein_centric_curve` takes a `level=` parameter for this rather
than carrying a second copy of the threshold sweep. Read the pair:

| | high `family_fmax` | low `family_fmax` |
|---|---|---|
| **wide gap** | recognises the family, cannot delineate it | — |
| **narrow gap** | recognises and delineates | has lost the family signal |

Two properties of the family level worth knowing before reading a gap:

- It is **exactly invariant to redundant calls**. Rows are collapsed to one per
  `(query_acc, pfam_id)` keeping the best score, so a family transferred fifty times onto
  one protein is one family prediction. Verified on the mini run by doubling every call
  with a worse-scoring displaced copy: `family_fmax` moved by 0.00e+00 on all seven arms
  tested while `fmax` fell (hmmscan 0.803 → 0.536, foldseek/yeast 0.057 → 0.048). This is
  a separate mechanism from the fragment dedup, which is about one alignment reported once
  per overlapping AlphaFold fragment.
- `family_fmax >= fmax` is **usual but not guaranteed**, and nothing clamps it. Ignoring
  placement can only help precision, but the family reading also swaps the recall
  denominator from instances to families, and a per-protein macro-average is not invariant
  under that swap. One protein with three instances of family A, all found, and one
  instance of family B, missed: interval recall 3/4, family recall 1/2. Measured on the
  mini run at **4 rows out of 2762**, three of them cells where one protein carries a
  tandem array (an IGSF decoy at 51 instances per family, `fmax` 0.831 against
  `family_fmax` 0.727) and one a 0.0012 threshold-grid difference.

`wfmax` and `smin` are weakened relative to real CAFA, and the difference matters.
CAFA weights GO terms by *information accretion*, defined against the ontology DAG as a
term's information content conditioned on its parents. Pfam is flat -- clans are a shallow
grouping, not a subsumption hierarchy -- so there are no parents to condition on and
information accretion degenerates to plain information content, `IC = -log2 P(family)`.
That is still a real weighting (a rare family counts for more than a ubiquitous one) but
it is not the CAFA quantity. Do not describe these as information-accretion weighted.

### Domain boundary

| column | meaning |
|---|---|
| `ndo` | normalized domain overlap: correctly labelled residues / true domain residues |
| `interval_semantics` | `alignment` or `motif` -- boundary metrics only compare within one |
| `residue_precision` / `_recall` / `_f1` | the same overlap from both directions |
| `dbd_median` / `dbd_mean` | boundary distance in residues, over correct calls only |
| `precision_iou80` / `recall_iou80` | the strict "correctly parsed" criterion used by structure parsers |
| `domain_count_accuracy` / `domain_count_mcc` | single- vs multi-domain call, over proteins with a prediction |

`ndo` is the residue-level normalized overlap CASP's NDO score is built from, not CASP's
full scoring matrix. `dbd` is reported over correct calls only -- the distance from a
wrong domain to a right one is not a boundary measurement.

### Point features

Two separate decisions, both about the same thing: a point feature asserts a residue and
this benchmark scores intervals.

#### They are scored by containment, not IoU

A Swiss-Prot `ACT_SITE`, `BINDING` or `SITE` asserts a **residue**, not an interval;
`build_swissprot_truth.py` widens it by one and `build_mcsa_truth.py` widens a catalytic
residue by `--window` purely so an interval exists to score. Judging those by interval IoU
is not a strict standard, it is an **unsatisfiable** one:

> IoU against a 1-residue interval is `1 / call_length`. At `--min-overlap 0.5` a true
> positive needs a call of at most 2 residues. Measured on the mini run: 97,706 calls
> across 32 arms, shortest 3 residues, **none** ≤ 2 — so the best IoU any tool could reach
> on a point feature was 0.333 against a 0.5 cutoff.

Every point stratum therefore scored exactly 0 on **every** metric, `fmax` included —
`protein_centric_curve` at its default `level="interval"` gates on `is_tp` too, so it did
not rescue those rows. (`family_fmax` does not rescue them either, and is not a substitute
for the fix below: it ignores placement, but a point instance whose family is never
transferred at all is still a miss at either level.) That is not a hard benchmark, it is
an unanswerable one, and it manufactures exactly the short-feature deficit the
reduced-alphabet question exists to test.

`--point-semantics cover` (the default) scores a point instance by containment instead:
did the call cover the annotated residue. `--point-semantics iou` restores the old
behaviour. `point_semantics` is stamped on every metric row beside `interval_semantics`,
and `point_fraction` records what share of each cell is point features — at 1.0, every
number on that row is a containment result and must not share an axis with a placement
result.

The cost, stated rather than hidden: containment favours long calls, since a 400-residue
region covering a catalytic residue counts the same as a tight one. Two things bound it.
`assign_instances` is one-to-one, so a call claims at most one instance; and precision
still counts every call, so a tool that carpets the protein pays for it.

Range instances are offered every call **before** point instances are offered any.
Containment maxes out at 1.0, so without that ordering a point feature outbid every
interval and a call correctly delineating a `DOMAIN` was consumed by an incidental
`ACT_SITE` inside it. Measured over the 24 range-only cells on the MHC set: **0** changed
on `n_instances_found`, `recall_reachable`, `n_tp_calls` or `fmax`. `precision` and
`coverage` do move (10 and 20 cells, ~0.0004 in precision) because a call whose true match
is a point feature outside a range cut is no longer charged there as a false positive — it
becomes gray. That is the intended consequence, not leakage.

This changes published Swiss-Prot and M-CSA numbers, including the ungrouped `all` row.
M-CSA is affected hardest: every instance is a widened catalytic residue, so at
`--window 5` the truth interval is 10 residues against a median call of 169, and that arm
was reporting near-zero throughout.

#### They are excluded from every boundary metric

`n_point_instances_excluded` records how many. A Swiss-Prot `ACT_SITE` or `BINDING` residue is a single position that
`build_swissprot_truth.py` widens by one, and an M-CSA catalytic residue is widened by a
window, purely so an interval exists to score at all. There is no boundary to be right or
wrong about at that length, so including them measured the widening rather than the
prediction. Both sides are cut -- the truth rows and the calls that matched them -- so
numerator and denominator keep describing the same set. The Pfam and Pfam-N truth sets have
no `is_point` column and are untouched. This changed the Swiss-Prot and M-CSA boundary
numbers when it landed; every M-CSA row is point features, so that arm now reports no
boundary metrics at all, which is what `build_mcsa_truth.py`'s own docstring already said
should happen.

### Threshold-based and threshold-free

| column | meaning |
|---|---|
| `precision` | of the calls reported, the fraction correctly placed |
| `recall` | of all human domain instances, the fraction recovered |
| `recall_reachable` | of the instances that *could* be transferred, the fraction recovered |
| `f1` | harmonic mean of `precision` and `recall` |
| `f1_reachable` | harmonic mean of `precision` and `recall_reachable` |
| `roc_auc` | P(a correctly placed call outranks an incorrectly placed one) |
| `auprc` | average precision over score-ranked calls, recall against reachable |
| `best_f1` | the best F1 at any score threshold |
| `best_f1_threshold` | the score achieving it, plus `_precision` and `_recall_reachable` |
| `median_iou_tp` | how a correct call is placed |

### Reseek and ProstT5: the two that position the paper

**Reseek** (Edgar 2024, Bioinformatics btae687) searches structures over a mega-alphabet of
~8.6e10 states. It scales alphabet size *up* where kmerseek scales it down to two letters,
and both working is itself the result worth reporting: it says alphabet size is not the
binding constraint. Foldseek's ">20 letters gives only incremental gains" finding is
disputed, so this is a live disagreement rather than settled ground -- worth stating as
such rather than picking a side. Run at `-sensitive`, not `-fast`: the claim under test is
remote-homolog detection, and benchmarking an incumbent at its weaker setting is the tell
reviewers look for. Reseek reports a p-value and no E-value, so the p-value takes the
E-value slot (same direction, lower is better).

**ProstT5** predicts 3Di directly from amino acid sequence, so run through Foldseek's
`--prostt5-model` it needs no structures on either side. That makes it the closest
published thing to what kmerseek claims: structural signal without structure prediction.
That makes it the baseline the paper most has to differentiate from. Two differentiators
the pipeline should make visible rather than assert: it still depends on Foldseek, and it
still needs a target database. It is also the only structure-flavoured arm that runs where
AlphaFold coverage is too thin for Foldseek or Reseek, which is the regime the
invertebrate claim lives in -- so it belongs *outside* the structure guard, and it is.

A database built this way carries predicted 3Di only, with no Ca coordinates, so TMalign
alignment types and TM-score/LDDT outputs are unavailable for that arm. The columns this
pipeline uses are all sequence-space and unaffected.

### Folddisco is scored differently, on purpose

Folddisco is not an aligner. Its hits are discontinuous residue sets such as
`A56,A99,A195`, not intervals. Every other arm reports a span, so the residue set is reduced to its
envelope (first matched residue to last), and the number of residues matched is
carried alongside it.

That reduction only overstates: a 3-residue motif spanning 56..195 yields a 139-residue
envelope while touching 3 residues. **Scoring that envelope by interval IoU would measure
the reduction, not the prediction**, so this arm runs with `--interval-semantics motif`:
a call is correct when its envelope *covers* the true domain, rather than when it
coincides with it. Every row carries `interval_semantics` so the two are never silently
compared, and both `iou` and `cover` are recorded on every call either way.

What this means when reading results:

- `fmax`, `precision`, `recall_reachable`, `auprc`, `smin` stay comparable across all
  arms -- they ask which family was found on which protein.
- `ndo`, `dbd_*`, `residue_*` and `precision_iou80` are **not** comparable between
  Folddisco and the aligners. Those measure boundary precision, and Folddisco is not
  making a boundary claim.

Folddisco queries one structure per invocation, so the ~19.4k human queries are spread
across `params.folddisco_chunks` (default 20) SLURM tasks per species. Its target index is
kept under `storeDir` and reused by every chunk. It ranks by `idf` (motif rarity, higher
is better) with `rmsd` in the E-value slot; it emits no E-value of its own.

### Truth sets, and what each is circular with

| truth set | built by | circular with | scale |
|---|---|---|---|
| `pfam` | always | **the profile baselines** — Pfam-A domains ARE profile HMMs | 50,185 human instances |
| `swissprot` | `--swissprot_dat` | nothing; literature-curated features | 142,857 human features |
| `pfamn` | `make pfamn-truth` | nothing — it exists where the HMMs FAILED | streamed from Pfam35.0 |
| `mcsa` | `make mcsa-truth` | nothing; function defined by mechanism | **106 human proteins** |

Every metric row carries `truth_set` and they are never pooled in the leaderboard: a mean
across a circular and a non-circular truth set has no interpretation.

Pfam-N is the one that creates credit. The gray-zone convention below stops the
benchmark charging for calls where Pfam-A is silent, but exclusion only removes them from
the denominator. Pfam-N is the explicit "Pfam-A HMMs missed these" label set, so a region
found in that silence can be adjudicated instead of merely ignored. It is published for
Pfam35.0 only — releases 36 and 37 do not carry it, verified 2026-08-20 — so it is a frozen
2022 resource and its accessions should be read against Pfam35. The source is ~17.4 GB of
Stockholm alignments, streamed and filtered rather than stored.

M-CSA is a vignette, not a statistic. 1003 curated entries intersect this benchmark at
106 human proteins, 0.5% of the query set. Read it as the MHC block is read. do not
let it carry a headline claim. Catalytic residues are single positions, widened to a small
window and flagged `is_point`, so boundary IoU against them is meaningless and recall is
the question — does the tool put a region on the catalytic machinery. Numbering comes from
the API's `residue_sequences`, which is UniProt-numbered; the `curated_data.csv` flat file
is PDB-numbered with a separate chain column and is silently wrong if used directly.

The higher-value use of M-CSA is not this truth set: Folddisco published an M-CSA benchmark
during review (713 queries, sensitivity-to-first-FP) with per-query results deposited, so
running that query set would make kmerseek directly comparable to a published NBT table.
That is a separate exercise from the QfO sweep and is not implemented here.

### The gray zone: Pfam silence is not a negative

Pfam-A annotates a fraction of residues and is silent everywhere else. Counting a call in
silent territory as a false positive asserts that the annotation looked there and found
nothing, which it did not -- and that is backwards for the claim under test, since
a cryptic domain Pfam never annotated is the thing the method is supposed to find. Scored
that way the benchmark punishes the hypothesis instead of testing it.

So calls are split three ways, following the convention Foldseek and Folddisco use on
SCOPe:

| class | meaning | counted |
|---|---|---|
| true positive | right family, right place | yes |
| confident FP | lands mostly INSIDE annotated territory, wrong family | yes |
| gray | lands mostly OUTSIDE any annotation -- unknown | excluded from the denominator |

`--gray-min-annotated-fraction` (default 0.5) is the share of a call that must sit in
annotated territory to be judged confidently wrong. It is measured against the call, not
the annotation, so a long call is not excused by clipping one domain's edge.

`coverage` travels with every metric and is the fraction of calls that were scoreable
at all. A precision of 0.9 on 12% of calls is a different claim from the same number on
90%, and the column is there so that distinction cannot be lost.

`precision_strict` is reported alongside `precision`, counting gray as false positives.
The gap between the two is the effect of this convention. On the mini set: hp-pbotc k19 against Pfam moves 0.162 -> 0.204 with
20% of calls gray, while protein k10 has no gray calls at all and does not move. The
convention helps a method to the degree it predicts where Pfam is silent, which
is the thing worth measuring.

Exclusion removes those calls from the denominator. It does not add any true positives: converting gray
calls into scoreable true positives needs a label set that says "Pfam-A missed these",
which is Pfam-N and is not implemented yet.

### Splits: the leaderboard is the held-out half

Every metric row carries `split` (`all` / `selection` / `heldout`). None of these tools
learns from the data, so there is no model to overfit -- but **picking the best of 113
alphabet x ksize combos on the same instances you report is model selection**, and
scoring the winner on the data that chose it is optimistically biased. Tune on
`selection`; report `heldout`. `make run` prints the leaderboard over `heldout` already.

The split is grouped by **Pfam family**, not by protein, and that choice is deliberate.
Splitting on proteins lets the same family sit on both sides, so a ksize tuned on PF00001
gets tested on PF00001 again and the held-out score measures memorised families rather
than generalisation. Grouping by family means the held-out half is families the sweep
never saw. It is hash-based on `(seed, pfam_id)`, so it reproduces without a state file.
Defaults: 4470 selection families / 4439 heldout.

One arm is exempt from this and should never be read as a competitor: **hmmscan against
Pfam-A is near-circular**. The ground truth *is* Pfam annotation, largely produced by
hmmscan against Pfam-A, so it is being scored against its own output. It marks the
ceiling of direct annotation, nothing more.

### Strata: results cut by biology

Every row also carries `stratum_axis` and `stratum`, cutting on the same axes the
200-series notebooks use:

| axis | source | coverage of ~19.4k query proteins |
|---|---|---|
| `hgnc` | HGNC gene group | 19,226 (4,222 groups; >= 30 proteins each, zinc fingers excluded) |
| `mhc` | `ou.MHC_CLASSES`, 7 classes | 25 genes |
| `geneset` | curated sets below | 6-463 each |
| `plddt` | mean pLDDT, bins 0-50/50-70/70-90/90-100 | rises to ~full once `make fetch-structures` completes |
| `disorder` | fraction of residues with pLDDT < 50 | same as pLDDT |
| `omega` | dN/dS from the human-mouse-dnds-omega pipeline | **1,289 only** |
| `feature_length_bin` | the truth interval's own residue length | every instance, every truth set |
| `feature_type` | Swiss-Prot FT type | Swiss-Prot truth only; **null** elsewhere |

`geneset` carries the 200-series' curated sets: `mhc_class_i_heavy` (6),
`antiviral_restriction_factor` (21), `igsf_decoy` (6), `fast_evolving_family` (462),
`olfactory_receptor`, `cytochrome_p450_2_3`. The MIN_STRATUM_PROTEINS floor of 30 does not
apply to `mhc`, `geneset`, `identity`, `feature_length_bin` or `feature_type` -- those
vocabularies are fixed and biologically defined rather than data-derived, and the floor
exists to stop ~4,200 HGNC groups from producing single-protein strata, which is not a
problem any of them has. `ACT_SITE` and `DNA_BIND` are small in every proteome that will
ever be measured; dropping them would delete the short-feature end of the gradient
`feature_length_bin` exists to measure. Every row reports its own `n_stratum_proteins` and
`n_truth_instances`.

#### `feature_length_bin` and `feature_type`

`feature_length_bin` bins each truth interval by its own residue length --
`1` (point features), `2-15`, `16-30`, `31-60`, `61-120`, `121-250`, `251+` -- on
`domain_end - domain_start`, the same length convention the boundary metrics sum over. The
unit is the annotation, not the protein: a 21-residue TRANSMEM helix and a 400-residue
kinase domain in the same protein are different measurement problems for a k-mer method,
and at k=19 the first admits three k-mers while the second admits 380.

The quantity to read is `best_f1` against **`median_feature_length / ksize`**, not against
raw length. Every method finds short features less reliably; the claim specific to a coarse
alphabet is that it needs a long window, so the axis is how many k-mers the feature can
hold. `median_feature_length` is on every metric row and is measured over the instances in
that cell, never a bin midpoint.

`feature_type` is one stratum per Swiss-Prot FT type -- `ACT_SITE`, `BINDING`, `DNA_BIND`,
`MOTIF`, `REGION`, `TRANSMEM`, `DOMAIN`, `REPEAT`, `ZN_FING`, `COILED`, `SITE`,
`INTRAMEM`. The Swiss-Prot truth set puts the FT type in the `pfam_id` column, keeping the
name for schema compatibility. For Pfam and Pfam-N that column holds a family accession
with no type variation to cut on, and for M-CSA an entry id, so the axis is **null** on
those sets rather than empty-stringed -- null drops the axis, an empty string would invent
a stratum named after nothing. The vocabulary is imported from `build_swissprot_truth.py`
rather than re-listed, so the two cannot drift.

Both axes are **instance-level**, which is load-bearing. Strata are applied to calls by
protein, but one protein can carry a 90%-identity domain and a 25%-identity one, or a
TRANSMEM and a DOMAIN. `restrict_tp_to_cut()` in `evaluate_domain_calls.py` clears `is_tp`
on any call whose matched instance is outside the cut before either `compute_metrics` or
`operating_points` sees the table, and marks it gray rather than a false positive -- it is a
right answer about something the cut does not measure. Without that, `recall_reachable`
exceeds 1.0 (2.11 observed on the MHC smoke set) and `best_f1` is computed from an inflated
recall. `coverage` reports the share, as it does for every other gray call.

Two things carried over from the notebooks that are easy to lose:

Zinc-finger groups are dropped from the `hgnc` axis, not just flagged. Tandem C2H2
arrays inflate k-mer sharing through repeat content rather than homology (notebook 206
section 6), and they are the single largest HGNC group in the query set -- 1,153 proteins,
5.9%. Left in, the one family the axis cannot trust would dominate it. They stay in the
covariate table under `hgnc_group_excluded` if you want to look at them deliberately.

Class I and class II are separate strata, never pooled. Notebook 211 found they answer
the k-size question in opposite directions, so a single "MHC" number hides the result.

Gene sets live in `bin/gene_sets.py`, copied from `notebooks/ortholog_analysis_utils.py`
because the container has no notebook tree. `bin/check_gene_sets.py` diffs the two wherever
both are importable, so the copy cannot drift silently -- run it if you change either.

`species_mya` is on every metric row. The 200-series used human-mouse percent identity as
its divergence axis because it had one target species; here the species IS that axis, so
plot against `species_mya` directly (mouse 100 ... ecoli 2000).

pLDDT and its disorder proxy are parsed from the AlphaFold `.cif` files the Foldseek arm
already stages, so they cost no extra download.

dN/dS covers ~7% of query proteins. Any omega-stratified result describes that subset,
not the proteome. Treat it as a probe, not a genome-wide claim.

The upstream `dS` column is corrupt and this pipeline does not read it.
`human-mouse-dnds-omega/bin/compute_omega.py` parses codeml output with `r"dS\s*=\s*(...)"`,
and `re.search` matches that inside `dN/dS=` first, so the published `dS` column is a copy
of `omega` in all 1335 rows. `dN` and `omega` themselves are correct (omega's median of
0.143 matches published human-mouse dN/dS). `dS` is reconstructed here as `dN/omega`, which
is exact, and surfaces as `dS_recovered`. Fix the parser upstream and drop the workaround.

`all_domain_curves.parquet` holds the full PR and ROC operating points per
(tool, variant, species, split) -- `score_threshold`, `precision`, `recall_reachable`,
`f1`, `tpr`, `fpr` -- thinned to at most 2000 points for plotting. Curves are emitted for
the ungrouped cut only; one per stratum across a 1017-combo sweep would dwarf the metrics
they support. Every scalar above is
computed on the *full* curve before thinning, so the two always agree.

### Three things that will mislead you if skipped

Compare on `recall_reachable`, not `recall`. A human Pfam family absent from a target
proteome cannot be transferred by any search, so raw recall understates every tool by a
species-specific amount. Ecoli has 971 of human's 8909 families; mouse has 8805. Raw
`recall` is kept only so the size of that gap stays visible.

`roc_auc` is conditioned on the calls a tool made. Its negatives are the false
positives the tool itself reported; a domain never called at all is not in the universe,
because an unranked prediction cannot be ranked. A tool can score a high `roc_auc` by
reporting a handful of confident calls and missing everything else. Read it next to
`recall_reachable`, never alone. `auprc` does not have this problem -- its recall
denominator is the full reachable set.

Prefer `best_f1` over `f1` when comparing tools. `f1` sits at whatever cutoff each
tool happens to default to, which differs between HMMER, MMseqs2 and kmerseek and is not
a property of the method. `best_f1` and the threshold-free `roc_auc`/`auprc` are the
comparable numbers.

`roc_auc` is null, not 0.0, when a tool produced only correct calls or only incorrect
ones -- there is no ranking question to answer, and 0.0 would rank it below a coin flip.
