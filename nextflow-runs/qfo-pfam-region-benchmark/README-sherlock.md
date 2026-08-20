# QfO Pfam region benchmark on Sherlock

Domain finding, not orthology. Every tool searches human query proteins against a QfO
target proteome; each aligned region is turned into a Pfam domain call by transferring
the families annotated on the overlapped target interval; the call is scored against the
human protein's real domain instances. Right family in the wrong place is a false
positive, which is the whole reason this scores regions rather than protein pairs.

## How the comparisons are structured

**Human is always the query. It is never a target and never appears in the species list.**

Every search takes the whole human proteome as queries and searches it against one target
proteome, producing a result named `human_vs_<target>`. So `--target_species yeast,ecoli`
is two targets and therefore two searches -- `human_vs_yeast` and `human_vs_ecoli` -- not
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

**The targets are not a random subset.** They are chosen *because* they share Pfam
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
discriminates, which is the point.

**It does not give the held-out split signal.** 241 query families split in half leaves
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

**If your compute nodes have no outbound internet**, the jobs fail immediately with a
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

**ProstT5 weights** (~1-2 GB, once):

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

**If your compute nodes have no outbound internet**, the jobs fail immediately with a
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

**ProstT5 weights** (~1-2 GB, once):

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

**Every run target already passes `-resume`**, so `make run` picks up where the last one
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

## Metrics

Everything is computed in the pipeline. `all_domain_metrics.csv` is the whole table;
nothing here needs recomputing in a notebook.

### CAFA-style

| column | meaning |
|---|---|
| `fmax` | CAFA's headline metric: max protein-centric F over thresholds |
| `wfmax` | same, weighted by family information content |
| `smin` | min sqrt(remaining uncertainty^2 + misinformation^2), in bits. **Lower is better** |
| `smin_ru` / `smin_mi` | the two error terms at that threshold: information missed, information invented |

`fmax` is *macro-averaged over proteins* -- precision over proteins that predicted
something, recall over every protein with a true domain. `best_f1` below is
micro-averaged over calls. They are different numbers on purpose; a few domain-dense
proteins can move `best_f1` but not `fmax`.

**`wfmax` and `smin` are weakened relative to real CAFA, and the difference matters.**
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
| `median_iou_tp` | how precisely a correct call is placed |

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
`--prostt5-model` it needs **no structures on either side**. That makes it the closest
published thing to what kmerseek claims -- structural signal without structure prediction --
and therefore the baseline the paper most has to differentiate from. Two differentiators
the pipeline should make visible rather than assert: it still depends on Foldseek, and it
still needs a target database. It is also the only structure-flavoured arm that runs where
AlphaFold coverage is too thin for Foldseek or Reseek, which is exactly the regime the
invertebrate claim lives in -- so it belongs *outside* the structure guard, and it is.

A database built this way carries predicted 3Di only, with no Ca coordinates, so TMalign
alignment types and TM-score/LDDT outputs are unavailable for that arm. The columns this
pipeline uses are all sequence-space and unaffected.

### Folddisco is scored differently, on purpose

Folddisco is not an aligner. Its hits are *discontinuous* residue sets -- `A56,A99,A195` --
not intervals. Every other arm reports a span, so the residue set is reduced to its
envelope (first matched residue to last), and the number of residues genuinely matched is
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

### The gray zone: Pfam silence is not a negative

Pfam-A annotates a fraction of residues and is silent everywhere else. Counting a call in
silent territory as a false positive asserts that the annotation looked there and found
nothing, which it did not -- and that is exactly backwards for the claim under test, since
a cryptic domain Pfam never annotated is the thing the method is supposed to find. Scored
that way the benchmark punishes the hypothesis instead of testing it.

So calls are split three ways, following the convention Foldseek and Folddisco use on
SCOPe:

| class | meaning | counted |
|---|---|---|
| true positive | right family, right place | yes |
| confident FP | lands mostly INSIDE annotated territory, wrong family | yes |
| gray | lands mostly OUTSIDE any annotation -- unknown | **excluded from the denominator** |

`--gray-min-annotated-fraction` (default 0.5) is the share of a call that must sit in
annotated territory to be judged confidently wrong. It is measured against the call, not
the annotation, so a long call is not excused by clipping one domain's edge.

**`coverage` travels with every metric** and is the fraction of calls that were scoreable
at all. A precision of 0.9 on 12% of calls is a different claim from the same number on
90%, and the column is there so that distinction cannot be lost.

**`precision_strict` is reported alongside `precision`**, counting gray as false positives.
The gap between the two IS the effect of this convention, so it can never be mistaken for
a free improvement. On the mini set: hp-pbotc k19 against Pfam moves 0.162 -> 0.204 with
20% of calls gray, while protein k10 has no gray calls at all and does not move. The
convention helps a method precisely to the degree it predicts where Pfam is silent, which
is the thing worth measuring.

Exclusion stops the benchmark bleeding credit. It does not create any: converting gray
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

`geneset` carries the 200-series' curated sets: `mhc_class_i_heavy` (6),
`antiviral_restriction_factor` (21), `igsf_decoy` (6), `fast_evolving_family` (462),
`olfactory_receptor`, `cytochrome_p450_2_3`. The MIN_STRATUM_PROTEINS floor of 30 does not
apply to `mhc` or `geneset` -- those sets are small on purpose and are the point of cutting
on them. Every row reports its own n.

Two things carried over from the notebooks that are easy to lose:

**Zinc-finger groups are dropped from the `hgnc` axis, not just flagged.** Tandem C2H2
arrays inflate k-mer sharing through repeat content rather than homology (notebook 206
section 6), and they are the single largest HGNC group in the query set -- 1,153 proteins,
5.9%. Left in, the one family the axis cannot trust would dominate it. They stay in the
covariate table under `hgnc_group_excluded` if you want to look at them deliberately.

**Class I and class II are separate strata, never pooled.** Notebook 211 found they answer
the k-size question in opposite directions, so a single "MHC" number hides the result.

Gene sets live in `bin/gene_sets.py`, copied from `notebooks/ortholog_analysis_utils.py`
because the container has no notebook tree. `bin/check_gene_sets.py` diffs the two wherever
both are importable, so the copy cannot drift silently -- run it if you change either.

`species_mya` is on every metric row. The 200-series used human-mouse percent identity as
its divergence axis because it had one target species; here the species IS that axis, so
plot against `species_mya` directly (mouse 100 ... ecoli 2000).

pLDDT and its disorder proxy are parsed from the AlphaFold `.cif` files the Foldseek arm
already stages, so they cost no extra download.

**dN/dS covers ~7% of query proteins.** Any omega-stratified result describes that subset,
not the proteome. Treat it as a probe, not a genome-wide claim.

**The upstream `dS` column is corrupt and this pipeline does not read it.**
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

**Compare on `recall_reachable`, not `recall`.** A human Pfam family absent from a target
proteome cannot be transferred by any search, so raw recall understates every tool by a
species-specific amount. Ecoli has 971 of human's 8909 families; mouse has 8805. Raw
`recall` is kept only so the size of that gap stays visible.

**`roc_auc` is conditioned on the calls a tool made.** Its negatives are the false
positives the tool itself reported; a domain never called at all is not in the universe,
because an unranked prediction cannot be ranked. A tool can score a high `roc_auc` by
reporting a handful of confident calls and missing everything else. Read it next to
`recall_reachable`, never alone. `auprc` does not have this problem -- its recall
denominator is the full reachable set.

**Prefer `best_f1` over `f1` when comparing tools.** `f1` sits at whatever cutoff each
tool happens to default to, which differs between HMMER, MMseqs2 and kmerseek and is not
a property of the method. `best_f1` and the threshold-free `roc_auc`/`auprc` are the
comparable numbers.

`roc_auc` is null, not 0.0, when a tool produced only correct calls or only incorrect
ones -- there is no ranking question to answer, and 0.0 would rank it below a coin flip.
