# Running the k-mer spectra pipeline on Sherlock

Everything below needs to run from your own terminal, not through Claude Code: Sherlock login
requires an interactive Duo push, and the `sherlock` SSH alias signs in via the 1Password SSH
agent, whose socket (`~/.1password/agent.sock`) isn't reachable from this sandboxed session.

A `Makefile` in this directory wraps every command below. `build-image`/`push-image`/
`sync-pipeline`/`sync-data`/`pull-k2`/`pull-k3`/`pull-uniref50`/`pull-uniref90`/
`pull-uniref100` run on your Mac; `download-uniref50`/`download-uniref90`/
`download-uniref100`/`run`/`status` run on Sherlock (`ssh sherlock`, `cd $SCRATCH/kmer-spectra`,
`make <target>`).

**One `make run` sweeps every selected dataset in a single nextflow invocation** -- pass
`DATASETS=k2,k3,uniref50` (comma-separated, matching `main.nf`'s dataset names: `k2`, `k3`,
`uniref50`, `uniref90`, `uniref100`) to choose which; default is `k2,k3`, the two with a
validated memory profile. There's one shared `work/`/trace/timeline/report for the whole run
now, not one per dataset -- no more juggling separate tmux panes just to keep runs from
colliding.

Account: group `ayeletv`, using the `hns` school-condo partition (`groups` / `sh_part` on
Sherlock). `hns` had ~7.4k jobs queued vs. ~95k on the public `normal` partition at last check --
worth re-running `sh_part` before a big submission to confirm that's still true.

## 1. Build and push the image (from your machine, once local resources are free)

The pipeline's Dockerfile does a real `cargo build --release` from source -- avoid running this
while another local job is using CPU.

Sherlock's compute nodes are amd64. A plain `docker build` on Apple Silicon produces an arm64
image, which Apptainer refuses to run (`the image's architecture (arm64) could not run on the
host's (amd64)`) -- pass `--platform linux/amd64` explicitly. `make push-image` (below) does
this for you.

```bash
cd nextflow-runs/kmer-spectra
make push-image
# equivalent to:
#   docker build --platform linux/amd64 -t kmerseek-spectra:latest \
#     --build-arg GIT_SHA=f006f821829e7354579267cc5336b341427105b3 \
#     -f Dockerfile \
#     /Users/olga/code/kmerseek-kmer-frequency-histogram
#   docker tag kmerseek-spectra:latest docker.io/olgabot/kmerseek:2026-08-19-kmer-spectra
#   docker push docker.io/olgabot/kmerseek:2026-08-19-kmer-spectra
```

(`/Users/olga/code/kmerseek-kmer-frequency-histogram` is just this Mac's local directory name for
the kmerseek checkout -- it's actually on branch `olgabot/kmer-stats-review-fixes` now, which
superseded `olgabot/kmer-frequency-histogram` via PR #36. The `--stats-only` flag this pipeline
now passes to `kmerseek index` requires that branch.)

If Apptainer already cached a bad (arm64) image under `$SCRATCH/apptainer-cache/` from an earlier
run, delete it after pushing the fixed image so the next task re-pulls instead of reusing the
stale `.img` -- a task failing with `the image's architecture (arm64) could not run on the host's
(amd64)` even though `docker inspect docker.io/olgabot/kmerseek:<tag>` on your Mac shows
`amd64 linux` means this, not a bad push:

```bash
rm "$SCRATCH/apptainer-cache/olgabot-kmerseek-2026-08-19-kmer-spectra.img"
```

Then resume the run rather than starting over -- completed tasks are cached by Nextflow's own
`work/` dir, independent of the Apptainer image cache:

```bash
make run NF_ARGS=-resume
```

`nextflow.config`'s `sherlock` profile already points at
`docker://olgabot/kmerseek:2026-08-19-kmer-spectra` -- no edit needed once the push above completes.

**If you're mid-run on the previous image**: the 2026-08-19 image adds `--stats-only` to every
`kmerseek index` call (see `main.nf`'s script block) -- this fixes both the SearchCache
`"Invalid argument: value is too large"` crash (hp k=30) and the chunked-signature-storage
`"IO error: ... Input/output error"` crash (hp_kyte_doolittle k=30) by never persisting a
searchable index at all, since this pipeline only ever wanted the spectrum CSV. Old image, no
`--stats-only`: tasks past the failure ksize keep failing. New image: `-resume` picks up cleanly,
no need to restart from scratch.

## 2. Confirm Sherlock filesystem paths

From a Sherlock login node:

```bash
echo "$SCRATCH"
echo "$GROUP_HOME"
```

The `sherlock` profile caches Apptainer images under `$SCRATCH/apptainer-cache` automatically.
Use `$SCRATCH` for the pipeline's `--outdir` and working directory too -- `$HOME` has a small
quota and isn't meant for pipeline output.

## 3. Transfer the pipeline directory and input fasta

Pipeline code goes over git, not rsync -- `$SCRATCH/kmer-spectra` on Sherlock is a symlink into
a sparse checkout of this repo, not a plain rsync target. **One-time setup**, from a Sherlock
login node:

```bash
cd "$SCRATCH"
git clone --no-checkout --filter=blob:none https://github.com/seanome/2024-kmerseek-analysis.git kmer-spectra-analysis
cd kmer-spectra-analysis
git sparse-checkout init --cone
git sparse-checkout set nextflow-runs/kmer-spectra
git checkout olgabot/kmer-spectra-analysis
ln -s "$SCRATCH/kmer-spectra-analysis/nextflow-runs/kmer-spectra" "$SCRATCH/kmer-spectra"
```

`data/` and `results-<dataset>/` live inside that sparse-checked-out directory as untracked,
gitignored content alongside the tracked pipeline files, the same as on your Mac -- `git pull`
never touches them.

`$SCRATCH/kmer-spectra` is only for your own `cd` convenience. `main.nf`'s `--fasta` paths are
relative to this directory itself (its launchDir), not that symlink, because Apptainer's
`autoMounts` binds paths based on the pipeline's real (non-symlinked) directory tree, and a path
reached through the `$SCRATCH/kmer-spectra` symlink is a second, unrelated absolute path outside
that tree that never gets bound -- the input stages in fine on the host but the symlink is
dangling inside the container, producing a "FASTA file not found" error that's confusing because
the file plainly does exist when you check.

After the one-time setup, syncing code and data is:

```bash
cd nextflow-runs/kmer-spectra   # on your Mac
make sync-pipeline sync-data
```

`sync-pipeline` pushes your local commits and fast-forwards the Sherlock checkout with `git
pull --ff-only` -- it fails loudly instead of silently overwriting if the two have diverged,
so commit your changes on the Mac first. `sync-data` still uses rsync (with `--checksum`,
since these files change identity but not necessarily size/mtime -- cheap here since the files
are only ~100-150 MB each): data files aren't code, so git isn't the right tool for them.

## 4. Launch the pipeline

Nextflow's own head process is lightweight (mostly polling SLURM), but it needs to survive an
SSH disconnect and the docs ask that anything nontrivial not run bare on a login node -- run it
inside `tmux`. One `make run` sweeps every dataset in `DATASETS` (default `k2,k3`) in a single
nextflow invocation and blocks in the foreground:

```bash
ssh sherlock
tmux new -s kmer-spectra
cd "$SCRATCH/kmer-spectra"
make run                        # DATASETS=k2,k3 by default
# or: DATASETS=k2,k3,uniref50 make run

# detach: ctrl-b d ; reattach later with: tmux attach -t kmer-spectra
```

From another pane or session, `make status` shows the SLURM queue.

Each `(dataset, alphabet, ksize)` combo becomes its own `sbatch` job on `hns` (`--account=ayeletv`),
up to 20 concurrent across *all* selected datasets together (`maxForks = 20` in the profile --
raise or lower depending on how `hns` is looking that day). That's a real improvement over
running datasets as separate invocations: Nextflow's scheduler now knows about all of them at
once instead of each assuming it owns the whole `maxForks` budget by itself.

## 5. Pull results back

```bash
cd nextflow-runs/kmer-spectra   # on your Mac
make pull-k2 pull-k3            # or: make pull-all
```

`pull-uniref50`/`pull-uniref90`/`pull-uniref100` aren't in `pull-all` -- pull them explicitly
once their runs actually exist, rather than have a routine `make pull-all` fail on a results
dir that isn't there yet.

## 6. UniRef50/90/100

Same pipeline, three much bigger real-sequence databases in place of Swiss-Prot -- no shuffled
controls for these yet (see below). Verified against `curl -sIL` on 2026-08-18, not a guess:

| database  | compressed fasta | vs. Swiss-Prot |
|-----------|-------------------|----------------|
| Swiss-Prot | 89 MB            | 1x (baseline)  |
| UniRef50  | 8.8 GB            | ~94x           |
| UniRef90  | 32.1 GB           | ~344x          |
| UniRef100 | 63.1 GB           | ~676x          |

Download directly on Sherlock (avoids moving 8-63 GB through your Mac and back twice). Same
stall-detection retry loop as the bulk-download lesson from other multi-GB EBI/UniProt
transfers: a bare curl on a file this size reliably stalls on a dead connection and then hangs
for hours before curl's own timeout notices, so this resumes (`-C -`) and aborts within ~60s of
a stall instead, looping until the file matches the known `Content-Length`:

```bash
ssh sherlock
cd "$SCRATCH/kmer-spectra"
make download-uniref50     # ~10-20 min depending on hns's network path
make download-uniref90     # ~40-70 min
make download-uniref100    # ~80-140 min
# or all three back to back (sequential, not concurrent -- ~2-4h total):
make download-uniref
```

Then add it to a `make run` -- these are real biological sequences, not a null control, so the
full 9-alphabet sweep runs (unlike k2/k3's `--hp_only`), with the HP-family ksize floor raised
from 15 to 18 by default (`main.nf`'s `hp_kmin_uniref` param):

```bash
DATASETS=k2,k3,uniref50 make run NF_ARGS=-resume
# or by itself: DATASETS=uniref50 make run NF_ARGS=-resume
```

**Run uniref50 first, by itself, before adding 90 or 100.** `hp_kmin_uniref`'s companion,
`mem_scale_uniref` (default 1), leaves `indexAndSpectrum` at Swiss-Prot's own memory tiers
(96 GB for HP-family, 16 GB for protein/dayhoff, capped at 176 GB) -- deliberately *not*
pre-scaled by the ~94x/344x/676x size ratios above, because that ratio is a fasta-byte-size
proxy, not a measurement of kmerseek's actual peak RAM at this scale, and guessing memory
numbers without evidence is exactly what cost real SLURM re-queue cycles getting Swiss-Prot's
HP-family combos right (see `main.nf`'s `taskMemory` comment for that history). Once uniref50
finishes or fails, check `kmer_spectra.*.trace.txt`'s `peak_rss` column and set a real
`mem_scale_uniref` for 90/100 from that, e.g.:

```bash
DATASETS=uniref90 make run NF_ARGS="-resume --mem_scale_uniref 4"
```

**If a combo dies even after retrying at 176 GB (hns's practical per-node ceiling), it doesn't
fit on hns at all -- no `mem_scale_uniref` will fix that, since `NODE_MEM_CAP` (default 176 GB,
`params.node_mem_cap_gb`) stops retries from ever requesting more than a node actually has.**
This is exactly what happened on the first real uniref50 attempt: `hp` k=26 was `Killed` at
130K/~65M sequences processed, on an alphabet that already needed the 96 GB tier just for
Swiss-Prot's 573K sequences. Switch to Sherlock's `bigmem` partition (up to 4096 GB/node vs
hns's ~186 GB) and raise the memory cap to match:

```bash
DATASETS=uniref50 make run NF_ARGS="-resume --queue bigmem --node_mem_cap_gb 3800 --node_mem_min_gb 256 --mem_scale_uniref 8"
```

**`bigmem` also enforces its own hard minimum (currently 256 GB) -- sbatch rejects the
submission outright below it, before the job even queues.** `--node_mem_min_gb 256` is required
alongside `--queue bigmem`, not optional: `mem_scale_uniref` scales protein/dayhoff's 16 GB base
tier the same as HP-family's 96 GB one, so `mem_scale_uniref=8` alone asks bigmem for only
128 GB (16*8) on protein/dayhoff combos -- under the floor. `node_mem_min_gb` raises anything
below it up to the floor without touching combos already above it (HP-family at scale=8 is
already 768 GB, well clear of 256 GB).

Tune `mem_scale_uniref`/`node_mem_cap_gb` down once you've seen a real `peak_rss` in the trace
file -- 8x/3800GB here is headroom to get a first successful run, not a measurement.
`node_mem_min_gb` should stay at bigmem's actual floor regardless (confirm it hasn't changed:
the error message states it directly if sbatch rejects a submission).

**Shuffled (ushuffle order-2/order-3) null controls for UniRef50/90/100 don't exist yet.** The
existing `uniprot_sprot.ushuffle_k2/k3.fasta.gz` were generated by a one-off script outside this
repo (using the Python `ushuffle` package at `~/code/ushuffle`), not something this pipeline can
reproduce yet -- shuffling hundreds of millions of sequences at UniRef90/100 scale is its own
nontrivial job, not a quick addition. Worth confirming uniref50's real-sequence run is feasible
on `hns` memory/time budgets before building that out, since it'd only multiply the same risk
across more variants without new information.
