# Running the k-mer spectra pipeline on Sherlock

Everything below needs to run from your own terminal, not through Claude Code: Sherlock login
requires an interactive Duo push, and the `sherlock` SSH alias signs in via the 1Password SSH
agent, whose socket (`~/.1password/agent.sock`) isn't reachable from this sandboxed session.

A `Makefile` in this directory wraps every command below. `build-image`/`push-image`/
`sync-pipeline`/`sync-data`/`pull-k2`/`pull-k3` run on your Mac; `run-k2`/`run-k3`/`status`
run on Sherlock (`ssh sherlock`, `cd $SCRATCH/kmer-spectra`, `make <target>`). `run-k2` and
`run-k3` each launch nextflow in the foreground from their own `run-kN/` subdirectory, so
their `work/`, trace, timeline, and report files don't collide -- run them in two separate
tmux panes to get both ksizes running at the same time.

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
#     --build-arg GIT_SHA=b2dfde27f368a4e99a73b429d0c772ce932fd9e3 \
#     -f Dockerfile \
#     /Users/olga/code/kmerseek-kmer-frequency-histogram
#   docker tag kmerseek-spectra:latest docker.io/olgabot/kmerseek:2026-08-17-kmer-spectra
#   docker push docker.io/olgabot/kmerseek:2026-08-17-kmer-spectra
```

If Apptainer already cached a bad (arm64) image under `$SCRATCH/apptainer-cache/` from an earlier
run, delete it after pushing the fixed image so the next task re-pulls instead of reusing the
stale `.img`:

```bash
rm "$SCRATCH/apptainer-cache/olgabot-kmerseek-2026-08-17-kmer-spectra.img"
```

`nextflow.config`'s `sherlock` profile already points at
`docker://olgabot/kmerseek:2026-08-17-kmer-spectra` -- no edit needed once the push above completes.

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

`data/`, `run-k2/`, `run-k3/`, and `results-ushuffle-*/` live inside that sparse-checked-out
directory as untracked, gitignored content alongside the tracked pipeline files, the same as on
your Mac -- `git pull` never touches them.

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
inside `tmux`, not bare on the login node. `make run-k2` and `make run-k3` each block in the
foreground and launch from their own `run-kN/` subdirectory, so the two runs' `work/`, trace,
timeline, and report files don't collide. Run them in two separate tmux panes to get both ksizes
running at the same time:

```bash
ssh sherlock
tmux new -s kmer-spectra
# split the window (ctrl-b %), then in each pane:
cd "$SCRATCH/kmer-spectra" && make run-k2
cd "$SCRATCH/kmer-spectra" && make run-k3

# detach: ctrl-b d ; reattach later with: tmux attach -t kmer-spectra
```

From a third pane or another session, `make status` shows the SLURM queue.

Each `(alphabet, ksize)` combo becomes its own `sbatch` job on `hns` (`--account=ayeletv`), up to
20 concurrent per run (`maxForks = 20` in the profile -- raise or lower depending on how `hns` is
looking that day). With both k2 and k3 running, that's up to 40 concurrent jobs against the
`ayeletv` account.

## 5. Pull results back

```bash
cd nextflow-runs/kmer-spectra   # on your Mac
make pull-k2 pull-k3            # or: make pull-all
```
