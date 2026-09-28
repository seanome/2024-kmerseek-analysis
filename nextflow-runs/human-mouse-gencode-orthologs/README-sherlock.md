# Running the family-AUC / RBH-F1 / metric-leaderboard sweeps on Sherlock

Everything below needs to run from your own terminal, not through Claude Code: Sherlock login
requires an interactive Duo push, and the `sherlock` SSH alias signs in via the 1Password SSH
agent, whose socket (`~/.1password/agent.sock`) isn't reachable from this sandboxed session.

## Why this exists

`run_family_auc_direct.sh` / `run_rbh_f1_direct.sh` (in this directory) already do the real
work -- read the existing kmerseek search results, bootstrap an AUC/F1 per (encoding, ksize)
combo -- but sequentially, on one Mac. That's the part that was "taking forever," not the
kmerseek search itself (already done; the ~267GB of results this needs already exist locally).
This setup reruns the SAME computation via Nextflow + SLURM on Sherlock, one task per combo in
parallel, instead of one combo after another on a laptop.

Deliberately NOT re-running the search: `params.skip_search = true` (set by `-profile sherlock`)
skips `indexDatabase`/`searchHumanVsMouse`/`convertResultsToParquet`/`evaluateOrthologs`/
`multiQC` entirely, rather than relying on Nextflow's `-resume`/storeDir to skip them. They
wouldn't skip cleanly anyway: `convertResultsToParquet` deletes each combo's raw
`results.csv.zst` after converting it to parquet, so a combo that's already fully processed
looks *missing* to `searchHumanVsMouse`'s own storeDir check, not done -- resuming would
re-trigger a multi-day kmerseek re-search across ~100 combos for no reason. See
`run_rbh_f1_direct.sh`'s header comment for the original (Mac-local) version of this same
problem.

A `Makefile` in this directory wraps every command below. `build-image`/`push-image`/
`sync-pipeline`/`sync-data`/`pull` run on your Mac; `run`/`status` run on Sherlock (`ssh
sherlock`, `cd $SCRATCH/human-mouse-orthologs-analysis/nextflow-runs/human-mouse-gencode-orthologs`,
`make <target>`).

Account: group `ayeletv`, using the `hns` school-condo partition -- see `sh_part` on Sherlock
to confirm it's still shorter than the public `normal` partition before a big submission.

## 1. Build and push the image (from your machine)

This image is just `polars`/`numpy`/`scipy`/`scikit-learn`/`pyarrow`/`requests` (see
`docker/Dockerfile.python-sweep`), pinned to the exact versions in the local
`2025-kmerseek-analysis` conda env -- no `cargo build`, no kmerseek binary. It does NOT replace
`kmerseek:0.3.1`/`kmerseek:main` (the search-step containers); those stay as they are since
search never runs under `-profile sherlock`.

Sherlock's compute nodes are amd64; a plain `docker build` on Apple Silicon produces an arm64
image Apptainer refuses to run. `make push-image` passes `--platform linux/amd64` for you.

```bash
cd nextflow-runs/human-mouse-gencode-orthologs
make push-image
```

If Apptainer already cached a stale (arm64, or an older tag's) image under
`$SCRATCH/apptainer-cache/` on Sherlock, delete it after pushing so the next task re-pulls:

```bash
rm "$SCRATCH/apptainer-cache/olgabot-kmerseek-python-sweep-2026-08-18.img"
```

## 2. Confirm Sherlock filesystem paths

From a Sherlock login node:

```bash
echo "$SCRATCH"
```

Use `$SCRATCH` for the checkout and all data below -- `$HOME` has a small quota.

## 3. One-time checkout setup

Pipeline code (and the one shared Python module it imports) go over git, not rsync:

```bash
ssh sherlock
cd "$SCRATCH"
git clone --no-checkout --filter=blob:none https://github.com/seanome/2024-kmerseek-analysis.git human-mouse-orthologs-analysis
cd human-mouse-orthologs-analysis
git sparse-checkout init --cone
git sparse-checkout set --no-cone nextflow-runs/human-mouse-gencode-orthologs 'notebooks/ortholog_analysis_utils.py'
git checkout olgabot/human-mouse-orthologs-sherlock
```

`nextflow-runs/human-mouse-gencode-orthologs/data/` (the rsync targets below) lives inside that
sparse-checked-out directory as untracked, gitignored content alongside the tracked pipeline
files -- `git pull` never touches it. This mirrors `../kmer-spectra`'s Sherlock setup, and for
the same reason: Apptainer's `autoMounts` binds paths from the real (non-symlinked) directory
tree Nextflow actually runs in, and `nextflow.config`'s `sherlock` profile points every
`KMERSEEK_*` env var and `params.outdir`/`params.of_tsv` at `${launchDir}/data/...` -- i.e.
relative to wherever this checkout actually is, not a hardcoded path.

After the one-time setup, syncing code is:

```bash
cd nextflow-runs/human-mouse-gencode-orthologs   # on your Mac
make sync-pipeline
```

## 4. Sync data (from your Mac)

```bash
make sync-data
```

This rsyncs three things into `nextflow-runs/human-mouse-gencode-orthologs/data/` on Sherlock:

- `results-human-mouse-orthologs/` minus `indices/` (~255GB -- the 74GB of kmerseek indices
  aren't read by `computeFamilyAuc`/`computeRbhF1`/`computeMetricLeaderboard`, only by the
  search step, which never runs here)
- `results-human-mouse-orthologs-hp-v040/` (~12GB -- the sibling pipeline's k=18-19 HP-only
  results; `ortholog_analysis_utils.EXTRA_DATA_DIRS` falls back to this dir for combos the main
  pipeline doesn't have)
- the one OrthoFinder TSV `computeRbhF1` needs (~24MB), not the full 2.6GB `OrthoFinder/` results
  tree

It does NOT sync the human/mouse FASTA files -- with `skip_search = true`, nothing on Sherlock
ever reads `params.human_fasta`/`params.mouse_fasta`.

~267GB over `ssh` will take a while and can stall. `sync-data` uses `--partial --timeout=120`
so it can resume instead of restarting from zero -- if it dies partway, just run `make
sync-data` again.

## 5. Launch the pipeline

Run inside `tmux`, not bare on the login node (Nextflow's head process needs to survive an SSH
disconnect, and the docs ask that nontrivial work not run directly on a login node):

```bash
ssh sherlock
tmux new -s human-mouse-orthologs
cd "$SCRATCH/human-mouse-orthologs-analysis/nextflow-runs/human-mouse-gencode-orthologs"
make run

# detach: ctrl-b d ; reattach later with: tmux attach -t human-mouse-orthologs
```

Each combo becomes its own `sbatch` job on `hns` (`--account=ayeletv`), up to 4 concurrent
(`maxForks` on `computeMetricLeaderboard`/`computeRbhF1` in the top-level `process{}` block --
raise or lower via `-profile sherlock --maxForks N` depending on how `hns` looks that day and
how many of the >100M-row combos are in flight, since those measured 43-56GB peak RSS locally).

From a third pane or another session, `make status` shows the SLURM queue.

To resume an interrupted run (Nextflow's own `-resume`, using its `work/` dir -- this is
independent of the `params.skip_search` decision above):

```bash
make run NF_ARGS=-resume
```

## 6. Pull results back

```bash
cd nextflow-runs/human-mouse-gencode-orthologs   # on your Mac
make pull
```

This pulls only the three sweeps' own outputs (`206_family_auc/`, `200_rbh_f1/`,
`200_metric_leaderboard/`, and their three `*_all_combos.csv` aggregates) back into the same
local paths notebook 200/206 already read from -- not the (already-present, unchanged) input
results files.
