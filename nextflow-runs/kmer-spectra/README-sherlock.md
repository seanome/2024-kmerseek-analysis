# Running the k-mer spectra pipeline on Sherlock

Everything below needs to run from your own terminal, not through Claude Code: Sherlock login
requires an interactive Duo push, and the `sherlock` SSH alias signs in via the 1Password SSH
agent, whose socket (`~/.1password/agent.sock`) isn't reachable from this sandboxed session.

Account: group `ayeletv`, using the `hns` school-condo partition (`groups` / `sh_part` on
Sherlock). `hns` had ~7.4k jobs queued vs. ~95k on the public `normal` partition at last check --
worth re-running `sh_part` before a big submission to confirm that's still true.

## 1. Build and push the image (from your machine, once local resources are free)

The pipeline's Dockerfile does a real `cargo build --release` from source -- avoid running this
while another local job is using CPU.

```bash
cd nextflow-runs/kmer-spectra
docker build -t kmerseek-spectra:latest \
  --build-arg GIT_SHA=b2dfde27f368a4e99a73b429d0c772ce932fd9e3 \
  -f Dockerfile \
  /Users/olga/code/kmerseek-kmer-frequency-histogram

# Sherlock's Apptainer pulls from a registry (no local Docker/Apptainer bridge on macOS).
# Push to Docker Hub, which Sherlock can reach anonymously:
docker tag kmerseek-spectra:latest docker.io/olgabot/kmerseek:2026-08-17-kmer-spectra
docker push docker.io/olgabot/kmerseek:2026-08-17-kmer-spectra
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

```bash
rsync -avz --progress \
  /Users/olga/code/2024-kmerseek-analysis/.claude/worktrees/kmer-spectra-analysis/nextflow-runs/kmer-spectra/ \
  sherlock:'$SCRATCH/kmer-spectra/'

rsync -avz --progress -e ssh --checksum \
  /Users/olga/data/uniprot/uniprot_sprot.ushuffle_k2.fasta.gz \
  /Users/olga/data/uniprot/uniprot_sprot.ushuffle_k3.fasta.gz \
  sherlock:'$SCRATCH/kmer-spectra/data/'
```

(`--checksum` because these files change identity but not necessarily size/mtime; the flag is
cheap here since the files are only ~100-150 MB each.)

## 4. Launch the pipeline

Nextflow's own head process is lightweight (mostly polling SLURM), but it needs to survive an
SSH disconnect and the docs ask that anything nontrivial not run bare on a login node. Use
`tmux`/`screen`, or launch inside an `sh_dev` session:

```bash
ssh sherlock
tmux new -s kmer-spectra
cd "$SCRATCH/kmer-spectra"
ml load nextflow   # or use whatever module/venv provides nextflow on Sherlock -- check `ml avail nextflow`
nextflow run main.nf -profile sherlock \
  --fasta "$SCRATCH/kmer-spectra/data/uniprot_sprot.ushuffle_k2.fasta.gz" \
  --outdir "$SCRATCH/kmer-spectra/results-ushuffle-k2" \
  --hp_only true

# detach: ctrl-b d ; reattach later with: tmux attach -t kmer-spectra
```

Repeat for the k3 fasta with a separate `--outdir`. Each `(alphabet, ksize)` combo becomes its
own `sbatch` job on `hns` (`--account=ayeletv`), up to 20 concurrent (`maxForks = 20` in the
profile -- raise or lower depending on how `hns` is looking that day).

## 5. Pull results back

```bash
rsync -avz --progress \
  sherlock:'$SCRATCH/kmer-spectra/results-ushuffle-k2/' \
  /Users/olga/data/kmerseek-kmer-spectra-ushuffle-k2/

rsync -avz --progress \
  sherlock:'$SCRATCH/kmer-spectra/results-ushuffle-k3/' \
  /Users/olga/data/kmerseek-kmer-spectra-ushuffle-k3/
```
