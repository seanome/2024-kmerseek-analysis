# E3 — detection-gap benchmark, missing baseline arms

Companion pipeline to `notebooks/340_e3_detection_gap_baselines.ipynb`.
It only produces the arms that are **not already on disk**. Nothing here duplicates
`../pfam-benchmark-tools` (DIAMOND, MMseqs2 seq-seq, MMseqs2 iterative, phmmer) or
`../qfo-pfam-benchmark` (kmerseek HP).

Every arm writes the same 4-column TSV the notebook harness
(`notebooks/e3_detection_gap_utils.py`) reads:

```
human_accession <TAB> species_accession <TAB> score <TAB> evalue
```

gzipped to `~/data/e3-detection-gap/<arm>/human_vs_<species>.<arm>.tsv.gz`.

## Arm status as of 2026-08-11

| Arm | Priority | Data on disk? | What it needs |
|---|---|---|---|
| kmerseek HP (k=10–40) | — | **yes** | — |
| DIAMOND | — | **yes** | — |
| MMseqs2 seq-seq | — | **yes** | — |
| phmmer | — | **yes** | — |
| MMseqs2 iterative profile | — | **yes** | — |
| HHblits vs UniRef30 | 1 | **no** — existing files are 20-byte empty gzips | UniRef30/UniRef50 download; rerun `../pfam-benchmark-tools` with `--hhblits_db` |
| jackhmmer 3 iter | 1 | no | UniRef50 (or UniRef30) fasta |
| InterProScan Pfam (ceiling) | 2 | no | local InterProScan install (already at `~/data/interproscan`) |
| FoldSeek on AFDB | 4 | no — but **all 54_339 structures are already downloaded** | nothing; just run it |
| ESM-2 windowed | 3 | no | a GPU |

The HHblits files under `~/data/pfam-benchmark-tools/results/hhblits/` are **not**
"HHblits found nothing". They are empty because `../pfam-benchmark-tools/main.nf` was
run with `params.hhblits_db = null`, which makes `hhblitsBuildDB` fall back to
single-sequence a3m profiles with `-n 0` iterations. The arm has never actually run.

## Handoff commands

Nextflow: `/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow`. Resume is `-resume`
(single dash — `--resume` would be parsed as a pipeline parameter).

### 1. FoldSeek — no download needed, run this first

```bash
cd /Users/olga/code/2024-kmerseek-analysis/nextflow-runs/e3-detection-gap
/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf \
    -profile local -resume \
    --arms foldseek \
    --species ciona,fly,worm,mouse,zebrafish
```

### 2. jackhmmer — needs a background database

UniRef50 is ~15 GB compressed and is enough for a 3-iteration profile; UniRef30
(the HH-suite clustered database) is ~50 GB. On this project **egress volume, not
compute, is the budget constraint**, so prefer UniRef50 unless a reviewer asks for
UniRef30 by name. Bare background `curl` on a multi-GB EBI file silently stalls, so
always resume with a speed floor:

```bash
mkdir -p ~/data/uniref
curl -L -C - --speed-limit 50000 --speed-time 120 \
    -o ~/data/uniref/uniref50.fasta.gz \
    https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip ~/data/uniref/uniref50.fasta.gz

cd /Users/olga/code/2024-kmerseek-analysis/nextflow-runs/e3-detection-gap
/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf \
    -profile local -resume \
    --arms jackhmmer \
    --jackhmmer_db $HOME/data/uniref/uniref50.fasta \
    --species ciona,fly,worm
```

### 3. HHblits — reuse the existing pipeline, do not rewrite it

```bash
# UniRef30 for HH-suite: https://wwwuser.gwdg.de/~compbiol/uniclust/current_release/
cd /Users/olga/code/2024-kmerseek-analysis/nextflow-runs/pfam-benchmark-tools
/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf -resume \
    --hhblits_db $HOME/data/uniclust30/UniRef30_2023_02 \
    --skip_psalm true
```

Then point the harness at it: the notebook's `hhblits` arm already reads
`~/data/pfam-benchmark-tools/results/hhblits/human_vs_<species>.hhblits.tsv.gz`
and will flip from `no data (empty output)` to `ok` on its own.

### 4. InterProScan Pfam ceiling

```bash
cd /Users/olga/code/2024-kmerseek-analysis/nextflow-runs/e3-detection-gap
/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf \
    -profile local -resume \
    --arms interproscan \
    --species ciona
```

### 5. ESM-2 PLM arm — GPU, hand off

```bash
cd /Users/olga/code/2024-kmerseek-analysis/nextflow-runs/e3-detection-gap
/Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow run main.nf \
    -profile awsbatch -resume \
    --arms esm2 \
    --species ciona
```

`-profile awsbatch` reads queue/bucket/region from environment variables
(`NF_BATCH_QUEUE`, `NF_WORKDIR_S3`, `AWS_REGION`) sourced from a **gitignored** `.env`.
This repo is public — never commit real account, queue or bucket names.

## Notes that cost time to rediscover

- Inline Python uses the absolute conda interpreter as the shebang
  (`#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3`). `beforeScript`
  and the `conda` directive do not reliably activate an environment in Nextflow 25.x,
  and `/usr/bin/env python3` resolves to the system Python.
- Processes with an inline-Python shebang set `container = null` in `nextflow.config`;
  otherwise the shebang path does not exist inside the image.
- `docker.runOptions = '--entrypoint ""'` — Nextflow calls `/bin/bash` inside the
  container and an image `ENTRYPOINT` breaks it.
- No `/usr/bin/time -l` anywhere: it is macOS-only, Debian silently skips it, and the
  failure surfaces much later as a missing-output error.
- `peak_rss` and `%cpu` are not recorded on macOS without containers. The cost /
  throughput table in notebook 340 needs a **Linux** run to be quotable.
