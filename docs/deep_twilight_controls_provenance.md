# Deep-twilight controls (notebook 252): where every input and number comes from

Branch `olgabot/deep-twilight-controls`, PR #55. Every number in notebook 252 and in the PR
description is printed by a notebook cell or written by a script in this repository, listed
below in the order they run. The pipeline results are not in git. They are on Sherlock under
`/scratch/users/olgabot/2024-kmerseek-analysis-deep-twilight/nextflow-runs/deep-twilight-controls/run/results/`,
and `make pull-results` copies them to `nextflow-runs/deep-twilight-controls/results/` on the
Mac. Their SHA-256 checksums are below so a rerun can check it read the same files.

## Inputs

| input | path | what it is | version / date | SHA-256 |
|---|---|---|---|---|
| 16 family proteins | `nextflow-runs/deep-twilight-controls/assets/deep_twilight_proteins.fasta` (in git) | UniProtKB entries, full sequence including signal peptides, written by `scripts/make_deep_twilight_pairs_tsv.py` | UniProt release 2026_03 (REST API), fetched 2026-09-25 | `4f5772cd8f95bd0920ef9640593f51956b55e74aa68f018fe09f604e2521fde4` |
| functional labels | `tables/deep_twilight_pairs.tsv` (in git) | each protein's labelled residues, copied from its own Swiss-Prot features by the same script; which feature carries which label is the `RULES` table in the script | same fetch | `7536a06f1ea1514e21eeb176e2bcc061c2f3f17be25c9140033853906c00319d` |
| human proteome | Mac: `/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota/UP000005640_9606.fasta`; Sherlock: `/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/qfo/Eukaryota/UP000005640_9606.fasta` (the two copies have the same MD5, `74bf5689099474ca85a96b090c722ece`) | Quest for Orthologs reference proteome, 20_600 proteins; the 8 human family members in it are identical to their UniProt 2026_03 sequences | QfO release 2020_04 | `3793b9bfcc956a278494734bb775a010513f5764d8047fcdb283d2aeaa7fefb8` |
| human AlphaFold models | Sherlock: `/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/structures/human/`, unpacked from `_archives/UP000005640_9606_HUMAN_v6.tar` (5_177_506_304 bytes, dated 2026-08-19) | AlphaFold DB human proteome; the Foldseek database holds 20_336 models for the 20_608 database proteins (the other 272 have no model) | AlphaFold DB v6 | not recorded |
| family AlphaFold models | fetched at run time by `bin/fetch_afdb.py`; versions in `results/structures/afdb_models.tsv` | current model per accession from the AlphaFold DB API; each model's sequence checked equal to the UniProt sequence | v6 (model date 2025-08-01), fetched 2026-09-25 | per file, not recorded |
| kmerseek alphabet and k ladder | `nextflow-runs/deep-twilight-controls/assets/arms.tsv` (in git) | alphabet, k, bits per seed, mismatch penalty and give-up margin from nb 241's `arms.csv`, without gbmr7 k8 and k10 (commit c0be2ef) | 150 rows | `6f406a0c83cf7d46a6797dc3e67059aaf658e950f5f664d96ea1f8503c7a223a` |

## Tools

| tool | version | where it runs |
|---|---|---|
| Nextflow | 25.04.7 (`ml load biology nextflow` on Sherlock) | pipeline head |
| kmerseek | branch `olgabot/run-evalue`, commit `5fdfdcc5e30dc1e78110ff0c24846637e0509439`; image `docker.io/olgabot/kmerseek@sha256:3e4f4d77f996ee97addbc235b68b3afd2f21130865e0eb2cba7abf7968a2d140` (tag `2026-09-25-run-evalue-5fdfdcc-rtcompat`), built from `nextflow-runs/deep-twilight-controls/Dockerfile` by `make push-image` | `kmerseekArm`; the same image runs every python step (python 3.13, polars[rtcompat] 1.43.2, numpy 2.5.2, pyarrow 25.0.1) |
| HMMER (phmmer) | 3.4, `quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c` | `phmmerSearch`: `--max -E 1000 --domE 1000 --incE 1000 --incdomE 1000 -A` |
| MMseqs2 | 18.8cc5c, `quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef` | `mmseqsSearch`: `-a -s 7.5 --exhaustive-search 1 -e 1000 --max-seqs 100000` |
| Foldseek | 10.941cd33, `quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a` | `foldseekSearch`: `-a --exhaustive-search 1 -e 1000 --max-seqs 100000` |
| EMBOSS needle | 6.6.0.0, `quay.io/biocontainers/emboss@sha256:a9bf499a690de7950a3f553793ea45daa947c63946a43edaefdbcfb0e9960a97` | `needleIdentity`: EBLOSUM62, gap open 10, gap extend 0.5 |
| Python for the notebook | 3.13.11, polars 1.34.0, matplotlib 3.10.8 (conda env `2025-kmerseek-analysis`) | `scripts/make_nb252.py`, notebook 252 |

## Scripts, in order

| step | command | reads | writes |
|---|---|---|---|
| 1. Labels and sequences | `python scripts/make_deep_twilight_pairs_tsv.py` | UniProt REST | `tables/deep_twilight_pairs.tsv`, `assets/deep_twilight_proteins.fasta` |
| 2. Image | `make push-image` in `nextflow-runs/deep-twilight-controls` | kmerseek at `KMERSEEK_SHA` (`git archive`) | the kmerseek image |
| 3. Pipeline | `make run` in `nextflow-runs/deep-twilight-controls` on Sherlock | 1, human proteome, human AlphaFold models, `assets/arms.tsv` | `results/database/`, `results/kmerseek/<alphabet>.k<k>.{hits.parquet,pairs.tsv,log}`, `results/baselines/{phmmer,mmseqs2,foldseek}.tsv`, `results/identity/needle_identity.tsv`, `results/structures/` |
| 4. Scoring | the pipeline's `scoreTransfer` process (`bin/score_transfer.py`). In this run the head job was cancelled before it ran, so it was run on the Mac in the same image, on the pulled results, with the command in `scoreTransfer` | 3, 1 | `results/outcomes.parquet`, `results/calls.parquet`, `results/pair_kmers.parquet` |
| 5. Notebook | `python scripts/make_nb252.py`, then `jupyter nbconvert --to notebook --execute --inplace notebooks/252_deep_twilight_controls.ipynb` | 4, 1, `results/kmerseek/*.log` (extension status) | the notebook, `figures/252_deep_twilight_label_transfer.png` |

## Result files read by the notebook

| file | SHA-256 |
|---|---|
| `results/outcomes.parquet` | `290bb8f717f0f454bc305f931e2d1e54917488a3b60bb94b252d2f327305104e` |
| `results/calls.parquet` | `a3f7345178be5ff4d03c8b871e8b4d3afe698fd5b08187df1d5b3ef9815ac614` |
| `results/pair_kmers.parquet` | `ded321b52ab99cd2d6e8f003dc36fa5a6268b3f7e1f36505577c16b9a2f9d5f4` |
| `results/database/database.fasta` | `39a8abb7b59c23ae16a356ed19e1964fd483c493147b417799f317a7d384cd8a` |
| `results/identity/needle_identity.tsv` | `9eeb2c11af472c317a1655d0f58500f71167f94ca85b7b60b57c9f2944f22d93` |
| `results/baselines/phmmer.tsv` | `27e57c52c22489c3d4c44e92faa5f7ede8117ec60dfa060a020894ce74a150b9` |
| `results/baselines/mmseqs2.tsv` | `c4c2543acbdbd80d29d0ce88200186a25aa09c186ee3b952120dfaacfe993199` |
| `results/baselines/foldseek.tsv` | `ccdd6d55639aa2bcb9fc900e0950baf535c5ea7c224958619bb5ea1a934c7b7c` |

## Family references

Each DOI in notebook 252 was read from Crossref (`api.crossref.org/works/<doi>`) and resolves
at doi.org, checked 2026-09-25: `10.1016/0022-2836(80)90373-3`,
`10.1146/annurev.pp.35.060184.002303`, `10.1016/s0021-9258(18)95873-4`,
`10.1007/bf02102453`, `10.1002/j.1460-2075.1988.tb03109.x`.
