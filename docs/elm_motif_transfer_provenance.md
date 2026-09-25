# ELM motif transfer (notebook 250): where every input and number comes from

Branch `olgabot/elm-motif-transfer`, PR #51. Every number in notebook 250, in the PR
description and in `tables/250_*.csv` is written by a script in this repository, listed below
in the order they run. Data files live outside git on Olga's Mac; their paths, dates and
SHA-256 checksums are here so a rerun can check it read the same files.

## Inputs

| input | path on the Mac | what it is | version / date | SHA-256 |
|---|---|---|---|---|
| ELM instances | `/Users/olga/data/elm-motif-transfer/elm_instances.tsv` | `http://elm.eu.org/instances.tsv?q=*`, downloaded by `scripts/fetch_elm.py` | ELM download format 1.4, downloaded 2026-09-24 19:56 (date in the file header); 4_277 instances | `d40168eb3d9714bf87c1400f2a2a71f6a0fa3a9d27be593c4d1550684e06e18b` |
| ELM classes | `/Users/olga/data/elm-motif-transfer/elm_classes.tsv` | `http://elm.eu.org/elms/elms_index.tsv`, same script | format 1.4, downloaded 2026-09-24 07:44; 353 classes | `aef1c5a26dde5d2d9dab310163b11c11a0694517576ac898a55b07dfbb193f2a` |
| human proteome | `/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota/UP000005640_9606.fasta` | Quest for Orthologs reference proteome, the one the region benchmark searches | QfO release 2020_04 | `3793b9bfcc956a278494734bb775a010513f5764d8047fcdb283d2aeaa7fefb8` |
| nine target proteomes and their `.idmapping` files | same release folder; file names from `nextflow-runs/qfo-pfam-region-benchmark/assets/qfo_species.tsv` | proteomes, and UniProt cross-references including each protein's OMA group | QfO release 2020_04 | not recorded per file |
| midi-plus human queries | `/Users/olga/data/qfo-pfam-region-midi-plus/extract/244_human_queries.fasta` | the 998 human sequences the midi-plus run searched, written by `scripts/prep_244_instance_covariates.py` | 2026-09-24 | `95f77e8c9ddfb91249ef294d1d758048dc4ccc9725280cbceb71d7f265b13425` |
| human AlphaFold models | `/Users/olga/data/qfo-pfam-region-midi-plus/structures_human/` (midi-plus proteins) and `/Users/olga/data/alphafold_structures/` (all others) | AlphaFold DB models; the cache resolves the current model through the AFDB API, `AF-<acc>-F1-model_v6.cif` at the time of download | v6 | per file, not recorded |
| human Swiss-Prot features | `/Users/olga/data/qfo-pfam-region-midi-plus/truth_swissprot/human_swissprot_truth.parquet` | midi-plus run output (`bin/build_swissprot_truth.py`) | 2026-09-11 | not recorded |
| Swiss-Prot flat file | `/Users/olga/data/uniprot/uniprot_sprot.dat.gz` | UniProtKB/Swiss-Prot, read for MOTIF and REGION features with `/note` and `/evidence` on the orthologs | file dated 2025-07-03; the release number is not written in the file on disk | `9dbe6dee59163d4c47ff450e6923b71c5063f0f8d45503012846e1c439a71fd3` |
| midi-plus trace | `/Users/olga/data/qfo-pfam-region-midi-plus/traces/midi-plus.trace.txt` | Nextflow trace of the midi-plus run on Sherlock; peak memory of every kmerseek search on the nine proteomes | file dated 2026-09-02 | `cc3f8eff45a782df7b9dec3a9d8b3dd5791d8d32ae2f5bc7ff6e46d1d3316c37` |
| kappa on Pfam pairs | `nextflow-runs/qfo-pfam-region-benchmark/assets/kappa_by_alphabet.pfam_a_38.2_seed_pairs_20-30pct_identity.tsv` (in git, from PR #52) | kappa per alphabet on Pfam-A 38.2 seed pairs at 20-30% identity, notebook 230 | 2026-09-21 | in git |
| HGNC | `/Users/olga/data/qfo-pfam-region-midi-plus/hgnc_complete_set.txt` | gene groups for the query split | 2026-09-11 | not recorded |

## Tools

| tool | version | where it runs |
|---|---|---|
| Python | 3.13.11 (conda env `2025-kmerseek-analysis`) | every script |
| polars / numpy | 1.34.0 / 2.3.5 | every script |
| metapredict | 3.0.1, threshold 0.5 (in the same env; the amd64 `olgabot/metapredict` image crashes under Rosetta on Apple Silicon) | `prep_elm_inputs.py`, `elm_stage0_flanks.py` |
| MAFFT | 7.526 (2024/Apr/26), conda env `orthofinder`, `--anysymbol --localpair --maxiterate 1000` | `elm_stage0_flanks.py` |
| kmerseek | 0.4.0, image `kmerseek-region:0.4.0-rc5-arm64` (image id `sha256:2691409980c4…`), built from `nextflow-runs/qfo-pfam-region-benchmark/Dockerfile` | `elm_seed_floor.py` (`index --stats-only`) |

## Scripts, in order

| step | command | reads | writes |
|---|---|---|---|
| 1. ELM tables | `python scripts/fetch_elm.py` | ELM URLs | `elm_instances.tsv`, `elm_classes.tsv`, `elm_instances_tp.parquet` (4_047 "true positive" instances, coordinates 0-based end-exclusive) |
| 2. Stage 0 | `python scripts/elm_stage0_flanks.py` (or `make run-elm-stage0` in `nextflow-runs/qfo-pfam-region-benchmark`) | 1, QfO human proteome and `.idmapping`, nine target proteomes, AlphaFold models, human Swiss-Prot features | `stage0/stage0_instances.parquet` (2_160 usable instances), `stage0/stage0_pairs.parquet` (3_871 pairs with a 1:1 OMA ortholog), `stage0/alignments/`, figures and tables `250_stage0_*` |
| 3. Kappa on ELM pairs | `python scripts/elm_kappa.py` | 2, `notebooks/hp_conservation_utils.py` | `stage0/elm_kappa_*.parquet`, `tables/250_elm_kappa_by_species.csv` |
| 4. Tiers | `python scripts/elm_tiers.py` | 1, 2, Swiss-Prot flat file | `tiers/regex_on_target.parquet`, `tier_experimental`, `tier_swissprot`, `tier_regex_projected`, `regex_fails_stratum` |
| 5. Seed lengths | `python scripts/elm_seed_floor.py` | QfO human proteome, kmerseek image, midi-plus trace | `seed_floor/human.<alphabet>.k<k>.csv`, `seed_floor/scan.csv`, `tables/250_two_k_per_alphabet.csv` |
| 6. Notebook | `python scripts/make_nb250.py`, then `jupyter nbconvert --execute` on `notebooks/250_elm_motif_transfer.ipynb` | 1-5 | the notebook, `figures/250_*`, `tables/250_*` |

All data outputs are under `/Users/olga/data/elm-motif-transfer/` (`stage0/`, `tiers/`,
`seed_floor/`). `scripts/prep_elm_inputs.py` and `scripts/reduce_elm_landing.py` belong to the
first, pooled-target design (2026-09-24) and are not used by the steps above.

## How the ortholog pairs are made

Nothing in this project infers orthology. A human protein and a protein from one of the nine
species form a pair when both carry the same OMA group in their QfO `.idmapping` file and each
is the only member of that group from its species (1:1). This is the rule of
`nextflow-runs/qfo-dnds-omega/bin/build_ortholog_pairs.py`. E. coli has no OMA cross-references
in this release, so no pairs. Chicken's are sparse: that script records 5_038 of 17_837
chicken proteins with one.

## Truth

The only truth is on the human side: ELM instances ELM marks "true positive" (supported by
experiments), kept when the instance lies inside the QfO 2020_04 human sequence and its class
regex matches a stretch overlapping it. The orthologs and their labels (step 4) are not truth
for the cover score.
