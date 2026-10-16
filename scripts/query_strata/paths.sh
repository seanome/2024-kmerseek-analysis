# Paths and pinned images shared by the query-strata jobs. Sourced, never run.
#
# The 998 queries are the midi-plus query set of the QfO region benchmark: the human
# proteome file under data/midi-plus/qfo is the subset make_mini_testset.py wrote
# (sha256 04965fb6a2b0f578dbe2c31dc97a5de4f27c5bc990286fd01858eb0ebdb053a7, checked
# 2026-10-16). The 15 targets are the species in all_domain_metrics.parquet
# (sha256 8ba26616...), read from assets/qfo_species.tsv.

QFO_PIPE=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark
QUERY_FASTA=$QFO_PIPE/data/midi-plus/qfo/Eukaryota/UP000005640_9606.fasta
# The whole QfO 2020_04 human proteome the 998 queries were drawn from (20,600 proteins).
HUMAN_PROTEOME=$QFO_PIPE/data/qfo/Eukaryota/UP000005640_9606.fasta
QFO_DIR=$QFO_PIPE/data/midi-plus/qfo

WD=/scratch/users/olgabot/leaderboard-query-strata
IMAGES=$WD/images

# Pinned by digest. mmseqs2 is the image every MMSeqs2 process of the pipeline uses
# (main.nf), so the identity here comes from the same build as the benchmark's searches.
MMSEQS_REF=quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef
BLAST_REF=quay.io/biocontainers/blast@sha256:9dfc69f990c0aeb936276ee591ed32919a79f46dfa34060055e05a050a17959c   # 2.16.0--h66d330f_5
BUSCO_REF=quay.io/biocontainers/busco@sha256:acf4e2f5bf6eb5c195b4165b4261e233974df5b23795319e2e4185569c9c903c   # 6.1.0--pyhdfd78af_2
DEEPTMHMM_REF=docker.io/deeptmhmm/deeptmhmm@sha256:e527883fd2114007c6208c3d764fece40016cc95e209eab93016644c3e7ccb16   # :latest, 2023-02-28
# Python with polars, the image the pipeline's own python processes run in.
PY_IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-2026-08-24-reduced-alphabets.img

MMSEQS_IMG=$IMAGES/mmseqs2.sif
BLAST_IMG=$IMAGES/blast-2.16.0.sif
BUSCO_IMG=$IMAGES/busco-6.1.0.sif
DEEPTMHMM_IMG=$IMAGES/deeptmhmm.sif

BUSCO_LINEAGE=primates_odb12

# label <tab> proteome file, the 15 scored targets of the midi-plus run.
TARGETS="arabidopsis Eukaryota/UP000006548_3702
btaurus Eukaryota/UP000009136_9913
chicken Eukaryota/UP000000539_9031
ciona Eukaryota/UP000008144_7719
clupus Eukaryota/UP000002254_9615
ecoli Bacteria/UP000000625_83333
fly Eukaryota/UP000000803_7227
mdomestica Eukaryota/UP000002280_13616
mouse Eukaryota/UP000000589_10090
ptroglodytes Eukaryota/UP000002277_9598
rnorvegicus Eukaryota/UP000002494_10116
worm Eukaryota/UP000001940_6239
xtropicalis Eukaryota/UP000008143_8364
yeast Eukaryota/UP000002311_559292
zebrafish Eukaryota/UP000000437_7955"
