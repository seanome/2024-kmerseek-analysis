MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
NEXTFLOW     ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
export NXF_JAVA_HOME ?= /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm

# Swiss-Prot release 2026_03, the release the chromosome 6 query list was taken from
# (assets/chr6_queries.2026_03.tsv). md5 in build_summary/inputs_and_outputs.md5.
SWISSPROT_DAT ?= $(HOME)/data/uniprot/uniprot_sprot.2026_03.dat.gz
OUTDIR        ?= $(HOME)/data/regions-index-pilot/results
LAUNCH_DIR    ?= $(HOME)/data/regions-index-pilot/launch
# A kmerseek v0.4.0 (0aed5ca) binary for the local profile; Sherlock uses the image.
KMERSEEK      ?= kmerseek

.PHONY: fetch-inputs build-local test-local fetch-swissprot-sherlock run-sherlock running

# --- Sherlock --------------------------------------------------------------------------
# Paths on Sherlock. $(SCRATCH) is purged after 90 days untouched, so the results that matter
# get copied back (the comparison tables are small).
SH_DATA       ?= $(SCRATCH)/regions-index-pilot
SH_DAT        ?= $(SH_DATA)/uniprot_sprot.2026_03.dat.gz
SH_OUTDIR     ?= $(SH_DATA)/results
SH_LAUNCH_DIR := $(MAKEFILE_DIR)runs/full
SWISSPROT_URL := https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
SWISSPROT_MD5 := a31b5d0756f0a8f94164bf20b703812c
# The Nextflow head runs as a SLURM job inside tmux (scripts/nf-head), never on a login
# node, whose processes are killed after ten CPU-minutes.
HEAD_SUBMIT = HEAD_ACCOUNT=ayeletv HEAD_PARTITION=hns HEAD_QOS=normal HEAD_TIME=2-00:00:00 \
              HEAD_CPUS=2 HEAD_MEM=16G HEAD_SESSION=regions-pilot HEAD_JOBNAME=nfdrv-regions-pilot \
              HEAD_HINT_WATCH="make running" HEAD_HINT_STOP="scancel --name=nfdrv-regions-pilot" \
              $(MAKEFILE_DIR)../../scripts/nf-head
# First asks per task, from the laptop measurements in BUILD_REPORT.md ("Resources").
SH_RESOURCES ?= --kmerseek_index_memory_gb 120 --kmerseek_index_hours 2 \
                --kmerseek_search_memory_gb 100 --kmerseek_search_hours 2 \
                --mmseqs_memory_gb 64 --mmseqs_hours 8

## Re-fetch the chromosome 6 query list and DisProt disorder into assets/ (committed).
fetch-inputs:
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 $(MAKEFILE_DIR)tools/fetch_inputs.py --assets $(MAKEFILE_DIR)assets

## Build the queries, truth table, both indexes and both decoy copies on this machine.
## About 3 minutes on the laptop.
build-local:
	@mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && $(NEXTFLOW) run $(MAKEFILE_DIR)main.nf -entry build -profile local \
	  --swissprot_dat $(SWISSPROT_DAT) --outdir $(OUTDIR) -resume

## The whole pipeline on 20 queries, on this machine: every index at full size, every
## search, scoring and comparison.
test-local:
	@mkdir -p $(LAUNCH_DIR)-test
	cd $(LAUNCH_DIR)-test && $(NEXTFLOW) run $(MAKEFILE_DIR)main.nf -profile local \
	  --swissprot_dat $(SWISSPROT_DAT) --outdir $(OUTDIR)-test --kmerseek $(KMERSEEK) \
	  --query_limit 20 -resume

## Download Swiss-Prot 2026_03 to Sherlock scratch and check its md5. UniProt's
## current_release moves on: if the md5 differs, the release changed and the query list
## (assets/chr6_queries.2026_03.tsv) no longer matches it, so stop.
fetch-swissprot-sherlock:
	@mkdir -p $(SH_DATA)
	curl -C - --retry 5 --speed-limit 100000 --speed-time 60 -o $(SH_DAT) $(SWISSPROT_URL)
	@test "$$(md5sum $(SH_DAT) | cut -d' ' -f1)" = "$(SWISSPROT_MD5)" \
	  && echo "md5 matches Swiss-Prot 2026_03" \
	  || { echo "md5 differs: not the 2026_03 release"; exit 1; }

## The full run on Sherlock: 975 queries, both indexes, both decoy windows, four kmerseek
## settings, MMseqs2 and the composition classifier. Submits the Nextflow head as a SLURM job.
run-sherlock:
	@mkdir -p $(SH_LAUNCH_DIR)
	cd $(SH_LAUNCH_DIR) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run $(MAKEFILE_DIR)main.nf -profile sherlock -resume \
	    --swissprot_dat $(SH_DAT) --outdir $(SH_OUTDIR) $(SH_RESOURCES)

## What is running, from any folder (scripts/nf-running, PR 70).
running:
	@python3 $(MAKEFILE_DIR)../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
