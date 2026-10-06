MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
NEXTFLOW     ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
export NXF_JAVA_HOME ?= /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm

# Swiss-Prot release 2026_03, the release the chromosome 6 query list was taken from
# (assets/chr6_queries.2026_03.tsv). md5 in build_summary/inputs_and_outputs.md5.
SWISSPROT_DAT ?= $(HOME)/data/uniprot/uniprot_sprot.2026_03.dat.gz
OUTDIR        ?= $(HOME)/data/regions-index-pilot/results
LAUNCH_DIR    ?= $(HOME)/data/regions-index-pilot/launch

.PHONY: fetch-inputs build-local running

## Re-fetch the chromosome 6 query list and DisProt disorder into assets/ (committed).
fetch-inputs:
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 $(MAKEFILE_DIR)tools/fetch_inputs.py --assets $(MAKEFILE_DIR)assets

## Build the queries, truth table, both indexes and both decoy copies on this machine.
## About 3 minutes on the laptop.
build-local:
	@mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && $(NEXTFLOW) run $(MAKEFILE_DIR)main.nf -profile local \
	  --swissprot_dat $(SWISSPROT_DAT) --outdir $(OUTDIR) -resume

## What is running, from any folder (scripts/nf-running, PR 70).
running:
	@python3 $(MAKEFILE_DIR)../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
