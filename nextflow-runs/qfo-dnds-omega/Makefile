# qfo-dnds-omega/Makefile
#
# Per-ortholog-pair omega (pairwise dN/dS) for human against QfO target species,
# from the cDNA shipped with the QfO release.
#
#   make build   - build the mafft + paml image
#   make test    - 20 pairs per species, validates the codeml parser end to end
#   make run     - full run for the species where dS is still informative
#   make run-all - every species, including the ones where dS is saturated
#
# `make test` first. The previous pipeline shipped a codeml parser that silently put
# omega in the dS column; the test profile is small enough to eyeball, and dS and
# omega differing is the thing to look for.

NF      := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
JAVA_HOME := /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm
IMAGE   := qfo-dnds-omega:1.0
QFO_DIR := $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
OUTDIR  := results

export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)

.PHONY: build test run run-all clean

build:
	docker build -f docker/Dockerfile.omega -t $(IMAGE) .

test: build
	$(NF) run main.nf -profile test \
	    --qfo_dir $(QFO_DIR) \
	    --omega_container $(IMAGE) \
	    --outdir results-test \
	    -resume

run: build
	$(NF) run main.nf -profile standard \
	    --qfo_dir $(QFO_DIR) \
	    --omega_container $(IMAGE) \
	    --outdir $(OUTDIR) \
	    -resume

# Saturated species included. Every row past chicken carries dS_saturated=true and
# should not be read as a selection measurement.
run-all: build
	$(NF) run main.nf -profile standard \
	    --qfo_dir $(QFO_DIR) \
	    --omega_container $(IMAGE) \
	    --species mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis \
	    --outdir $(OUTDIR) \
	    -resume

clean:
	/bin/rm -rf work .nextflow* results-test

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, from any pipeline, then one row per job
## of the runs under this folder. scripts/nf-running does the work. Runs on Sherlock only,
## since it reads the queue. Read TIME as squeue's [DD-[HH:]]MM:SS: `4:03` is four minutes
## and `1-02:19:10` is a day. PENDING jobs show 0:00.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

.PHONY: running
running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/qfo-dnds-omega"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
