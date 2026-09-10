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
