# How much of an invertebrate proteome does conventional sequence search miss?
#
# The dark set is every protein that phmmer, jackhmmer AND mmseqs2 all fail to place into
# reviewed Swiss-Prot with the query's clade removed. It needs no answer key, which is the
# point: C. gigas has 16 reviewed entries and Botryllus has 0, so neither can be scored
# until the structural key exists -- but "no arm hit this" is a property of the searches
# alone. This runs while Chainsaw and InterProScan are still going, and the number it
# returns decides whether the expensive half is worth running.
#
#   make run-dark-set SPECIES=botryllus     full proteome
#   make run-dark-set-mini SPECIES=botryllus  200 proteins, one node, sanity only
#
# The mini is a plumbing check, NOT a small answer: a dark fraction measured on 200
# proteins is not the proteome's dark fraction.

SHELL := /bin/bash

MAC_PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
ifneq ($(wildcard $(MAC_PYTHON)),)
PYTHON ?= $(MAC_PYTHON)
else
PYTHON ?= python3
endif

# Pinned on the mac, from the module tree on Sherlock. `ml load biology nextflow` needs
# both words: nextflow lives under the biology hierarchy, so `ml load nextflow` resolves
# only in a shell that already has biology loaded.
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
ifneq ($(wildcard $(MAC_NEXTFLOW)),)
NEXTFLOW ?= $(MAC_NEXTFLOW)
PROFILE  ?= standard
else
NEXTFLOW ?= nextflow
PROFILE  ?= sherlock
endif

SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
OUTDIR      ?= $(SCRATCH_DIR)/invertebrate-dark-set/results
# One reference per excluded clade, shared by every species that asks for that clade.
REF_CACHE   ?= $(SCRATCH_DIR)/invertebrate-dark-set/reference
QFO_DIR     ?= $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
SWISSPROT   ?= $(HOME)/data/uniprot/uniprot_sprot.dat.gz

SHERLOCK_HOST ?= sherlock
REPO_URL      ?= https://github.com/seanome/2024-kmerseek-analysis/
BRANCH        ?= $(shell git rev-parse --abbrev-ref HEAD)
SSH_MUX       := ssh -o ControlMaster=auto -o ControlPath=$$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m
SHERLOCK_CLONE     ?= $$SCRATCH/2024-kmerseek-$(notdir $(BRANCH))
SHERLOCK_CLONE_DIR := $(SHERLOCK_CLONE)/nextflow-runs/invertebrate-dark-set

SPECIES ?=
NF_ARGS ?=
MINI_N  ?= 200

.PHONY: help check-species check-python sherlock-clone sync run-dark-set run-dark-set-mini queue-empty

help:
	@echo "Dark set: proteins no sequence arm can place into the reference"
	@echo
	@echo "  sherlock-clone          push and clone this branch to \$$SCRATCH"
	@echo "  check-python            verify polars before a run, not during one"
	@echo "  run-dark-set-mini       SPECIES=<botryllus|cgigas>  plumbing check, $(MINI_N) proteins"
	@echo "  run-dark-set            SPECIES=<botryllus|cgigas>  full proteome"
	@echo
	@echo "  PROFILE=$(PROFILE)  OUTDIR=$(OUTDIR)"
	@echo "  REF_CACHE=$(REF_CACHE)  (one file per excluded clade, shared across species)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi

check-python:
	@$(PYTHON) -c 'import polars, sys; print("python OK:", sys.executable, "polars", polars.__version__)' \
	  || { echo; echo "PYTHON=$(PYTHON) cannot import polars."; \
	       echo "  make $(MAKECMDGOALS) PYTHON=/path/to/python3"; exit 1; }

sherlock-clone:
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) '\
	  set -e; \
	  if [ ! -d $(SHERLOCK_CLONE)/.git ]; then \
	    git clone --branch $(BRANCH) $(REPO_URL) $(SHERLOCK_CLONE); \
	  else \
	    cd $(SHERLOCK_CLONE) && git fetch origin $(BRANCH) \
	      && git checkout $(BRANCH) && git pull --ff-only; \
	  fi; \
	  echo "ready: $(SHERLOCK_CLONE_DIR)"'
	@echo
	@echo "  ssh $(SHERLOCK_HOST)"
	@echo "  cd $(SHERLOCK_CLONE_DIR)"

sync:
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) 'cd $(SHERLOCK_CLONE_DIR) && git pull --ff-only'

# -resume, single dash. `--resume` is a pipeline param and silently does nothing.
# Never piped through tee: the live task table is the point, and tee makes nextflow
# reprint the whole table on every update.
run-dark-set: check-species check-python queue-empty
	$(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) \
	  --qfo_dir $(QFO_DIR) \
	  --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) \
	  --reference_cache $(REF_CACHE) \
	  $(NF_ARGS)

run-dark-set-mini: check-species check-python
	@mkdir -p $(OUTDIR)/_mini
	@src=$(QFO_DIR)/Eukaryota/BOTSCH2026_30301.fasta; \
	 if [ "$(SPECIES)" != "botryllus" ]; then \
	   echo "the mini currently stages botryllus only; for cgigas pass --query_fasta yourself"; \
	   exit 1; fi; \
	 awk -v m=$(MINI_N) '/^>/{n++} n>m{exit} {print}' $$src > $(OUTDIR)/_mini/$(SPECIES).mini.fasta; \
	 echo "staged $$(grep -c '^>' $(OUTDIR)/_mini/$(SPECIES).mini.fasta) proteins"
	$(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) \
	  --query_fasta $(OUTDIR)/_mini/$(SPECIES).mini.fasta \
	  --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR)/mini \
	  --reference_cache $(REF_CACHE) \
	  $(NF_ARGS)

queue-empty:
	@if command -v squeue >/dev/null 2>&1; then \
	  n=$$(squeue -u $$USER -h 2>/dev/null | wc -l | tr -d ' '); \
	  if [ "$$n" -gt 0 ]; then \
	    echo "$$n job(s) already queued or running:"; squeue -u $$USER; \
	    echo; echo "wait, or override with SKIP_QUEUE_CHECK=1"; \
	    [ -n "$(SKIP_QUEUE_CHECK)" ] || exit 1; \
	  fi; \
	fi
