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
# Inputs are in different places on the laptop and on Sherlock, and hard-coding either
# produced a run that launched against files that were not there. On Sherlock the shared
# inputs are hard-linked into the region benchmark's data/ by its `sherlock-clone`, so that
# is the second place to look. Both are probed for a file that must exist; the first hit
# wins, and passing QFO_DIR= or SWISSPROT= on the command line still overrides.
PRIMARY_DATA ?= $(SCRATCH_DIR)/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data

QFO_CANDIDATES := $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143 \
                  $(PRIMARY_DATA)/qfo
QFO_DIR     ?= $(firstword $(foreach d,$(QFO_CANDIDATES),$(if $(wildcard $(d)/Eukaryota),$(d),)))

SWISSPROT_CANDIDATES := $(HOME)/data/uniprot/uniprot_sprot.dat.gz \
                        $(PRIMARY_DATA)/uniprot_sprot.dat.gz
SWISSPROT   ?= $(firstword $(foreach f,$(SWISSPROT_CANDIDATES),$(if $(wildcard $(f)),$(f),)))

BOTRYLLUS_FASTA ?= $(QFO_DIR)/Eukaryota/BOTSCH2026_30301.fasta

SHERLOCK_HOST ?= sherlock
REPO_URL      ?= https://github.com/seanome/2024-kmerseek-analysis/
BRANCH        ?= $(shell git rev-parse --abbrev-ref HEAD)
SSH_MUX       := ssh -o ControlMaster=auto -o ControlPath=$$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m
SHERLOCK_CLONE     ?= $$SCRATCH/2024-kmerseek-$(notdir $(BRANCH))
SHERLOCK_CLONE_DIR := $(SHERLOCK_CLONE)/nextflow-runs/invertebrate-dark-set

SPECIES ?=
NF_ARGS ?=
WITH_KMERSEEK ?=
KMERSEEK_ARGS := $(if $(WITH_KMERSEEK),--with_kmerseek true,)
MINI_N  ?= 200

.PHONY: help check-species check-inputs check-python sherlock-clone sync run-dark-set run-dark-set-mini queue-note

help:
	@echo "Dark set: proteins no sequence arm can place into the reference"
	@echo
	@echo "  sherlock-clone          push and clone this branch to \$$SCRATCH"
	@echo "  check-python            host python diagnostic; NOT needed for a run"
	@echo "  run-dark-set-mini       SPECIES=<botryllus|cgigas>  plumbing check, $(MINI_N) proteins"
	@echo "  run-dark-set            SPECIES=<botryllus|cgigas>  full proteome"
	@echo "                          add WITH_KMERSEEK=1 for the kmerseek arm"
	@echo
	@echo "  PROFILE=$(PROFILE)  OUTDIR=$(OUTDIR)"
	@echo "  REF_CACHE=$(REF_CACHE)  (one file per excluded clade, shared across species)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi

# NOT a prerequisite of the run targets, and deliberately so. Every python step in this
# pipeline (buildReference, computeDarkSet) carries `label python_scoring`, which the
# sherlock profile runs INSIDE params.kmerseek_image -- so the host python is never used and
# gating the run on it blocks a run that would have worked. It did exactly that on Sherlock,
# where PYTHON falls back to a bare python3 with no polars. Kept as a manual diagnostic for
# anyone running bin/ scripts by hand.
# Nothing is launched until the inputs actually resolve. A missing reference or query does
# not fail loudly downstream -- it produces an empty result that reads as a real one.
check-inputs:
	@ok=1; \
	 if [ -z "$(QFO_DIR)" ] || [ ! -d "$(QFO_DIR)/Eukaryota" ]; then \
	   echo "QfO tree not found. Looked for Eukaryota/ under:"; \
	   for d in $(QFO_CANDIDATES); do echo "    $$d"; done; \
	   echo "  override: make <target> QFO_DIR=/path/to/release"; ok=0; \
	 else echo "QFO_DIR   = $(QFO_DIR)"; fi; \
	 if [ -z "$(SWISSPROT)" ] || [ ! -f "$(SWISSPROT)" ]; then \
	   echo "uniprot_sprot.dat.gz not found. Looked at:"; \
	   for f in $(SWISSPROT_CANDIDATES); do echo "    $$f"; done; \
	   echo "  override: make <target> SWISSPROT=/path/to/uniprot_sprot.dat.gz"; ok=0; \
	 else echo "SWISSPROT = $(SWISSPROT)"; fi; \
	 [ "$$ok" = 1 ] || exit 1

check-python:
	@$(PYTHON) -c 'import polars, sys; print("host python OK:", sys.executable, "polars", polars.__version__)' \
	  || { echo; echo "host PYTHON=$(PYTHON) cannot import polars."; \
	       echo "This does NOT block a pipeline run -- the python steps run in"; \
	       echo "params.kmerseek_image. It only matters for running bin/ scripts by hand:"; \
	       echo "  make <target> PYTHON=\"apptainer exec \$$SCRATCH/containers/kmerseek.sif python3\""; \
	       exit 1; }

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
run-dark-set: check-species check-inputs queue-note
	$(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) \
	  --qfo_dir $(QFO_DIR) \
	  --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) \
	  --reference_cache $(REF_CACHE) \
	  $(KMERSEEK_ARGS) $(NF_ARGS)

run-dark-set-mini: check-species check-inputs
	@mkdir -p $(OUTDIR)/_mini
	@# set -e plus an explicit count. Without both, a failed awk left an empty FASTA, printed
	@# "staged 0 proteins", and make went on to launch nextflow against it -- a whole run
	@# whose only possible answer was "every protein is dark".
	@set -e; \
	 if [ "$(SPECIES)" != "botryllus" ]; then \
	   echo "the mini stages botryllus only; for cgigas pass --query_fasta yourself"; exit 1; fi; \
	 src="$(BOTRYLLUS_FASTA)"; \
	 [ -f "$$src" ] || { echo "no botryllus proteome at $$src"; exit 1; }; \
	 awk -v m=$(MINI_N) '/^>/{n++} n>m{exit} {print}' "$$src" > $(OUTDIR)/_mini/$(SPECIES).mini.fasta; \
	 n=$$(grep -c '^>' $(OUTDIR)/_mini/$(SPECIES).mini.fasta || true); \
	 [ "$$n" -gt 0 ] || { echo "staged 0 proteins from $$src -- refusing to launch"; exit 1; }; \
	 echo "staged $$n proteins"
	$(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) \
	  --query_fasta $(OUTDIR)/_mini/$(SPECIES).mini.fasta \
	  --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR)/mini --run_tag mini \
	  --reference_cache $(REF_CACHE) \
	  $(KMERSEEK_ARGS) $(NF_ARGS)

# Advisory, NOT a block. The original gate refused to launch while any job was queued,
# which on this account is never true -- the region benchmark keeps several hundred jobs
# pending for days, so the gate would have blocked this run indefinitely. It also could not
# have told the two apart: nextflow names jobs after the process, and both pipelines have
# nf-phmme, nf-jackh and nf-mmseq.
#
# Double-launching the same run is already prevented by nextflow's own session lock in the
# launch directory, which is the mechanism that actually works. So this reports the queue
# depth as context for how long a submission will sit, and gets out of the way.
queue-note:
	@if command -v squeue >/dev/null 2>&1; then \
	  n=$$(squeue -u $$USER -h 2>/dev/null | wc -l | tr -d ' '); \
	  r=$$(squeue -u $$USER -h -t RUNNING 2>/dev/null | wc -l | tr -d ' '); \
	  echo "queue: $$n job(s) total, $$r running -- this run's tasks queue behind them"; \
	fi
