# QfO Pfam region/domain-finding benchmark.
#
# build/push/sync/fetch/pull targets run on your Mac.
# run-*/status targets run ON Sherlock (ssh sherlock; cd $SCRATCH/qfo-pfam-region-benchmark).
# See README-sherlock.md for the full runbook.

SHELL := /bin/bash

# --- Mac-side config -------------------------------------------------------
KMERSEEK_SHA   ?= 0302511391c5f8eef499179430764d3d0881e529
IMAGE_TAG      ?= 2026-08-18-region-scoring
IMAGE_LOCAL    := kmerseek-region:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)
KMERSEEK_SRC   := /Users/olga/code/kmerseek-region-scoring

SHERLOCK_HOST  ?= sherlock
REPO_URL       ?= https://github.com/seanome/2024-kmerseek-analysis.git
BRANCH         ?= olgabot/qfo-pfam-region-benchmark

# $SCRATCH, never $HOME -- the home quota is small and the work dir is not.
REMOTE_REPO    := $$SCRATCH/2024-kmerseek-analysis
# The pipeline directory inside that checkout. Every remote target runs from here: it is
# both inside the git tree (so `git pull` works) and the directory holding main.nf (so the
# run targets' `../main.nf` from run/ resolves). Those two roles are why this is the
# subdirectory and not the repo root.
REMOTE_DIR     := $(REMOTE_REPO)/nextflow-runs/qfo-pfam-region-benchmark

ANALYSIS_ROOT  := /Users/olga/code/2024-kmerseek-analysis
ANNOTATIONS    := $(ANALYSIS_ROOT)/results/pfam_benchmark/annotations
QFO_DIR        := /Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
STRUCT_DIR     := /Users/olga/data/alphafold_proteomes
ACC_DIR        := /Users/olga/data/alphafold_proteomes/_accessions
LOCAL_RESULTS  := /Users/olga/data/qfo-pfam-region-benchmark
MINI_DIR       := /Users/olga/data/qfo-pfam-region-benchmark-mini
MHC_DIR        := /Users/olga/data/qfo-pfam-region-benchmark-mhc

# Two species and two combos: enough to exercise every stage in minutes.
MINI_SPECIES ?= yeast,ecoli
MINI_COMBOS  ?= protein:10,hp-pbotc-1st-ed:19

PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# Sherlock compute nodes are amd64; an Apple Silicon build defaults to arm64, which
# Apptainer refuses to run.
DOCKER_PLATFORM := linux/amd64

PROFILE := sherlock

.PHONY: bootstrap-sherlock build-image build-image-local push-image structure-lists fetch-structures \
        mini-testset mini-testset-sherlock mhc-testset run-mhc-local \
        run-mini run-mini-local pull-mini \
        sync-pipeline sync-data sync-structures pull-results \
        run run-kmerseek run-baselines status clean-cache

## ---- Mac side -----------------------------------------------------------

build-image:
	docker build --platform $(DOCKER_PLATFORM) -t $(IMAGE_LOCAL) \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

# Native-arch build, for running the mini set on this Mac. The amd64 image above cannot
# execute here: polars requires AVX2, Apple Silicon emulation does not provide it, and the
# python steps die with SIGSEGV. Sherlock nodes are real amd64, so only build-image is
# pushed.
build-image-local:
	docker build -t kmerseek-region:local \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

push-image: build-image
	docker tag $(IMAGE_LOCAL) $(IMAGE_REMOTE)
	docker push $(IMAGE_REMOTE)

# One accession list per species: the proteins the Foldseek arm needs structures for.
structure-lists:
	$(PYTHON) bin/write_structure_lists.py \
	  --annotations $(ANNOTATIONS) \
	  --outdir      $(ACC_DIR)

# ~73k structures / ~36 GB missing from the flat cache as of 2026-08-18. Resumable --
# re-run it after an interruption rather than starting over.
fetch-structures: structure-lists
	bin/fetch_alphafold_structures.sh $(STRUCT_DIR) $(ACC_DIR)

## ---- mini smoke test ----
##
## Builds a few-hundred-protein set whose targets were chosen BECAUSE they share Pfam
## families with the queries, so real true positives exist. A random subset would share
## almost nothing, every metric would read 0.0, and a broken scoring path would look
## exactly like a working one.

mini-testset:
	$(PYTHON) bin/make_mini_testset.py \
	  --annotations $(ANNOTATIONS) \
	  --qfo-dir     $(QFO_DIR) \
	  --outdir      $(MINI_DIR) \
	  --species     $(MINI_SPECIES)

# MHC-only set: the 25 genes of notebooks 210-216 against mouse. A vignette and a smoke
# test, NOT a population statistic -- n=25 across 7 classes must never carry a headline
# number. Deliberately not size-matched to the default set for that reason.
mhc-testset:
	$(PYTHON) bin/make_mini_testset.py \
	  --annotations $(ANNOTATIONS) \
	  --qfo-dir     $(QFO_DIR) \
	  --outdir      $(MHC_DIR) \
	  --species     mouse \
	  --gene-set    mhc \
	  --hgnc        $(HGNC_FILE) \
	  --n-targets   400

run-mhc-local: 
	mkdir -p run-mhc && cd run-mhc && \
	  nextflow run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MHC_DIR)/qfo \
	    --annotations     $(MHC_DIR)/annotations \
	    --structures      $(MHC_DIR)/structures \
	    --outdir          $(MHC_DIR)/results \
	    --species         mouse \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(NF_ARGS) 2>&1 | tee nextflow.mhc.log

# Run it here first (needs build-image-local), before spending Sherlock queue time.
run-mini-local: 
	mkdir -p run-mini && cd run-mini && \
	  nextflow run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MINI_DIR)/qfo \
	    --annotations     $(MINI_DIR)/annotations \
	    --structures      $(MINI_DIR)/structures \
	    --outdir          $(MINI_DIR)/results \
	    --species         $(MINI_SPECIES) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(NF_ARGS) 2>&1 | tee nextflow.mini.log

# NOT rsynced to Sherlock. The mini set is DERIVED -- subset FASTAs and subset annotation
# parquets cut from inputs `make sync-data` already put there -- so shipping it would
# duplicate state that can simply be regenerated. The generator itself is code and travels
# by git like everything else. Run `make mini-testset-sherlock` on the cluster instead.
#
# Structures are the one exception and are handled by `make sync-structures`: they are
# real input data, not derived, and nothing on Sherlock can synthesise them.

pull-mini:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/mini/results/' $(LOCAL_RESULTS)-mini/

# Code moves by git, never rsync: rsync duplicates state outside git history and defeats
# the point of running this from a worktree.
# One-time setup: clone the repo onto $SCRATCH and create the data directories the sync
# targets write into. Safe to re-run; it does not touch an existing checkout's state.
bootstrap-sherlock:
	ssh $(SHERLOCK_HOST) '\
	  set -e; \
	  if [ ! -d $(REMOTE_REPO)/.git ]; then \
	    git clone --branch $(BRANCH) $(REPO_URL) $(REMOTE_REPO); \
	  else \
	    cd $(REMOTE_REPO) && git fetch origin $(BRANCH) && git checkout $(BRANCH) && git pull --ff-only; \
	  fi; \
	  mkdir -p $(REMOTE_DIR)/data/annotations \
	           $(REMOTE_DIR)/data/qfo \
	           $(REMOTE_DIR)/data/covariates \
	           $(REMOTE_DIR)/data/structures; \
	  echo "ready: $(REMOTE_DIR)"'

sync-pipeline:
	git push origin $$(git branch --show-current)
	ssh $(SHERLOCK_HOST) 'cd $(REMOTE_DIR) && git pull --ff-only'

# Data may move by rsync. Annotations are small; proteomes are ~1 GB; structures ~60 GB.
HGNC_FILE  := /Users/olga/data/gencode/results-human-mouse-orthologs/hgnc_complete_set.txt
SPROT_DAT  := /Users/olga/data/uniprot/uniprot_sprot.dat.gz
OMEGA_FILE := $(ANALYSIS_ROOT)/nextflow-runs/human-mouse-dnds-omega/results/omega_results.tsv

sync-data:
	rsync -avz --progress -e ssh $(ANNOTATIONS)/ \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/annotations/'
	rsync -avz --progress -e ssh $(QFO_DIR)/ \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/qfo/'
	rsync -avz --progress -e ssh $(HGNC_FILE) $(OMEGA_FILE) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/covariates/'
	rsync -avz --progress -e ssh $(SPROT_DAT) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/'

sync-structures:
	rsync -av --progress --copy-links -e ssh $(STRUCT_DIR)/ \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/structures/' \
	  --exclude '_archives'

pull-results:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/metrics/' $(LOCAL_RESULTS)/metrics/
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/all_domain_metrics.parquet' $(LOCAL_RESULTS)/
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/truth/' $(LOCAL_RESULTS)/truth/

## ---- Sherlock side ------------------------------------------------------
##
## Launch from run/ so Nextflow's workDir, trace, timeline and report all land in one
## place. Paths passed to nextflow are relative to run/ on purpose: workDir and launchDir
## live under the real (non-symlinked) checkout path, and Apptainer's autoMounts binds
## based on that real tree. A file reached through a $SCRATCH symlink is a second,
## unrelated absolute path that never gets bound, and the staged input ends up dangling
## inside the container. Relative paths never cross the symlink boundary.
##
## Every run target blocks in the FOREGROUND and streams live output. They do not start
## tmux themselves -- you run them inside your own tmux session and watch. Never wrap these
## in `tmux new-session`: a target that detaches into its own session hides the output you
## are trying to watch, and nesting tmux inside tmux makes the keybindings a fight.

define launch
	mkdir -p run
	cd run && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) -resume $(2) $(NF_ARGS) \
	    --qfo_dir     ../data/qfo \
	    --annotations ../data/annotations \
	    --structures  ../data/structures \
	    --hgnc_file   ../data/covariates/hgnc_complete_set.txt \
	    --omega_file  ../data/covariates/omega_results.tsv \
	    --swissprot_dat ../data/uniprot_sprot.dat.gz \
	    --outdir      ../results \
	    2>&1 | tee -a nextflow.$(1).log
endef

run:
	$(call launch,qfo-region,)

# Split arms so the 1017-job kmerseek sweep and the 9-species baselines can be queued
# and debugged independently.
run-kmerseek:
	$(call launch,qfo-kmerseek,--skip_baselines true)

run-baselines:
	$(call launch,qfo-baselines,--skip_kmerseek true)

# Build the mini set ON Sherlock from the inputs sync-data already staged. Runs inside the
# same container the pipeline uses, so it needs no cluster python environment of its own.
mini-testset-sherlock:
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/make_mini_testset.py \
	    --annotations     data/annotations \
	    --qfo-dir         data/qfo \
	    --outdir          data/mini \
	    --species         $(MINI_SPECIES) \
	    --structure-cache data/structures

# Same code path as the real run, scoped down. Do this before `make run`.
run-mini: mini-testset-sherlock
	mkdir -p run-mini
	cd run-mini && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE),mini -resume \
	    --qfo_dir         ../data/mini/qfo \
	    --annotations     ../data/mini/annotations \
	    --structures      ../data/mini/structures \
	    --outdir          ../data/mini/results \
	    --species         $(MINI_SPECIES) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    $(NF_ARGS) 2>&1 | tee -a nextflow.mini.log

status:
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}

clean-cache:
	rm -rf $$SCRATCH/apptainer-cache
