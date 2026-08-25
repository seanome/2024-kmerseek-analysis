# QfO Pfam region/domain-finding benchmark.
#
# build/push/sync/fetch/pull targets run on your Mac.
# run-*/status targets run ON Sherlock (ssh sherlock; cd $SCRATCH/qfo-pfam-region-benchmark).
# See README-sherlock.md for the full runbook.

SHELL := /bin/bash

# --- Mac-side config -------------------------------------------------------
KMERSEEK_SHA   ?= 3fdfd51a1f3c68925e12eac76d19d5d021c29837
IMAGE_TAG      ?= 2026-08-24-reduced-alphabets
IMAGE_LOCAL    := kmerseek-region:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)
FOLDDISCO_TAG    ?= 2026-08-20-noentrypoint
FOLDDISCO_REMOTE := docker.io/olgabot/folddisco:$(FOLDDISCO_TAG)
KMERSEEK_SRC   := /Users/olga/code/kmerseek/.claude/worktrees/reduced-alphabets

SHERLOCK_HOST  ?= sherlock

# Multiplex ssh connections. Every rsync and ssh here otherwise opens its own connection
# and triggers its own two-factor prompt; with ControlMaster the first one authenticates
# and the rest ride the same socket for ControlPersist. This applies to all targets, not
# just sync-data -- pull-results and sync-pipeline were equally annoying.
SSH_MUX  := ssh -o ControlMaster=auto -o ControlPath=$$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m
RSYNC    := rsync -avz --progress -e "$(SSH_MUX)"
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

# One species at a time for the login-node fallback, e.g. make fetch-structures-login SPECIES_ONE=human
SPECIES_ONE    ?= human

# Two species and two combos: enough to exercise every stage in minutes.
# TARGET proteomes. Human is the query and is never listed here.
MINI_TARGETS ?= yeast,ecoli
MINI_COMBOS  ?= protein20:10,hp_pbotc_1st_ed2:19

# Midi: every human chr6 gene as the query, FULL proteomes as targets. Real scale on the
# target side, ~5% of the query load -- 964 proteins carrying 2358 of the 50185 human Pfam
# domain instances. chr6 because it carries the MHC, so this subsumes the MHC vignette.
MIDI_TARGETS ?= mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli

PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# Mac-side Nextflow. Pinned to the conda env rather than relying on PATH; on Sherlock the
# run targets use `ml load nextflow` instead.
NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

# Sherlock compute nodes are amd64; an Apple Silicon build defaults to arm64, which
# Apptainer refuses to run.
DOCKER_PLATFORM := linux/amd64

PROFILE := sherlock

# SLURM coordinates for the few things this Makefile submits itself. Nextflow gets its own
# from the sherlock profile in nextflow.config; these are only for prefetch-images.
SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns
# mksquashfs sizes its cache off the node's total RAM, not the cgroup it is confined to, so
# on a login node it happily over-allocates and dies with "Out of memory (cache_alloc)".
# Converting inside a job with a real allocation is the fix.
PREFETCH_MEM    ?= 48G
PREFETCH_TIME   ?= 2:00:00

# --- MultiQC report --------------------------------------------------------
# Which run's resource numbers the report describes. Defaults to the newest trace the main
# run left behind; override for a specific one:
#   make multiqc TRACE=run/qfo_pfam_region.2026-08-24.trace.txt
TRACE          ?= $(shell ls -t run/*.trace.txt 2>/dev/null | head -1)
MULTIQC_OUTDIR ?= ../results
# Local equivalents, pointed at the MHC smoke run by default.
TRACE_LOCAL          ?= $(shell ls -t run-mhc/*.trace.txt 2>/dev/null | head -1)
MULTIQC_OUTDIR_LOCAL ?= $(MHC_DIR)/results

# Every container the pipeline can pull, read straight out of main.nf plus the kmerseek
# image, so this list cannot drift from what the processes actually declare.
ALL_CONTAINERS := $(shell sed -n "s/^[[:space:]]*container[[:space:]]*'\\(.*\\)'.*/\\1/p" main.nf | sort -u) \
                  docker.io/olgabot/kmerseek:$(IMAGE_TAG) $(FOLDDISCO_REMOTE)

.PHONY: bootstrap-sherlock build-image build-image-local push-image \
        build-folddisco-image push-folddisco-image structure-lists fetch-structures \
        mini-testset mini-testset-sherlock mhc-testset run-mhc-local \
        midi-testset-sherlock run-midi \
        fetch-structures-mac \
        run-mini run-mini-local pull-mini \
        sync-pipeline sync-data sync-structures pull-results \
        run run-kmerseek run-baselines status clean-cache \
        prefetch-images clean-image-cache fetch-structures-sherlock prostt5-weights \
        pfamn-download pfamn-truth mcsa-truth help \
        fetch-structures-login \
        multiqc multiqc-local pull-report

help:
	@echo "Mac:"
	@echo "  build-image           Build the amd64 Kmerseek image for Sherlock"
	@echo "  build-image-local     Build the native Kmerseek image for local Docker"
	@echo "  push-image            Push the Kmerseek image to Docker Hub"
	@echo "  push-folddisco-image  Push the FoldDisco image to Docker Hub"
	@echo "  sync-pipeline         Push the current branch and update Sherlock"
	@echo "  sync-data             Copy benchmark inputs to Sherlock"
	@echo "  sync-structures       Copy AlphaFold structures to Sherlock"
	@echo "  multiqc-local         Build the MultiQC report from a local run"
	@echo "  pull-report           Fetch the MultiQC report back from Sherlock"
	@echo ""
	@echo "Sherlock:"
	@printf '%s\n' '  prefetch-images       Pull all pinned images into $$SCRATCH cache'
	@echo "  mini-testset-sherlock Build the mini data set"
	@echo "  run-mini             Run the mini smoke test"
	@echo "  run-midi             chr6 queries vs FULL proteomes (includes the MHC)"
	@echo "  run-kmerseek         Run only the Kmerseek sweep"
	@echo "  run-baselines        Run only the baseline tools"
	@echo "  run                  Run the complete benchmark"
	@echo "  multiqc              Rebuild the MultiQC report from published results"
	@echo "  prostt5-weights      Download ProstT5 weights on a login node"
	@echo "  status               Show this user's SLURM jobs"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-image-cache    Remove partial Apptainer pulls"
	@echo "  clean-cache          Remove all cached Apptainer images"


# --- which machine am I on? -------------------------------------------------
# Half these targets push data from the Mac to the cluster and half run on the
# cluster itself. Running one on the wrong side produces a confusing failure --
# `make sync-data` on Sherlock tries to resolve the host `sherlock` from Sherlock
# and dies in rsync -- so each target says where it belongs and refuses politely.
# $SCRATCH is set on Sherlock and unset on the Mac, which is the cheapest reliable
# discriminator that needs no network.
ON_CLUSTER := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

MAC_PIPELINE_DIR := /Users/olga/code/2024-kmerseek-analysis/.claude/worktrees/qfo-pfam-region-benchmark/nextflow-runs/qfo-pfam-region-benchmark

define require_mac
	@if [ "$(ON_CLUSTER)" = "yes" ]; then \
	  echo ""; \
	  echo "  '$@' runs on your MAC, not on the cluster."; \
	  echo "  It moves data from the Mac to $(SHERLOCK_HOST), so it cannot work from here."; \
	  echo ""; \
	  echo "  On your Mac:"; \
	  echo "    cd $(MAC_PIPELINE_DIR)"; \
	  echo "    make $@"; \
	  echo ""; \
	  exit 1; \
	fi
endef

define require_cluster
	@if [ "$(ON_CLUSTER)" != "yes" ]; then \
	  echo ""; \
	  echo "  '$@' runs ON SHERLOCK, not on your Mac (no \$$SCRATCH here)."; \
	  echo ""; \
	  echo "  ssh $(SHERLOCK_HOST)"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark"; \
	  echo "    make $@"; \
	  echo ""; \
	  exit 1; \
	fi
endef

## ---- Mac side -----------------------------------------------------------

build-image:
	$(call require_mac)
	docker build --platform $(DOCKER_PLATFORM) -t $(IMAGE_LOCAL) \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

# Native-arch build, for running the mini set on this Mac. The amd64 image above cannot
# execute here: polars requires AVX2, Apple Silicon emulation does not provide it, and the
# python steps die with SIGSEGV. Sherlock nodes are real amd64, so only build-image is
# pushed.
build-image-local:
	$(call require_mac)
	docker build -t kmerseek-region:local \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

# folddisco with ENTRYPOINT cleared. The upstream image's entrypoint becomes the SIF
# runscript under Apptainer and stops Nextflow running /bin/bash, so the task exits 1 with
# no output whatsoever. Docker does not reproduce it.
build-folddisco-image:
	$(call require_mac)
	docker build --platform $(DOCKER_PLATFORM) -t folddisco-noentry:latest \
	  -f Dockerfile.folddisco .

push-folddisco-image: build-folddisco-image
	$(call require_mac)
	docker tag folddisco-noentry:latest $(FOLDDISCO_REMOTE)
	docker push $(FOLDDISCO_REMOTE)

push-image: build-image
	$(call require_mac)
	docker tag $(IMAGE_LOCAL) $(IMAGE_REMOTE)
	docker push $(IMAGE_REMOTE)

# One accession list per species: the proteins the Foldseek arm needs structures for.
structure-lists:
	$(call require_mac)
	$(PYTHON) bin/write_structure_lists.py \
	  --annotations $(ANNOTATIONS) \
	  --outdir      $(ACC_DIR)

# Mac-side fetch. Prefer `make fetch-structures`, which downloads on the cluster:
# pulling ~60 GB to a laptop and pushing it back over ssh is strictly worse than
# downloading it where it will be used. Kept for wanting the structures locally too.
fetch-structures-mac: structure-lists
	$(call require_mac)
	bin/fetch_alphafold_structures.sh $(STRUCT_DIR) $(ACC_DIR)

# Everything the cluster needs, in one go. With connection multiplexing these share a
# single authenticated ssh, so this is one two-factor prompt rather than one per target.
sync-all: sync-pipeline sync-data
	@echo "pipeline and data synced. Structures download on the cluster: make fetch-structures"

## ---- mini smoke test ----
##
## Builds a few-hundred-protein set whose targets were chosen BECAUSE they share Pfam
## families with the queries, so real true positives exist. A random subset would share
## almost nothing, every metric would read 0.0, and a broken scoring path would look
## exactly like a working one.

mini-testset:
	$(call require_mac)
	$(PYTHON) bin/make_mini_testset.py \
	  --annotations $(ANNOTATIONS) \
	  --qfo-dir     $(QFO_DIR) \
	  --outdir      $(MINI_DIR) \
	  --target_species $(MINI_TARGETS)

# MHC-only set: the 25 genes of notebooks 210-216 against mouse. A vignette and a smoke
# test, NOT a population statistic -- n=25 across 7 classes must never carry a headline
# number. Deliberately not size-matched to the default set for that reason.
mhc-testset:
	$(call require_mac)
	$(PYTHON) bin/make_mini_testset.py \
	  --annotations $(ANNOTATIONS) \
	  --qfo-dir     $(QFO_DIR) \
	  --outdir      $(MHC_DIR) \
	  --target_species mouse \
	  --gene-set    mhc \
	  --hgnc        $(HGNC_FILE) \
	  --n-targets   400

run-mhc-local: 
	$(call require_mac)
	mkdir -p run-mhc && cd run-mhc && \
	  nextflow run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MHC_DIR)/qfo \
	    --annotations     $(MHC_DIR)/annotations \
	    --structures      $(MHC_DIR)/structures \
	    --outdir          $(MHC_DIR)/results \
	    --target_species  mouse \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(NF_ARGS)

# Run it here first (needs build-image-local), before spending Sherlock queue time.
run-mini-local: 
	$(call require_mac)
	mkdir -p run-mini && cd run-mini && \
	  nextflow run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MINI_DIR)/qfo \
	    --annotations     $(MINI_DIR)/annotations \
	    --structures      $(MINI_DIR)/structures \
	    --outdir          $(MINI_DIR)/results \
	    --target_species  $(MINI_TARGETS) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(NF_ARGS)

# NOT rsynced to Sherlock. The mini set is DERIVED -- subset FASTAs and subset annotation
# parquets cut from inputs `make sync-data` already put there -- so shipping it would
# duplicate state that can simply be regenerated. The generator itself is code and travels
# by git like everything else. Run `make mini-testset-sherlock` on the cluster instead.
#
# Structures are the one exception and are handled by `make sync-structures`: they are
# real input data, not derived, and nothing on Sherlock can synthesise them.

# Same entry point as `make multiqc`, against a local run. Override both for a run other
# than the MHC smoke test:
#   make multiqc-local TRACE_LOCAL=run-mini/qfo_pfam_region.<date>.trace.txt \
#                      MULTIQC_OUTDIR_LOCAL=/Users/olga/data/qfo-pfam-region-benchmark-mini/results
multiqc-local:
	$(call require_mac)
	@if [ -z "$(TRACE_LOCAL)" ]; then \
	  echo "No trace file under run-mhc/. Pass one explicitly:"; \
	  echo "  make multiqc-local TRACE_LOCAL=run-mhc/qfo_pfam_region.<date>.trace.txt"; \
	  exit 1; \
	fi
	@echo "reporting on trace: $(TRACE_LOCAL)"
	mkdir -p run-report
	cd run-report && \
	  $(NEXTFLOW) run ../main.nf -entry report -profile standard \
	    --kmerseek_image kmerseek-region:local \
	    --outdir     $(MULTIQC_OUTDIR_LOCAL) \
	    --report_trace $(abspath $(TRACE_LOCAL)) \
	    --qfo_dir    $(QFO_DIR) \
	    $(NF_ARGS)

# The report is one self-contained HTML plus its exported figures. The figures come down
# too: the svg is what goes into a paper, and re-exporting it locally would need the whole
# results tree rather than the 25 MB this pulls.
pull-report:
	$(call require_mac)
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/qfo_pfam_region_multiqc.html' $(LOCAL_RESULTS)/
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/qfo_pfam_region_multiqc_plots/' \
	  $(LOCAL_RESULTS)/qfo_pfam_region_multiqc_plots/

pull-mini:
	$(call require_mac)
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/mini/results/' $(LOCAL_RESULTS)-mini/

# Code moves by git, never rsync: rsync duplicates state outside git history and defeats
# the point of running this from a worktree.
# One-time setup: clone the repo onto $SCRATCH and create the data directories the sync
# targets write into. Safe to re-run; it does not touch an existing checkout's state.
bootstrap-sherlock:
	$(call require_mac)
	$(SSH_MUX) $(SHERLOCK_HOST) '\
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
	$(call require_mac)
	git push origin $$(git branch --show-current)
	$(SSH_MUX) $(SHERLOCK_HOST) 'cd $(REMOTE_DIR) && git pull --ff-only'

# Data may move by rsync. Annotations are small; proteomes are ~1 GB; structures ~60 GB.
HGNC_FILE  := /Users/olga/data/gencode/results-human-mouse-orthologs/hgnc_complete_set.txt
SPROT_DAT  := /Users/olga/data/uniprot/uniprot_sprot.dat.gz
OMEGA_FILE := $(ANALYSIS_ROOT)/nextflow-runs/human-mouse-dnds-omega/results/omega_results.tsv

# ONE rsync, so it authenticates once.
#
# The four inputs live under different local roots and land in different remote
# subdirectories, which rsync cannot express in a single invocation. So a staging tree of
# symlinks is built first, mirroring the remote layout exactly, and that one tree is sent
# with --copy-links so the real bytes go across rather than the links.
#
# Staging costs nothing: symlinks are created, not data, and rsync still transfers only
# what changed. SSH multiplexing above means the ssh in sync-pipeline reuses this
# connection too, so a full sync is one authentication rather than five.
SYNC_STAGE := .sync-staging

sync-data:
	$(call require_mac)
	@rm -rf $(SYNC_STAGE)
	@mkdir -p $(SYNC_STAGE)/covariates
	@ln -s $(ANNOTATIONS) $(SYNC_STAGE)/annotations
	@ln -s $(QFO_DIR)     $(SYNC_STAGE)/qfo
	@ln -s $(HGNC_FILE)   $(SYNC_STAGE)/covariates/
	@ln -s $(OMEGA_FILE)  $(SYNC_STAGE)/covariates/
	@ln -s $(SPROT_DAT)   $(SYNC_STAGE)/
	$(RSYNC) --copy-links $(SYNC_STAGE)/ \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/'
	@rm -rf $(SYNC_STAGE)

sync-structures:
	$(call require_mac)
	$(RSYNC) --copy-links $(STRUCT_DIR)/ \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/structures/' \
	  --exclude '_archives'

pull-results:
	$(call require_mac)
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/metrics/' $(LOCAL_RESULTS)/metrics/
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/all_domain_metrics.parquet' $(LOCAL_RESULTS)/
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/results/truth/' $(LOCAL_RESULTS)/truth/


# Pull every container ONCE, serially, into the Apptainer cache before any pipeline run.
#
# Nextflow launches all its pulls concurrently, and Apptainer's OCI blob cache is not
# concurrency-safe: simultaneous pulls into one cacheDir race and leave a half-written
# blob, which surfaces as the useless "unexpected end of JSON input". That is what killed
# hhblitsBuildDB while foldseek, pulling at the same moment, happened to win.
#
# Serial pulls also give each image the full timeout rather than five sharing it, and make
# a genuine registry failure legible instead of hiding behind a race.
#
# Nextflow finds a pre-pulled image by an exact filename: the container ref with '/' and
# the digest ':' turned into '-', plus .img. Names are computed to match.
prefetch-images:
	$(call require_cluster)
	@mkdir -p $${SCRATCH}/apptainer-cache $${SCRATCH}/apptainer-tmp
	@set -e; \
	export APPTAINER_CACHEDIR=$${SCRATCH}/apptainer-cache; \
	export APPTAINER_TMPDIR=$${SCRATCH}/apptainer-tmp; \
	for ref in $(ALL_CONTAINERS); do \
	  name=$$(echo "$$ref" | sed -e 's|^docker://||' -e 's|/|-|g' -e 's|:|-|g'); \
	  out="$${SCRATCH}/apptainer-cache/$${name}.img"; \
	  if [ -s "$$out" ]; then echo "have  $$name"; continue; fi; \
	  echo "pull  $$name"; \
	  rm -f "$$out" "$$out".pulling.* 2>/dev/null || true; \
	  srun --account=$(SLURM_ACCOUNT) --partition=$(SLURM_PARTITION) \
	       --mem=$(PREFETCH_MEM) --time=$(PREFETCH_TIME) --cpus-per-task=4 \
	       apptainer pull --name "$$out" "docker://$$ref" || \
	    { echo "FAILED: $$ref"; rm -f "$$out"; exit 1; }; \
	done; \
	echo "all containers cached in $${SCRATCH}/apptainer-cache"

# Remove half-written blobs left by a racing or interrupted pull. Run this if a pull fails
# with "unexpected end of JSON input" -- the partial entry is sticky and the next attempt
# reuses it.
clean-image-cache:
	$(call require_cluster)
	rm -rf $${SCRATCH}/apptainer-cache/*.pulling.* $${HOME}/.apptainer/cache/blob
	@echo "cleared partial pulls; re-run 'make prefetch-images'"


# Download structures ON the cluster. Sherlock's connection is far faster than pulling
# ~60 GB to a laptop and pushing it back over ssh, so this is the preferred route and
# `make sync-structures` becomes unnecessary.
#
# The Mac's flat AlphaFold cache buys almost nothing here: 8 of the 10 species come from
# whole-proteome tarballs that are downloaded in full regardless, and only chicken and
# ciona (~8.4k structures) are fetched per accession. So starting cold on the cluster costs
# little more than starting from the cache on the Mac.
#
# Blocks in the foreground and streams progress -- run it inside your own tmux, it takes a
# while. Resumable: re-run after an interruption rather than starting over.
#
# Must run where there is outbound internet. On Sherlock that is a login node, which is why
# this is not submitted through SLURM; it is I/O, not compute.
fetch-structures: fetch-structures-sherlock

# Download structures where they will be used.
# Download as SLURM jobs, one per species. The login-node version was SIGKILLed mid-
# transfer (exit 137) -- Sherlock enforces limits there and a multi-GB sustained download
# trips them. Blocks in the foreground so progress is visible; run inside your own tmux.
fetch-structures-sherlock:
	$(call require_cluster)
	@mkdir -p data/structures/_accessions
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/write_structure_lists.py \
	    --annotations data/annotations \
	    --outdir      data/structures/_accessions
	ml load nextflow && \
	  nextflow run main.nf -profile $(PROFILE) -resume \
	    --fetch_structures true \
	    --structures  $$PWD/data/structures \
	    --annotations $$PWD/data/annotations \
	    --qfo_dir     $$PWD/data/qfo \
	    $(NF_ARGS)

# Fallback for a cluster whose compute nodes have no outbound internet: run the download
# on the login node one species at a time, with low parallelism to stay under its limits.
# Slower and killable, but it needs no route from the batch partition.
fetch-structures-login:
	$(call require_cluster)
	@mkdir -p data/structures/_accessions
	PARALLEL=2 FLAT_CACHE=data/structures bin/fetch_alphafold_structures.sh \
	  data/structures data/structures/_accessions $(SPECIES_ONE)


# ProstT5 weights, on the login node. Sherlock's COMPUTE nodes have no outbound internet --
# confirmed by the pipeline's own preflight, which failed in 30 seconds rather than
# stalling -- so this cannot run as a SLURM job. Login nodes can reach out.
#
# ~1-2 GB, once. The run targets pass --prostt5_weights at this path, and the pipeline
# errors with a pointer here if it is missing rather than trying to download.
FOLDSEEK_IMG := $(shell sed -n "s|.*container '\(quay.io/biocontainers/foldseek@[^']*\)'.*|\1|p" main.nf | head -1)

# The weights are tied to the foldseek release that downloaded them: after the 9 -> 10 bump
# createdb refuses them with "Found ProstT5 model weights for previous Foldseek release".
# FOLDSEEK_IMG is read out of main.nf, so this target always downloads for whatever version
# the pipeline currently pins -- but `foldseek databases` will not write over an existing
# directory, so an old one has to be moved aside first. Moved, not deleted: they are a few
# GB over a slow link, and if the new download fails you still have something that works.
prostt5-weights:
	$(call require_cluster)
	@mkdir -p data/prostt5
	@if [ -d data/prostt5/weights ]; then \
	  stamp=$$(date +%Y%m%d-%H%M%S); \
	  echo "moving existing weights to data/prostt5/weights.old-$$stamp"; \
	  /bin/mv -f data/prostt5/weights data/prostt5/weights.old-$$stamp; \
	fi
	@rm -rf data/prostt5/tmp
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://$(FOLDSEEK_IMG) \
	  foldseek databases ProstT5 data/prostt5/weights data/prostt5/tmp
	@rm -rf data/prostt5/tmp
	@echo "weights at data/prostt5/weights (foldseek $(FOLDSEEK_IMG))"
	@echo "if that worked, the data/prostt5/weights.old-* directories can be removed"


# Pfam-N truth: the "Pfam-A HMMs missed these" labels, streamed from EBI and filtered to
# our accessions. ~17.4 GB crosses the wire and nothing is stored; expect 30-60 min.
# Login node, because compute nodes here have no outbound internet.
#
# curl runs on the HOST and the container only parses stdin. The pipeline image has python
# and polars but no curl, and adding it there for one step is not worth a rebuild and push.
# Running curl on the host also keeps its retry and speed-floor behaviour, which matters
# over 17.4 GB -- and the parser refuses a truncated stream rather than writing a partial
# truth set, since a missing Pfam-N label reads as "Pfam-N had nothing here".
PFAMN_RSYNC := rsync://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-N.gz
PFAMN_GZ   := data/pfamn/Pfam-N.gz

# rsync, not curl. EBI serves this over rsync, which resumes a partial transfer natively --
# the first attempt piped curl straight into the parser and died 27.8M lines in, and a pipe
# cannot resume at all. Re-running this costs the remainder, not the whole 17.4 GB.
pfamn-download:
	$(call require_cluster)
	@mkdir -p data/pfamn
	rsync -P $(PFAMN_RSYNC) $(PFAMN_GZ)
	@echo "downloaded: $$(du -h $(PFAMN_GZ) | cut -f1)"

# Parse the downloaded file. Separate from the download so a parse failure does not cost
# the transfer, and re-running is free once the file is on disk.
pfamn-truth: pfamn-download
	$(call require_cluster)
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/build_pfamn_truth.py \
	    --source      $(PFAMN_GZ) \
	    --annotations data/annotations \
	    --truth-out   data/pfamn/human_pfamn_truth.parquet \
	    --map-outdir  data/pfamn \
	    --summary-out data/pfamn/pfamn_summary.json
	@echo "Pfam-N truth built. $(PFAMN_GZ) can be deleted if disk is tight:"
	@echo "  rm $(PFAMN_GZ)"


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

# All run targets already pass -resume. Do NOT write `make run -resume`: make consumes
# the flag itself and dies with "invalid option -- 'u'" before nextflow is reached.
#
# To pass extra flags through to nextflow, use NF_ARGS:
#   make run NF_ARGS="-with-dag dag.html"
#   make run NF_ARGS="--skip_folddisco true"
# NO `| tee` on any nextflow invocation. Nextflow's live display repaints the task table in
# place; a pipe turns that into thousands of reprinted tables scrolling past. The pretty log
# is the point -- it is how you watch a 13_000-job run. Nothing is lost by dropping the pipe:
# Nextflow already writes a full .nextflow.log in the launch dir. And do not "fix" a mangled
# display with -ansi-log false either -- that removes the display instead of the cause.
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
	    --pfamn_dir      ../data/pfamn          \
	    --prostt5_weights ../data/prostt5/weights \
	    --outdir      ../results
endef

run: prefetch-images
	$(call require_cluster)
	$(call launch,qfo-region,)

# Split arms so the 1017-job kmerseek sweep and the 9-species baselines can be queued
# and debugged independently.
run-kmerseek: prefetch-images
	$(call require_cluster)
	$(call launch,qfo-kmerseek,--skip_baselines true)

run-baselines: prefetch-images
	$(call require_cluster)
	$(call launch,qfo-baselines,--skip_kmerseek true)

# Build the mini set ON Sherlock from the inputs sync-data already staged. Runs inside the
# same container the pipeline uses, so it needs no cluster python environment of its own.
mini-testset-sherlock:
	$(call require_cluster)
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/make_mini_testset.py \
	    --annotations     data/annotations \
	    --qfo-dir         data/qfo \
	    --outdir          data/mini \
	    --target_species  $(MINI_TARGETS) \
	    --structure-cache data/structures

# ---- midi: chr6 queries, full targets ----
midi-testset-sherlock:
	$(call require_cluster)
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/make_mini_testset.py \
	    --annotations     data/annotations \
	    --qfo-dir         data/qfo \
	    --outdir          data/midi \
	    --gene-set        chr6 \
	    --full-targets \
	    --hgnc            data/covariates/hgnc_complete_set.txt \
	    --target_species  $(MIDI_TARGETS) \
	    --structure-cache data/structures

# Full sherlock profile, NOT mini: the targets are real proteomes and need real resources.
# --db_cache points at the full run's outdir so the target-side databases this builds are
# the ones `make run` will reuse. Results stay in data/midi/results -- same target files,
# different query set, so the calls and metrics must not land on top of the full run's.
run-midi: prefetch-images midi-testset-sherlock
	$(call require_cluster)
	mkdir -p run-midi
	cd run-midi && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) -resume \
	    --qfo_dir         ../data/midi/qfo \
	    --annotations     ../data/midi/annotations \
	    --structures      ../data/midi/structures \
	    --outdir          ../data/midi/results \
	    --db_cache        ../results \
	    --target_species  $(MIDI_TARGETS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir       ../data/pfamn \
	    --prostt5_weights ../data/prostt5/weights \
	    $(NF_ARGS)

# Same code path as the real run, scoped down. Do this before `make run`.
run-mini: prefetch-images mini-testset-sherlock
	$(call require_cluster)
	mkdir -p run-mini
	cd run-mini && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE),mini -resume \
	    --qfo_dir         ../data/mini/qfo \
	    --annotations     ../data/mini/annotations \
	    --structures      ../data/mini/structures \
	    --outdir          ../data/mini/results \
	    --target_species  $(MINI_TARGETS) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir        ../data/pfamn          \
	    --prostt5_weights ../data/prostt5/weights \
	    $(NF_ARGS)

# Rebuild the one HTML report from results already published, without re-running a single
# search. This is the normal way to get it after a long sweep: the trace is only complete
# once the run has ended.
#
# Launched from its own directory on purpose. Nextflow truncates its trace file and
# overwrites its execution report and timeline as a run starts, so a report run launched in
# run/ would destroy the very numbers it is reading. main.nf refuses outright in that case
# rather than quietly emitting empty resource sections.
multiqc: prefetch-images
	$(call require_cluster)
	@if [ -z "$(TRACE)" ]; then \
	  echo "No trace file under run/. Pass one explicitly:"; \
	  echo "  make multiqc TRACE=run/qfo_pfam_region.<date>.trace.txt"; \
	  exit 1; \
	fi
	@echo "reporting on trace: $(TRACE)"
	mkdir -p run-report
	cd run-report && \
	  ml load nextflow && \
	  nextflow run ../main.nf -entry report -profile $(PROFILE) \
	    --outdir     $(MULTIQC_OUTDIR) \
	    --report_trace $(abspath $(TRACE)) \
	    --qfo_dir    ../data/qfo \
	    $(NF_ARGS)

status:
	$(call require_cluster)
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}

clean-cache:
	rm -rf $$SCRATCH/apptainer-cache
