# k-mer spectra pipeline: build/push/sync targets run on your Mac.
# run-*/status targets run ON Sherlock (ssh sherlock; cd $SCRATCH/kmer-spectra).
# Run run-k2 and run-k3 in separate tmux panes -- they block in the foreground.
# See README-sherlock.md for the full narrative walkthrough.

SHELL := /bin/bash

# --- Mac-side config ---
GIT_SHA        ?= b2dfde27f368a4e99a73b429d0c772ce932fd9e3
IMAGE_TAG      ?= 2026-08-17-kmer-spectra
IMAGE_LOCAL    := kmerseek-spectra:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)
KMERSEEK_SRC   := /Users/olga/code/kmerseek-kmer-frequency-histogram
SHERLOCK_HOST  ?= sherlock
LOCAL_DATA_DIR := /Users/olga/data/uniprot
LOCAL_RESULTS_ROOT := /Users/olga/data

# Sherlock compute nodes are amd64; a Mac build defaults to the host's arch
# (arm64 on Apple Silicon), which apptainer refuses to run. Force amd64.
DOCKER_PLATFORM := linux/amd64

# --- Sherlock-side config (only used by targets that run there) ---
PROFILE := sherlock

.PHONY: build-image push-image sync-pipeline sync-data \
        run-k2 run-k3 status \
        pull-k2 pull-k3 pull-all

## ---- run these from your Mac ----

build-image:
	docker build --platform $(DOCKER_PLATFORM) -t $(IMAGE_LOCAL) \
	  --build-arg GIT_SHA=$(GIT_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

push-image: build-image
	docker tag $(IMAGE_LOCAL) $(IMAGE_REMOTE)
	docker push $(IMAGE_REMOTE)

sync-pipeline:
	git push origin $$(git branch --show-current)
	ssh $(SHERLOCK_HOST) 'cd $$SCRATCH/kmer-spectra && git pull --ff-only'

sync-data:
	rsync -avz --progress -e ssh --checksum \
	  $(LOCAL_DATA_DIR)/uniprot_sprot.ushuffle_k2.fasta.gz \
	  $(LOCAL_DATA_DIR)/uniprot_sprot.ushuffle_k3.fasta.gz \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/data/'

pull-k2:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-ushuffle-k2/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-ushuffle-k2/

pull-k3:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-ushuffle-k3/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-ushuffle-k3/

pull-all: pull-k2 pull-k3

## ---- run these ON Sherlock (ssh sherlock; cd $SCRATCH/kmer-spectra) ----
##
## Each ksize launches nextflow from its own run-kN/ subdirectory (as its
## Nextflow launchDir), so k2 and k3 can run at the same time without their
## workDir/trace/timeline/report files colliding. Both block in the
## foreground -- run run-k2 and run-k3 in separate tmux panes to get both
## ksizes running concurrently.

# Shared per-ksize launch recipe. Implemented as a define + $(call ...) rather
# than a `run-%:` pattern rule -- macOS ships GNU Make 3.81 (2006), which does
# not apply pattern rules to .PHONY targets, so `make run-k2` would silently
# no-op ("Nothing to be done"). Explicit targets below are also more portable
# to whatever `make` version happens to be on Sherlock.
define launch_ksize
	mkdir -p run-$(1)
	cd run-$(1) && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) \
	    --fasta $$SCRATCH/kmer-spectra/data/uniprot_sprot.ushuffle_$(1).fasta.gz \
	    --outdir $$SCRATCH/kmer-spectra/results-ushuffle-$(1) \
	    --hp_only true 2>&1 | tee nextflow.log
endef

run-k2:
	$(call launch_ksize,k2)

run-k3:
	$(call launch_ksize,k3)

status:
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
