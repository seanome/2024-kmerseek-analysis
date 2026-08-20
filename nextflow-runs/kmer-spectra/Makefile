# k-mer spectra pipeline: build/push/sync targets run on your Mac.
# run/status targets run ON Sherlock (ssh sherlock; cd $SCRATCH/kmer-spectra).
# One `make run` sweeps every dataset named in DATASETS (default: k2,k3) in a
# single nextflow invocation. See README-sherlock.md for the full walkthrough.

SHELL := /bin/bash

# --- Mac-side config ---
GIT_SHA        ?= 34002cbd473db82371887645e9d41e9a90f85942
IMAGE_TAG      ?= 2026-08-20-kmer-spectra
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
# Which datasets `make run` sweeps -- matches main.nf's own default (k2, k3;
# the two with a validated memory profile). Override per-invocation to add
# more once they're ready, e.g. `DATASETS=k2,k3,uniref50 make run`.
DATASETS ?= k2,k3

# UniRef50/90/100: real (non-shuffled) sequences, full 9-alphabet sweep, HP-family
# floor raised from Swiss-Prot's k15 by default (main.nf's hp_kmin_uniref) -- see
# README-sherlock.md for why. Content-Length in bytes (`curl -sIL <url>`), checked
# 2026-08-19 against ftp.ebi.ac.uk/pub/databases/uniprot/uniref/<name>/<name>.fasta.gz
# -- confirm again before a big download if it's been a while, UniProt updates
# these on a regular release schedule.
#
# Was ftp.uniprot.org until 2026-08-19, when it started 404ing on every path
# including its own root (not just these files -- a real outage, not a moved
# file). ftp.ebi.ac.uk mirrors byte-identical content (same Content-Length,
# checked for all three here) and is EBI infrastructure this project already
# relies on elsewhere for large downloads.
UNIREF_BASE_URL  := https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref
UNIREF50_BYTES   := 8770260598
UNIREF90_BYTES   := 32059052376
UNIREF100_BYTES  := 63101622504

.PHONY: build-image push-image sync-pipeline sync-data \
        download-uniref50 download-uniref90 download-uniref100 download-uniref \
        run resume status \
        pull-k2 pull-k3 pull-uniref50 pull-uniref90 pull-uniref100 pull-all

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
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-k2/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-ushuffle-k2/

pull-k3:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-k3/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-ushuffle-k3/

pull-uniref50:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-uniref50/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-uniref50/

pull-uniref90:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-uniref90/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-uniref90/

pull-uniref100:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-uniref100/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-uniref100/

# Not part of pull-all -- pull these explicitly once their runs actually exist,
# rather than have a routine `make pull-all` fail on a results dir that isn't
# there yet.
pull-all: pull-k2 pull-k3

## ---- run these ON Sherlock (ssh sherlock; cd $SCRATCH/kmer-spectra) ----
##
## ONE nextflow run sweeps every dataset in DATASETS (default k2,k3) -- a single
## shared workDir/trace/timeline/report, not one per dataset, and Nextflow's own
## maxForks caps concurrency across all selected datasets together instead of
## several uncoordinated invocations each assuming they own the whole node.
##
## No `| tee` on the nextflow command -- piping its stdout turns it into a
## non-tty, so Nextflow can't detect a terminal and falls back to plain
## sequential logging (a whole new status block per update) instead of
## repainting the live dashboard in place. Nextflow already writes its own
## .nextflow.log regardless (that's what's actually been used to debug every
## OOM/arch/path issue this session, not a tee'd copy), so there's nothing lost
## by letting stdout go straight to the terminal.
##
## main.nf's --fasta paths are relative to this directory (its launchDir), NOT
## $SCRATCH/kmer-spectra/... -- workDir/launchDir already live under the real
## (non-symlinked) checkout path, and Apptainer's autoMounts binds paths based
## on that real tree. A fasta file reached through the $SCRATCH/kmer-spectra
## symlink is a second, unrelated absolute path that never gets bound, so a
## staged input symlink would be dangling inside the container ("FASTA file not
## found" even though the file plainly exists on the host).

run:
	module load biology nextflow && \
	  nextflow run main.nf -profile $(PROFILE) $(NF_ARGS) --datasets $(DATASETS)

# Same as `run`, with -resume baked in -- shorthand for the common case of
# picking a run back up (new/failed datasets added, a fixed image pushed,
# etc.) without retyping NF_ARGS=-resume every time. Extra NF_ARGS still
# pass through, e.g. `NF_ARGS=-stub-run make resume`.
resume:
	module load biology nextflow && \
	  nextflow run main.nf -profile $(PROFILE) -resume $(NF_ARGS) --datasets $(DATASETS)

## ---- UniRef50/90/100 downloads: ON Sherlock ----
##
## UniRef50/90/100 are ~94x/344x/676x Swiss-Prot's compressed fasta size (verified
## via `curl -sIL`, not a guess) -- direct download here avoids moving 8-63 GB
## through the Mac and back. A bare curl on a multi-GB UniProt/EBI file reliably
## stalls (dead TCP connection) and then hangs for hours before curl's own
## timeout catches it, so this resumes (-C -) and aborts within ~60s of a stall
## (--speed-limit/--speed-time) instead, re-invoking in a loop until the file
## reaches the known Content-Length.
##
## -C - resumes from whatever's already on disk, which backfires if that partial
## file is actually a small error response (e.g. a 404 page cached during a
## source outage) rather than real partial data: the server has nothing to
## resume from at that byte offset, and curl/the server can end up producing
## confusing "FTP error" / "doesn't seem to support byte ranges" failures on
## every retry forever instead of just trying again from scratch. A multi-GB
## gzip will always be well past 1 MiB within the first successful response
## chunk, so anything smaller sitting on disk is assumed to be a stale error
## response and removed before the next attempt.
define download_uniref
	mkdir -p data
	until [ "$$(stat -c%s data/$(1).fasta.gz 2>/dev/null || echo 0)" -ge "$(2)" ]; do \
	  size=$$(stat -c%s data/$(1).fasta.gz 2>/dev/null || echo 0); \
	  if [ "$$size" -gt 0 ] && [ "$$size" -lt 1048576 ]; then \
	    echo "data/$(1).fasta.gz is only $$size bytes -- likely a cached error response, not real data. Removing before retry."; \
	    rm -f data/$(1).fasta.gz; \
	  fi; \
	  curl -C - --speed-limit 1024 --speed-time 60 --retry 3 \
	    -o data/$(1).fasta.gz \
	    $(UNIREF_BASE_URL)/$(1)/$(1).fasta.gz; \
	done
endef

download-uniref50:
	$(call download_uniref,uniref50,$(UNIREF50_BYTES))

download-uniref90:
	$(call download_uniref,uniref90,$(UNIREF90_BYTES))

download-uniref100:
	$(call download_uniref,uniref100,$(UNIREF100_BYTES))

# All three, sequentially (Make's default without -j) -- deliberately not
# concurrent, so three multi-GB downloads don't fight over the same network
# path at once. ~2-4h total (uniref50 ~10-20min, uniref90 ~40-70min, uniref100
# ~80-140min) -- run inside tmux, same as `make run`.
download-uniref: download-uniref50 download-uniref90 download-uniref100

status:
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
