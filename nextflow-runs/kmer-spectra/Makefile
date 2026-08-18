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

# UniRef50/90/100: real (non-shuffled) sequences, full 9-alphabet sweep, HP-family
# floor raised from Swiss-Prot's k15 -- see README-sherlock.md for why. Content-
# Length in bytes (`curl -sIL <url>`), checked 2026-08-18 against
# ftp.uniprot.org/pub/databases/uniprot/uniref/<name>/<name>.fasta.gz -- confirm
# again before a big download if it's been a while, UniProt updates these on
# a regular release schedule.
UNIREF_BASE_URL  := https://ftp.uniprot.org/pub/databases/uniprot/uniref
HP_KMIN_UNIREF   ?= 18
MEM_SCALE        ?= 1
UNIREF50_BYTES   := 8770260598
UNIREF90_BYTES   := 32059052376
UNIREF100_BYTES  := 63101622504

.PHONY: build-image push-image sync-pipeline sync-data \
        download-uniref50 download-uniref90 download-uniref100 \
        run-k2 run-k3 run-uniref50 run-uniref90 run-uniref100 status \
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
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-ushuffle-k2/' \
	  $(LOCAL_RESULTS_ROOT)/kmerseek-kmer-spectra-ushuffle-k2/

pull-k3:
	rsync -avz --progress \
	  $(SHERLOCK_HOST):'$$SCRATCH/kmer-spectra/results-ushuffle-k3/' \
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
## Each ksize launches nextflow from its own run-kN/ subdirectory (as its
## Nextflow launchDir), so k2 and k3 can run at the same time without their
## workDir/trace/timeline/report files colliding. Both block in the
## foreground -- run run-k2 and run-k3 in separate tmux panes to get both
## ksizes running concurrently.
##
## --fasta/--outdir are relative to run-kN/ (../data, ../results-ushuffle-kN),
## NOT $SCRATCH/kmer-spectra/... -- workDir/launchDir already live under the
## real (non-symlinked) checkout path, and Apptainer's autoMounts binds paths
## based on that real tree. A fasta file reached through the $SCRATCH/kmer-spectra
## symlink is a second, unrelated absolute path that never gets bound, so the
## staged input symlink was dangling inside the container ("FASTA file not
## found" even though the file exists on the host). Relative paths never cross
## the symlink boundary, so this can't happen.

# Shared per-ksize launch recipe. Implemented as a define + $(call ...) rather
# than a `run-%:` pattern rule -- macOS ships GNU Make 3.81 (2006), which does
# not apply pattern rules to .PHONY targets, so `make run-k2` would silently
# no-op ("Nothing to be done"). Explicit targets below are also more portable
# to whatever `make` version happens to be on Sherlock.
define launch_ksize
	mkdir -p run-$(1)
	cd run-$(1) && \
	  module load biology nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) $(NF_ARGS) \
	    --fasta ../data/uniprot_sprot.ushuffle_$(1).fasta.gz \
	    --outdir ../results-ushuffle-$(1) \
	    --hp_only true 2>&1 | tee nextflow.log
endef

run-k2:
	$(call launch_ksize,k2)

run-k3:
	$(call launch_ksize,k3)

## ---- UniRef50/90/100: download + run, both ON Sherlock ----
##
## UniRef50/90/100 are ~94x/344x/676x Swiss-Prot's compressed fasta size (verified
## via `curl -sIL`, not a guess) -- direct download here avoids moving 8-63 GB
## through the Mac and back. A bare curl on a multi-GB UniProt/EBI file reliably
## stalls (dead TCP connection) and then hangs for hours before curl's own
## timeout catches it, so this resumes (-C -) and aborts within ~60s of a stall
## (--speed-limit/--speed-time) instead, re-invoking in a loop until the file
## reaches the known Content-Length.
define download_uniref
	mkdir -p data
	until [ "$$(stat -c%s data/$(1).fasta.gz 2>/dev/null || echo 0)" -ge "$(2)" ]; do \
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

## Full 9-alphabet sweep (not --hp_only, unlike run-k2/run-k3 -- these are real
## sequences, not null controls), with the HP-family floor raised via
## --hp_kmin. MEM_SCALE defaults to 1 (Swiss-Prot's own tiers, unscaled) --
## deliberately NOT pre-scaled by UniRef's size ratio, since that ratio is a
## fasta-byte-size proxy, not a measurement of kmerseek's actual peak RAM at
## this scale, and guessing memory numbers without evidence is exactly what
## cost real SLURM re-queue cycles on Swiss-Prot. Run uniref50 first, check its
## trace file's peak_rss, and only then decide whether uniref90/100 need
## `MEM_SCALE=N make run-uniref90` -- see README-sherlock.md.
define launch_uniref
	mkdir -p run-$(1)
	cd run-$(1) && \
	  module load biology nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) $(NF_ARGS) \
	    --fasta ../data/$(1).fasta.gz \
	    --outdir ../results-$(1) \
	    --hp_kmin $(HP_KMIN_UNIREF) \
	    --mem_scale $(MEM_SCALE) 2>&1 | tee nextflow.log
endef

run-uniref50:
	$(call launch_uniref,uniref50)

run-uniref90:
	$(call launch_uniref,uniref90)

run-uniref100:
	$(call launch_uniref,uniref100)

status:
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
