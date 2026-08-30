# QfO Pfam region/domain-finding benchmark.
#
# build/push/sync/fetch/pull targets run on your Mac.
# run-*/status targets run ON Sherlock (ssh sherlock; cd $SCRATCH/qfo-pfam-region-benchmark).
# See README-sherlock.md for the full runbook.

SHELL := /bin/bash

# --- Mac-side config -------------------------------------------------------
# Resolved from the source worktree at build time, not pinned by hand. It is written into
# the image as provenance (/etc/kmerseek_git_sha) while the build itself COPYs that
# worktree, so a hand-pinned value silently labels the image with a commit it does not
# contain. Override on the command line to build a specific commit.
KMERSEEK_SHA   ?= $(shell git -C $(KMERSEEK_SRC) rev-parse HEAD 2>/dev/null)
IMAGE_TAG      ?= 2026-08-24-reduced-alphabets
IMAGE_LOCAL    := kmerseek-region:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)

# IMAGE_TAG is the single source of truth for which kmerseek image runs, and every Sherlock
# run target passes --kmerseek_image $(IMAGE_REMOTE) so that stays true.
#
# nextflow.config declares its own literal for params.kmerseek_image. That is the right
# default for a bare `nextflow run` with no Makefile involved, but as a value the run
# targets INHERITED it was a silent trap: IMAGE_TAG also builds ALL_CONTAINERS, so bumping
# the tag here made `make prefetch-images` pull and cache the new image, print a cheerful
# "have <new tag>", and then hand the run the old one from the config. Two strings that
# have to be edited together, where disagreeing produces results rather than an error, is
# the shape that gets blamed on the pipeline for a week. So the Makefile always overrides.
#
# The local run targets already did this with kmerseek-region:local. This is the same move
# for the cluster.
#
# A SECOND kmerseek image, for the two alphabets the running sweep's binary does not have.
# polarity4 and funcgroups8 landed in kmerseek dd630a8 (2026-08-25); IMAGE_TAG above was
# built from 3fdfd51a (2026-08-24), which predates it.
#
# The obvious move -- bump IMAGE_TAG and rebuild -- is the one to avoid. The container is
# part of every kmerseek task's hash, so a new tag re-pulls the image on every node and
# rehashes the whole matrix, and a sweep sitting on 2507 cached indexes and 1700 cached
# searches would resume by re-running all of it. A second tag leaves that cache alone: only
# `run-rannon-burstein` passes it, and only that run's tasks hash against it.
#
# Both images are built from the same KMERSEEK_SRC worktree, so this tag is only meaningful
# once that worktree is at a commit containing dd630a8. `make build-image-rannon-burstein`
# checks that rather than trusting it.
RB_IMAGE_TAG    ?= 2026-08-27-rannon-burstein
RB_IMAGE_REMOTE := docker.io/olgabot/kmerseek:$(RB_IMAGE_TAG)
# The kmerseek commit that added polarity4 and funcgroups8.
RB_REQUIRED_SHA ?= dd630a8

METAPREDICT_TAG    ?= 2026-08-27
METAPREDICT_REMOTE := docker.io/olgabot/metapredict:$(METAPREDICT_TAG)
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
# Pfam-A HMMs for the hmmscan annotation ceiling. main.nf defaults to ~/data/pfam/Pfam-A.hmm
# on whichever machine it runs, so the cluster needs its own copy -- see sync-pfam.
PFAM_HMM       ?= /Users/olga/data/pfam/Pfam-A.hmm

# Where sync-pfam puts it on the cluster, and therefore what the run targets pass. These
# two have to agree: main.nf sets params.run_hmmscan from file(pfam_hmm).exists(), so a
# path nothing wrote to turns the annotation ceiling off silently. That is what made a
# successful `make sync-pfam` still produce runs with no hmmscan arm -- the data was there
# and nothing pointed at it.
#
# $$SCRATCH, not $(SCRATCH): the run targets execute ON Sherlock, so this has to reach that
# shell as a literal $SCRATCH for it to expand. Unquoted for the same reason -- the single
# quotes sync-pfam wraps its rsync destination in are what keep ssh from expanding it on
# the Mac, and there is no ssh hop here.
PFAM_HMM_REMOTE ?= $$SCRATCH/data/pfam/Pfam-A.hmm
# Same shape as LOCAL_SKIP_ARMS: the whole flag or nothing, so `PFAM_HMM_REMOTE=` runs
# without --pfam_hmm rather than passing an empty path.
PFAM_HMM_ARG     = $(if $(PFAM_HMM_REMOTE),--pfam_hmm $(PFAM_HMM_REMOTE))

QFO_DIR        := /Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
STRUCT_DIR     := /Users/olga/data/alphafold_proteomes
ACC_DIR        := /Users/olga/data/alphafold_proteomes/_accessions
LOCAL_RESULTS  := /Users/olga/data/qfo-pfam-region-benchmark
MINI_DIR       := /Users/olga/data/qfo-pfam-region-benchmark-mini
MHC_DIR        := /Users/olga/data/qfo-pfam-region-benchmark-mhc

# Empty means all ten proteomes. Name one to fetch just that, e.g.
#   make fetch-structures SPECIES_ONE=mouse
SPECIES_ONE    ?=

# Parallel transfers on the login node. Modest on purpose: shared node, public EBI endpoint.
FETCH_PARALLEL ?= 4

# Two species and two combos: enough to exercise every stage in minutes.
# TARGET proteomes. Human is the query and is never listed here.
MINI_TARGETS ?= yeast,ecoli
MINI_COMBOS  ?= protein20:10,hp_pbotc_1st_ed2:19

# Midi: every human chr6 gene as the query, FULL proteomes as targets. Real scale on the
# target side, ~5% of the query load -- 964 proteins carrying 2358 of the 50185 human Pfam
# domain instances. chr6 because it carries the MHC, so this subsumes the MHC vignette.
MIDI_TARGETS ?= mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli

# Arms the local smoke tests leave out. The big stuff belongs on Sherlock.
#
# ProstT5 is the whole of it right now. prostt5Weights downloads the model over the
# network, and it has FAILED in every local run there has ever been -- exit 1 in half a
# second in both run-mhc/qfo_pfam_region.2026-08-24.trace.txt and the 2026-08-26 one, which
# then aborts everything queued behind it. The Sherlock targets never hit that process
# because they pass --prostt5_weights ../data/prostt5/weights, downloaded once by
# `make prostt5-weights` on a login node; the local targets have no such path.
#
# Even with weights on hand this would be the wrong place for it. prostt5Db runs a
# 3B-parameter T5 encoder over every proteome, on a GPU on Sherlock and on nothing at all
# here, and params.prostt5_memory is 64 GB with a comment in main.nf saying explicitly not
# to scale it down for a smoke test -- the model sets that floor, not the data set. The
# local runs exist to check pipeline wiring in minutes.
#
# Set to empty to run the arm anyway (expect a multi-GB download and a very long CPU pass):
#   make run-mhc-local LOCAL_SKIP_ARMS=
LOCAL_SKIP_ARMS ?= --skip_prostt5 true

PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# Mac-side Nextflow. Pinned to the conda env rather than relying on PATH; on Sherlock the
# run targets use `ml load nextflow` instead.
NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

# Sherlock compute nodes are amd64; an Apple Silicon build defaults to arm64, which
# Apptainer refuses to run.
DOCKER_PLATFORM := linux/amd64

# Where kmerseekIndex stores its target indexes: DB_CACHE in main.nf, which the run
# targets set to ../results so run-midi and the full run share one set.
REMOTE_INDEX_DIR ?= results/kmerseek_index

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

# Both multiqc targets launch nextflow from run-report/, so a relative --outdir is relative
# to THAT directory and not to this Makefile. The preflight check below has to look in the
# same place nextflow will, so relative values get resolved through run-report/ and
# absolute ones are left alone.
outdir_abs = $(if $(filter /%,$(1)),$(1),$(abspath run-report/$(1)))

MULTIQC_OUTDIR_ABS       := $(call outdir_abs,$(MULTIQC_OUTDIR))
MULTIQC_OUTDIR_LOCAL_ABS := $(call outdir_abs,$(MULTIQC_OUTDIR_LOCAL))

# The same resolution for any other launch directory. outdir_abs hard-codes run-report/,
# which is right for the two multiqc targets and wrong for everything else: every run
# target's paths are written against ITS launch directory, and make's own cwd is the
# pipeline directory one level up. `--report_outdirs ../data/midi/results` passed through
# $(abspath) without this correction resolves one directory too high and silently names a
# path that does not exist.
#   $(1) launch directory the path was written against
#   $(2) the path
rel_to_launch = $(if $(filter /%,$(2)),$(2),$(abspath $(1)/$(2)))

# Every container the pipeline can pull, read straight out of main.nf plus the kmerseek
# image, so this list cannot drift from what the processes actually declare.
ALL_CONTAINERS := $(shell sed -n "s/^[[:space:]]*container[[:space:]]*'\\(.*\\)'.*/\\1/p" main.nf | sort -u) \
                  docker.io/olgabot/kmerseek:$(IMAGE_TAG) $(FOLDDISCO_REMOTE) \
                  $(METAPREDICT_REMOTE)

# What `prefetch-images` actually pulls. Indirection so a run target that needs an image
# outside the standard set can add it with a target-specific override, which make also
# applies to the target's prerequisites -- see run-rannon-burstein. RB_IMAGE_REMOTE is
# deliberately NOT in ALL_CONTAINERS: that would make every `make run` pull and squashfs a
# second multi-GB image for arms it never executes.
PREFETCH_CONTAINERS ?= $(ALL_CONTAINERS)

.PHONY: bootstrap-sherlock build-image build-image-local push-image \
        build-folddisco-image push-folddisco-image \
        build-metapredict-image push-metapredict-image structure-lists fetch-structures \
        mini-testset mini-testset-sherlock mhc-testset run-mhc-local \
        midi-testset-sherlock run-midi fetch-structures-login \
        fetch-structures-mac \
        run-mini run-mini-local pull-mini \
        sync-pipeline sync-data sync-pfam sync-structures pull-results \
        run run-kmerseek run-baselines rannon-burstein-all status running kill-jobs clean-cache clean-indexes index-disk \
        gpu-benchmark gpu-benchmark-report \
        prefetch-images clean-image-cache fetch-structures-sherlock prostt5-weights \
        pfamn-download pfamn-truth mcsa-truth help \
        bpe-tokenizer hp-bpe-diagnostic sherlock-clone test \
        fetch-structures-login \
        multiqc multiqc-local multiqc-preflight multiqc-status \
        build-image-rannon-burstein push-image-rannon-burstein \
        run-rannon-burstein multiqc-rannon-burstein \
        pull-report pull-report-mini pull-report-midi

help:
	@echo "Mac:"
	@echo "  build-image           Build the amd64 Kmerseek image for Sherlock"
	@echo "  build-image-local     Build the native Kmerseek image for local Docker"
	@echo "  push-image            Push the Kmerseek image to Docker Hub"
	@echo "  push-image-rannon-burstein"
	@echo "                        Push a SECOND Kmerseek image carrying polarity4 and"
	@echo "                        funcgroups8, under its own tag so the running sweep's"
	@echo "                        cache is untouched"
	@echo "  push-folddisco-image  Push the FoldDisco image to Docker Hub"
	@echo "  sync-pipeline         Push the current branch and update Sherlock"
	@echo "  sync-data             Copy benchmark inputs to Sherlock"
	@echo "  sync-structures       Copy AlphaFold structures to Sherlock"
	@echo "  run-mhc-local         Smoke test: 25 MHC genes vs mouse, ProstT5 arm off"
	@echo "  run-mini-local        Smoke test: mini set, ProstT5 arm off"
	@echo "  multiqc-status        Show what the report would be built from"
	@echo "  multiqc-local         Build the MultiQC report from a local run"
	@echo "  test                  Run the pipeline tests on the committed fixture (~15 s)"
	@echo "  bpe-tokenizer         Fetch the ProtBERTa BPE tokenizers from Zenodo (186 KB)"
	@echo "  hp-bpe-diagnostic     Do ProtBERTa_2 token boundaries land on Pfam domains?"
	@echo "  pull-report           Fetch the MultiQC report back from Sherlock"
	@echo ""
	@echo "  The local runs are wiring smoke tests. Anything that needs a model download,"
	@echo "  a GPU or real proteomes belongs on Sherlock -- see LOCAL_SKIP_ARMS."
	@echo ""
	@echo "Sherlock:"
	@printf '%s\n' '  prefetch-images       Pull all pinned images into $$SCRATCH cache'
	@echo "  mini-testset-sherlock Build the mini data set"
	@echo "  run-mini             Run the mini smoke test"
	@echo "  run-midi             chr6 queries vs FULL proteomes (includes the MHC)"
	@echo "  fetch-structures     Download AlphaFold structures ON the login node (~36 GB)"
	@echo "  run-kmerseek         Run only the Kmerseek sweep"
	@echo "  run-baselines        Run only the baseline tools"
	@echo "  gpu-benchmark        Time foldseek+mmseqs search on CPU and GPU, one session"
	@echo "  gpu-benchmark-report Re-read a gpu-benchmark trace into queries/s"
	@echo "  run                  Run the complete benchmark"
	@echo "  run-rannon-burstein  polarity4 + funcgroups8 only, midi set, kmerseek only"
	@echo "  multiqc              Rebuild the MultiQC report from published results"
	@echo "  multiqc-rannon-burstein"
	@echo "                       One report over run-midi AND run-rannon-burstein, so the"
	@echo "                       two new alphabets appear next to every other tool"
	@echo "  multiqc-status       Show what that report would be built from"
	@echo "  prostt5-weights      Download ProstT5 weights on a login node"
	@echo "  status               Show this user's SLURM jobs"
	@echo "  rannon-burstein-all  Search polarity4/funcgroups8, then build the combined report"
	@echo "  running              Show the process and tag behind each running job"
	@echo "  kill-jobs            Cancel every job on the account (confirms first)"
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

# The name these two guards quote back at you. Normally the target being built, which is
# also the target you typed. A helper target that only ever runs as somebody else's
# prerequisite -- multiqc-preflight -- sets this to the name you actually typed, since
# being told to ssh over and run an internal target is not advice anyone can follow.
GUARD_NAME =
guard_name = $(if $(GUARD_NAME),$(GUARD_NAME),$@)

define require_mac
	@if [ "$(ON_CLUSTER)" = "yes" ]; then \
	  echo ""; \
	  echo "  '$(guard_name)' runs on your MAC, not on the cluster."; \
	  echo "  It moves data from the Mac to $(SHERLOCK_HOST), so it cannot work from here."; \
	  echo ""; \
	  echo "  On your Mac:"; \
	  echo "    cd $(MAC_PIPELINE_DIR)"; \
	  echo "    make $(guard_name)"; \
	  echo ""; \
	  exit 1; \
	fi
endef

define require_cluster
	@if [ "$(ON_CLUSTER)" != "yes" ]; then \
	  echo ""; \
	  echo "  '$(guard_name)' runs ON SHERLOCK, not on your Mac (no \$$SCRATCH here)."; \
	  echo ""; \
	  echo "  ssh $(SHERLOCK_HOST)"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark"; \
	  echo "    make $(guard_name)"; \
	  echo ""; \
	  exit 1; \
	fi
endef

# The report reads all_domain_metrics.parquet out of an outdir; it does not compute it.
# main.nf's `report` entry does say so, but only after nextflow has started, resolved a
# profile, pulled an image and reached the workflow body -- and only about the parquet
# path, which does not tell you which run was supposed to have written it. `make
# multiqc-local` hit this because the MHC smoke run had never finished: prostt5Weights
# failed, everything behind it aborted, aggregateMetrics never ran. So the same condition is
# checked here first, in a form that names the run and the command that produces the file.
#
#   $(1) outdir to look in, already absolute
#   $(2) run directory that outdir belongs to
#   $(3) make target that fills it
#   $(4) variable to override to point somewhere else
#   $(5) the target the user typed, spelled out rather than taken from $@, because this
#        also runs from a preflight prerequisite whose own name would mean nothing to them
define require_metrics
	@if [ ! -f "$(1)/all_domain_metrics.parquet" ]; then \
	  echo ""; \
	  echo "  '$(5)' has nothing to report on yet."; \
	  echo ""; \
	  echo "    run directory:  $(2)"; \
	  echo "    outdir:         $(1)"; \
	  echo "    looked for:     $(1)/all_domain_metrics.parquet"; \
	  echo ""; \
	  echo "  aggregateMetrics publishes that file at the END of a run, so a run that"; \
	  echo "  failed partway leaves the outdir looking like this. Produce it with:"; \
	  echo ""; \
	  echo "    make $(3)"; \
	  echo ""; \
	  echo "  or report on a different run's outdir:"; \
	  echo ""; \
	  echo "    make $(5) $(4)=<outdir>"; \
	  echo ""; \
	  echo "  'make multiqc-status' shows this without launching anything."; \
	  echo ""; \
	  exit 1; \
	fi
endef

# What the two multiqc targets would use, without starting nextflow. Reports on whichever
# side you are on, since the trace and outdir defaults differ between them.
define report_status
	@echo ""
	@echo "  make $(3)"
	@echo "    trace:   $(if $(1),$(1),NONE FOUND -- pass $(5)=<trace file>)"
	@echo "    outdir:  $(2)"
	@if [ -f "$(2)/all_domain_metrics.parquet" ]; then \
	  echo "    metrics: OK      $(2)/all_domain_metrics.parquet"; \
	else \
	  echo "    metrics: MISSING $(2)/all_domain_metrics.parquet"; \
	  echo "             run 'make $(4)' first"; \
	fi
	@if [ -f "$(2)/all_domain_curves.parquet" ]; then \
	  echo "    curves:  OK      $(2)/all_domain_curves.parquet"; \
	else \
	  echo "    curves:  missing -- PR and ROC sections will be skipped"; \
	fi
	@echo ""
endef

multiqc-status:
ifeq ($(ON_CLUSTER),yes)
	$(call report_status,$(TRACE),$(MULTIQC_OUTDIR_ABS),multiqc,run,TRACE)
else
	$(call report_status,$(TRACE_LOCAL),$(MULTIQC_OUTDIR_LOCAL_ABS),multiqc-local,run-mhc-local,TRACE_LOCAL)
endif

## ---- Mac side -----------------------------------------------------------

build-image:
	$(call require_mac)
	docker build --platform $(DOCKER_PLATFORM) -t $(IMAGE_LOCAL) \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

# The image carrying polarity4 and funcgroups8. Same Dockerfile and same source worktree as
# build-image; the only difference is the tag it gets pushed under, which is the whole point
# -- see the RB_IMAGE_TAG note at the top.
#
# The ancestry check is here rather than left to the run. KMERSEEK_SRC is a live worktree,
# so nothing stops it sitting on a commit older than dd630a8, and an image built from one
# would push, prefetch and launch perfectly happily before failing on the first task with
# "unexpected argument '--polarity4'" -- after the queue wait. `git merge-base
# --is-ancestor` answers the question in a second here instead.
build-image-rannon-burstein:
	$(call require_mac)
	@if ! git -C $(KMERSEEK_SRC) merge-base --is-ancestor $(RB_REQUIRED_SHA) HEAD; then \
	  echo ""; \
	  echo "  $(KMERSEEK_SRC) is at $(KMERSEEK_SHA),"; \
	  echo "  which does not contain $(RB_REQUIRED_SHA) -- the commit that added polarity4"; \
	  echo "  and funcgroups8. An image built from it would not have the alphabets."; \
	  echo ""; \
	  echo "    git -C $(KMERSEEK_SRC) log --oneline -5"; \
	  echo ""; \
	  exit 1; \
	fi
	@echo "building $(RB_IMAGE_REMOTE) from kmerseek $(KMERSEEK_SHA)"
	docker build --platform $(DOCKER_PLATFORM) -t kmerseek-region:rannon-burstein \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f Dockerfile \
	  $(KMERSEEK_SRC)

push-image-rannon-burstein: build-image-rannon-burstein
	$(call require_mac)
	docker tag kmerseek-region:rannon-burstein $(RB_IMAGE_REMOTE)
	docker push $(RB_IMAGE_REMOTE)

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
build-metapredict-image:
	$(call require_mac)
	docker build --platform $(DOCKER_PLATFORM) -t metapredict:latest \
	  -f Dockerfile.metapredict .

push-metapredict-image: build-metapredict-image
	$(call require_mac)
	docker tag metapredict:latest $(METAPREDICT_REMOTE)
	docker push $(METAPREDICT_REMOTE)

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
#
# It asks first. This is the one Mac target that commits the laptop to tens of GB over the
# wire, and the name sits one tab-completion away from `fetch-structures`, which is the
# cluster one you almost always want. Same read -p shape as clean-indexes. Skip the prompt
# with `make fetch-structures-mac CONFIRM=yes` when you mean it.
fetch-structures-mac: structure-lists
	$(call require_mac)
	@if [ "$(CONFIRM)" != "yes" ]; then \
	  echo ""; \
	  echo "  This downloads ~60 GB of AlphaFold structures to $(STRUCT_DIR) on THIS Mac."; \
	  echo "  For a Sherlock run you want 'make fetch-structures' instead, which downloads"; \
	  echo "  them on the login node where they will actually be used."; \
	  echo ""; \
	  read -p "  download to this Mac anyway? [y/N] " a; [ "$$a" = y ] || exit 1; \
	fi
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

# $(NEXTFLOW), not a bare `nextflow`: the same conda pin every other Mac-side target uses,
# so this does not depend on which env happens to be active in the shell.
run-mhc-local:
	$(call require_mac)
	mkdir -p run-mhc && cd run-mhc && \
	  $(NEXTFLOW) run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MHC_DIR)/qfo \
	    --annotations     $(MHC_DIR)/annotations \
	    --structures      $(MHC_DIR)/structures \
	    --outdir          $(MHC_DIR)/results \
	    $(BPE_NF_ARG) \
	    --target_species  mouse \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(LOCAL_SKIP_ARMS) \
	    $(NF_ARGS)

# Run it here first (needs build-image-local), before spending Sherlock queue time.
run-mini-local:
	$(call require_mac)
	mkdir -p run-mini && cd run-mini && \
	  $(NEXTFLOW) run ../main.nf -profile standard,mini -resume \
	    --kmerseek_image  kmerseek-region:local \
	    --qfo_dir         $(MINI_DIR)/qfo \
	    --annotations     $(MINI_DIR)/annotations \
	    --structures      $(MINI_DIR)/structures \
	    --outdir          $(MINI_DIR)/results \
	    $(BPE_NF_ARG) \
	    --target_species  $(MINI_TARGETS) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       $(HGNC_FILE) \
	    --omega_file      $(OMEGA_FILE) \
	    $(LOCAL_SKIP_ARMS) \
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
	$(call require_metrics,$(MULTIQC_OUTDIR_LOCAL_ABS),run-mhc,run-mhc-local,MULTIQC_OUTDIR_LOCAL,multiqc-local)
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
## Which run's outdir to pull from, relative to REMOTE_DIR. Every run target writes its
## MultiQC report to its own --outdir, so the full run's `results` is only one of them:
##   run       -> results              run-midi  -> data/midi/results
##   run-mini  -> data/mini/results
## Use the pull-report-mini / pull-report-midi shorthands rather than passing both vars.
REMOTE_RESULTS ?= results

pull-report:
	$(call require_mac)
	mkdir -p $(LOCAL_RESULTS)
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/$(REMOTE_RESULTS)/qfo_pfam_region_multiqc.html' $(LOCAL_RESULTS)/
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/$(REMOTE_RESULTS)/qfo_pfam_region_multiqc_plots/' \
	  $(LOCAL_RESULTS)/qfo_pfam_region_multiqc_plots/
	@echo; echo "open $(LOCAL_RESULTS)/qfo_pfam_region_multiqc.html"

pull-report-mini:
	@$(MAKE) pull-report REMOTE_RESULTS=data/mini/results LOCAL_RESULTS=$(MINI_DIR)/results

pull-report-midi:
	@$(MAKE) pull-report REMOTE_RESULTS=data/midi/results LOCAL_RESULTS=$(LOCAL_RESULTS)/midi

pull-mini:
	$(call require_mac)
	$(RSYNC) \
	  $(SHERLOCK_HOST):'$(REMOTE_DIR)/data/mini/results/' $(LOCAL_RESULTS)-mini/

# Code moves by git, never rsync: rsync duplicates state outside git history and defeats
# the point of running this from a worktree.
# One-time setup: clone the repo onto $SCRATCH and create the data directories the sync
# targets write into. Safe to re-run; it does not touch an existing checkout's state.
## A SECOND checkout on Sherlock, so a branch can run without touching the one that
## already has work in flight. Switching branches in the primary checkout would move
## main.nf under a running sweep, and Nextflow re-reads it on resume.
##
## The inputs are HARD-LINKED, not symlinked and not copied. This is not a micro-
## optimisation: `ln -s data` would make every relative path the run targets pass
## (`--qfo_dir ../data/qfo`) resolve through the link to an absolute path under the OTHER
## checkout, which Apptainer's autoMounts never binds -- the staged input then dangles
## inside the container. That failure is already documented above the Sherlock run targets.
## Hard links give a real directory tree at a real path under this clone, share the inode
## so ~36 GB of structures is not duplicated, and cost nothing.
##
## Only the INPUT subdirectories are linked. data/mini, data/midi and data/gpu hold run
## OUTPUTS, and sharing an inode with another run's published results is how two runs
## silently overwrite each other.
SHERLOCK_CLONE ?= $$SCRATCH/2024-kmerseek-$(notdir $(BRANCH))
SHERLOCK_CLONE_DIR := $(SHERLOCK_CLONE)/nextflow-runs/qfo-pfam-region-benchmark
SHERLOCK_SHARED_INPUTS := annotations qfo covariates structures prostt5 pfamn protberta

sherlock-clone:
	$(call require_mac)
	git push origin $$(git branch --show-current)
	$(SSH_MUX) $(SHERLOCK_HOST) '\
	  set -e; \
	  if [ ! -d $(SHERLOCK_CLONE)/.git ]; then \
	    git clone --branch $(BRANCH) $(REPO_URL) $(SHERLOCK_CLONE); \
	  else \
	    cd $(SHERLOCK_CLONE) && git fetch origin $(BRANCH) \
	      && git checkout $(BRANCH) && git pull --ff-only; \
	  fi; \
	  mkdir -p $(SHERLOCK_CLONE_DIR)/data; \
	  for d in $(SHERLOCK_SHARED_INPUTS); do \
	    src=$(REMOTE_DIR)/data/$$d; dst=$(SHERLOCK_CLONE_DIR)/data/$$d; \
	    [ -e "$$src" ] || continue; \
	    [ -e "$$dst" ] && { echo "have  $$d"; continue; }; \
	    cp -al "$$src" "$$dst" 2>/dev/null && echo "link  $$d" \
	      || { echo "copy  $$d (hard link failed: different filesystem?)"; \
	           cp -a "$$src" "$$dst"; }; \
	  done; \
	  for f in $(REMOTE_DIR)/data/uniprot_sprot.dat.gz; do \
	    [ -e "$$f" ] && ln -f "$$f" $(SHERLOCK_CLONE_DIR)/data/ 2>/dev/null || true; \
	  done; \
	  echo "ready: $(SHERLOCK_CLONE_DIR)"'
	@echo
	@echo "inputs hard-linked from $(REMOTE_DIR)/data -- no structures were copied."
	@echo
	@echo "Target databases are not duplicated either. Point db_cache at the primary"
	@echo "checkout so every foldseek / mmseqs / reseek / kmerseek index is built once:"
	@echo
	@echo "  ssh $(SHERLOCK_HOST)"
	@echo "  cd $(SHERLOCK_CLONE_DIR)"
	@echo "  make bpe-tokenizer"
	@echo '  make run-mini NF_ARGS="--db_cache $(REMOTE_DIR)/results"'

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

## Push from the Mac, then pull on Sherlock over ssh. Costs a two-factor prompt for the
## ssh hop; if you are already logged in to Sherlock, push here and run `make
## pull-pipeline` there instead, which does the same pull without the hop.
sync-pipeline:
	$(call require_mac)
	git push origin $$(git branch --show-current)
	$(SSH_MUX) $(SHERLOCK_HOST) 'cd $(REMOTE_DIR) && git pull --ff-only'

## The Sherlock half of sync-pipeline, run ON SHERLOCK. Same `git pull --ff-only`, minus
## the ssh from the Mac that makes you re-do two-factor for a command you are already
## sitting in front of. Assumes the Mac side has pushed; --ff-only so a diverged checkout
## stops rather than opening a merge in a shell you did not ask to be in.
pull-pipeline:
	$(call require_cluster)
	@echo "branch: $$(git -C $(REMOTE_REPO) rev-parse --abbrev-ref HEAD)"
	git -C $(REMOTE_REPO) pull --ff-only
	@echo
	@git -C $(REMOTE_REPO) log --oneline -6

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

## Pfam-A HMMs for the annotation ceiling. Not part of sync-data because it is 4.5 GB and
## only one arm needs it, but without it that arm silently does not run: params.run_hmmscan
## defaults to file(pfam_hmm).exists(), which was false on the cluster for every run of this
## pipeline and said nothing.
##
## All five files, not just the .hmm. hmmscan reads the pressed .h3m/.h3i/.h3f/.h3p; the
## .hmm alone switches the arm on and then fails inside the container on a database it
## cannot read. main.nf now checks for them up front rather than letting that happen.
##
## Running this is the whole of it. The destination below and PFAM_HMM_REMOTE above are the
## same path, and every run target passes --pfam_hmm $(PFAM_HMM_REMOTE), so the next run
## picks the arm up with no flag. A sync that still needed a flag afterwards was a sync
## that looked done and was not.
sync-pfam:
	$(call require_mac)
	@test -f $(PFAM_HMM) || { echo "no $(PFAM_HMM) on this Mac"; exit 1; }
	@ls $(PFAM_HMM).h3m $(PFAM_HMM).h3i $(PFAM_HMM).h3f $(PFAM_HMM).h3p >/dev/null 2>&1 \
	  || { echo "$(PFAM_HMM) is not pressed. Run: hmmpress $(PFAM_HMM)"; exit 1; }
	@du -ch $(PFAM_HMM) $(PFAM_HMM).h3* | tail -1
	$(SSH_MUX) $(SHERLOCK_HOST) 'mkdir -p $$SCRATCH/data/pfam'
	$(RSYNC) $(PFAM_HMM) $(PFAM_HMM).h3m $(PFAM_HMM).h3i $(PFAM_HMM).h3f $(PFAM_HMM).h3p \
	  $(SHERLOCK_HOST):'$$SCRATCH/data/pfam/'
	@echo
	@echo 'hmmscan annotation ceiling is now ENABLED for the run targets.'
	@echo 'They already pass --pfam_hmm $(PFAM_HMM_REMOTE) -- no extra flag needed.'
	@echo 'Point them elsewhere with PFAM_HMM_REMOTE=<path>, or PFAM_HMM_REMOTE= to run without it.'

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
	for ref in $(PREFETCH_CONTAINERS); do \
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
# Default to the LOGIN node. fetch-structures-sherlock routes the download through
# Nextflow, which puts it on a compute node -- and Sherlock compute nodes have no outbound
# internet, so it fails its own preflight every time. The login node has the fast link, so
# there is no reason to drag ~36 GB across an rsync from the Mac either: EBI -> Sherlock
# directly beats EBI -> Mac -> Sherlock on both hops.
#
# This blocks in the foreground so you can watch it. The transfers are resumable: proteome
# archives come over rsync, per-accession fetches use --continue-at with a speed floor, and
# re-running skips whatever is already on disk. Safe to Ctrl-C and re-run.
fetch-structures: fetch-structures-login

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
# All ten proteomes unless SPECIES_ONE names one. The accession lists have to exist first:
# chicken and ciona are not in AFDB's proteome set and are fetched accession by accession,
# so without the lists those two silently produce nothing. That step was missing here while
# fetch-structures-sherlock had it, which is why this target looked like it worked and then
# left two species empty.
fetch-structures-login:
	$(call require_cluster)
	@mkdir -p data/structures/_accessions
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/write_structure_lists.py \
	    --annotations data/annotations \
	    --outdir      data/structures/_accessions
	PARALLEL=$(FETCH_PARALLEL) FLAT_CACHE=data/structures bin/fetch_alphafold_structures.sh \
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


## ---- reduced-alphabet information ceiling ----
##
## The BPE boundary diagnostic needs the tokenizers Rannon & Burstein released with
## ProtBERTa. Zenodo record id, not the DOI string: the DOI resolves through a redirect
## that the compute nodes cannot follow, and the record id is what the files endpoint keys
## on. 186 KB, so this is a login-node download and not a job.
##
## The checksum is the one Zenodo reports for the record; the target refuses a mismatch
## rather than handing a truncated archive to the diagnostic, which would silently produce
## a shorter merge table and therefore longer tokens.
BPE_ZENODO_RECORD := 18256943
BPE_TARBALL       := data/protberta/ProtBERTa_tokenizers.tar.gz
BPE_SHA256        := d594310f524c69f6dbf9d445d049d09cd0370af443a40abf600692bad2c542f6
BPE_OUT           ?= data/protberta/hp_bpe_boundary.json

# Appended to every launch target, but only once the tarball is actually on disk. Named
# rather than left to NF_ARGS so that `make bpe-tokenizer` is the whole opt-in: a flag that
# has to be retyped on every run target is a flag that gets left off the run that matters.
# Evaluated at launch time via $(wildcard), so downloading the tokenizer between two runs
# turns the arm on without editing anything.
#
# ../data because the run targets launch nextflow from run/, and data/ is gitignored and
# lives beside main.nf.
BPE_NF_ARG = $(if $(wildcard $(BPE_TARBALL)),--bpe_tokenizer ../$(BPE_TARBALL),)

# Runs on the Mac or on a login node -- deliberately not guarded by require_cluster, since
# the diagnostic is meant to be run before any cluster time is committed.
## Everything downstream of search, on the committed 60-protein fixture. No containers, no
## QfO download, no cluster: ~15 s. This is what CI runs.
test:
	$(PYTHON) -m pytest tests/ -v
	$(PYTHON) tests/test_dedup_passes.py

bpe-tokenizer:
	@mkdir -p $(dir $(BPE_TARBALL))
	@if [ -s "$(BPE_TARBALL)" ]; then echo "have $(BPE_TARBALL)"; else \
	  curl -fL --retry 3 --retry-delay 5 \
	    "https://zenodo.org/records/$(BPE_ZENODO_RECORD)/files/ProtBERTa_tokenizers.tar.gz?download=1" \
	    -o "$(BPE_TARBALL)"; \
	fi
	@echo "$(BPE_SHA256)  $(BPE_TARBALL)" | shasum -a 256 -c - || \
	  { echo "checksum mismatch; delete $(BPE_TARBALL) and re-run"; exit 1; }

# The go/no-go measurement, before any compute is committed. ~30 s on one core at the
# default --max-proteins; the figure needs matplotlib, which the Mac env has and the
# pipeline container does not.
hp-bpe-diagnostic: bpe-tokenizer
	$(call require_mac)
	$(PYTHON) bin/hp_bpe_boundary_diagnostic.py \
	  --fasta       $(QFO_DIR)/Eukaryota/UP000005640_9606.fasta \
	  --annotations $(ANNOTATIONS) \
	  --tokenizer   $(BPE_TARBALL) \
	  --out         $(BPE_OUT) \
	  --figure      $(BPE_OUT:.json=.png)

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
	  nextflow run ../main.nf -profile $(PROFILE) -resume $(2) $(BPE_NF_ARG) $(NF_ARGS) \
	    --kmerseek_image $(IMAGE_REMOTE) \
	    --qfo_dir     ../data/qfo \
	    --annotations ../data/annotations \
	    --structures  ../data/structures \
	    --hgnc_file   ../data/covariates/hgnc_complete_set.txt \
	    --omega_file  ../data/covariates/omega_results.tsv \
	    --swissprot_dat ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir      ../data/pfamn          \
	    --prostt5_weights ../data/prostt5/weights \
	    $(PFAM_HMM_ARG) \
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

# ---- GPU vs CPU for the baseline search arms ------------------------------
#
# The paper says kmerseek finds remote homology fast without structure. A reviewer will
# ask whether the structure and profile baselines were given a GPU before that comparison
# was drawn. This target is the measurement that answers it.
#
# One invocation, both modes. --gpu_benchmark true runs the CPU and GPU arms of
# foldseekSearch and mmseqs2Search against the SAME databases in the SAME scheduler
# session, so the per-task comparison needs no cross-run correction, and it drops every
# other arm: phmmer, jackhmmer, hhblits, reseek and folddisco have no GPU path in the
# pinned binaries, so running them again would cost hours and measure nothing. It also
# implies --skip_kmerseek.
#
# Its own launch directory. Nextflow truncates its trace as a run starts, so launching
# this from run/ would destroy the trace `make multiqc` reads.
#
# Runs on the midi query set by default: 964 chr6 proteins against full target proteomes.
# That is real target-side scale, which is what the prefilter cost is driven by, at ~5% of
# the query load. Point it at the full query set with QUERY_DIR=../data/qfo once the shape
# of the answer is known.
GPU_BENCH_DIR     ?= run-gpu-bench
GPU_BENCH_QFO     ?= ../data/midi/qfo
GPU_BENCH_ANNOT   ?= ../data/midi/annotations
GPU_BENCH_STRUCTS ?= ../data/midi/structures
GPU_BENCH_OUTDIR  ?= ../data/gpu-bench/results
# Target databases are shared with the full run: they depend on the target proteome alone.
GPU_BENCH_DBCACHE ?= ../results
GPU_BENCH_TARGETS ?= $(MIDI_TARGETS)
GPU_BENCH_TRACE   ?= $(shell ls -t $(GPU_BENCH_DIR)/*.trace.txt 2>/dev/null | head -1)

gpu-benchmark: prefetch-images midi-testset-sherlock
	$(call require_cluster)
	mkdir -p $(GPU_BENCH_DIR)
	cd $(GPU_BENCH_DIR) && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) -resume \
	    --gpu_benchmark   true \
	    --qfo_dir         $(GPU_BENCH_QFO) \
	    --annotations     $(GPU_BENCH_ANNOT) \
	    --structures      $(GPU_BENCH_STRUCTS) \
	    --outdir          $(GPU_BENCH_OUTDIR) \
	    --db_cache        $(GPU_BENCH_DBCACHE) \
	    --target_species  $(GPU_BENCH_TARGETS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir       ../data/pfamn \
	    $(NF_ARGS)
	@$(MAKE) --no-print-directory gpu-benchmark-report

# Read the trace back into queries/s per tool per mode. Separate from the run so it can be
# re-run against an older trace without re-searching anything:
#   make gpu-benchmark-report GPU_BENCH_TRACE=run-gpu-bench/qfo_pfam_region.<date>.trace.txt
gpu-benchmark-report:
	@if [ -z "$(GPU_BENCH_TRACE)" ]; then \
	  echo "No trace under $(GPU_BENCH_DIR). Run 'make gpu-benchmark' first, or pass one:"; \
	  echo "  make gpu-benchmark-report GPU_BENCH_TRACE=<path>"; \
	  exit 1; \
	fi
	@echo "reporting on trace: $(GPU_BENCH_TRACE)"
	mkdir -p $(GPU_BENCH_OUTDIR)
	apptainer exec \
	  --bind $$PWD:$$PWD --pwd $$PWD \
	  docker://olgabot/kmerseek:$(IMAGE_TAG) \
	  python3 bin/gpu_search_benchmark.py \
	    --trace         $(GPU_BENCH_TRACE) \
	    --queries-fasta $(GPU_BENCH_QFO)/Eukaryota/UP000005640_9606.fasta \
	    --tasks-out     $(GPU_BENCH_OUTDIR)/gpu_benchmark_tasks.parquet \
	    --summary-out   $(GPU_BENCH_OUTDIR)/gpu_benchmark_summary.parquet \
	    --mqc-out       $(GPU_BENCH_OUTDIR)/qfo_gpu_search_mqc.json

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
#
# A variable rather than a literal because multiqc-rannon-burstein has to name the same
# directory to read this run's scored arms back. Written relative to run-midi/, like every
# other path on this target; $(call rel_to_launch,run-midi,...) is what makes it absolute.
MIDI_OUTDIR ?= ../data/midi/results

# polarity4 and funcgroups8 run inside this same command, on the RB image, WITHOUT that
# image touching anything else. --kmerseek_image stays IMAGE_REMOTE, so every python-labelled
# process -- truth building, covariates, scoring, aggregation, all of which take their
# container from that same param -- keeps its cache. --kmerseek_extra_image is consulted only
# by the kmerseek index and search tasks whose alphabet is one of the two, which is a
# per-task container rather than a per-run one.
#
# Both images are prefetched because a run now genuinely uses both.
run-midi: PREFETCH_CONTAINERS := $(RB_IMAGE_REMOTE) $(ALL_CONTAINERS)
run-midi: prefetch-images midi-testset-sherlock
	$(call require_cluster)
	mkdir -p run-midi
	cd run-midi && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) -resume \
	    --kmerseek_image  $(IMAGE_REMOTE) \
	    --kmerseek_extra_encodings true \
	    --kmerseek_extra_image     $(RB_IMAGE_REMOTE) \
	    --qfo_dir         ../data/midi/qfo \
	    --annotations     ../data/midi/annotations \
	    --structures      ../data/midi/structures \
	    --outdir          $(MIDI_OUTDIR) \
	    $(BPE_NF_ARG) \
	    --db_cache        ../results \
	    --target_species  $(MIDI_TARGETS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir       ../data/pfamn \
	    --prostt5_weights ../data/prostt5/weights \
	    $(PFAM_HMM_ARG) \
	    $(NF_ARGS)

# ---- polarity4 and funcgroups8 (Rannon & Burstein 2026) --------------------
#
# The two alphabets kmerseek gained in dd630a8, on the same midi query set as run-midi so
# their numbers are comparable to everything that run has already scored. Four things keep
# this off the main sweep's toes, and all four matter:
#
#   --kmerseek_image $(RB_IMAGE_REMOTE)   the only image with the alphabets. Kept out of
#       IMAGE_TAG so the running sweep's 2507 indexes and 1700 searches do not rehash.
#   RB_OUTDIR, its own results directory   scoreDomainCalls publishes per-arm parquets and
#       aggregateMetrics publishes all_domain_metrics.parquet. Sharing run-midi's outdir
#       would have this run's aggregate, which covers two alphabets and no baselines,
#       replace the one describing the whole sweep. The combined view comes from
#       `multiqc-rannon-burstein` below, which reads both directories.
#   RB_RUN_DIR, its own launch directory   Nextflow truncates its trace as a run starts, so
#       launching from run-midi/ would destroy the trace `make multiqc` reads. Same reason
#       gpu-benchmark has run-gpu-bench/.
#   --skip_baselines true   the baselines are a property of the query set and the target
#       proteomes, not of the alphabet. run-midi has already scored them; re-running
#       phmmer, jackhmmer, mmseqs2, foldseek, reseek, folddisco and ProstT5 over nine
#       proteomes would cost days and produce the same numbers.
#
# --run_hmmscan false for the same reason, and for one more. The Pfam annotation ceiling is
# a property of Pfam-A, so this run would spend hours on a whole-proteome hmmscan to
# reproduce run-midi's number exactly. It also defaults to ON whenever the Pfam HMM file is
# present, which it is on Sherlock, so it has to be turned off explicitly. And
# scoreHmmscanCeiling writes a FIXED filename, hmmscan.direct.all.metrics.parquet, into
# <outdir>/metrics -- two runs producing it means the same arm in both directories, which
# the combined report refuses rather than counting twice.
#
# --db_cache stays on ../results. The kmerseek index store is keyed by
# <species>.<label>.k<ksize>.lc<lowcomp>, and polarity4/funcgroups8 are new labels, so
# these runs add entries beside the existing ones rather than colliding with them.
#
# 2 alphabets x 10 ksizes x 2 low-complexity settings x 9 targets = 360 searches, plus the
# same number of index builds. The ksize ranges are in main.nf's EXTRA_ENCODINGS, derived
# from measured bits/symbol; --kmerseek_encodings names the alphabets and picks them up.
RB_RUN_DIR ?= run-rannon-burstein
RB_OUTDIR  ?= ../data/midi/results-rannon-burstein
RB_ENCODINGS ?= polarity4,funcgroups8

## One command: search the two Rannon & Burstein alphabets, then build the combined report.
##
## Sequential on purpose, not parallel. multiqc-rannon-burstein re-aggregates the per-arm
## parquets from BOTH outdirs, so it has to see this run's scoring finished; and make -j
## on these two would race the second against the first's still-writing metrics directory.
##
## The two halves stay separately runnable. This wrapper fails at the first, so a search
## that dies partway does not go on to build a report over half its arms -- and after
## fixing whatever failed, `make run-rannon-burstein` resumes from storeDir and
## `make multiqc-rannon-burstein` finishes the job without repeating the search.
##
## run-midi must already have completed. Its metrics are the other half of the combined
## report -- every baseline and all 17 existing alphabets -- and this run deliberately does
## not recompute them.
rannon-burstein-all:
	$(call require_cluster)
	@test -f $(MIDI_OUTDIR)/all_domain_metrics.parquet \
	  || { echo; echo "  run-midi has not produced $(MIDI_OUTDIR)/all_domain_metrics.parquet."; \
	       echo "  The combined report needs it for the baselines and the other 17 alphabets."; \
	       echo "  Run 'make run-midi' first, or pass MIDI_OUTDIR=<other outdir>."; echo; exit 1; }
	$(MAKE) run-rannon-burstein
	$(MAKE) multiqc-rannon-burstein
	@echo
	@echo "  Combined report written under $(RB_COMBINED_OUTDIR)."
	@echo "  Pull it with:  make pull-report REMOTE_RESULTS=$(RB_COMBINED_OUTDIR)"

run-rannon-burstein: PREFETCH_CONTAINERS := $(RB_IMAGE_REMOTE) $(ALL_CONTAINERS)
run-rannon-burstein: prefetch-images midi-testset-sherlock
	$(call require_cluster)
	mkdir -p $(RB_RUN_DIR)
	cd $(RB_RUN_DIR) && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE) -resume \
	    --kmerseek_image      $(RB_IMAGE_REMOTE) \
	    --kmerseek_encodings  $(RB_ENCODINGS) \
	    --skip_baselines      true \
	    --run_hmmscan         false \
	    --qfo_dir             ../data/midi/qfo \
	    --annotations         ../data/midi/annotations \
	    --structures          ../data/midi/structures \
	    --outdir              $(RB_OUTDIR) \
	    --db_cache            ../results \
	    --target_species      $(MIDI_TARGETS) \
	    --hgnc_file           ../data/covariates/hgnc_complete_set.txt \
	    --omega_file          ../data/covariates/omega_results.tsv \
	    --swissprot_dat       ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir           ../data/pfamn \
	    $(PFAM_HMM_ARG) \
	    $(NF_ARGS)

# Same code path as the real run, scoped down. Do this before `make run`.
run-mini: prefetch-images mini-testset-sherlock
	$(call require_cluster)
	mkdir -p run-mini
	cd run-mini && \
	  ml load nextflow && \
	  nextflow run ../main.nf -profile $(PROFILE),mini -resume \
	    --kmerseek_image  $(IMAGE_REMOTE) \
	    --qfo_dir         ../data/mini/qfo \
	    --annotations     ../data/mini/annotations \
	    --structures      ../data/mini/structures \
	    --outdir          ../data/mini/results \
    $(BPE_NF_ARG) \
	    --target_species  $(MINI_TARGETS) \
	    --kmerseek_combos $(MINI_COMBOS) \
	    --hgnc_file       ../data/covariates/hgnc_complete_set.txt \
	    --omega_file      ../data/covariates/omega_results.tsv \
	    --swissprot_dat   ../data/uniprot_sprot.dat.gz \
	    --pfamn_dir        ../data/pfamn          \
	    --prostt5_weights ../data/prostt5/weights \
	    $(PFAM_HMM_ARG) \
	    $(NF_ARGS)

# Rebuild the one HTML report from results already published, without re-running a single
# search. This is the normal way to get it after a long sweep: the trace is only complete
# once the run has ended.
#
# Launched from its own directory on purpose. Nextflow truncates its trace file and
# overwrites its execution report and timeline as a run starts, so a report run launched in
# run/ would destroy the very numbers it is reading. main.nf refuses outright in that case
# rather than quietly emitting empty resource sections.
#
# multiqc-preflight comes FIRST in the prerequisite list, ahead of prefetch-images. Make
# builds prerequisites left to right, so the "nothing to report on" check happens before
# any srun image pull rather than after several minutes of them. Same check the local
# target does inline; it needs a target of its own here only to get ahead of that pull.
multiqc-preflight: GUARD_NAME := multiqc
multiqc-preflight:
	$(call require_cluster)
	@if [ -z "$(TRACE)" ]; then \
	  echo "No trace file under run/. Pass one explicitly:"; \
	  echo "  make multiqc TRACE=run/qfo_pfam_region.<date>.trace.txt"; \
	  exit 1; \
	fi
	$(call require_metrics,$(MULTIQC_OUTDIR_ABS),run,run,MULTIQC_OUTDIR,multiqc)

#
# This one needs --kmerseek_image too, which is not obvious: multiqcReport pins its own
# multiqc container, but buildMultiqcInputs carries `label 'python'`, and the sherlock
# profile binds that label to params.kmerseek_image. The section-building step is the part
# that reads the parquets.
multiqc: multiqc-preflight prefetch-images
	$(call require_cluster)
	@echo "reporting on trace: $(TRACE)"
	mkdir -p run-report
	cd run-report && \
	  ml load nextflow && \
	  nextflow run ../main.nf -entry report -profile $(PROFILE) \
	    --kmerseek_image $(IMAGE_REMOTE) \
	    --outdir     $(MULTIQC_OUTDIR) \
	    --report_trace $(abspath $(TRACE)) \
	    --qfo_dir    ../data/qfo \
	    $(NF_ARGS)

# ---- one report over run-midi AND run-rannon-burstein ---------------------
#
# The point of running the new alphabets at all is seeing them next to the baselines and
# the seventeen alphabets run-midi already scored, and neither run's
# all_domain_metrics.parquet has both. --report_outdirs re-aggregates the per-arm parquets
# of both -- see the note on that param in main.nf for why the alternatives do not work --
# and publishes the combined aggregate plus the HTML into RB_COMBINED_OUTDIR, which is a
# THIRD directory so neither source run's own aggregate is touched.
#
# The trace is the one thing that cannot simply be unioned by Nextflow, because the report
# reads a single file. Concatenating them here is safe: both runs write the field list from
# nextflow.config's `trace` block, so the headers are identical and only the second file's
# needs dropping. Without this the resource sections would describe one of the two runs
# while the accuracy sections described both, which is the kind of mismatch nobody notices
# in an HTML report.
#
# The concatenated file lands in RB_RUN_DIR rather than run/ or run-midi/: those are launch
# directories with live trace observers, and a report run pointed at a file inside one is
# how the trace-truncation bug happened in the first place.
RB_COMBINED_OUTDIR ?= ../data/midi/results-combined
RB_TRACE           ?= $(shell ls -t $(RB_RUN_DIR)/*.trace.txt 2>/dev/null | head -1)
MIDI_TRACE         ?= $(shell ls -t run-midi/*.trace.txt 2>/dev/null | head -1)
RB_COMBINED_TRACE  := $(RB_RUN_DIR)/combined.trace.txt

multiqc-rannon-burstein: PREFETCH_CONTAINERS := $(RB_IMAGE_REMOTE) $(ALL_CONTAINERS)
multiqc-rannon-burstein: prefetch-images
	$(call require_cluster)
	@for t in "$(MIDI_TRACE)" "$(RB_TRACE)"; do \
	  if [ -z "$$t" ] || [ ! -f "$$t" ]; then \
	    echo ""; \
	    echo "  Missing a trace file."; \
	    echo "    run-midi:            $(if $(MIDI_TRACE),$(MIDI_TRACE),NONE under run-midi/)"; \
	    echo "    run-rannon-burstein: $(if $(RB_TRACE),$(RB_TRACE),NONE under $(RB_RUN_DIR)/)"; \
	    echo ""; \
	    echo "  Both runs have to have finished. Name them explicitly if the newest file"; \
	    echo "  under either launch directory is not the one you want:"; \
	    echo ""; \
	    echo "    make multiqc-rannon-burstein MIDI_TRACE=<path> RB_TRACE=<path>"; \
	    echo ""; \
	    exit 1; \
	  fi; \
	done
	@echo "combining traces: $(MIDI_TRACE) + $(RB_TRACE)"
	@cp -f $(MIDI_TRACE) $(RB_COMBINED_TRACE)
	@tail -n +2 $(RB_TRACE) >> $(RB_COMBINED_TRACE)
	mkdir -p run-report-rannon-burstein
	cd run-report-rannon-burstein && \
	  ml load nextflow && \
	  nextflow run ../main.nf -entry report -profile $(PROFILE) \
	    --kmerseek_image  $(RB_IMAGE_REMOTE) \
	    --report_outdirs  $(call rel_to_launch,run-midi,$(MIDI_OUTDIR)),$(call rel_to_launch,$(RB_RUN_DIR),$(RB_OUTDIR)) \
	    --outdir          $(RB_COMBINED_OUTDIR) \
	    --report_trace    $(abspath $(RB_COMBINED_TRACE)) \
	    --qfo_dir         ../data/midi/qfo \
	    $(NF_ARGS)

status:
	$(call require_cluster)
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}

## What each running job actually IS: process AND tag, in separate columns.
##
## Read from `### name:` on line 3 of the task's own .command.run, which Nextflow writes
## and which carries both. Nothing here is inferred. The first version of this target
## guessed the tag out of .command.sh with regexes, which gave `human_vs_ecoli` for six
## different tools and a bare work-directory hash for every process whose script matched
## neither pattern -- reseekConvert, scoreDomainCalls and the truth builders among them.
##
## The SLURM job name cannot be used for this: it truncates to `nf-kmers` for every
## kmerseek task whatever its alphabet, which is the whole reason `status` is not enough.
##
## Read TIME carefully. squeue formats it [DD-[HH:]]MM:SS, so `4:03` is four minutes and
## `1-02:19:10` is a day. That distinction has already cost a wrong call about whether a
## job was stuck. PENDING jobs show 0:00.
running:
	$(call require_cluster)
	@printf "%-10s %-4s %-10s %-24s %s\n" JOBID ST TIME PROCESS TAG
	@squeue -u $$USER -h --account=$${SLURM_ACCOUNT:-ayeletv} -o "%i %t %M" | while read j st tm; do \
	   w=$$(scontrol show job $$j 2>/dev/null | tr ' ' '\n' | grep -m1 WorkDir | cut -d= -f2); \
	   n=$$(sed -n "s/^### name: '\(.*\)'.*/\1/p" "$$w/.command.run" 2>/dev/null | head -1); \
	   if [ -z "$$n" ]; then n="(no .command.run under $$(basename "$$w" 2>/dev/null))"; fi; \
	   pr=$$(printf '%s' "$$n" | sed 's/ (.*//'); \
	   tg=$$(printf '%s' "$$n" | sed -n 's/.*(\(.*\))$$/\1/p'); \
	   printf "%-10s %-4s %-10s %-24s %s\n" "$$j" "$$st" "$$tm" "$$pr" "$$tg"; \
	 done; true

## Cancel every job on this account, then show what is left.
##
## This is part of restarting, not housekeeping. Nextflow does not reliably stop its jobs
## when it dies: a failed run prints "Finishing pending tasks before exit" and keeps
## unstaging for minutes, and an interrupted one leaves orphans that no later -resume will
## ever collect. Those orphans are exactly what produces storeDir "Directory not empty"
## collisions against the next run, because two writers reach the same store entry. Wait
## for this to come back empty before relaunching.
kill-jobs:
	$(call require_cluster)
	@squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
	@read -p "cancel ALL of the above? [y/N] " a; [ "$$a" = y ] || exit 1
	scancel -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
	@sleep 2; squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}

# Guarded like every other cluster target. Without it, $SCRATCH is empty on the Mac and
# this expands to `rm -rf /apptainer-cache` -- harmless only because that path happens not
# to exist.
clean-cache:
	$(call require_cluster)
	rm -rf $$SCRATCH/apptainer-cache

## Drop the cached kmerseek target indexes. They exist so run-midi and the full run share
## one set of index builds -- roughly 3294 RocksDB directories at a 2.1 GB median write,
## which is multiple TB on $SCRATCH. Run this only when the LAST run over these target
## proteomes has finished; anything still to come re-indexes from scratch.
clean-indexes:
	$(call require_cluster)
	@du -sh $(REMOTE_INDEX_DIR) 2>/dev/null || echo "no index cache at $(REMOTE_INDEX_DIR)"
	@read -p "delete the kmerseek index cache above? [y/N] " a; [ "$$a" = y ] || exit 1
	/bin/rm -rf $(REMOTE_INDEX_DIR)

REMOTE_FOLDDISCO_DIR ?= results/folddisco_index

## Drop the cached folddisco indexes. Needed once, after the fix that makes folddiscoIndex
## record ABSOLUTE structure paths: `folddisco query` reopens the target CIF files that the
## index points at, and every index built before that fix points at a relative path that
## resolves to nothing outside the task that built it. storeDir skips a rebuild whenever the
## output directory already exists and does NOT notice that the script changed, so the stale
## indexes are picked up and every query panics on "Failed to read CIF file" exactly as
## before. Deleting them is what makes the fix take effect.
clean-folddisco-index:
	$(call require_cluster)
	@du -sh $(REMOTE_FOLDDISCO_DIR) 2>/dev/null || echo "no folddisco index at $(REMOTE_FOLDDISCO_DIR)"
	@read -p "delete the folddisco indexes above? they rebuild on the next run [y/N] " a; [ "$$a" = y ] || exit 1
	/bin/rm -rf $(REMOTE_FOLDDISCO_DIR)

## How much disk the shared cache is holding, and how much is left. Check this before
## launching the full sweep. Everything under results/ is shared between run-midi and run:
## the target databases are identical for both, and the human entries are kept apart by a
## query-set digest in their names (human-<md5>_mmdb and friends).
index-disk:
	$(call require_cluster)
	@for d in kmerseek_index hhblits_db foldseek_db mmseqs_db prostt5_db reseek_db \
	          folddisco_index prostt5_weights; do \
	    printf '%-18s ' "$$d"; (du -sh results/$$d 2>/dev/null | cut -f1) || echo "-"; \
	  done
	@echo "--- orphaned pre-digest human entries (safe to delete) ---"
	@find results -maxdepth 2 \( -name 'human_*' -o -name 'human.bca' \) 2>/dev/null || true
	@echo "--- quota ---"; lfs quota -h -u $$USER $$SCRATCH 2>/dev/null || df -h $$SCRATCH
