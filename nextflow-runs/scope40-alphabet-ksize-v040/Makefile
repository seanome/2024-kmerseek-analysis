# SCOPe40 all-vs-all with kmerseek 0.4.0: 19 alphabets x their k_min..k_max (notebook 274)
# x six settings (exact, and extension at C = c_min, 1.1 c_min, c_opt, c_max, 2), scored by
# every output column as family, superfamily and fold sensitivity AUC.
#
#   make settings             rewrite assets/settings_per_alphabet.tsv from its sources
#   make plan                 print the pairs and searches, submit nothing
#   make measure-resources    write <launch dir>/resources_measured.tsv from every trace here
#   make stub                 wiring check, no data, no container
#   make run-sbatch           on Sherlock: the whole run, head job in tmux on a compute node
#   make watch / make stop    attach to the live table / stop the run cleanly
SHELL := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_DIR     := $(abspath $(MAKEFILE_DIR)/../..)

MAC_PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
PYTHON ?= $(if $(wildcard $(MAC_PYTHON)),$(MAC_PYTHON),python3)
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
ifneq ($(wildcard $(MAC_NEXTFLOW)),)
NEXTFLOW ?= NXF_JAVA_HOME=/Users/olga/anaconda3/envs/nf-core-v2/lib/jvm $(MAC_NEXTFLOW)
PROFILE  ?= standard
else
NEXTFLOW ?= nextflow
PROFILE  ?= sherlock
endif

SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
SCOPE_DIR   ?= $(SCRATCH_DIR)/scope
FASTA       ?= $(SCOPE_DIR)/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
FASTA_URL   := https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
# sha256 of the copy every earlier SCOPe40 notebook used (~/data/scope on the laptop).
FASTA_SHA256 := e6d3213b38225f617966686809fa5531f0de6a7d00d01692477476a1493ab037
OUTDIR      ?= $(SCOPE_DIR)/results-scope40-alphabet-ksize-v040
LAUNCH_DIR  ?= $(MAKEFILE_DIR)/runs/main
NF_ARGS     ?=

HEAD_ACCOUNT   ?= ayeletv
HEAD_PARTITION ?= hns
HEAD_QOS       ?= normal
HEAD_CPUS      ?= 2
HEAD_MEM       ?= 8G
HEAD_XMX       ?= 6g
HEAD_TIME      ?= 2-00:00:00
HEAD_SESSION   ?= scope40v040
HEAD_JOBNAME   ?= nfdrv-scope40-v040
export NXF_OPTS := -Xms1g -Xmx$(HEAD_XMX)

# -resume, single dash. Never piped through tee: the live task table is the point.
# The first asks the launch uses, measured from every trace. Written into the launch
# directory, not over the committed assets/resources_measured.tsv: on 2026-10-03 the
# rewritten copy in the Sherlock clone blocked `git pull`.
MEASURED    = $(LAUNCH_DIR)/resources_measured.tsv
NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
         --fasta $(FASTA) --outdir $(OUTDIR) --resources_measured $(MEASURED) $(NF_ARGS)

.PHONY: help settings plan fasta measure-resources stub run run-sbatch watch stop

help:
	@sed -n '1,12p' $(MAKEFILE_DIR)/Makefile

settings:
	$(PYTHON) $(REPO_DIR)/scripts/scope40_v040_settings.py

plan:
	$(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile stub -preview --fasta $(FASTA) $(NF_ARGS) 2>&1 \
	  | grep -E 'SCOPe40|ERROR|error'

# The FASTA, checked against the sha256 of the copy every earlier SCOPe40 notebook used.
fasta: $(FASTA)
$(FASTA):
	mkdir -p $(dir $@)
	curl -fL -C - --retry 5 --speed-limit 1000 --speed-time 60 -o $@.part $(FASTA_URL)
	echo "$(FASTA_SHA256)  $@.part" | { if command -v sha256sum >/dev/null; then sha256sum -c -; else shasum -a 256 -c -; fi; } && mv $@.part $@

# First asks for every task: the measured peak and run time of each process at each pair,
# from every trace this pipeline has written (laptop and Sherlock). run-sbatch runs this
# first, so a relaunch never repeats an ask an earlier run already outgrew.
measure-resources:
	@traces=$$(ls $(MAKEFILE_DIR)/runs/*/scope40_v040.*.trace.txt $(MAKEFILE_DIR)/assets/traces/*.trace.txt 2>/dev/null); \
	 test -n "$$traces" || { echo "no trace files found"; exit 1; }; \
	 mkdir -p $(LAUNCH_DIR); \
	 $(PYTHON) $(MAKEFILE_DIR)/tools/measure_resources.py $(MEASURED) $$traces

stub:
	mkdir -p $(MAKEFILE_DIR)/runs/stub
	cd $(MAKEFILE_DIR)/runs/stub && $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile stub -stub-run \
	  --outdir $(MAKEFILE_DIR)/runs/stub/out $(NF_ARGS)

run: $(FASTA) measure-resources
	mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && $(NF_RUN)

run-sbatch: $(FASTA) measure-resources
	@if squeue -u $$USER -h -o '%j' 2>/dev/null | grep -qx '$(HEAD_JOBNAME)'; then \
	   echo "$(HEAD_JOBNAME) is already queued or running; make watch, or make stop"; exit 1; fi
	mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && HEAD_ACCOUNT=$(HEAD_ACCOUNT) HEAD_PARTITION=$(HEAD_PARTITION) \
	  HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
	  HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
	  HEAD_HINT_WATCH="make watch" HEAD_HINT_STOP="make stop" \
	  $(MAKEFILE_DIR)/tools/nf-head $(NF_RUN)

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 test -n "$$jid" || { echo "no running driver named $(HEAD_JOBNAME)"; squeue -u $$USER -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

# TERM to the batch script only: tools/nf-head turns it into Ctrl-C, and Nextflow then
# cancels its own tasks. Finished searches stay cached; run-sbatch resumes.
stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | head -1); \
	 test -n "$$jid" || { echo "no driver named $(HEAD_JOBNAME)"; exit 0; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked driver $$jid to stop"
