# nf-core/rnaseq (STAR + RSEM, April 2024 genome) and nf-core/differentialabundance (DESeq2) on the
# 47 PEC samples of notebook 361, as the reference for its k-mer routes. Runs on Sherlock.
#
#   make run       the Nextflow head job, rnaseq then differentialabundance
#   make watch     attach to the live table (detach: Ctrl-b d)
#   make stop      ask the run to shut down cleanly
#   make running   every Nextflow run in the queue

SHELL := /bin/bash
# Both pipelines need Nextflow >= 25.10.4; Sherlock's newest module is older.
NEXTFLOW := $(SCRATCH)/bin/nextflow-25.10.4-dist
export NXF_OPTS := -Xms512m -Xmx4g
PEC_ROOT ?= $(SCRATCH)/nb361-pec
OUT ?= $(PEC_ROOT)/nfcore
HERE := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_JOBNAME ?= nf-pec361-nfcore
HEAD_SESSION ?= pec361nfcore

.PHONY: run watch stop running

run:
	mkdir -p $(OUT)/design
	$(PEC_ROOT)/env/bin/python $(HERE)/make_design.py $(HERE)/../../analysis/pec-domen2026-splash/design $(PEC_ROOT)/trimmed $(OUT)/design
	cd $(OUT) && ml load biology nextflow && NF_HEAD_DETACH=1 HEAD_ACCOUNT=ayeletv HEAD_PARTITION=normal,hns HEAD_QOS=normal \
	  HEAD_TIME=3-00:00:00 HEAD_CPUS=2 HEAD_MEM=8G HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
	  $(HERE)/../../scripts/nf-head $(HERE)/run_pipelines.sh $(HERE) $(OUT) $(NEXTFLOW) $(PEC_ROOT)/env/bin/python

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

running:
	@$(SCRATCH)/nf-running
