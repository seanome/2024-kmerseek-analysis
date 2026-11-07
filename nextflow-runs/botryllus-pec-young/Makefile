# Young Botryllus colonies, PEC-treated vs control, per time window, two ways:
# nf-core/rnaseq + nf-core/differentialabundance against the April 2024 genome, and SPLASH
# without a genome. Run on Sherlock from this directory.
#
#   make run     submit everything (splash_prep -> 6 SPLASH jobs; Nextflow head job)
#   make watch   attach to the running head's live task table (detach: Ctrl-b d)
#   make stop    ask the running head to shut down cleanly
#   make run-alone    refit with each comparison on its own libraries, on the counts above
#   make watch-alone, make stop-alone    the same two commands for that head

SHELL := /bin/bash

# Both pipelines require Nextflow >= 25.10.4. Sherlock's newest module is 25.04.7, so the
# 25.10.4 self-contained build is fetched from the release page; the module is still loaded
# for its Java. Same binary as nextflow-runs/botryllus-rnaseq-preprocess.
NXF_RELEASE := 25.10.4
NEXTFLOW    := $(SCRATCH)/bin/nextflow-$(NXF_RELEASE)-dist
export NXF_OPTS := -Xms512m -Xmx4g

OUT ?= $(SCRATCH)/botryllus-pec-young

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_TIME    ?= 2-00:00:00
HEAD_SESSION ?= botryllus-pec-young
HEAD_JOBNAME ?= nf-botryllus-pec-young

ALONE_SESSION := botryllus-pec-alone
ALONE_JOBNAME := nf-botryllus-pec-alone
ALONE_TIME    ?= 12:00:00

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

.PHONY: run watch stop run-alone watch-alone stop-alone

$(NEXTFLOW):
	mkdir -p $(dir $@)
	curl -fsSL -o $@.part https://github.com/nextflow-io/nextflow/releases/download/v$(NXF_RELEASE)/nextflow-$(NXF_RELEASE)-dist
	chmod +x $@.part && mv $@.part $@

run: $(NEXTFLOW)
	ml load biology nextflow && \
	  REPO=$(MAKEFILE_DIR) OUT=$(OUT) NEXTFLOW=$(NEXTFLOW) \
	  HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) HEAD_QOS=$(HEAD_QOS) \
	  HEAD_TIME=$(HEAD_TIME) HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
	  HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
	  $(MAKEFILE_DIR)/submit_all.sh

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

# Refuses to submit a second copy while one is queued or running, as submit_all.sh does.
run-alone: $(NEXTFLOW)
	@queued=$$(squeue -u $$USER -h -n $(ALONE_JOBNAME) -o "%i %T"); \
	 [ -z "$$queued" ] || { echo "already queued or running: $$queued"; exit 1; }
	mkdir -p $(OUT)/each-alone-head
	ml load biology nextflow && cd $(OUT)/each-alone-head && \
	  NF_HEAD_DETACH=1 HEAD_HINT_WATCH="make watch-alone" HEAD_HINT_STOP="make stop-alone" \
	  HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) HEAD_QOS=$(HEAD_QOS) \
	  HEAD_TIME=$(ALONE_TIME) HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
	  HEAD_SESSION=$(ALONE_SESSION) HEAD_JOBNAME=$(ALONE_JOBNAME) \
	  $(MAKEFILE_DIR)/../../scripts/nf-head $(MAKEFILE_DIR)/run_each_alone.sh $(MAKEFILE_DIR) $(OUT) $(NEXTFLOW)

watch-alone:
	@$(MAKE) --no-print-directory watch HEAD_JOBNAME=$(ALONE_JOBNAME) HEAD_SESSION=$(ALONE_SESSION)

stop-alone:
	@$(MAKE) --no-print-directory stop HEAD_JOBNAME=$(ALONE_JOBNAME)
