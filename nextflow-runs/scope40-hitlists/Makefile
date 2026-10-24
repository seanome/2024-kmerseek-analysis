# SCOPe40 all-against-all with kept hit lists, for the leaderboard's SCOPe40 page.
#
#   make test                 score_hitlist.py against Foldseek's bench.noselfhit.awk (laptop is fine)
#   make lint                 nextflow lint, no run
#   make pdbstyle             on Sherlock: fetch the SCOPe 2.08 40% domain structures (login node)
#   make score-kmerseek       on Sherlock: list and score the existing kmerseek pair tables
#   make run-sbatch           on Sherlock: the searches, head job in tmux on a compute node
#   make watch / make stop    attach to the live table / stop the run cleanly
# Never `make run-sbatch` on the laptop: the searches are Sherlock-sized.
SHELL := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

NEXTFLOW ?= nextflow
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
LINT_NF := $(if $(wildcard $(MAC_NEXTFLOW)),NXF_JAVA_HOME=/Users/olga/anaconda3/envs/nf-core-v2/lib/jvm $(MAC_NEXTFLOW),$(NEXTFLOW))
MAC_PYTHON := /Users/olga/code/protein-function-annotation-leaderboard/.venv/bin/python
PYTHON ?= $(if $(wildcard $(MAC_PYTHON)),$(MAC_PYTHON),python3)

SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
SCOPE_DIR   ?= $(SCRATCH_DIR)/scope
FASTA       ?= $(SCOPE_DIR)/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
PDBSTYLE    ?= $(SCOPE_DIR)/pdbstyle-2.08
PDBSTYLE_URL := https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz
OUTDIR      ?= $(SCOPE_DIR)/results-scope40-hitlists
LAUNCH_DIR  ?= $(MAKEFILE_DIR)/runs/main
NF_ARGS     ?=

HEAD_ACCOUNT   ?= ayeletv
HEAD_PARTITION ?= hns
HEAD_QOS       ?= normal
HEAD_CPUS      ?= 2
HEAD_MEM       ?= 8G
HEAD_XMX       ?= 6g
HEAD_TIME      ?= 1-00:00:00
HEAD_SESSION   ?= scope40hits
HEAD_JOBNAME   ?= nfdrv-scope40-hits
export NXF_OPTS := -Xms1g -Xmx$(HEAD_XMX)

# -resume, single dash. Never piped through tee.
NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile sherlock -resume \
         --fasta $(FASTA) --pdbstyle $(PDBSTYLE) --outdir $(OUTDIR) $(NF_ARGS)

.PHONY: test lint pdbstyle score-kmerseek run-sbatch watch stop

test:
	$(PYTHON) $(MAKEFILE_DIR)/tests/test_score_hitlist.py

lint:
	$(LINT_NF) lint $(MAKEFILE_DIR)/main.nf $(MAKEFILE_DIR)/nextflow.config

# 15,177 .ent files; the tarball's top folder is dropped so files land in 2-letter folders.
pdbstyle:
	@test -n "$$SCRATCH" || { echo "run this on Sherlock"; exit 1; }
	mkdir -p $(PDBSTYLE)
	curl -sfL $(PDBSTYLE_URL) | tar -xzf - --strip-components=1 -C $(PDBSTYLE)
	@n=$$(find $(PDBSTYLE) -name '*.ent' | wc -l); echo "$$n structures in $(PDBSTYLE)"; \
	 test "$$n" -eq 15177 || { echo "expected 15177"; exit 1; }

score-kmerseek:
	@test -n "$$SCRATCH" || { echo "run this on Sherlock"; exit 1; }
	bash $(MAKEFILE_DIR)/score_existing_kmerseek.sbatch
	n=$$(wc -l < $(OUTDIR)/kmerseek/tables.txt); \
	 cd $(MAKEFILE_DIR)/../.. && sbatch --array=0-$$((n - 1)) nextflow-runs/scope40-hitlists/score_existing_kmerseek.sbatch

run-sbatch:
	@test -n "$$SCRATCH" || { echo "run this on Sherlock"; exit 1; }
	@test -d $(PDBSTYLE) || { echo "no $(PDBSTYLE): make pdbstyle first"; exit 1; }
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
	 test -n "$$jid" || { echo "no running driver named $(HEAD_JOBNAME)"; exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | head -1); \
	 test -n "$$jid" || { echo "no driver named $(HEAD_JOBNAME)"; exit 0; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked driver $$jid to stop"
