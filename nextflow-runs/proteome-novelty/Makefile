# proteome-novelty: notebook 345. Recurring-word discovery curves for six clades of the
# QfO 2020_04 panel, a Markov-chain null, MMseqs2 families, and the held-out prediction test.
#
#   make test          (laptop) unit tests of bin/novelty.py against brute force
#   make stub          (laptop) the whole DAG with -stub-run, no containers, seconds
#   make run           (Sherlock) submit the Nextflow head as a SLURM job in tmux
#   make watch         (Sherlock) attach read-only to the head's live task table
#   make stop          (Sherlock) ask the head to shut down cleanly
#   make running       (Sherlock) what is in the queue
#   make pull-results  (laptop) copy the six collected tables to tables/345/

SHELL := /bin/bash

NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
NEXTFLOW_JVM := $(patsubst %/bin/nextflow,%/lib/jvm,$(NEXTFLOW))
ifneq ($(wildcard $(NEXTFLOW_JVM)/bin/java),)
export JAVA_HOME := $(NEXTFLOW_JVM)
endif
PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# Sherlock copy of the QfO 2020_04 release (the region benchmark's). checkFasta stops the
# run if any FASTA differs from assets/proteomes.tsv.
QFO_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/qfo
# Laptop copy, for `make stub` only (stub tasks read nothing; the files must exist).
LAPTOP_QFO_DIR ?= /Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143

# A clone of its own, so this branch never moves the main Sherlock checkout under a run.
REMOTE_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis-nb345/nextflow-runs/proteome-novelty

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_TIME    ?= 2-00:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= nb345-novelty
HEAD_JOBNAME ?= nf-nb345-novelty
export NXF_OPTS := -Xms512m -Xmx6g

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head

.PHONY: test stub run watch stop pull-results run-sweep stub-sweep pull-sweep

test:
	$(PYTHON) -m pytest -q $(MAKEFILE_DIR)/tests

stub:
	mkdir -p run-stub
	cd run-stub && $(NEXTFLOW) run ../main.nf -stub-run -profile stub \
	  --qfo_dir $(LAPTOP_QFO_DIR) --outdir results

run:
	mkdir -p run
	cd run && if command -v ml >/dev/null 2>&1; then ml load biology nextflow; fi; \
	  $(HEAD_SUBMIT) nextflow run ../main.nf -profile sherlock -resume --qfo_dir $(QFO_DIR)

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

pull-results:
	mkdir -p $(MAKEFILE_DIR)/../../tables/345
	rsync -av sherlock:$(REMOTE_DIR)/run/results/ $(MAKEFILE_DIR)/../../tables/345/

# --- Every alphabet and k of the region benchmark's sweep (notebook 345, section 8) -------
## Same launch folder as `run`, with -resume, so its 325 finished tasks are reused. Real
## proteomes are counted at all 182 settings in assets/sweep_settings.txt; the synthetic
## copies stay at the six pre-registered settings. Results go to run/results-sweep, so the
## pre-registered tables in run/results are never overwritten.
SWEEP_SETTINGS := $(shell grep -E '^[a-z]' $(MAKEFILE_DIR)/assets/sweep_settings.txt | paste -sd, -)
PREREG_SETTINGS := hp_pbotc_1st_ed:21,hp_pbotc_1st_ed:30,hp_pbotc_1st_ed:42,protein:5,protein:7,protein:10

run-sweep:
	mkdir -p run
	cd run && if command -v ml >/dev/null 2>&1; then ml load biology nextflow; fi; \
	  $(HEAD_SUBMIT) nextflow run ../main.nf -profile sherlock -resume --qfo_dir $(QFO_DIR) \
	    --settings '$(SWEEP_SETTINGS)' --null_settings '$(PREREG_SETTINGS)' --outdir results-sweep

stub-sweep:
	mkdir -p run-stub
	cd run-stub && $(NEXTFLOW) run ../main.nf -stub-run -profile stub \
	  --qfo_dir $(LAPTOP_QFO_DIR) --outdir results-sweep \
	  --settings '$(SWEEP_SETTINGS)' --null_settings '$(PREREG_SETTINGS)'

pull-sweep:
	mkdir -p $(MAKEFILE_DIR)/../../tables/345_sweep
	rsync -av --exclude parts/ sherlock:$(REMOTE_DIR)/run/results-sweep/ $(MAKEFILE_DIR)/../../tables/345_sweep/

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, from any pipeline, then one row per job
## of the runs under this folder. scripts/nf-running does the work. Runs on Sherlock only,
## since it reads the queue. Read TIME as squeue's [DD-[HH:]]MM:SS: `4:03` is four minutes
## and `1-02:19:10` is a day. PENDING jobs show 0:00.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

.PHONY: running
running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis-nb345/nextflow-runs/proteome-novelty"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
