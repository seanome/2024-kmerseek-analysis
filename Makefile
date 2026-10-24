# Targets that are about the whole repo, not one pipeline. Each pipeline has its own
# Makefile under nextflow-runs/<pipeline>/.

SHERLOCK_HOST ?= sherlock
ON_CLUSTER := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

.PHONY: help running leaderboard-tables leaderboard-fetch

help:
	@echo "Sherlock:"
	@echo "  running    Every Nextflow run in the SLURM queue, from any pipeline: clone,"
	@echo "             run folder, head job, tasks by process and state, and jobs that"
	@echo "             are not part of a run"
	@echo ""
	@echo "Leaderboard:"
	@echo "  leaderboard-fetch   Copy the region benchmark's traces and query summary from"
	@echo "                      Sherlock into LEADERBOARD_DATA"
	@echo "  leaderboard-tables  Write results/leaderboard/ (results.parquet, compute.parquet,"
	@echo "                      coverage.md, provenance.json) from LEADERBOARD_DATA"
	@echo ""
	@echo "Each pipeline's own targets: make -C nextflow-runs/<pipeline> help"

## Every Nextflow run in the queue, whichever pipeline or clone it came from.
##
## `make running` inside a pipeline folder shows the same blocks, then one row per job of
## the runs under that folder. Read TIME carefully: squeue formats it [DD-[HH:]]MM:SS, so
## `4:03` is four minutes and `1-02:19:10` is a day. PENDING jobs show 0:00.
running:
	@if [ "$(ON_CLUSTER)" != "yes" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh $(SHERLOCK_HOST)"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $$SCRATCH/nf-running

## The protein annotation leaderboard's tables (seanome/protein-function-annotation-leaderboard
## copies results/leaderboard/ from a commit of this repo).
##
## Source: the QfO Pfam region benchmark's midi-plus run, 998 human queries against 15 QfO
## proteomes, finished 2026-09-13 on Sherlock. all_domain_metrics.parquet is the run's own
## output (sha256 8ba26616...); the traces are both launch folders' trace files, because the
## run was resumed from run-midi-plus and run-midi-plus-all-qfo.
LEADERBOARD_DATA ?= $(HOME)/data/qfo-pfam-region-benchmark/midi-plus-ladder-2026-09-13
LEADERBOARD_PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
REGION_RUN := /scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark

leaderboard-fetch:
	mkdir -p $(LEADERBOARD_DATA)/traces/run-midi-plus $(LEADERBOARD_DATA)/traces/run-midi-plus-all-qfo
	rsync -a "$(SHERLOCK_HOST):$(REGION_RUN)/data/midi-plus/results/all_domain_metrics.parquet" $(LEADERBOARD_DATA)/
	rsync -a "$(SHERLOCK_HOST):$(REGION_RUN)/data/midi-plus/midi_testset_summary.json" $(LEADERBOARD_DATA)/
	rsync -a "$(SHERLOCK_HOST):$(REGION_RUN)/run-midi-plus/qfo_pfam_region.2026-09-*.trace.txt" $(LEADERBOARD_DATA)/traces/run-midi-plus/
	rsync -a "$(SHERLOCK_HOST):$(REGION_RUN)/run-midi-plus-all-qfo/qfo_pfam_region.2026-09-*.trace.txt" $(LEADERBOARD_DATA)/traces/run-midi-plus-all-qfo/

leaderboard-tables:
	$(LEADERBOARD_PYTHON) scripts/build_leaderboard_tables.py \
	  --metrics $(LEADERBOARD_DATA)/all_domain_metrics.parquet \
	  --summary $(LEADERBOARD_DATA)/midi_testset_summary.json \
	  --traces $(LEADERBOARD_DATA)/traces/*/qfo_pfam_region.*.trace.txt \
	  --outdir results/leaderboard \
	  --metrics-commit a47ea255
