# Targets that are about the whole repo, not one pipeline. Each pipeline has its own
# Makefile under nextflow-runs/<pipeline>/.

SHERLOCK_HOST ?= sherlock
ON_CLUSTER := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

.PHONY: help running leaderboard-alphabets

help:
	@echo "Sherlock:"
	@echo "  running    Every Nextflow run in the SLURM queue, from any pipeline: clone,"
	@echo "             run folder, head job, tasks by process and state, and jobs that"
	@echo "             are not part of a run"
	@echo ""
	@echo "Leaderboard:"
	@echo "  leaderboard-alphabets  Write results/leaderboard/alphabets/ (the alphabet page's"
	@echo "                         tables) from notebooks 230 and 291 and PR 116's CSV"
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

## The leaderboard site's alphabet page (seanome/protein-function-annotation-leaderboard copies
## results/leaderboard/alphabets/ from a commit of this repo). Reads notebook 291's stored
## table and the per-pair tables of notebook 230 from ~/data (about 100 MB), so it runs on the
## laptop in about 20 s with a peak of 3.75 GB resident (measured 2026-10-16 with
## /usr/bin/time -l); see scripts/build_leaderboard_alphabets.py.
LEADERBOARD_PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
leaderboard-alphabets:
	$(LEADERBOARD_PYTHON) scripts/build_leaderboard_alphabets.py --outdir results/leaderboard/alphabets
