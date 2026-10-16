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
	@echo "  leaderboard-fetch   Copy every input provenance.json lists from Sherlock into"
	@echo "                      LEADERBOARD_DATA (a mirror of /scratch/users/olgabot)"
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
## One long table of every finished benchmark output: the QfO region benchmark (mini, midi,
## midi-plus), ELM cover, the dark set (Pfam Fmax, reach, best-hit accuracy), notebook
## 241's three cases, the deep-twilight controls, the SCOPe40 kmerseek 0.4 sweeps, and
## the Pfam protein-pair benchmark of other tools (pfam_pair_roc).
## results/leaderboard/inventory.md lists every output found and why some are left out.
##
## LEADERBOARD_DATA mirrors /scratch/users/olgabot: a file at $SCRATCH/x/y is read from
## $(LEADERBOARD_DATA)/x/y. leaderboard-fetch copies exactly the inputs provenance.json
## lists, over the data transfer node. Inputs that exist only on the laptop
## (/Users/olga/x/y) are copied to $(LEADERBOARD_DATA)/laptop/x/y.
LEADERBOARD_DATA ?= $(HOME)/data/leaderboard-inputs
LEADERBOARD_PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
SHERLOCK_DTN ?= sherlock-dtn

leaderboard-fetch:
	mkdir -p $(LEADERBOARD_DATA)
	$(LEADERBOARD_PYTHON) -c 'import json; [print(p.removeprefix("/scratch/users/olgabot/")) for p in json.load(open("results/leaderboard/provenance.json"))["inputs"] if p.startswith("/scratch/")]' \
	  > $(LEADERBOARD_DATA)/.fetch-list.txt
	pixi exec rsync -a --files-from=$(LEADERBOARD_DATA)/.fetch-list.txt \
	  "$(SHERLOCK_DTN):/scratch/users/olgabot/" $(LEADERBOARD_DATA)/
	$(LEADERBOARD_PYTHON) -c 'import json; [print(p.removeprefix("/Users/olga/")) for p in json.load(open("results/leaderboard/provenance.json"))["inputs"] if p.startswith("/Users/olga/")]' \
	  > $(LEADERBOARD_DATA)/.fetch-list-laptop.txt
	mkdir -p $(LEADERBOARD_DATA)/laptop
	pixi exec rsync -a --files-from=$(LEADERBOARD_DATA)/.fetch-list-laptop.txt \
	  /Users/olga/ $(LEADERBOARD_DATA)/laptop/

leaderboard-tables:
	$(LEADERBOARD_PYTHON) scripts/build_leaderboard_long.py \
	  --data $(LEADERBOARD_DATA) \
	  --outdir results/leaderboard
