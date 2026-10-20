# Targets that are about the whole repo, not one pipeline. Each pipeline has its own
# Makefile under nextflow-runs/<pipeline>/.

SHERLOCK_HOST ?= sherlock
ON_CLUSTER := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

.PHONY: help running leaderboard-tables leaderboard-fetch leaderboard-timetree

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
	@echo "  query-strata-fetch  Copy the query-strata job outputs from Sherlock"
	@echo "  query-strata        Write results/leaderboard/query_strata/ and recount its bins"
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
## 241's three cases, the deep-twilight controls, and the SCOPe40 kmerseek 0.4 sweeps.
## results/leaderboard/inventory.md lists every output found and why some are left out.
##
## LEADERBOARD_DATA mirrors /scratch/users/olgabot: a file at $SCRATCH/x/y is read from
## $(LEADERBOARD_DATA)/x/y. leaderboard-fetch copies exactly the inputs provenance.json
## lists, over the data transfer node.
LEADERBOARD_DATA ?= $(HOME)/data/leaderboard-inputs
LEADERBOARD_PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
SHERLOCK_DTN ?= sherlock-dtn

leaderboard-fetch:
	mkdir -p $(LEADERBOARD_DATA)
	$(LEADERBOARD_PYTHON) -c 'import json; [print(p.removeprefix("/scratch/users/olgabot/")) for p in json.load(open("results/leaderboard/provenance.json"))["inputs"]]' \
	  > $(LEADERBOARD_DATA)/.fetch-list.txt
	pixi exec rsync -a --files-from=$(LEADERBOARD_DATA)/.fetch-list.txt \
	  "$(SHERLOCK_DTN):/scratch/users/olgabot/" $(LEADERBOARD_DATA)/

## TimeTree's divergence time from human for each QfO relative; leaderboard-tables reads it.
leaderboard-timetree:
	$(LEADERBOARD_PYTHON) scripts/fetch_timetree_human_pairwise.py \
	  --species nextflow-runs/qfo-pfam-region-benchmark/assets/qfo_species.tsv \
	  --out results/leaderboard/timetree_human_pairwise.tsv

leaderboard-tables:
	$(LEADERBOARD_PYTHON) scripts/build_leaderboard_long.py \
	  --data $(LEADERBOARD_DATA) \
	  --outdir results/leaderboard

## Query-side strata of the 998 midi-plus queries (results/leaderboard/query_strata/).
## The five jobs in scripts/query_strata/ run on Sherlock and write under
## $SCRATCH/leaderboard-query-strata/out; query-strata-fetch copies their small outputs
## (and the run's own covariates and truth tables) into LEADERBOARD_DATA; query-strata
## builds the tables and the covariates file main.nf --query_covariates reads, then
## recounts every bin from the raw outputs with its own code.
QUERY_STRATA_FILES = \
  leaderboard-query-strata/out/identity/ \
  leaderboard-query-strata/out/seg/ \
  leaderboard-query-strata/out/length/ \
  leaderboard-query-strata/out/deeptmhmm/predicted_topologies.3line \
  leaderboard-query-strata/out/deeptmhmm/TMRs.gff3 \
  leaderboard-query-strata/out/deeptmhmm/queries.fasta \
  leaderboard-query-strata/out/busco/busco_version.txt \
  leaderboard-query-strata/out/busco/busco_human/run_primates_odb12/full_table.tsv \
  leaderboard-query-strata/out/busco/busco_human/short_summary.specific.primates_odb12.busco_human.json \
  leaderboard-query-strata/out/busco/busco_human/short_summary.specific.primates_odb12.busco_human.txt \
  leaderboard-query-strata/out/pfam_breadth/pfam_family_breadth.tsv \
  2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results/truth/human_query_covariates.parquet \
  2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results/truth/human_domain_truth.parquet \
  2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/pfamn/human_pfamn_truth.parquet

.PHONY: query-strata-fetch query-strata
query-strata-fetch:
	mkdir -p $(LEADERBOARD_DATA)
	printf '%s\n' $(QUERY_STRATA_FILES) > $(LEADERBOARD_DATA)/.query-strata-list.txt
	pixi exec rsync -a -r --files-from=$(LEADERBOARD_DATA)/.query-strata-list.txt \
	  "$(SHERLOCK_DTN):/scratch/users/olgabot/" $(LEADERBOARD_DATA)/

query-strata:
	$(LEADERBOARD_PYTHON) scripts/query_strata/build_query_strata.py \
	  --data $(LEADERBOARD_DATA) --outdir results/leaderboard/query_strata
	$(LEADERBOARD_PYTHON) scripts/query_strata/check_query_strata.py \
	  --data $(LEADERBOARD_DATA) --strata results/leaderboard/query_strata \
	  --results results/leaderboard/results.parquet
