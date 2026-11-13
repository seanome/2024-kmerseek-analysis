# Notebook 290: how often chance alone gives a dark protein a kmerseek region.
# Shuffled copies of every query chunk of the dark-set run (PR 57, runs/ladder-0.4-midi),
# searched with the run's six kmerseek settings, indexes, image and flags.
#
# Runs ON SHERLOCK, from a copy of this folder taken with git archive (a full clone stalls
# on lustre):
#   cd $SCRATCH/2024-kmerseek-analysis && git fetch origin olgabot/dark-set-shuffle-control
#   mkdir -p $SCRATCH/290-shuffle-control/repo
#   git archive FETCH_HEAD scripts/290_shuffle_control nextflow-runs/scope40-lowcomplexity-null/bin | tar -x -C $SCRATCH/290-shuffle-control/repo
#   cd $SCRATCH/290-shuffle-control/repo/scripts/290_shuffle_control
#   make plan shuffles pilot
#
# Nothing here writes outside $(BASE). The run's indexes, chunks and region files are
# mounted read-only in every task, and no target builds or calibrates an index.

BASE     ?= $(SCRATCH)/290-shuffle-control
RUN_REPO ?= /scratch/users/olgabot/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set
RUN_DIR  ?= $(RUN_REPO)/runs/ladder-0.4-midi
IMG      ?= /scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
STAGE    := $(CURDIR)
REPO     := $(abspath $(CURDIR)/../..)
ON_CLUSTER := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

.PHONY: help running plan shuffles pilot-dry pilot full-dry full status guard

help:
	@echo "Notebook 290 (dark-set shuffle control). On Sherlock, in this folder:"
	@echo "  plan       check the run's 294 search commands and write plan/ (chunks, tasks, resources)"
	@echo "  shuffles   write the dipeptide and 10-residue-window copies of all 49 chunks"
	@echo "  pilot-dry  print the pilot's sbatch commands without submitting"
	@echo "  pilot      65 searches (one chunk per species) and 30 run-file reads"
	@echo "  full-dry   print the full run's sbatch commands"
	@echo "  full       every chunk: 588 searches and 294 run-file reads (done tasks are skipped)"
	@echo "  status     finished tasks per setting and kind, and this experiment's queue"
	@echo "  running    every Nextflow run in the SLURM queue"

## Every Nextflow run in the queue, whichever pipeline or clone it came from.
running:
	@if [ "$(ON_CLUSTER)" != "yes" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $$SCRATCH/nf-running

## Stop if the dark-set run or its repair is in the queue: it may still be writing files
## this experiment reads.
guard:
	@if squeue --me -h -o "%Z %j" | grep -q "invertebrate-dark-set"; then \
	  echo "A dark-set run job is in the queue; not submitting. squeue --me -o '%i %j %Z':"; \
	  squeue --me -h -o "%i %j %Z" | grep "invertebrate-dark-set"; exit 1; \
	fi

plan: guard
	python3 $(STAGE)/tasks.py $(RUN_DIR) $(RUN_REPO) $(BASE)/plan

shuffles: $(BASE)/plan/chunks.tsv
	srun -A ayeletv -p hns,normal -c 1 --mem 4G -t 00:30:00 -J 290-shuffles \
	  apptainer exec --no-home -B $(BASE) -B $(REPO):$(REPO):ro -B $(RUN_DIR):$(RUN_DIR):ro $(IMG) \
	  python3 $(STAGE)/shuffle_chunks.py $(BASE)/plan/chunks.tsv $(BASE)/chunks

pilot-dry: $(BASE)/chunks/shuffles.tsv
	python3 $(STAGE)/submit.py pilot $(BASE) $(STAGE) $(IMG) --dry-run

pilot: guard $(BASE)/chunks/shuffles.tsv
	python3 $(STAGE)/submit.py pilot $(BASE) $(STAGE) $(IMG)

full-dry: $(BASE)/chunks/shuffles.tsv
	python3 $(STAGE)/submit.py full $(BASE) $(STAGE) $(IMG) --dry-run

full: guard $(BASE)/chunks/shuffles.tsv
	python3 $(STAGE)/submit.py full $(BASE) $(STAGE) $(IMG)

status:
	@for sp in human worm yeast borrelia botryllus; do \
	  printf "%-10s" $$sp; \
	  for kind in dipeptide window10 real realfile; do \
	    printf "  %s %3d" $$kind $$(ls $(BASE)/out/$$sp/*.$$kind.*.check.tsv 2>/dev/null | wc -l); \
	  done; echo; \
	done
	@squeue --me -h -o "%.18i %.40j %.8T %.10M %R" | grep " 290-" || echo "no 290 jobs in the queue"
