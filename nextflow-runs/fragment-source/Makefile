# Notebook 354: does a reduced-alphabet kmerseek index find the source protein of a short
# de novo peptide call better than an exact full-alphabet index plus blastp-short?
#
# Runs on Sherlock as SLURM jobs chained by dependencies. From a clone of this branch:
#   cd $SCRATCH/2024-kmerseek-analysis-nb354/nextflow-runs/fragment-source
#   make all            # download -> build kmerseek -> freeze -> dev -> test -> summarise
# or one stage at a time: make download, make build, make freeze, make dev, make test,
# make summarise. Then on the laptop: make pull.
#
# Sizing (measured 2026-10-20 on the frozen 20,734-protein database):
#   download   330 MB at about 1 MB/s from PRIDE: 8 min (job 48678939). 1 h asked.
#   build      cargo build of kmerseek 4d7e27b: 4 min 17 s (job 48679601). 40 min asked.
#   freeze     under 1 min, peak under 2 GB (srun job 48680049). 20 min, 8 GB asked.
#   dev        18 kmerseek settings: one massspec18 k6 index took 10 s and 0.74 GB on 4
#              cores, a 10-query search 5 s and 0.9 GB (job 48680816). Lower k builds and
#              searches more postings; 8 workers x 3 GB, plus blastp-short of 2,750 queries
#              on 8 threads. 2 h, 32 GB asked.
#   test       one thread for the timed runs, 8 for the rest; blastp-short of 2,500 control
#              queries dominates. 4 h, 32 GB asked.
#   summarise  about 1,000 alignments: minutes. 30 min, 8 GB asked.

OUT ?= $(SCRATCH)/fragment-source-354
SHERLOCK_HOST ?= sherlock
REMOTE_OUT ?= /scratch/users/olgabot/fragment-source-354
NOTEBOOK_DATA := ../../notebooks/354_data
KMERSEEK_SRC ?= $(OUT)/kmerseek-4d7e27b
KMERSEEK := $(KMERSEEK_SRC)/target/release/kmerseek

RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
BIN := $(RUNNING_DIR)/bin
SB := $(RUNNING_DIR)/sbatch
JOBENV := OUT=$(OUT),BIN=$(BIN),KMERSEEK=$(KMERSEEK),KMERSEEK_SRC=$(KMERSEEK_SRC)
RUN := sbatch --parsable --export=ALL,$(JOBENV)

.PHONY: all download build freeze dev test summarise pull test-code running

## Submit every stage as one chain. Prints the job ids.
all:
	@mkdir -p $(OUT)/logs
	@cd $(OUT)/logs && \
	dl=$$($(RUN) $(SB)/download.sbatch) && \
	bu=$$($(RUN) $(SB)/build_kmerseek.sbatch) && \
	fr=$$($(RUN) --dependency=afterok:$$dl --job-name=nb354-freeze -c 1 --mem=8G -t 00:20:00 --export=ALL,$(JOBENV),STAGE=freeze $(SB)/run.sbatch) && \
	de=$$($(RUN) --dependency=afterok:$$fr:$$bu --job-name=nb354-dev -c 8 --mem=32G -t 02:00:00 --export=ALL,$(JOBENV),STAGE=dev $(SB)/run.sbatch) && \
	te=$$($(RUN) --dependency=afterok:$$de --job-name=nb354-test -c 8 --mem=32G -t 04:00:00 --export=ALL,$(JOBENV),STAGE=test $(SB)/run.sbatch) && \
	su=$$($(RUN) --dependency=afterok:$$te --job-name=nb354-summarise -c 1 --mem=8G -t 00:30:00 --export=ALL,$(JOBENV),STAGE=summarise $(SB)/run.sbatch) && \
	echo "download $$dl, build $$bu -> freeze $$fr -> dev $$de -> test $$te -> summarise $$su; logs in $(OUT)/logs"

download:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && $(RUN) $(SB)/download.sbatch
build:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && $(RUN) $(SB)/build_kmerseek.sbatch
freeze:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && $(RUN) --job-name=nb354-freeze -c 1 --mem=8G -t 00:20:00 --export=ALL,$(JOBENV),STAGE=freeze $(SB)/run.sbatch
## dev, then test and summarise chained on it.
dev:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && \
	de=$$($(RUN) --job-name=nb354-dev -c 8 --mem=32G -t 02:00:00 --export=ALL,$(JOBENV),STAGE=dev $(SB)/run.sbatch) && \
	echo "dev $$de"
test:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && \
	te=$$($(RUN) --job-name=nb354-test -c 8 --mem=32G -t 04:00:00 --export=ALL,$(JOBENV),STAGE=test $(SB)/run.sbatch) && \
	su=$$($(RUN) --dependency=afterok:$$te --job-name=nb354-summarise -c 1 --mem=8G -t 00:30:00 --export=ALL,$(JOBENV),STAGE=summarise $(SB)/run.sbatch) && \
	echo "test $$te -> summarise $$su"
summarise:
	@mkdir -p $(OUT)/logs && cd $(OUT)/logs && $(RUN) --job-name=nb354-summarise -c 1 --mem=8G -t 00:30:00 --export=ALL,$(JOBENV),STAGE=summarise $(SB)/run.sbatch

## Copy the small result tables to the notebook's data folder. Run on the laptop.
pull:
	mkdir -p $(NOTEBOOK_DATA)
	rsync -av $(SHERLOCK_HOST):$(REMOTE_OUT)/work/results/ $(NOTEBOOK_DATA)/results/
	rsync -av $(SHERLOCK_HOST):$(REMOTE_OUT)/work/dev_choice.json $(SHERLOCK_HOST):$(REMOTE_OUT)/work/dev_table.tsv $(NOTEBOOK_DATA)/

test-code:
	python3 -m pytest -q tests

running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis-nb354/nextflow-runs/fragment-source"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
