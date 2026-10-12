# Time-split test for notebook 280: would kmerseek have placed Pfam domains before Pfam
# did? t0 = Pfam 33.1 (May 2020), t1 = Pfam 38.2 (June 2026). Runs on Sherlock.
#
#   make download   fetch Pfam 33.1, 38.2, 38.0 pfamseq and Swiss-Prot 2019_08 (batch job)
#   make stop1      build the query set, count every step, identity of 500 instances
#   make running    what is running on this account
#
# Code is read from this checkout; data and outputs go to RUN_DIR.
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
RUN_DIR      ?= $(SCRATCH)/280-time-split
SBATCH_ENV    = --chdir=$(RUN_DIR) --export=ALL,TIME_SPLIT_BIN=$(MAKEFILE_DIR)/bin

.PHONY: download stop1 running

download:
	@mkdir -p $(RUN_DIR)/data
	sbatch $(SBATCH_ENV) $(MAKEFILE_DIR)/bin/download.sbatch

stop1:
	sbatch $(SBATCH_ENV) $(MAKEFILE_DIR)/bin/stop1.sbatch

## Read TIME carefully. squeue formats it [DD-[HH:]]MM:SS, so `4:03` is four minutes.
running:
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
