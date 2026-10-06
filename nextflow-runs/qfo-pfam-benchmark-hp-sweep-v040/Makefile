NF := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

.PHONY: run resume

run:
	$(NF) run kmerseek_qfo_pfam_benchmark_hp_sweep.nf

resume:
	$(NF) run kmerseek_qfo_pfam_benchmark_hp_sweep.nf -resume

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
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-benchmark-hp-sweep-v040"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
