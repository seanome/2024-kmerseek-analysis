# Notebook 346: the two Sherlock jobs (Slurm sbatch scripts, no Nextflow).
#
#   scripts/346_swissprot_scan.sbatch     pairs 3 to 6 against Swiss-Prot, 16 array tasks,
#                                         then one merge task (Python and numpy only)
#   scripts/346_kmerseek_ced9_human.sbatch  kmerseek's own search, CED-9 against human, k 6, 8, 10
#
# Run from a clone of its own, so this branch never moves the main Sherlock checkout:
#   /scratch/users/olgabot/2024-kmerseek-analysis-nb346
#
# On the Mac:      make push-inputs     copy Swiss-Prot to Sherlock (once)
#                  make pull-results    copy the results back to ~/data/linked-pairs
# On Sherlock:     make run             submit both jobs (the merge waits for the scan)
#                  make run-kmerseek    submit only the kmerseek job (k 6, 8, 10)
#                  make running         this notebook's jobs in the queue
#                  make done            elapsed time and peak memory of finished jobs

REMOTE_CLONE ?= /scratch/users/olgabot/2024-kmerseek-analysis-nb346
JOB_DIR      ?= /scratch/users/olgabot/346_linked_pairs
SPROT        ?= /Users/olga/data/uniprot/uniprot_sprot.fasta.gz
LOCAL_DATA   ?= /Users/olga/data/linked-pairs
DTN          ?= sherlock-dtn
REPO_ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
ON_CLUSTER   := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)
JOBS         := 346-swissprot-scan,346-kmerseek-ced9

.PHONY: help push-inputs run run-kmerseek running done pull-results

help:
	@sed -n '2,17p' Makefile

## Mac: Swiss-Prot (the same file the laptop scans read; the scan checks its 572,970 proteins).
push-inputs:
	ssh $(DTN) mkdir -p $(JOB_DIR)/inputs $(JOB_DIR)/logs
	pixi exec rsync -av $(SPROT) $(DTN):$(JOB_DIR)/inputs/uniprot_sprot.fasta.gz

## Sherlock: submit from the repository root, which both sbatch scripts read as SLURM_SUBMIT_DIR.
run:
	@[ "$(ON_CLUSTER)" = yes ] || { echo "run submits Slurm jobs: run it on Sherlock"; exit 1; }
	mkdir -p $(JOB_DIR)/logs $(JOB_DIR)/results
	cd $(REPO_ROOT) && \
	  scan=$$(sbatch --parsable scripts/346_swissprot_scan.sbatch | cut -d';' -f1) && \
	  echo "Swiss-Prot scan: job $$scan" && \
	  merge=$$(sbatch --parsable --dependency=afterok:$$scan --array=0 --export=ALL,NB346_MERGE=1 \
	           scripts/346_swissprot_scan.sbatch | cut -d';' -f1) && \
	  echo "merge (waits for $$scan): job $$merge" && \
	  km=$$(sbatch --parsable scripts/346_kmerseek_ced9_human.sbatch | cut -d';' -f1) && \
	  echo "kmerseek k = 6, 8, 10: job $$km"

run-kmerseek:
	@[ "$(ON_CLUSTER)" = yes ] || { echo "run-kmerseek submits a Slurm job: run it on Sherlock"; exit 1; }
	mkdir -p $(JOB_DIR)/logs $(JOB_DIR)/results
	cd $(REPO_ROOT) && km=$$(sbatch --parsable scripts/346_kmerseek_ced9_human.sbatch | cut -d';' -f1) && \
	  echo "kmerseek k = 6, 8, 10: job $$km"

running:
	@[ "$(ON_CLUSTER)" = yes ] || { echo "running reads the Slurm queue: run it on Sherlock"; exit 1; }
	squeue -u $$USER -n $(JOBS) -o "%.18i %.20j %.8T %.10M %.10l %R"

done:
	@[ "$(ON_CLUSTER)" = yes ] || { echo "done reads sacct: run it on Sherlock"; exit 1; }
	sacct -u $$USER --name=$(JOBS) -S now-7days -o JobID%20,JobName%20,State,Elapsed,MaxRSS

## Mac: the merged Swiss-Prot scores and kmerseek's output, where the notebook reads them.
pull-results:
	mkdir -p $(LOCAL_DATA)
	pixi exec rsync -av $(DTN):$(JOB_DIR)/results/346_scan_swissprot.parquet $(LOCAL_DATA)/
	pixi exec rsync -av --include='346_kmerseek_*' --exclude='*' $(DTN):$(JOB_DIR)/results/ $(LOCAL_DATA)/
