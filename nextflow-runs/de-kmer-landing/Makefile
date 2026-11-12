# Notebook 360: differentially expressed k-mers landed on Pfam domains by kmerseek, against
# Trinity + DIAMOND + borrowed Pfam, on simulated reads with a known answer.
# Rules: notebooks/360_data/preregistered.md.
#
#   make key        (Sherlock) build the answer keys and search databases (one SLURM job)
#   make smoke      (Sherlock) Ciona only, two kmerseek settings, to check every step runs
#   make run        (Sherlock) the full run, resuming whatever smoke built
#   make watch      (Sherlock) attach read-only to the head's live task table
#   make stop       (Sherlock) ask the head to shut down cleanly
#   make running    (Sherlock) what is running, per process and tag
#   make preview    (anywhere) check the workflow wiring without running anything

SHELL := /bin/bash

NB360_DIR ?= /scratch/users/olgabot/360-de-kmer-landing
KEY_DIR   ?= $(NB360_DIR)/key
PFAM_SEED ?= $(NB360_DIR)/downloads/Pfam-A.seed.gz
RUN_DIR   ?= $(NB360_DIR)/run

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_TIME    ?= 2-00:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= nb360
HEAD_JOBNAME ?= nf-nb360
export NXF_OPTS := -Xms512m -Xmx6g

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head
NF_ARGS = $(MAKEFILE_DIR)/main.nf -profile sherlock -resume --key_dir $(KEY_DIR) --pfam_seed $(PFAM_SEED)

.PHONY: key smoke run watch stop running preview measure-resources

key:
	cd $(MAKEFILE_DIR)/../.. && sbatch scripts/nb360/build_answer_key.sbatch

smoke: measure-resources
	mkdir -p $(RUN_DIR) && cd $(RUN_DIR) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run $(NF_ARGS) --organisms ciona --only_settings protein20.k7,hp_pbotc_1st_ed2.k30

run: measure-resources
	mkdir -p $(RUN_DIR) && cd $(RUN_DIR) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run $(NF_ARGS)

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

running:
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"

preview:
	nextflow run $(MAKEFILE_DIR)/main.nf -preview --key_dir $(KEY_DIR) --pfam_seed $(PFAM_SEED)

measure-resources:
	python3 $(MAKEFILE_DIR)/bin/measure_landing_resources.py $(RUN_DIR) $(MAKEFILE_DIR)/assets/kmerseek_settings.tsv

repair-landings: measure-resources
	mkdir -p $(RUN_DIR) && cd $(RUN_DIR) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run $(MAKEFILE_DIR)/repair_landings.nf -profile sherlock -resume --key_dir $(KEY_DIR)
