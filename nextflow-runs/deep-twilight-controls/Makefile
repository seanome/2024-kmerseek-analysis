# Deep-twilight controls: 16 family proteins, 150 kmerseek arms plus phmmer, MMseqs2 and
# Foldseek, all searched against the QfO human proteome plus the 8 non-human family members.
#
#   make push-image      (laptop) build the kmerseek image from KMERSEEK_SHA and push it
#   make smoke           (laptop) two arms and every baseline, local Docker, a few minutes
#   make run             (Sherlock) submit the Nextflow head as a SLURM job in tmux
#   make watch           (Sherlock) attach read-only to the head's live task table
#   make stop            (Sherlock) ask the head to shut down cleanly
#   make pull-results    (laptop) copy results/ back for notebook 252

SHELL := /bin/bash

KMERSEEK_REPO ?= /Users/olga/code/kmerseek
# The search image (params.kmerseek_search_image). The python steps still run in
# 2026-09-25-run-evalue-5fdfdcc-rtcompat (params.kmerseek_image), built from 5fdfdcc.
KMERSEEK_SHA  ?= 90c581a011d42002eeeb204ec6efea79ece0359d
IMAGE_TAG     ?= 2026-09-25-ka-fixes-90c581a
IMAGE_REMOTE  := docker.io/olgabot/kmerseek:$(IMAGE_TAG)

NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
NEXTFLOW_JVM := $(patsubst %/bin/nextflow,%/lib/jvm,$(NEXTFLOW))
ifneq ($(wildcard $(NEXTFLOW_JVM)/bin/java),)
export JAVA_HOME := $(NEXTFLOW_JVM)
endif

# The region benchmark already holds the human proteome and its AlphaFold DB v6 models.
REGION_DATA      ?= /scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data
HUMAN_FASTA      ?= $(REGION_DATA)/qfo/Eukaryota/UP000005640_9606.fasta
HUMAN_STRUCTURES ?= $(REGION_DATA)/structures/human

# Laptop copies for `make smoke`
SMOKE_HUMAN_FASTA ?= /Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota/UP000005640_9606.fasta

REMOTE_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis-deep-twilight/nextflow-runs/deep-twilight-controls

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_TIME    ?= 12:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= deep-twilight
HEAD_JOBNAME ?= nf-deep-twilight
export NXF_OPTS := -Xms512m -Xmx6g

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head

.PHONY: push-image smoke run watch stop pull-results

# Built from a clean `git archive` of KMERSEEK_SHA, so nothing uncommitted in a kmerseek
# worktree can reach the image.
push-image:
	@ctx=$$(mktemp -d); \
	 git -C $(KMERSEEK_REPO) archive $(KMERSEEK_SHA) | tar -x -C $$ctx && \
	 docker build --platform linux/amd64 -t $(IMAGE_REMOTE) \
	   --build-arg GIT_SHA=$(KMERSEEK_SHA) -f $(MAKEFILE_DIR)/Dockerfile $$ctx && \
	 docker run --rm --platform linux/amd64 $(IMAGE_REMOTE) python3 -c "import polars, numpy; polars.DataFrame({\"a\": [1]}).sum()" && \
	 docker push $(IMAGE_REMOTE); \
	 status=$$?; /bin/rm -rf $$ctx; exit $$status

# Two arms, every baseline, and a structure directory holding only the family members'
# human models, on the laptop. Checks the plumbing, not the numbers.
smoke:
	mkdir -p run-smoke/human_structures
	head -3 assets/arms.tsv > run-smoke/arms.tsv
	cd run-smoke && $(NEXTFLOW) run ../main.nf -profile standard -resume \
	  --arms $(MAKEFILE_DIR)/run-smoke/arms.tsv \
	  --human_fasta $(SMOKE_HUMAN_FASTA) \
	  --human_structures $(MAKEFILE_DIR)/run-smoke/human_structures

run:
	mkdir -p run
	cd run && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run ../main.nf -profile sherlock -resume \
	    --human_fasta $(HUMAN_FASTA) --human_structures $(HUMAN_STRUCTURES)

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

pull-results:
	mkdir -p results
	rsync -av sherlock:$(REMOTE_DIR)/run/results/ results/
