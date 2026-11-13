# Notebook 344: kmerseek region E-values on 309_000 decoy queries (20_600 human proteins x
# 5 copies x 3 shuffles), searched against the human proteome.
#
#   make push-image    (laptop) build the image from KMERSEEK_SHA and push it
#   make stub          (laptop) the whole graph on CED-9 and BCL2, no kmerseek, seconds
#   make pilot         (Sherlock) index + one chunk of each kind, to measure the searches
#   make measure       (Sherlock) size the full run from the pilot's trace
#   make run           (Sherlock) everything; reuses the pilot's tasks
#   make watch / stop / running
#   make pull-results  (laptop) copy results/ back for the notebook

SHELL := /bin/bash

KMERSEEK_REPO ?= /Users/olga/code/kmerseek
# seanome/kmerseek PR 134 head (olgabot/run-evalue-checks), which holds PR 133.
KMERSEEK_SHA  ?= 0a229700bc9dbf3d04f9e4f141024ff33b5c4e7e
IMAGE_TAG     ?= 2026-10-18-nb344-0a22970
IMAGE_REMOTE  := docker.io/olgabot/kmerseek:$(IMAGE_TAG)

NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
NEXTFLOW_JVM := $(patsubst %/bin/nextflow,%/lib/jvm,$(NEXTFLOW))
ifneq ($(wildcard $(NEXTFLOW_JVM)/bin/java),)
export JAVA_HOME := $(NEXTFLOW_JVM)
endif

# The QfO 2020_04 human proteome the region benchmark already holds, 20_600 proteins.
# The md5 is the laptop copy's; MAKE_DECOYS stops if Sherlock's differs.
REGION_DATA ?= /scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data
HUMAN_FASTA ?= $(REGION_DATA)/qfo/Eukaryota/UP000005640_9606.fasta
HUMAN_MD5   ?= 74bf5689099474ca85a96b090c722ece

# A clone of its own, so checking out this branch cannot disturb runs in the shared clone.
REMOTE_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis-nb344/nextflow-runs/region-evalue-calibration

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_TIME    ?= 48:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= nb344
HEAD_JOBNAME ?= nf-nb344
export NXF_OPTS := -Xms512m -Xmx6g

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head
RUN_ARGS = -profile sherlock -resume --human_fasta $(HUMAN_FASTA) --human_md5 $(HUMAN_MD5)

.PHONY: push-image stub pilot measure run watch stop running pull-results

# Built from a clean `git archive` of KMERSEEK_SHA, so nothing uncommitted in a kmerseek
# worktree can reach the image. Checks kmerseek and segmasker run before pushing.
push-image:
	@ctx=$$(mktemp -d); \
	 git -C $(KMERSEEK_REPO) archive $(KMERSEEK_SHA) | tar -x -C $$ctx && \
	 docker build --platform linux/amd64 -t $(IMAGE_REMOTE) \
	   --build-arg GIT_SHA=$(KMERSEEK_SHA) -f $(MAKEFILE_DIR)/Dockerfile $$ctx && \
	 docker run --rm --platform linux/amd64 $(IMAGE_REMOTE) bash -c 'kmerseek --version && segmasker -version && cat /etc/kmerseek_git_sha' && \
	 docker push $(IMAGE_REMOTE); \
	 status=$$?; /bin/rm -rf $$ctx; exit $$status

# Every process on CED-9 and BCL2: the real decoy maker, SEG parser, region filter and
# collector, with stand-ins for segmasker and kmerseek. Checks the plumbing and that the
# decoy files keep what each shuffle must keep.
stub:
	mkdir -p run-stub
	cd run-stub && $(NEXTFLOW) run ../main.nf -profile stub -stub-run \
	  --human_fasta $(MAKEFILE_DIR)/assets/test/ced9_bcl2.fasta --chunk_size 4
	@test -s run-stub/results/decoys/decoys.tsv
	@for k in composition dipeptide window10; do \
	   test -s run-stub/results/regions/$$k.regions.tsv.gz || { echo "no $$k regions"; exit 1; }; \
	 done
	@echo "stub passed: $$(($$(wc -l < run-stub/results/decoys/decoys.tsv) - 1)) decoys"

define require_cluster
@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
  echo ""; \
  echo "  this target runs ON SHERLOCK, not on your Mac:"; \
  echo "    ssh sherlock"; \
  echo "    cd $(REMOTE_DIR)"; \
  echo ""; \
  exit 1; \
fi
endef

# The index and the first chunk of each kind (6_000 decoys), in run/ so `make run` reuses them.
pilot:
	$(call require_cluster)
	mkdir -p run
	cd run && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run ../main.nf $(RUN_ARGS) --max_chunks 1 --trace_label pilot

measure:
	$(call require_cluster)
	python3 measure_resources.py $$(ls -t run/region_evalue_calibration.pilot.*.trace.txt | head -1) run/measured.json

run:
	$(call require_cluster)
	@test -s run/measured.json || { echo "run make pilot, then make measure, first"; exit 1; }
	cd run && ml load biology nextflow && \
	  $(HEAD_SUBMIT) nextflow run ../main.nf $(RUN_ARGS) -params-file measured.json

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

running:
	$(call require_cluster)
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"

# Everything but the decoy FASTAs (about 160 MB); the notebook reads decoys.tsv, the SEG
# tables and the region files.
pull-results:
	mkdir -p results
	rsync -av --exclude 'decoys/*.fasta' sherlock:$(REMOTE_DIR)/run/results/ results/
