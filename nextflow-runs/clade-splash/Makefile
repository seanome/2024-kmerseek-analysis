# clade-splash: notebook 357. Clade-grouped SPLASH (OASIS test) on reduced-alphabet protein
# anchors, UniProt reference proteomes, one species per family, NCBI classes as samples.
#
#   make test          (laptop) unit tests of bin/clade_splash.py
#   make oasis         (laptop) fetch OASIS.py at the pinned commit, for the tests
#   make pilot         (Sherlock) 8 species from each of the 6 largest classes
#   make run           (Sherlock) the full run
#   make watch         (Sherlock) attach read-only to the head's live task table
#   make stop          (Sherlock) ask the head to shut down cleanly
#   make pull-results  (laptop) copy the small result tables to tables/357/

SHELL := /bin/bash

NEXTFLOW ?= /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
NEXTFLOW_JVM := $(patsubst %/bin/nextflow,%/lib/jvm,$(NEXTFLOW))
ifneq ($(wildcard $(NEXTFLOW_JVM)/bin/java),)
export JAVA_HOME := $(NEXTFLOW_JVM)
endif
PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

OASIS_COMMIT := 00ab197550303e713343b68ea97daf36deb76946
OASIS_SHA256 := 8116c5c206051b254698448ed74611f64e6cc94fea11452622b544c818207125

# Pfam-A.hmm, Pfam 38 (file dated July 2025).
PFAM_HMM ?= /scratch/users/olgabot/data/pfam/Pfam-A.hmm

# Clones of their own, so this branch never moves the main Sherlock checkout under a run.
# The full run has its own clone too: a running head reads bin/ at task time, so the pilot's
# clone could not take later commits while the full run was going.
REMOTE_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis-nb357-run/nextflow-runs/clade-splash
PILOT_DIR ?= /scratch/users/olgabot/2024-kmerseek-analysis-nb357/nextflow-runs/clade-splash

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_TIME    ?= 3-00:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= nb357-splash
HEAD_JOBNAME ?= nf-nb357-splash
export NXF_OPTS := -Xms512m -Xmx6g

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head

.PHONY: test oasis pilot run watch stop pull-results pull-pilot running stub

oasis: $(MAKEFILE_DIR)/assets/OASIS.py
$(MAKEFILE_DIR)/assets/OASIS.py:
	curl -sfL https://raw.githubusercontent.com/refresh-bio/SPLASH/$(OASIS_COMMIT)/OASIS/OASIS.py -o $@.tmp
	echo "$(OASIS_SHA256)  $@.tmp" | shasum -a 256 -c - && mv $@.tmp $@

test: oasis
	$(PYTHON) -m pytest -q $(MAKEFILE_DIR)/tests

stub:
	mkdir -p run-stub
	cd run-stub && $(NEXTFLOW) run ../main.nf -stub-run -profile stub --pfam_hmm $(MAKEFILE_DIR)/tests/data/Pfam-A.hmm --outdir results --proteins_per_chunk 0 --fasta_cache $(MAKEFILE_DIR)/run-stub/fasta-cache

pilot: HEAD_TIME := 0-12:00:00
pilot: HEAD_SESSION := nb357-pilot
pilot: HEAD_JOBNAME := nf-nb357-pilot
pilot:
	mkdir -p run-pilot
	cd run-pilot && if command -v ml >/dev/null 2>&1; then ml load biology nextflow; fi; \
	  $(HEAD_SUBMIT) nextflow run ../main.nf -profile sherlock -resume --pfam_hmm $(PFAM_HMM) \
	    --max_per_class 8 --max_classes 6 --proteins_per_chunk 200000

run:
	mkdir -p run
	cd run && if command -v ml >/dev/null 2>&1; then ml load biology nextflow; fi; \
	  $(HEAD_SUBMIT) nextflow run ../main.nf -profile sherlock -resume --pfam_hmm $(PFAM_HMM)

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

# The notebook reads only these; the per-bucket tables stay on Sherlock.
RESULT_FILES := counts.tsv significant.parquet pfam_by_anchor.parquet \
                alignment_control.parquet top_occurrences.parquet residue_totals.json fetch_stats.tsv species/ candidates/ provenance/

pull-results:
	mkdir -p $(MAKEFILE_DIR)/../../tables/357
	cd $(MAKEFILE_DIR)/../../tables/357 && for f in $(RESULT_FILES); do \
	  rsync -av sherlock:$(REMOTE_DIR)/run/results/$$f ./$$f || exit 1; done

pull-pilot:
	mkdir -p $(MAKEFILE_DIR)/../../tables/357_pilot
	cd $(MAKEFILE_DIR)/../../tables/357_pilot && for f in $(RESULT_FILES); do \
	  rsync -av sherlock:$(PILOT_DIR)/run-pilot/results/$$f ./$$f || exit 1; done

running:
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
