# Short disulfide-rich peptides by cysteine framework.
#
# `pilot` runs on this Mac and proves the pipeline end to end. Every `run-*` target runs
# ON SHERLOCK (ssh sherlock; cd $SCRATCH/2024-kmerseek-analysis/nextflow-runs/cys-frameworks).
#
# A pilot's numbers are not the experiment's numbers: its reference holds every toxin but
# only a sample of everything else, so precision comes out better than it is.

SHELL := /bin/bash

NF     := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
# Nextflow needs a JVM, and the one in the nf-core-v2 environment is under lib/jvm.
NXF_JAVA_HOME := /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm
PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# kmerseek branch olgabot/cys-runs-040 at c39370c: v0.4.0 + PR 141 (an index entry keyed by
# its sequence) + cys2 (PR 147) + cys_hp3 (PR 148). The key fix has to be in, or a cys2
# index silently drops every target that shares a k-mer set with one already stored.
KMERSEEK_SHA   ?= c39370c4e8d3b1b89cee726945bb0aa1bf90a8a3
KMERSEEK_SRC   ?= /Users/olga/code/kmerseek-cys-runs
KMERSEEK_BIN   ?= $(KMERSEEK_SRC)/target/release
HMMER_BIN      ?= /Users/olga/anaconda3/envs/hmmer/bin
# The bin/ scripts start with `#!/usr/bin/env python3`, so the environment that has polars
# goes on PATH for a local run and the container's python3 answers on Sherlock. One
# mechanism, not two.
PYTHON_BIN     ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin

# IMAGE_TAG is the one place the image is named. Every run target passes
# --pipeline_image $(IMAGE_REMOTE), so the tag here is what runs even though
# nextflow.config also declares a default: two strings that must be edited together, where
# disagreeing produces results rather than an error, is the shape that costs a week.
IMAGE_TAG      ?= 2026-10-15-cys-frameworks-c39370c
IMAGE_LOCAL    := kmerseek-cys:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)

# On Sherlock: make DATA=$$SCRATCH/cys-frameworks run-phylum
DATA      ?= $(HOME)/data/cys-frameworks
SWISSPROT ?= $(DATA)/uniprot_sprot.fasta.gz
TABLES    := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../tables/cys_frameworks

.PHONY: help check-kmerseek build-image push-image data pilot pilot-docker run-phylum run-genus running

help:
	@echo ""
	@echo "  make build-image  build the amd64 image from the pinned kmerseek commit"
	@echo "  make push-image   build it and push to $(IMAGE_REMOTE)"
	@echo "  make data         download ConoServer, Swiss-Prot and the NCBI taxonomy, then"
	@echo "                    build the query, reference and settings tables"
	@echo "  make pilot        10 queries per framework, one clade, a sampled reference;"
	@echo "                    runs here, proves the pipeline end to end"
	@echo "  make run-phylum   ON SHERLOCK: 3 held-out references (one per phylum)"
	@echo "  make run-genus    ON SHERLOCK: one per genus with >= 20 queries"
	@echo "  make running      ON SHERLOCK: what is in the queue"
	@echo ""

# kmerseek is pinned by commit, and this checks the binary was built from that commit
# rather than trusting the path. A cys2 index built by anything older than PR 141 drops
# targets silently, which is the one failure that would not show up as an error.
check-kmerseek:
	@test -x $(KMERSEEK_BIN)/kmerseek || { \
	  echo "No kmerseek at $(KMERSEEK_BIN)/kmerseek."; \
	  echo "Build it:"; \
	  echo "  cd $(KMERSEEK_SRC)"; \
	  echo "  export SDKROOT=\$$(xcrun --show-sdk-path)"; \
	  echo "  export BINDGEN_EXTRA_CLANG_ARGS=\"-isysroot \$$SDKROOT\""; \
	  echo "  cargo build --release --bins"; \
	  exit 1; }
	@have=$$(git -C $(KMERSEEK_SRC) rev-parse HEAD); \
	 if [ "$$have" != "$(KMERSEEK_SHA)" ]; then \
	   echo "$(KMERSEEK_SRC) is at $$have, not the pinned $(KMERSEEK_SHA)."; \
	   echo "Check it out and rebuild, or pass KMERSEEK_SHA=$$have if that is deliberate."; \
	   exit 1; \
	 fi
	@echo "kmerseek $(KMERSEEK_SHA) at $(KMERSEEK_BIN)/kmerseek"

# The build context is the kmerseek worktree, not this repo, and the Dockerfile COPYs that
# worktree, so the image holds whatever commit is checked out there. check-kmerseek is a
# dependency so the tag cannot name a commit the image does not contain.
build-image: check-kmerseek
	docker build --platform linux/amd64 \
	  -t $(IMAGE_LOCAL) -t $(IMAGE_REMOTE) \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/Dockerfile \
	  $(KMERSEEK_SRC)
	@echo ""
	@echo "kmerseek commit inside the image:"
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) cat /etc/kmerseek_git_sha
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) kmerseek --version
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) \
	  python3 -c "import polars; print('polars', polars.__version__)"

push-image: build-image
	docker push $(IMAGE_REMOTE)
	@echo ""
	@echo "Pin this digest in nextflow.config if you want the run to name an immutable image:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(IMAGE_REMOTE)

data:
	$(PYTHON) $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../scripts/download_cys_framework_data.py $(DATA)
	$(PYTHON) $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../scripts/build_cys_framework_data.py $(DATA) $(TABLES)

pilot: check-kmerseek
	mkdir -p pilot
	cd pilot && \
	  PATH="$(KMERSEEK_BIN):$(HMMER_BIN):$(PYTHON_BIN):$$PATH" \
	  NXF_JAVA_HOME=$(NXF_JAVA_HOME) \
	  $(NF) run ../main.nf -profile standard -resume \
	    --swissprot_fasta $(SWISSPROT) \
	    --level phylum \
	    --only Cnidaria \
	    --sample_non_toxins 20000 \
	    --queries_per_framework 10 \
	    --outdir $(PWD)/pilot/results

# On Sherlock, nextflow comes from the module and the data from $$SCRATCH, so DATA and NF
# are overridden there rather than hard-coded to Mac paths.
run-phylum:
	mkdir -p run-phylum
	cd run-phylum && \
	  ml load biology nextflow && \
	  nextflow run ../main.nf -profile sherlock -resume \
	    --pipeline_image $(IMAGE_REMOTE) \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level phylum \
	    --outdir $(CURDIR)/run-phylum/results

run-genus:
	mkdir -p run-genus
	cd run-genus && \
	  ml load biology nextflow && \
	  nextflow run ../main.nf -profile sherlock -resume \
	    --pipeline_image $(IMAGE_REMOTE) \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level genus \
	    --genus_min_queries 20 \
	    --outdir $(CURDIR)/run-genus/results

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, then one row per job of the runs under
## this folder. Runs on Sherlock only, since it reads the queue.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/cys-frameworks"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
