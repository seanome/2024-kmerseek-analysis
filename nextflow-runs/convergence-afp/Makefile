# Convergence search kill test: ice-binding proteins across lineages.
#
# On the Mac:
#   make test               the whole pipeline on a 3_000-entry reference, 2 settings, Docker
#   make stage              copy Swiss-Prot 2026_03 to Sherlock through the transfer node
#   make build-image        build the kmerseek PR 141 image for Sherlock (linux/amd64)
#   make push-image         push it to Docker Hub (asks nothing; run it yourself)
#   make pull               copy the results back for the notebook
#
# On Sherlock (cd $SCRATCH/2024-kmerseek-convergence-afp/nextflow-runs/convergence-afp):
#   make plan                    print the memory and time each kmerseek setting will ask
#   make convergence_afp_pilot   every method, kmerseek on the pilot settings only
#   make convergence_afp         every method, all 230 kmerseek settings (resumes the pilot)
#   make running                 what is running

SHELL := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_DIR     := $(abspath $(MAKEFILE_DIR)/../..)
ON_CLUSTER   := $(shell [ -n "$$SCRATCH" ] && [ -d "$$SCRATCH" ] && echo yes)

MAC_PYTHON   := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
ifeq ($(ON_CLUSTER),yes)
PYTHON   ?= python3
NEXTFLOW ?= nextflow
PROFILE  ?= sherlock
else
PYTHON   ?= $(MAC_PYTHON)
NEXTFLOW ?= $(MAC_NEXTFLOW)
PROFILE  ?= standard
export NXF_JAVA_HOME ?= /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm
endif

RELEASE        := 2026_03
SPROT_MD5      := bc9d398533e6df582b563c6c03093bd0
MAC_SPROT      := /Users/olga/data/uniprot/uniprot_sprot.$(RELEASE).fasta.gz
SHERLOCK_USER_SCRATCH := /scratch/users/olgabot
SHERLOCK_CLONE := $(SHERLOCK_USER_SCRATCH)/2024-kmerseek-convergence-afp
SHERLOCK_DIR   := $(SHERLOCK_CLONE)/nextflow-runs/convergence-afp
SPROT          := $(MAKEFILE_DIR)/data/uniprot_sprot.$(RELEASE).fasta.gz

KMERSEEK_SRC   ?= $(HOME)/code/kmerseek
KMERSEEK_SHA   := 3c353d4f1552b95cf30fa1182718b3511c9e07c3
IMAGE          ?= docker.io/olgabot/kmerseek:0.4.0-pr141-3c353d4

RUN        ?= convergence-afp
LAUNCH_DIR := $(MAKEFILE_DIR)/runs/$(RUN)
OUTDIR     ?= $(MAKEFILE_DIR)/results
# The dark-set midi run's traces: its index tasks built these 230 settings' indexes, with
# the same E-value fit, on a reference the size of this one. Every clone on $SCRATCH is
# searched, and the plan prints which traces it read.
DARK_TRACES ?= $(wildcard $(SHERLOCK_USER_SCRATCH)/2024-kmerseek*/nextflow-runs/invertebrate-dark-set/runs/*/trace*.txt)
OWN_TRACES  = $(wildcard $(LAUNCH_DIR)/trace*.txt)
FLOOR_GB   ?= 64
FLOOR_H    ?= 4
PROSTT5    ?= $(SHERLOCK_USER_SCRATCH)/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/prostt5/weights

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns
HEAD_TIME ?= 7-00:00:00
HEAD_CPUS ?= 2
HEAD_MEM  ?= 16G
HEAD_XMX  ?= 12g
export NXF_OPTS := -Xms1g -Xmx$(HEAD_XMX)
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=normal HEAD_TIME=$(HEAD_TIME) HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=convergence HEAD_JOBNAME=nfdrv-$(RUN) \
              HEAD_HINT_WATCH="make watch" HEAD_HINT_STOP="make stop" \
              $(REPO_DIR)/scripts/nf-head

.PHONY: help assets test test-foldseek-db stage build-image push-image plan check-inputs \
        convergence_afp convergence_afp_pilot running watch stop pull

help:
	@sed -n '1,/^SHELL/p' $(firstword $(MAKEFILE_LIST)) | grep '^#' | sed 's/^# \{0,1\}//'

define require_cluster
	@if [ "$(ON_CLUSTER)" != "yes" ]; then \
	  echo "  this target runs ON SHERLOCK:"; \
	  echo "    ssh sherlock"; \
	  echo "    cd $(SHERLOCK_DIR) && make $@"; exit 1; fi
endef

# ---- inputs ---------------------------------------------------------------------------

# Rebuild the committed assets from their sources (Mac; see the script's docstring).
assets:
	cd $(REPO_DIR) && git -C $(KMERSEEK_SRC) show $(KMERSEEK_SHA):src/rust/alphabets.rs > /tmp/alphabets.$(KMERSEEK_SHA).rs && \
	  $(PYTHON) scripts/convergence_afp_assets.py \
	    --dark-results $(HOME)/data/invertebrate-dark-set/results-0.4/protein-evalues \
	    --alphabets-rs /tmp/alphabets.$(KMERSEEK_SHA).rs \
	    --alphabets-rs-source "kmerseek $(shell echo $(KMERSEEK_SHA) | cut -c1-7):src/rust/alphabets.rs" \
	    --dark-branch-commit ec91da5

# The data transfer node, never the login node, for the copy.
stage:
	@test -f $(MAC_SPROT) || { echo "missing $(MAC_SPROT)"; exit 1; }
	@test "$$(md5 -q $(MAC_SPROT))" = "$(SPROT_MD5)" || { echo "$(MAC_SPROT) is not the file the truth set was built against"; exit 1; }
	ssh sherlock-dtn "mkdir -p $(SHERLOCK_DIR)/data"
	rsync -av --partial $(MAC_SPROT) sherlock-dtn:$(SHERLOCK_DIR)/data/

check-inputs:
	@test -f $(SPROT) || { echo "missing $(SPROT): run 'make stage' on the Mac"; exit 1; }
	@test "$$(md5sum $(SPROT) | cut -d' ' -f1)" = "$(SPROT_MD5)" || { echo "$(SPROT) md5 is not $(SPROT_MD5)"; exit 1; }
	@echo "inputs ok: Swiss-Prot $(RELEASE), md5 $(SPROT_MD5)"

# Built from the region benchmark's Dockerfile with the kmerseek checkout as context, so
# python3, polars and numpy are in the image with the binary.
build-image:
	@d=$$(mktemp -d); git -C $(KMERSEEK_SRC) worktree add --detach $$d $(KMERSEEK_SHA) && \
	  docker build --platform linux/amd64 -f $(MAKEFILE_DIR)/../qfo-pfam-region-benchmark/Dockerfile \
	    --build-arg GIT_SHA=$(KMERSEEK_SHA) -t $(IMAGE) $$d; rc=$$?; \
	  git -C $(KMERSEEK_SRC) worktree remove --force $$d; exit $$rc
	docker run --rm --platform linux/amd64 $(IMAGE) sh -c 'cat /etc/kmerseek_git_sha; kmerseek --version; python3 -c "import polars, numpy; print(polars.__version__, numpy.__version__)"'

push-image:
	docker push $(IMAGE)

# ---- the laptop test ------------------------------------------------------------------

TEST_DIR      := $(MAKEFILE_DIR)/runs/test
TEST_IMAGE    ?= local/kmerseek:0.4.0-pr141-3c353d4-arm64
TEST_SETTINGS ?= hp_pbotc_1st_ed2.k19.s1.lcfalse.extend-c1.59,protein20.k5.s10.lcfalse.extend-c0.14
TEST_PROSTT5  ?= /Users/olga/data/disprot-region-transfer/prostt5/weights
MAC_FOLDSEEK  := /Users/olga/anaconda3/envs/foldseek-10.941cd33/bin/foldseek

# A target database of the six query models, standing in for AFDB Swiss-Prot.
test-foldseek-db:
	@mkdir -p $(TEST_DIR)/foldseek_db
	$(MAC_FOLDSEEK) createdb $(REPO_DIR)/data/convergence_afp/structures $(TEST_DIR)/foldseek_db/db
	awk '{print $$2}' $(TEST_DIR)/foldseek_db/db.lookup | sed -E 's/^AF-([A-Z0-9]+)-F[0-9]+.*/\1/' \
	  | sort -u > $(TEST_DIR)/foldseek_db/accessions.txt

test: test-foldseek-db
	cd $(TEST_DIR) && $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile standard -resume \
	  --swissprot_fasta $(MAC_SPROT) --subset 3000 --n_draws 2 \
	  --kmerseek_image $(TEST_IMAGE) --settings $(TEST_SETTINGS) \
	  --kmerseek_default_memory_gb 6 --kmerseek_default_time_h 2 \
	  --foldseek_target_db $(TEST_DIR)/foldseek_db --prostt5_weights $(TEST_PROSTT5) \
	  --outdir $(TEST_DIR)/results --run_tag test

# ---- Sherlock -------------------------------------------------------------------------

# Every setting's first ask, from this run's own traces (the pilot), the dark-set traces
# and its committed table. Written into the launch directory, never into a tracked file.
plan:
	$(call require_cluster)
	@mkdir -p $(LAUNCH_DIR)
	@echo "dark-set traces: $(words $(DARK_TRACES)) file(s)"; for f in $(DARK_TRACES); do echo "  $$f"; done
	@echo "this run's traces: $(words $(OWN_TRACES)) file(s)"
	$(PYTHON) $(MAKEFILE_DIR)/bin/plan_resources.py \
	  --settings $(MAKEFILE_DIR)/assets/kmerseek_settings.tsv \
	  --dark-measured $(MAKEFILE_DIR)/assets/dark_set_measured \
	  $(if $(DARK_TRACES),--dark-trace $(DARK_TRACES),) \
	  $(if $(OWN_TRACES),--trace $(OWN_TRACES),) \
	  --floor-gb $(FLOOR_GB) --floor-h $(FLOOR_H) --out $(LAUNCH_DIR)/resources.tsv

NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
  --swissprot_fasta $(SPROT) --kmerseek_image $(IMAGE) --outdir $(OUTDIR) \
  --resources $(LAUNCH_DIR)/resources.tsv \
  $(if $(wildcard $(PROSTT5)),--prostt5_weights $(PROSTT5),) --run_tag $(RUN)

convergence_afp_pilot: check-inputs plan
	cd $(LAUNCH_DIR) && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_RUN) --settings pilot $(NF_ARGS)

convergence_afp: check-inputs plan
	cd $(LAUNCH_DIR) && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_RUN) --settings all $(NF_ARGS)

running:
	@python3 $(REPO_DIR)/scripts/nf-running --jobs "$(MAKEFILE_DIR)"

watch:
	$(call require_cluster)
	@j=$$(squeue -u $$USER -h -n nfdrv-$(RUN) -o %i | head -1); test -n "$$j" || { echo "no nfdrv-$(RUN) job"; exit 1; }; \
	  sock=$$(cat $(LAUNCH_DIR)/.nf-head.sock); \
	  srun --jobid=$$j --overlap --pty tmux -S "$$sock" attach -r -t convergence

stop:
	$(call require_cluster)
	scancel -n nfdrv-$(RUN)

# Results back to the Mac: the ranked lists only (tens of MB), through the transfer node.
pull:
	mkdir -p $(HOME)/data/convergence-afp
	rsync -av sherlock-dtn:$(SHERLOCK_DIR)/results/{inputs,hits} $(HOME)/data/convergence-afp/
