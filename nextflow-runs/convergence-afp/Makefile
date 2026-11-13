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
#   make convergence_afp_after_pilot   queue convergence_afp to start when the pilot ends cleanly
#   make convergence_afp_and_score  one job: the searches, then the scoring
#   make score                   turn the ranked lists into the notebook's tables
#   make running                 what is running
#   make adhesive_q25460         the adhesive case study: mussel foot protein 1 (Q25460)
#                                as the one query, every method, all 230 settings
#   make watch RUN=adhesive-q25460, make stop RUN=adhesive-q25460

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
# The tables of raised index asks the dark-set launches measured from their logs.
DARK_MEASURED ?= $(dir $(wildcard $(SHERLOCK_USER_SCRATCH)/2024-kmerseek*/nextflow-runs/invertebrate-dark-set/runs/*/measured/kmerseek_index_memory_measured.tsv))
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
        convergence_afp convergence_afp_pilot convergence_afp_after_pilot cancel-after-pilot \
        score pull-scored convergence_afp_and_score \
        running watch stop pull adhesive_q25460

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
	  --dark-measured $(MAKEFILE_DIR)/assets/dark_set_measured $(DARK_MEASURED) \
	  $(if $(DARK_TRACES),--dark-trace $(DARK_TRACES),) \
	  $(if $(OWN_TRACES),--trace $(OWN_TRACES),) \
	  --floor-gb $(FLOOR_GB) --floor-h $(FLOOR_H) --out $(LAUNCH_DIR)/resources.tsv

# `ml` is a shell function from Lmod, missing in a batch job's shell unless exported; this
# loads Lmod first when it is absent, the way scripts/nf-head does.
LOAD_NEXTFLOW = { command -v ml >/dev/null 2>&1 || . /share/software/user/open/lmod/lmod/init/bash; } && ml load biology nextflow

NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
  --swissprot_fasta $(SPROT) --kmerseek_image $(IMAGE) --outdir $(OUTDIR) \
  --resources $(LAUNCH_DIR)/resources.tsv \
  $(if $(wildcard $(PROSTT5)),--prostt5_weights $(PROSTT5),) --run_tag $(RUN)

# A second driver in the same launch folder cannot run (Nextflow's session lock refuses it)
# but still overwrites the running driver's .nf-head.* files, which `make watch` and
# convergence_afp_after_pilot read. So refuse before submitting. Added 2026-10-14, after a
# second `make convergence_afp` during the pilot did exactly that.
define refuse_second_driver
	@j=$$(squeue -u $$USER -h -n nfdrv-$(RUN) -t PENDING,RUNNING,CONFIGURING -o %i | head -1); if [ -n "$$j" ]; then \
	  echo "nfdrv-$(RUN) is already queued or running as job $$j; not submitting a second driver."; \
	  echo "To start the full run when it ends cleanly: make convergence_afp_after_pilot"; exit 1; fi
endef

convergence_afp_pilot: check-inputs plan
	$(refuse_second_driver)
	cd $(LAUNCH_DIR) && $(LOAD_NEXTFLOW) && $(HEAD_SUBMIT) $(NF_RUN) --settings pilot $(NF_ARGS)

convergence_afp: check-inputs plan
	$(refuse_second_driver)
	cd $(LAUNCH_DIR) && $(LOAD_NEXTFLOW) && $(HEAD_SUBMIT) $(NF_RUN) --settings all $(NF_ARGS)

# Queue the full run behind the pilot's driver job. A small job waits for the pilot's head
# to exit with status 0 (afterok: a pilot that failed or was stopped does not start the full
# run), then runs `make convergence_afp` here: `make plan` then reads the pilot's trace, and
# the new head resumes the pilot's launch directory. If the pilot head is cancelled the
# waiting job is removed too (--kill-on-invalid-dep). Cancel it with `make cancel-after-pilot`.
PILOT_JOBID ?= $(shell squeue -u $$USER -h -n nfdrv-$(RUN) -o %i 2>/dev/null | head -1)
convergence_afp_after_pilot:
	$(call require_cluster)
	@test -n "$(PILOT_JOBID)" || { echo "no nfdrv-$(RUN) job in the queue to wait for"; exit 1; }
	@squeue -h -j $(PILOT_JOBID) -o '%i %j %t' | grep -q nfdrv-$(RUN) || \
	  { echo "job $(PILOT_JOBID) is not a queued or running nfdrv-$(RUN)"; exit 1; }
	sbatch --parsable --job-name=nfnext-$(RUN) --account=$(SLURM_ACCOUNT) --partition=$(SLURM_PARTITION) \
	  --qos=normal --time=01:00:00 --cpus-per-task=1 --mem=4G \
	  --dependency=afterok:$(PILOT_JOBID) --kill-on-invalid-dep=yes \
	  --output=$(LAUNCH_DIR)/nfnext.%j.log \
	  --wrap "cd $(MAKEFILE_DIR) && NF_HEAD_DETACH=1 make convergence_afp"
	@echo "queued after pilot head $(PILOT_JOBID); check with: squeue -u $$USER -n nfnext-$(RUN)"

cancel-after-pilot:
	scancel -n nfnext-$(RUN)

NF_SCORE = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -entry score -resume \
  --kmerseek_image $(IMAGE) --outdir $(OUTDIR) --run_tag $(RUN)-score

# Scoring only: reads the published hits, writes results/scored. Minutes, one small job,
# so it runs in the foreground rather than through a driver job.
score: check-inputs
	$(call require_cluster)
	@mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && $(LOAD_NEXTFLOW) && $(NF_SCORE) $(NF_ARGS)

# One driver job that runs the searches and then the scoring, so there is nothing to
# submit by hand in between. tools/run-and-score splits the two command lines and stops
# before scoring if the searches fail. Everything already searched is a cache hit, so
# after a change to the scans or the scoring this costs only what actually changed.
convergence_afp_and_score: check-inputs plan
	$(refuse_second_driver)
	cd $(LAUNCH_DIR) && $(LOAD_NEXTFLOW) && $(HEAD_SUBMIT) \
	  $(MAKEFILE_DIR)/tools/run-and-score $(NF_RUN) --settings all $(NF_ARGS) -- $(NF_SCORE)

# ---- adhesive case study: mussel foot protein 1 as the one query -----------------------

# The same pipeline with --truth_dir pointed at the adhesive class
# (scripts/convergence_adhesive_q25460_inputs.py fills it): Q25460 and its 20 null A and
# 20 null B draws against Swiss-Prot 2026_03 plus the 3 truth entries that are only in
# TrEMBL (2 sandcastle worm cement proteins, 1 barnacle). The reference is the same size
# as the ice-binding run's, so each setting's first ask is planned from that run's
# traces. Foldseek reuses that run's AFDB Swiss-Prot database. The scoring entry is
# written for the ice-binding lineages, so this run stops at the ranked lists.
ADH_RUN     := adhesive-q25460
ADH_TRUTH   := $(REPO_DIR)/data/convergence_adhesive
ADH_REF     := $(MAKEFILE_DIR)/data/uniprot_sprot.$(RELEASE).plus_adhesive_trembl.fasta
AFP_LAUNCH  := $(MAKEFILE_DIR)/runs/convergence-afp
AFP_FOLDSEEK_DB := $(MAKEFILE_DIR)/results/db_cache/foldseek/afdb_swissprot

# Target-specific values, so `plan` and the head-job wiring see this run, not the
# ice-binding one. LAUNCH_DIR is set with :=, so it is overridden here too.
adhesive_q25460: RUN = $(ADH_RUN)
adhesive_q25460: LAUNCH_DIR = $(MAKEFILE_DIR)/runs/$(ADH_RUN)
adhesive_q25460: OUTDIR = $(MAKEFILE_DIR)/results-$(ADH_RUN)
adhesive_q25460: OWN_TRACES = $(wildcard $(AFP_LAUNCH)/trace*.txt)

# Built once: a rebuilt file has a new mtime, and Nextflow would then redo every task.
$(ADH_REF): $(SPROT) $(ADH_TRUTH)/extra_reference.fasta
	{ gzip -dc $(SPROT); cat $(ADH_TRUTH)/extra_reference.fasta; } > $@.tmp && mv -f $@.tmp $@
	@echo "$@: $$(grep -c '>' $@) entries"

ADH_NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
  --swissprot_fasta $(ADH_REF) --truth_dir $(ADH_TRUTH) --kmerseek_image $(IMAGE) \
  --outdir $(OUTDIR) --resources $(LAUNCH_DIR)/resources.tsv --settings all \
  --foldseek_target_db $(AFP_FOLDSEEK_DB) \
  $(if $(wildcard $(PROSTT5)),--prostt5_weights $(PROSTT5),) --run_tag $(RUN)

adhesive_q25460: check-inputs $(ADH_REF) plan
	@test -f $(ADH_TRUTH)/queries.tsv || { echo "missing $(ADH_TRUTH)/queries.tsv"; exit 1; }
	@test -f $(AFP_FOLDSEEK_DB)/db || { echo "missing $(AFP_FOLDSEEK_DB)/db (built by the ice-binding run)"; exit 1; }
	$(refuse_second_driver)
	cd $(LAUNCH_DIR) && $(LOAD_NEXTFLOW) && $(HEAD_SUBMIT) $(ADH_NF_RUN) $(NF_ARGS)

# The scored tables back to the Mac for the notebook: a few MB, not the 1.2 GB of hits.
pull-scored:
	mkdir -p $(HOME)/data/convergence-afp
	rsync -av sherlock-dtn:$(SHERLOCK_DIR)/results/scored $(HOME)/data/convergence-afp/

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
