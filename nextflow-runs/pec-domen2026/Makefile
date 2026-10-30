# Notebook 361 pipeline (Domen et al. 2026 PEC RNA-seq without a genome). Runs on Sherlock.
#
#   make run        every contrast + the known-answer control
#   make run-test   young_132d + the control only (most results already exist; a wiring check)
#   make watch      attach to the live Nextflow table (read-only); Ctrl-b d to leave
#   make stop       ask the run to cancel its jobs and exit
#   make running    every Nextflow run in the queue, then this folder's jobs
#
# The Nextflow head runs in its own batch job (scripts/nf-head), never on a login node.

PIPELINE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_DIR     := $(abspath $(PIPELINE_DIR)/../..)
CODE_DIR     := $(REPO_DIR)/analysis/pec-domen2026-splash
PEC_ROOT     ?= $(SCRATCH)/nb361-pec
# other run's verified FASTQs; download_run.py re-checks each MD5 before hard-linking
REUSE_DIR    ?= $(SCRATCH)/botryllus-shazam-PEC-20261023/raw

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns
HEAD_QOS     ?= normal
HEAD_TIME    ?= 4-00:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 16G
HEAD_SESSION ?= pec361
HEAD_JOBNAME ?= nf-pec361
export NXF_OPTS := -Xms1g -Xmx12g

HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(REPO_DIR)/scripts/nf-head

NF_COMMON = nextflow run $(PIPELINE_DIR)/main.nf -profile sherlock -resume \
            --pec_root $(PEC_ROOT) --code_dir $(CODE_DIR) --reuse_dir $(REUSE_DIR)

define require_cluster
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  This runs ON SHERLOCK, not on your Mac:"; \
	  echo "    ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis-nb361/nextflow-runs/pec-domen2026"; \
	  echo ""; \
	  exit 1; \
	fi
endef

.PHONY: run run-test watch attach stop status running

run:
	$(call require_cluster)
	mkdir -p run && cd run && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_COMMON) $(NF_ARGS)

run-test: HEAD_TIME := 6:00:00
run-test:
	$(call require_cluster)
	mkdir -p run-test && cd run-test && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(NF_COMMON) --contrasts young_132d $(NF_ARGS)

watch: HEAD_ATTACH := -r
attach: HEAD_ATTACH :=
watch attach:
	$(call require_cluster)
	@jid="$(JOBID)"; \
	 if [ -z "$$jid" ]; then jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); fi; \
	 if [ -z "$$jid" ]; then echo "no $(HEAD_JOBNAME) job is RUNNING:"; squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "    %i %T %r"; exit 1; fi; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach $(HEAD_ATTACH) -t $(HEAD_SESSION)

## Ctrl-C to Nextflow (via the head's trap), so it cancels its own jobs instead of orphaning them.
stop:
	$(call require_cluster)
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 if [ -z "$$jid" ]; then echo "no $(HEAD_JOBNAME) job is running"; exit 1; fi; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down; check with: make running"

status:
	$(call require_cluster)
	squeue -u $$USER -o "%.12i %.32j %.8T %.10M %R" | grep -E "JOBID|nf-|pec361"

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, from any pipeline, then one row per job
## of the runs under this folder. scripts/nf-running does the work. Read TIME as squeue's
## [DD-[HH:]]MM:SS: `4:03` is four minutes and `1-02:19:10` is a day.
running:
	$(call require_cluster)
	@python3 $(REPO_DIR)/scripts/nf-running --jobs "$(PIPELINE_DIR)"
