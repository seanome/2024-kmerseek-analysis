# Notebook 361: reads per gene with no genome, for the Fig. 4C-D score. Runs on Sherlock.
#
#   make pilot    the first 1/1000 and 1/100 of each run's reads: checks the pipeline, sizes the rest
#   make run      all four shares of reads (1/1000, 1/100, 1/10, all); finished ones are reused
#   make watch    attach to the live Nextflow table (read-only); Ctrl-b d to leave
#   make running  every Nextflow run in the queue, then this folder's jobs
#
# The Nextflow head runs in its own batch job (scripts/nf-head), never on a login node.

PIPELINE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_DIR     := $(abspath $(PIPELINE_DIR)/../..)
CODE_DIR     := $(REPO_DIR)/analysis/pec-domen2026-splash
PEC_ROOT     ?= $(SCRATCH)/nb361-pec
OUT_ROOT     ?= $(PEC_ROOT)/human/fig4_score
SHARES       ?= 1000,100,10,1

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= normal,hns
HEAD_QOS     ?= normal
HEAD_TIME    ?= 12:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_SESSION ?= pec361fs
HEAD_JOBNAME ?= nf-pec361-fig4-score
export NXF_OPTS := -Xms1g -Xmx6g

HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(REPO_DIR)/scripts/nf-head

NF_COMMON = nextflow run $(PIPELINE_DIR)/main.nf -profile sherlock -resume \
            --pec_root $(PEC_ROOT) --out_root $(OUT_ROOT) --code_dir $(CODE_DIR)

define require_cluster
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  This runs ON SHERLOCK, not on your Mac:"; \
	  echo "    ssh sherlock"; \
	  echo "    cd $(PIPELINE_DIR)"; \
	  echo ""; \
	  exit 1; \
	fi
endef

.PHONY: pilot run watch running

pilot: HEAD_TIME := 2:00:00
pilot:
	$(call require_cluster)
	mkdir -p run && cd run && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_COMMON) --shares 1000,100 $(NF_ARGS)

run:
	$(call require_cluster)
	mkdir -p run && cd run && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_COMMON) --shares $(SHARES) $(NF_ARGS)

watch:
	$(call require_cluster)
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 if [ -z "$$jid" ]; then echo "no $(HEAD_JOBNAME) job is RUNNING:"; squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "    %i %T %r"; exit 1; fi; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

running:
	$(call require_cluster)
	@python3 $(REPO_DIR)/scripts/nf-running --jobs "$(PIPELINE_DIR)"
