# Notebook 361, human exercise series (GSE99963, GSE144304, GSE120862). Runs on Sherlock.
#
#   make pilot      gse99963_sed_post_vs_pre only (14 runs): the sizing check
#   make run        every contrast of every series (SERIES=... to limit)
#   make swaps      18 swapped-label runs of GSE99963 and GSE144304, in run-swaps/
#   make watch      attach to the live Nextflow table (read-only); Ctrl-b d to leave
#   make watch-swaps  the same for the swapped-label run
#   make stop       ask the run to cancel its jobs and exit
#   make running    every Nextflow run in the queue, then this folder's jobs
#
# The Nextflow head runs in its own batch job (scripts/nf-head), never on a login node.

PIPELINE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_DIR     := $(abspath $(PIPELINE_DIR)/../..)
CODE_DIR     := $(REPO_DIR)/analysis/pec-domen2026-splash
PEC_ROOT     ?= $(SCRATCH)/nb361-pec
OUT_ROOT     ?= $(PEC_ROOT)/human
# the 108 GSE99963 + GSE144304 runs downloaded and MD5-checked on 2026-11-02 for the read search
REUSE_DIR    ?= $(PEC_ROOT)/sra_search/reads
SERIES       ?= GSE99963,GSE144304,GSE120862
RUN_DIR      ?= run

SLURM_ACCOUNT   ?= ayeletv
# the head needs up to 4 days; the normal partition refuses that (QOSMaxWallDurationPerJobLimit)
SLURM_PARTITION ?= hns
HEAD_QOS     ?= normal
HEAD_TIME    ?= 4-00:00:00
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 16G
HEAD_SESSION ?= pec361h
HEAD_JOBNAME ?= nf-pec361-human
export NXF_OPTS := -Xms1g -Xmx12g

HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(REPO_DIR)/scripts/nf-head

comma := ,
NF_COMMON = nextflow run $(PIPELINE_DIR)/main.nf -profile sherlock -resume -with-trace -with-report -with-timeline \
            --pec_root $(PEC_ROOT) --out_root $(OUT_ROOT) --code_dir $(CODE_DIR) --reuse_dir $(REUSE_DIR)

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

.PHONY: pilot run fig4 swaps watch watch-swaps attach stop status running

pilot: HEAD_TIME := 1-00:00:00
pilot:
	$(call require_cluster)
	mkdir -p run-pilot && cd run-pilot && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(NF_COMMON) --series GSE99963 --contrasts gse99963_sed_post_vs_pre $(NF_ARGS)

run:
	$(call require_cluster)
	mkdir -p $(RUN_DIR) && cd $(RUN_DIR) && ml load biology nextflow && $(HEAD_SUBMIT) $(NF_COMMON) --series $(SERIES) $(NF_ARGS)

## Domen et al. 2026 Fig. 4 groups; every route, so they compare with every human panel
fig4: HEAD_JOBNAME := nf-pec361-human-fig4
fig4: HEAD_SESSION := pec361hf
fig4:
	$(call require_cluster)
	mkdir -p run-fig4 && cd run-fig4 && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(subst -profile sherlock,-profile sherlock$(comma)fig4,$(NF_COMMON)) \
	  --series GSE99963,GSE144304,GSE120862 \
	  --contrasts gse99963_fig4_post_vs_pre,gse144304_fig4_fit_vs_young,gse120862_fig4_trained_vs_untrained \
	  --routes compactors,compactors_p05,kamrat_ttest,kamrat_p05,kamrat_deseq2_p05 $(NF_ARGS)

swaps: HEAD_JOBNAME := nf-pec361-human-swaps
swaps: HEAD_SESSION := pec361hs
swaps:
	$(call require_cluster)
	mkdir -p run-swaps && cd run-swaps && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(subst -profile sherlock,-profile sherlock$(comma)label_swap,$(NF_COMMON)) \
	  --series GSE99963,GSE144304 --label_swap $(NF_ARGS)

watch-swaps: HEAD_JOBNAME := nf-pec361-human-swaps
watch-swaps: HEAD_SESSION := pec361hs
watch-swaps: watch

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

## Every Nextflow run with jobs in the SLURM queue, then one row per job of the runs under this
## folder. Read TIME as squeue's [DD-[HH:]]MM:SS.
running:
	$(call require_cluster)
	@python3 $(REPO_DIR)/scripts/nf-running --jobs "$(PIPELINE_DIR)"
