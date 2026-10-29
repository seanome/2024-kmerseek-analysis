# nf-core/rnaseq 3.27.0, pre-processing only, on the Botryllus germline young-vs-old reads.
# Run on Sherlock from this directory. The Nextflow head runs as a SLURM job (scripts/nf-head).
#
#   make test    2 samples x 200_000 read pairs, checks the setup end to end (about 40 min)
#   make run     all 18 samples
#   make watch [HEAD_JOBNAME=nf-botryllus-rnaseq-test HEAD_SESSION=botryllus-rnaseq-test]
#                attach to the running head's live task table (detach: Ctrl-b d)
#   make stop    ask the running head to shut down cleanly

SHELL := /bin/bash

PIPELINE  := nf-core/rnaseq
REVISION  := 3.27.0
# The pipeline requires Nextflow >= 25.10.4. Sherlock's newest module is 25.04.7, a
# self-contained build that ignores NXF_VER, so the 25.10.4 self-contained build is fetched
# from the Nextflow release page. The module is still loaded, for its Java 18.
NXF_RELEASE := 25.10.4
NEXTFLOW    := $(SCRATCH)/bin/nextflow-$(NXF_RELEASE)-dist
export NXF_OPTS := -Xms512m -Xmx4g

OUTROOT  ?= $(SCRATCH)/botryllus-rnaseq-preprocess
TESTROOT ?= $(SCRATCH)/botryllus-rnaseq-preprocess-test
TEST_READS ?= 200000

SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns,normal
HEAD_QOS     ?= normal
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 8G
HEAD_TIME    ?= 2-00:00:00
HEAD_SESSION ?= botryllus-rnaseq
HEAD_JOBNAME ?= nf-botryllus-rnaseq

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              $(MAKEFILE_DIR)/../../scripts/nf-head

NF_RUN = $(NEXTFLOW) run $(PIPELINE) -r $(REVISION) -resume \
           -c $(MAKEFILE_DIR)/sherlock.config -params-file $(MAKEFILE_DIR)/params.yaml

.PHONY: test run watch stop

$(NEXTFLOW):
	mkdir -p $(dir $@)
	curl -fsSL -o $@.part https://github.com/nextflow-io/nextflow/releases/download/v$(NXF_RELEASE)/nextflow-$(NXF_RELEASE)-dist
	chmod +x $@.part && mv $@.part $@

# First TEST_READS pairs of the deepest and the shallowest sample.
test: HEAD_TIME    = 06:00:00
test: HEAD_JOBNAME = nf-botryllus-rnaseq-test
test: HEAD_SESSION = botryllus-rnaseq-test
test: $(NEXTFLOW)
	mkdir -p $(TESTROOT)/reads
	set -e; \
	 echo "sample,fastq_1,fastq_2,strandedness" > $(TESTROOT)/samplesheet.csv; \
	 grep -E '^(TL3_GSC3_old|TL14_STD2_young),' $(MAKEFILE_DIR)/samplesheet.csv | \
	 while IFS=, read -r s r1 r2 strand; do \
	   zcat "$$r1" | head -n $$(( 4 * $(TEST_READS) )) | gzip > $(TESTROOT)/reads/$${s}_R1.fastq.gz; \
	   zcat "$$r2" | head -n $$(( 4 * $(TEST_READS) )) | gzip > $(TESTROOT)/reads/$${s}_R2.fastq.gz; \
	   echo "$$s,$(TESTROOT)/reads/$${s}_R1.fastq.gz,$(TESTROOT)/reads/$${s}_R2.fastq.gz,$$strand"; \
	 done >> $(TESTROOT)/samplesheet.csv
	cd $(TESTROOT) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(NF_RUN) --input $(TESTROOT)/samplesheet.csv --outdir $(TESTROOT)/results

run: $(NEXTFLOW)
	mkdir -p $(OUTROOT)
	cd $(OUTROOT) && ml load biology nextflow && \
	  $(HEAD_SUBMIT) $(NF_RUN) --input $(MAKEFILE_DIR)/samplesheet.csv --outdir $(OUTROOT)/results

watch:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; squeue -u $$USER -h -n $(HEAD_JOBNAME); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

stop:
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(HEAD_JOBNAME) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"
