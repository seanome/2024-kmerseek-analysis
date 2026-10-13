# Time-split test for notebook 280: would kmerseek have placed Pfam domains before Pfam
# did? t0 = Pfam 33.1 (May 2020), t1 = Pfam 38.2 (June 2026). Runs on Sherlock.
#
#   make download        fetch Pfam 33.1, 38.2, 38.0 pfamseq and Swiss-Prot 2019_08 (batch job)
#   make stop1           build the query set, count every step, identity of 500 instances
#   make inputs          Stop 2/3 search inputs: reference, settings, pilot and full query sets
#   make smoke           2 queries x 1 alphabet-k through every tool, to check the wiring
#   make pilot           Stop 2: 200 sample queries + 20 positive controls + decoys, every tool
#   make full            Stop 3: the 8_329 approved queries + decoys, every tool
#   make running         what is running on this account
#
# Code is read from this checkout; data, work dirs and outputs go to RUN_DIR. Nothing is
# shared with the dark-set run.
SHELL        := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
RUN_DIR      ?= $(SCRATCH)/280-time-split
SBATCH_ENV    = --chdir=$(RUN_DIR) --export=ALL,TIME_SPLIT_BIN=$(MAKEFILE_DIR)/bin
PY            = apptainer exec --bind /scratch $(RUN_DIR)/images/kmerseek-0.4.0-rc5.sif python3
SB            = sbatch --account=ayeletv --partition=hns,normal --chdir=$(RUN_DIR) --output=%x-%j.log

NF_COMMON = $(MAKEFILE_DIR)/main.nf -profile sherlock \
            --reference_dir $(RUN_DIR)/inputs/reference \
            --pfam_hmm $(RUN_DIR)/data/Pfam33.1/Pfam-A.hmm.gz \
            --index_cache $(RUN_DIR)/kmerseek_index --hmm_cache $(RUN_DIR)/hmm

# The head runs on hns alone: the normal partition caps a job at 2 days, and with
# "hns,normal" sbatch refuses a 7-day head (QOSMaxWallDurationPerJobLimit), as the dark set found.
HEAD = HEAD_ACCOUNT=ayeletv HEAD_PARTITION=hns HEAD_QOS=normal HEAD_TIME=7-00:00:00 \
       HEAD_CPUS=2 HEAD_MEM=16G HEAD_HINT_WATCH="tmux attach" HEAD_HINT_STOP="scancel"

.PHONY: download stop1 controls inputs smoke pilot full running

download:
	@mkdir -p $(RUN_DIR)/data
	sbatch $(SBATCH_ENV) $(MAKEFILE_DIR)/bin/download.sbatch

stop1:
	sbatch $(SBATCH_ENV) $(MAKEFILE_DIR)/bin/stop1.sbatch

# Identity of 3_000 more random instances, to find positive controls (>= 90% identical to
# their closest t0 Swiss-Prot domain over >= 80% of the t1 domain).
controls:
	$(SB) --job-name=280-controls --cpus-per-task=8 --mem=32G --time=4:00:00 \
	  --wrap 'set -e; W=$(RUN_DIR)/controls_identity; mkdir -p $$W/m8; \
	  $(PY) $(MAKEFILE_DIR)/bin/stop1_identity.py prepare --qs query_set --work $$W --n 3000 --seed 282; \
	  while IFS=$$'"'"'\t'"'"' read -r i fam; do [ -s $$W/m8/$$i.m8 ] && continue; \
	    apptainer exec --bind /scratch images/fasta3-36.3.8i.sif ssearch36 -q -m 8CB -s BL62 -f -11 -g -1 \
	      -E 1000000 -b 5 -d 0 -T 8 $$W/q/$$i.fa $$W/lib/$$fam.fa > $$W/m8/$$i.m8.tmp && mv $$W/m8/$$i.m8.tmp $$W/m8/$$i.m8; \
	  done < $$W/jobs.tsv; \
	  $(PY) $(MAKEFILE_DIR)/bin/stop1_identity.py collect --qs query_set --work $$W --no-examples'

inputs:
	$(SB) --job-name=280-inputs --cpus-per-task=4 --mem=64G --time=2:00:00 \
	  --wrap 'set -e; B=$(MAKEFILE_DIR); A=$$B/assets; \
	  $(PY) $$B/bin/make_search_inputs.py reference --qs query_set --out inputs/reference; \
	  $(PY) $$B/bin/make_search_inputs.py settings --ranges $$A/kmerseek_ksize_range_per_alphabet.tsv --kappa $$A/kappa_by_alphabet.tsv --out inputs; \
	  $(PY) $$B/bin/make_search_inputs.py queries --qs query_set --sample query_set/proposed_sample_queries.csv --n-pilot 200 --controls controls_identity/identity_sample.csv --n-controls 20 --out inputs/pilot; \
	  $(PY) $$B/bin/make_search_inputs.py queries --qs query_set --sample query_set/proposed_sample_queries.csv --out inputs/full; \
	  $(PY) $$B/bin/make_search_inputs.py queries --qs query_set --sample query_set/proposed_sample_queries.csv --n-pilot 2 --out inputs/smoke'

# Wiring check: two sample queries and their decoys, one cheap alphabet-k.
smoke:
	cd $(RUN_DIR) && mkdir -p launch-smoke && cd launch-smoke && \
	  printf 'alphabet\tksize\tc_opt\nhp_pbotc_1st_ed2\t29\t1.59\n' > settings.smoke.tsv && \
	  $(subst HEAD_TIME=7-00:00:00,HEAD_TIME=8:00:00,$(HEAD)) HEAD_SESSION=nf280smoke HEAD_JOBNAME=nfdrv-280-smoke \
	  $(MAKEFILE_DIR)/../../scripts/nf-head bash -lc 'ml load biology nextflow && nextflow run $(NF_COMMON) --settings settings.smoke.tsv \
	    --inputs $(RUN_DIR)/inputs/smoke --outdir $(RUN_DIR)/results-smoke --run_tag smoke -resume'

pilot:
	cd $(RUN_DIR) && mkdir -p launch-pilot && cd launch-pilot && \
	  $(HEAD) HEAD_SESSION=nf280pilot HEAD_JOBNAME=nfdrv-280-pilot \
	  $(MAKEFILE_DIR)/../../scripts/nf-head bash -lc 'ml load biology nextflow && nextflow run $(NF_COMMON) \
	    --settings $(RUN_DIR)/inputs/settings.tsv --inputs $(RUN_DIR)/inputs/pilot --outdir $(RUN_DIR)/results-pilot --run_tag pilot --queue_size 100 -resume'

full:
	cd $(RUN_DIR) && mkdir -p launch-full && cd launch-full && \
	  $(HEAD) HEAD_SESSION=nf280full HEAD_JOBNAME=nfdrv-280-full \
	  $(MAKEFILE_DIR)/../../scripts/nf-head bash -lc 'ml load biology nextflow && nextflow run $(NF_COMMON) \
	    --settings $(RUN_DIR)/inputs/settings.tsv --inputs $(RUN_DIR)/inputs/full --outdir $(RUN_DIR)/results-full --run_tag full --read_test_half true -resume'

## Read TIME carefully. squeue formats it [DD-[HH:]]MM:SS, so `4:03` is four minutes.
running:
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
