# Time-split test for notebook 280: would kmerseek have placed Pfam domains before Pfam
# did? t0 = Pfam 33.1 (May 2020), t1 = Pfam 38.2 (June 2026). Runs on Sherlock.
#
#   make download        fetch Pfam 33.1, 38.2, 38.0 pfamseq and Swiss-Prot 2019_08 (batch job)
#   make stop1           build the query set, count every step, identity of 500 instances
#   make inputs          Stop 2/3 search inputs: reference, settings, pilot and full query sets
#   make smoke           2 queries x 1 alphabet-k through every tool, to check the wiring
#   make pilot           Stop 2: 200 sample queries + 20 positive controls + decoys, every tool
#   make score-partial   score the pilot searches finished so far (tune half only)
#   make full            Stop 3: the 8_329 approved queries + decoys, every tool
#   make seed-reference  second reference: Pfam 33.1 seed members, with counts.json
#   make hash-check OLD=<commit>   smoke-size run of OLD, then of this checkout with -resume,
#                        in hashcheck/: every task hash the two share is listed side by side
#   make seed-pilot      Stop B: the pilot's queries against the seed reference
#   make seed-self-check queries already in a Pfam 33.1 seed, and the instances they cover
#   make plddt           AlphaFold DB models of every real query; mean pLDDT per instance
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
# The seed reference: its own reference dir, index tag, index cache and outdirs, so no
# index or output of the Swiss-Prot runs can be read as a seed one. hmmscan is read from
# the Swiss-Prot run of the same queries.
SEED      = pfamseed33_1
NF_SEED   = $(MAKEFILE_DIR)/main.nf -profile sherlock \
            --reference_dir $(RUN_DIR)/inputs/reference-$(SEED) --reference_tag $(SEED) \
            --pfam_hmm $(RUN_DIR)/data/Pfam33.1/Pfam-A.hmm.gz \
            --index_cache $(RUN_DIR)/kmerseek_index_$(SEED) --hmm_cache $(RUN_DIR)/hmm

# The head runs on hns alone: the normal partition caps a job at 2 days, and with
# "hns,normal" sbatch refuses a 7-day head (QOSMaxWallDurationPerJobLimit), as the dark set found.
HEAD = HEAD_ACCOUNT=ayeletv HEAD_PARTITION=hns HEAD_QOS=normal HEAD_TIME=7-00:00:00 \
       HEAD_CPUS=2 HEAD_MEM=16G HEAD_HINT_WATCH="tmux attach" HEAD_HINT_STOP="scancel"

.PHONY: download stop1 controls inputs smoke pilot score-partial full seed-reference hash-check seed-pilot seed-self-check plddt running

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

# Score the pilot searches that have finished, while the rest still run. Reads what the
# pilot published to results-pilot/ (kmerseek search outputs and their stats files, MMseqs2,
# phmmer, hmmscan) and writes results-pilot/scores-partial/. The test half is dropped, as
# in the pilot's own scoring step. Sizing: the scorer holds one search output at a time;
# the largest pilot file is 1.1 GB gzipped (~27 million rows). The first try (64 GB, job
# 47666740) read all 185 files and was killed holding every decoy-family pair; those now
# go to disk. 100 GB fits a normal node (128,000 MB; 128G does not). The log prints peak
# memory per step: read it before changing the ask.
score-partial:
	$(SB) --job-name=280-score-partial --cpus-per-task=4 --mem=100G --time=4:00:00 \
	  --wrap 'set -e; R=$(RUN_DIR)/results-pilot; O=$$R/scores-partial; command rm -rf $$O; mkdir -p $$O; \
	  $(PY) $(MAKEFILE_DIR)/bin/score_transfers.py --kmerseek $$R/kmerseek --stats $$R/kmerseek \
	    --mmseqs2 $$R/mmseqs2 --phmmer $$R/phmmer --hmmscan $$R/hmmscan --inputs $(RUN_DIR)/inputs/pilot \
	    --reference $(RUN_DIR)/inputs/reference --settings $(RUN_DIR)/inputs/settings.tsv --out $$O'

full:
	cd $(RUN_DIR) && mkdir -p launch-full && cd launch-full && \
	  $(HEAD) HEAD_SESSION=nf280full HEAD_JOBNAME=nfdrv-280-full \
	  $(MAKEFILE_DIR)/../../scripts/nf-head bash -lc 'ml load biology nextflow && nextflow run $(NF_COMMON) \
	    --settings $(RUN_DIR)/inputs/settings.tsv --inputs $(RUN_DIR)/inputs/full --outdir $(RUN_DIR)/results-full --run_tag full --read_test_half true -resume'

# Seed members of every Pfam 33.1 family, gaps removed, one entry per distinct sequence.
# A member that is a query, or lies inside one (any queries.fasta of any run, decoys
# included), is dropped and counted. Sizing: the seed file is 145 MB gzipped, 1.24 million
# members; the query k-mer index of 16_658 sequences holds about 7 million 20-mers.
seed-reference:
	$(SB) --job-name=280-seed-reference --cpus-per-task=2 --mem=32G --time=2:00:00 \
	  --wrap 'set -e; O=inputs/reference-$(SEED); mkdir -p $$O; \
	  $(PY) $(MAKEFILE_DIR)/bin/make_search_inputs.py reference-pfam-seed --seed data/Pfam33.1/Pfam-A.seed.gz \
	    --sprot inputs/reference/reference.fasta \
	    --queries inputs/smoke/queries.fasta inputs/pilot/queries.fasta inputs/full/queries.fasta --out $$O; \
	  cat inputs/reference/pfam_clans_t1.tsv > $$O/pfam_clans_t1.tsv'

# Do the Swiss-Prot tasks keep their hashes under this checkout? Runs the smoke set (2
# queries + decoys, hp_pbotc_1st_ed2 k29, real Swiss-Prot reference) with the code of
# OLD, then with this checkout's HEAD and -resume, in one launch dir with its own index
# cache, so nothing of the pilot is read or locked. Checks out OLD in THIS clone for the
# first run: use it only in a clone no run depends on.
hash-check:
	@test -n "$(OLD)" || { echo "make hash-check OLD=<commit>"; exit 1; }
	cd $(MAKEFILE_DIR) && test -z "$$(git status --porcelain --untracked-files=no)"
	$(SB) --job-name=nfdrv-280-hashcheck --cpus-per-task=2 --mem=8G --time=6:00:00 --partition=hns \
	  --wrap 'set -e; ml load biology nextflow; NEW=$$(cd $(MAKEFILE_DIR) && git rev-parse HEAD); \
	  H=$(RUN_DIR)/hashcheck; mkdir -p $$H; cd $$H; \
	  printf "alphabet\tksize\tc_opt\nhp_pbotc_1st_ed2\t29\t1.59\n" > settings.tsv; \
	  ARGS="$(subst --index_cache $(RUN_DIR)/kmerseek_index,--index_cache $(RUN_DIR)/hashcheck/kmerseek_index,$(NF_COMMON)) --settings settings.tsv --inputs $(RUN_DIR)/inputs/smoke --outdir $$H/results"; \
	  (cd $(MAKEFILE_DIR) && git checkout -q $(OLD)); nextflow run $$ARGS --run_tag old; \
	  (cd $(MAKEFILE_DIR) && git checkout -q $$NEW); nextflow run $$ARGS --run_tag new -resume; \
	  echo "old=$(OLD) new=$$NEW"'

seed-self-check:
	$(SB) --job-name=280-seed-self-check --cpus-per-task=1 --mem=16G --time=1:00:00 \
	  --wrap '$(PY) $(MAKEFILE_DIR)/bin/seed_self_check.py --run-dir $(RUN_DIR) --out $(RUN_DIR)/inputs/reference-$(SEED)/self_check'

seed-pilot:
	cd $(RUN_DIR) && mkdir -p launch-pilot-$(SEED) && cd launch-pilot-$(SEED) && \
	  $(HEAD) HEAD_SESSION=nf280seedpilot HEAD_JOBNAME=nfdrv-280-seed-pilot \
	  $(MAKEFILE_DIR)/../../scripts/nf-head bash -lc 'ml load biology nextflow && nextflow run $(NF_SEED) \
	    --settings $(RUN_DIR)/inputs/settings.tsv --inputs $(RUN_DIR)/inputs/pilot --outdir $(RUN_DIR)/results-pilot-$(SEED) \
	    --hmmscan_dir $(RUN_DIR)/results-pilot/hmmscan --run_tag seed-pilot --queue_size 100 -resume'

# Mean pLDDT of each instance's query residues inside its t1 domain, from the query's
# AlphaFold DB model (one file per accession, fetched by the QfO benchmark's script; 8_349
# real queries, 4 connections). A model of a different sequence counts as no model.
plddt:
	$(SB) --job-name=280-plddt --cpus-per-task=4 --mem=8G --time=12:00:00 \
	  --wrap 'set -e; P=inputs/plddt; mkdir -p $$P/acc $$P/structures; \
	  cut -f1 inputs/pilot/queries.tsv inputs/full/queries.tsv | grep -v -e "^DECOY_" -e "^query$$" | sort -u > $$P/acc/timesplit.accessions; \
	  FLAT_CACHE=$$P/no-cache PARALLEL=4 ALLOW_LARGE_PER_ACCESSION=1 \
	    bash $(MAKEFILE_DIR)/../qfo-pfam-region-benchmark/bin/fetch_alphafold_structures.sh $$P/structures $$P/acc timesplit; \
	  $(PY) $(MAKEFILE_DIR)/bin/instance_plddt.py --structures $$P/structures/timesplit \
	    --inputs inputs/pilot inputs/full --out $$P/instance_plddt.tsv'

## Read TIME carefully. squeue formats it [DD-[HH:]]MM:SS, so `4:03` is four minutes.
running:
	@python3 $(MAKEFILE_DIR)/../../scripts/nf-running --jobs "$(MAKEFILE_DIR)"
