# scope40-lowcomplexity-null/Makefile
#
# Null distribution for calibrating kmerseek's low-complexity k-mer mask.
# 10 dipeptide-preserving shuffles of SCOPe40 x 6 HP alphabets x k=22,24,26,28,
# plus the unshuffled set through the identical path for the true-positive cost.

NF          := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
PYTHON      := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
KMERSEEK    := $(HOME)/code/kmerseek/target/release/kmerseek
SCOPE_FASTA := $(HOME)/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
OUTDIR      := $(HOME)/data/scope/results-lowcomplexity-null

.PHONY: check test run resume clean-work clean-test

# Preflight: the two binaries this pipeline shells out to must both exist, and
# kmerseek must be new enough to emit the moltype_seq column the analysis reads.
check:
	@test -x $(KMERSEEK) || { echo "MISSING: $(KMERSEEK) — build with 'cargo build --release' in ~/code/kmerseek"; exit 1; }
	@test -x $(PYTHON)   || { echo "MISSING: $(PYTHON)"; exit 1; }
	@test -f $(SCOPE_FASTA) || { echo "MISSING: $(SCOPE_FASTA)"; exit 1; }
	@echo "kmerseek: $$($(KMERSEEK) --version)"
	@$(PYTHON) -c "import polars, numpy; print('polars', polars.__version__, '| numpy', numpy.__version__)"
	@$(PYTHON) -c "import sys; sys.path.insert(0, 'bin'); import hp_alphabets; print('HP tables OK:', len(hp_alphabets.HP_TABLES), 'alphabets')"
	@echo "preflight OK"

# 300 sequences, 2 shuffles, 2 alphabets, 2 k-sizes — exercises every process.
test: check
	$(NF) run main.nf -profile test,local -resume

# Full run: 11 seqsets x 6 alphabets x 4 ksizes = 264 index+search tasks.
run: check
	$(NF) run main.nf \
	    -profile local \
	    --fasta  $(SCOPE_FASTA) \
	    --outdir $(OUTDIR) \
	    -resume

resume:
	$(NF) run main.nf -profile local --fasta $(SCOPE_FASTA) --outdir $(OUTDIR) -resume

clean-work:
	/bin/rm -rf work .nextflow .nextflow.log*

clean-test:
	/bin/rm -rf results-test

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, from any pipeline, then one row per job
## of the runs under this folder. scripts/nf-running does the work. Runs on Sherlock only,
## since it reads the queue. Read TIME as squeue's [DD-[HH:]]MM:SS: `4:03` is four minutes
## and `1-02:19:10` is a day. PENDING jobs show 0:00.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

.PHONY: running
running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/scope40-lowcomplexity-null"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
