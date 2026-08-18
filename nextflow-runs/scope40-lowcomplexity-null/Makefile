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
