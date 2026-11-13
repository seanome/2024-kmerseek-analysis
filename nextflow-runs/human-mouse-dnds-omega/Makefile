# human-mouse-dnds-omega/Makefile
#
# Real per-gene omega (pairwise dN/dS) for notebook 206's mechanism test.
#
# One-time setup:
#   1. make build          — builds the omega-tools image (mafft + paml + biopython)
#   2. Run notebook 206 section 8 up through "Preparing gene pairs for the omega
#      pipeline" — it writes 206_omega_gene_pairs.csv to DATA_DIR.
#   3. make test            — 5 gene pairs, validates the codeml .mlc parser
#   4. make run              — full gene set from the notebook's CSV
#
# NOT YET RUN as of writing — see main.nf header. Start with `make test`.

NF           := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
IMAGE        := human-mouse-dnds-omega:1.0
DATA_DIR     := $(HOME)/data/gencode/results-human-mouse-orthologs
GENE_PAIRS   := $(DATA_DIR)/206_omega_gene_pairs.csv
OUTDIR       := results

.PHONY: build test run clean

build:
	docker build -f docker/Dockerfile.omega -t $(IMAGE) .

test: build
	$(NF) run main.nf -profile test \
	    --gene_pairs $(GENE_PAIRS) \
	    --omega_container $(IMAGE) \
	    --outdir results-test \
	    -resume

run: build
	$(NF) run main.nf -profile standard \
	    --gene_pairs $(GENE_PAIRS) \
	    --omega_container $(IMAGE) \
	    --outdir $(OUTDIR) \
	    -resume

clean:
	/bin/rm -rf work .nextflow* results-test

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
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/human-mouse-dnds-omega"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
