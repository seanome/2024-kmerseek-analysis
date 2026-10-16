NF          := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
PDBSTYLE_URL := http://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz
TARBALL      := $(HOME)/Downloads/pdbstyle-sel-gs-bib-40-2.08.tgz
PDBSTYLE_DIR := $(HOME)/data/scope/pdbstyle-2.08

.PHONY: all run resume extract check-structures help

all: check-structures run

# ── Step 1: download ────────────────────────────────────────────────────────
# The SCOPe server has bot-protection that blocks curl.
# Download the tarball manually in a browser, then run 'make extract':
#
#   $(PDBSTYLE_URL)
#
# The tarball (~1.0 GB) will land in ~/Downloads by default.

# ── Step 2: extract ─────────────────────────────────────────────────────────
extract: $(TARBALL)
	mkdir -p $(PDBSTYLE_DIR)
	tar -xzf $(TARBALL) -C $(PDBSTYLE_DIR) --strip-components=1
	@echo "Extracted to $(PDBSTYLE_DIR)"
	@echo "Structure count: $$(find $(PDBSTYLE_DIR) -name '*.ent' | wc -l | tr -d ' ') .ent files"

$(TARBALL):
	@echo ""
	@echo "  Tarball not found at $(TARBALL)"
	@echo "  SCOPe blocks curl with bot-protection. Download it manually:"
	@echo ""
	@echo "    $(PDBSTYLE_URL)"
	@echo ""
	@echo "  Save to $(TARBALL), then re-run 'make extract'."
	@echo ""
	@exit 1

# ── Step 3: verify structures are in place ──────────────────────────────────
check-structures:
	@n=$$(find $(PDBSTYLE_DIR) -name '*.ent' 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$n" -lt 15000 ]; then \
	    echo "ERROR: only $$n .ent files found in $(PDBSTYLE_DIR) (expected ~15,177)"; \
	    echo "Run 'make extract' first."; \
	    exit 1; \
	fi
	@echo "OK: $$(find $(PDBSTYLE_DIR) -name '*.ent' | wc -l | tr -d ' ') pdbstyle structures found"

# ── Step 4: run pipeline ────────────────────────────────────────────────────
run: check-structures
	$(NF) run main.nf

resume:
	$(NF) run main.nf -resume

help:
	@echo "Targets:"
	@echo "  make extract          Extract pdbstyle tarball from ~/Downloads"
	@echo "  make check-structures Verify all ~15,177 .ent files are present"
	@echo "  make run              Run full Nextflow pipeline (FoldSeek + MMseqs2)"
	@echo "  make resume           Resume a previous pipeline run"
	@echo "  make running          On Sherlock: every Nextflow run in the queue, and this folder's jobs"
	@echo ""
	@echo "Download the tarball manually (bot-protected URL):"
	@echo "  $(PDBSTYLE_URL)"

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
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/scope40-foldseek-pdb"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
