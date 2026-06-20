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
	@echo ""
	@echo "Download the tarball manually (bot-protected URL):"
	@echo "  $(PDBSTYLE_URL)"
