PYTHON    := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
REPO_ROOT := $(abspath ../..)
AF2_CACHE := $(HOME)/data/alphafold_structures
AF2_OUTDIR := $(REPO_ROOT)/results/pfam_benchmark/alphafold_structures/all_species
PFAM_HMM_DIR := $(HOME)/data/pfam
PFAM_HMM := $(PFAM_HMM_DIR)/Pfam-A.hmm
HMMER_IMAGE := quay.io/biocontainers/hmmer:3.4--hdbdd923_1

.PHONY: all fetch-envelope-coords build-architectures fetch-diamond stratify-pairs ksize-table fetch-alphafold-structures fetch-pfam-hmm help

all: fetch-envelope-coords build-architectures fetch-diamond stratify-pairs ksize-table

# ── Step 1: fetch Pfam envelope coordinates ─────────────────────────────────
# UniProt's own API doesn't expose Pfam domain positions (verified against
# TP53/P04637: zero Domain-type features, xref_pfam has no start/end).
# Downloads (resumable) Pfam-A.regions.tsv.gz from EBI (~4.7 GB) and filters
# it to our 10 species' QfO accessions.
#
# Long, network-bound, stall-prone: run it yourself and watch it.
# Safe to rerun this exact target if it stalls — curl -C - resumes.
fetch-envelope-coords:
	cd $(REPO_ROOT) && $(PYTHON) notebooks/fetch_pfam_envelope_coords.py

# ── Step 2: rebuild domain/architecture parquets ────────────────────────────
# Fast, local, no network. Requires Step 1's output.
build-architectures:
	cd $(REPO_ROOT) && $(PYTHON) notebooks/build_pfam_architectures.py --species all --force

# ── Step 3: fresh DIAMOND --ultra-sensitive (E<=10), all 9 species ──────────
# NOT the pfam-benchmark-tools output — that's verified broken (empty for
# 6/9 species, a container bug). ~10-15 min, real HSPs, run directly.
fetch-diamond:
	cd $(REPO_ROOT) && zsh scripts/run_diamond_fresh.sh

# ── Step 4: hard/easy stratification + full NW identity + 20% holdout ──────
# Fast (~3.5 min for all 9 species, parallelized). Requires steps 1-3.
stratify-pairs:
	cd $(REPO_ROOT) && $(PYTHON) notebooks/construct_pair_strata.py --species all --workers 8

# ── Step 5: region_length - k + 1 diagnostic table (k=6..30) ───────────────
# Fast, local. Requires step 1 (envelope coordinates).
ksize-table:
	cd $(REPO_ROOT) && $(PYTHON) notebooks/pfam_envelope_ksize_table.py

# ── Step 6: AlphaFold structures for FoldSeek (targeted, not full proteomes)─
# ~55,527 unique accessions across the stratified pair set, avg ~777 KB
# each -> ~43 GB total. This is a LONG job (likely 1-2+ hours of individual
# HTTPS requests) — run it yourself and watch it.
# Resumable: already-downloaded accessions are cached and skipped on rerun,
# so if it's interrupted just rerun this exact target.
# Requires step 4 (defines which accessions are needed).
fetch-alphafold-structures:
	cd $(REPO_ROOT) && $(PYTHON) nextflow-runs/alphafold-foldseek-336/bin/download_af2_structures.py \
		results/pfam_benchmark/pairs_stratified/all_accessions_for_af2.tsv \
		--outdir $(AF2_OUTDIR) \
		--cache $(AF2_CACHE) \
		--max-workers 15

# ── Step 7: Pfam-A.hmm for hmmscan ──────────────────────────────────────────
# ~399 MB compressed. Downloads + gunzips + hmmpress'es (via the hmmer 3.4
# biocontainer, no local hmmer binary on this machine). Once this exists at
# ~/data/pfam/Pfam-A.hmm, pfam-benchmark-tools/main.nf's hmmscan step
# auto-activates (params.run_hmmscan = file(params.pfam_hmm).exists()).
# Moderate size — safe to run directly, still resumable (-C -) if it stalls.
fetch-pfam-hmm:
	mkdir -p $(PFAM_HMM_DIR)
	if [ -f $(PFAM_HMM).h3p ]; then \
		echo "Already pressed: $(PFAM_HMM)"; \
	else \
		curl -L -C - -o $(PFAM_HMM_DIR)/Pfam-A.hmm.gz https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz && \
		gunzip -f $(PFAM_HMM_DIR)/Pfam-A.hmm.gz && \
		docker run --rm --entrypoint "" -v $(PFAM_HMM_DIR):/data $(HMMER_IMAGE) hmmpress /data/Pfam-A.hmm; \
	fi

help:
	@echo "Targets (run in order for a from-scratch build):"
	@echo "  make fetch-envelope-coords       Download + filter Pfam-A regions bulk file (long — run & watch)"
	@echo "  make build-architectures         Rebuild per-species domain/architecture parquets (fast, needs 1)"
	@echo "  make fetch-diamond               Fresh DIAMOND --ultra-sensitive, all 9 species (~10-15 min)"
	@echo "  make stratify-pairs              Hard/easy stratification + NW identity + 20% holdout (~3.5 min)"
	@echo "  make ksize-table                 region_length-k+1 diagnostic table, k=6..30 (fast)"
	@echo "  make fetch-alphafold-structures  Targeted AF structure pull for FoldSeek (long — run & watch, ~43GB)"
	@echo "  make fetch-pfam-hmm              Pfam-A.hmm + hmmpress for hmmscan (~399MB, moderate)"
	@echo "  make all                         Steps 1,2,3,4,5 in order (NOT structures/hmm — run those separately)"
