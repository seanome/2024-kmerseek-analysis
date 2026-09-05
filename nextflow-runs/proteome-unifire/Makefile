# UniFIRE over the non-vertebrate query proteomes.
#
# Not a Nextflow pipeline -- it is a container invocation. It lives under nextflow-runs/
# anyway so every benchmark run in this repo is in one place and follows the same Sherlock
# conventions. (EBI does publish a Nextflow wrapper image,
# dockerhub.ebi.ac.uk/uniprot-public/unifire/nextflow:0.1.0, if that ever becomes the
# easier path; this Makefile drives the plain image.)
#
# What UniFIRE is, and what it is NOT here. UniFIRE applies UniProt's own annotation rule
# systems -- UniRule, ARBA and PIRSR -- to private sequences. It is the machine that
# generates UniProt's automatic annotations, which makes it two things at once:
#
#   the deployment baseline    what someone with a new bivalve proteome actually runs
#                              today is InterProScan + UniRule. That is the bar kmerseek
#                              has to clear to be interesting, and it is a much harder bar
#                              than phmmer.
#   NOT a source of truth      its output is rule-based inference, circular with every
#                              profile arm and with the Swiss-Prot key itself. Nothing
#                              from here may be scored as an answer key.
#
# The measurement this exists for: how many domains in a non-vertebrate proteome get no
# UniRule annotation at all. UniRule conditions are taxon-scoped and thin outside
# well-studied clades, so that number is the proteome-annotate premise stated as a
# measurement rather than an assertion.
#
# ---------------------------------------------------------------------------------------
# Usage
#   make prepare-fasta SPECIES=botryllus       # add OX= headers -- REQUIRED before anything
#   make pull-unifire                          # LOGIN NODE. lite ~4 GB, full ~60 GB
#   make run-unifire-mini SPECIES=botryllus    # 200 proteins through the full image
#   make run-unifire      SPECIES=botryllus    # full proteome
#   make run-unifire-lite SPECIES=botryllus IPRSCAN_XML=/path/from/the/interproscan/run
# ---------------------------------------------------------------------------------------

SHELL := /bin/bash

# Tags resolved from the project's own container registry on 2026-09-05
# (gitlab.ebi.ac.uk/api/v4/projects/1200/registry/repositories). 2026.3 is the newest
# non-SNAPSHOT, non-dev tag. Override rather than editing a guess in.
UNIFIRE_VERSION ?= 2026.3
UNIFIRE_REGISTRY ?= dockerhub.ebi.ac.uk/uniprot-public/unifire
UNIFIRE_IMAGE   := $(UNIFIRE_REGISTRY):$(UNIFIRE_VERSION)
UNIFIRE_LITE    := $(UNIFIRE_REGISTRY):$(UNIFIRE_VERSION)-lite

SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
OUTDIR      ?= $(SCRATCH_DIR)/proteome-unifire/results
WORKDIR     ?= $(SCRATCH_DIR)/proteome-unifire/work
SIF_DIR     ?= $(SCRATCH_DIR)/containers

QFO_DIR         ?= $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
BOTRYLLUS_FASTA ?= $(QFO_DIR)/Eukaryota/BOTSCH2026_30301.fasta
CGIGAS_FASTA    ?= $(SCRATCH_DIR)/proteomes/UP000005408_29159.fasta
PYTHON          ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# UniFIRE asks for 24 GB minimum. The full image runs InterProScan inside it, which is the
# part that actually wants the memory and the hours.
MEM       ?= 48G
CPUS      ?= 16
TIME      ?= 24:00:00
PARTITION ?= normal
MINI_N    ?= 200

SPECIES ?=
ifeq ($(SPECIES),botryllus)
  SRC_FASTA := $(BOTRYLLUS_FASTA)
  TAXID     := 30301
  ORGANISM  := Botryllus schlosseri
else ifeq ($(SPECIES),cgigas)
  SRC_FASTA := $(CGIGAS_FASTA)
  TAXID     := 29159
  ORGANISM  := Magallana gigas
else
  SRC_FASTA :=
  TAXID     :=
  ORGANISM  :=
endif

PREPARED := $(WORKDIR)/$(SPECIES)/$(SPECIES)_unifire.fasta

.PHONY: help check-species prepare-fasta pull-unifire run-unifire run-unifire-mini \
        run-unifire-lite summarise queue-empty

help:
	@echo "UniFIRE over the non-vertebrate query proteomes"
	@echo
	@echo "  prepare-fasta      SPECIES=<botryllus|cgigas>  add OX= headers (REQUIRED FIRST)"
	@echo "  pull-unifire       LOGIN NODE: pull $(UNIFIRE_VERSION) (lite ~4 GB, full ~60 GB)"
	@echo "  run-unifire-mini   SPECIES=<...>  $(MINI_N) proteins"
	@echo "  run-unifire        SPECIES=<...>  full proteome, full image"
	@echo "  run-unifire-lite   SPECIES=<...> IPRSCAN_XML=<path>  reuse existing IPS output"
	@echo "  summarise          SPECIES=<...>  predictions -> parquet + the coverage number"
	@echo
	@echo "  UNIFIRE_IMAGE=$(UNIFIRE_IMAGE)"
	@echo "  OUTDIR=$(OUTDIR)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi
	@if [ -z "$(SRC_FASTA)" ]; then \
	  echo "unknown SPECIES '$(SPECIES)'. Known: botryllus, cgigas"; exit 1; fi
	@if [ ! -f "$(SRC_FASTA)" ]; then \
	  echo "no proteome for $(SPECIES) at $(SRC_FASTA)"; \
	  if [ "$(SPECIES)" = "cgigas" ]; then \
	    echo "  stage it from the interproscan run: make stage-cgigas"; fi; exit 1; fi

# UniRule conditions are taxon-scoped. Without OX= most rules cannot fire and the run
# finishes with a near-empty prediction set that reads as a biological result rather than a
# missing flag, so this is a hard prerequisite of every run target below.
prepare-fasta: check-species
	@mkdir -p $(WORKDIR)/$(SPECIES)
	$(PYTHON) bin/prepare_unifire_fasta.py \
	  --in "$(SRC_FASTA)" --out "$(PREPARED)" \
	  --taxid $(TAXID) --organism "$(ORGANISM)"

# LOGIN NODE. Compute nodes have no outbound internet (established 2026-08-20 by the
# ProstT5 weights job), so a pull from a compute node hangs rather than failing fast.
# --platform linux/amd64 is not optional when the pull happens on an Apple Silicon laptop:
# the arm64 manifest runs nowhere on the cluster. A stale .sif from a bad pull is not
# re-fetched by apptainer, so it is removed rather than reused.
pull-unifire:
	@mkdir -p $(SIF_DIR)
	@if command -v apptainer >/dev/null 2>&1; then \
	  echo "==> apptainer pull $(UNIFIRE_LITE)"; \
	  /bin/rm -f $(SIF_DIR)/unifire-$(UNIFIRE_VERSION)-lite.sif; \
	  apptainer pull $(SIF_DIR)/unifire-$(UNIFIRE_VERSION)-lite.sif docker://$(UNIFIRE_LITE); \
	  echo "==> apptainer pull $(UNIFIRE_IMAGE) (~60 GB, includes InterProScan)"; \
	  /bin/rm -f $(SIF_DIR)/unifire-$(UNIFIRE_VERSION).sif; \
	  apptainer pull $(SIF_DIR)/unifire-$(UNIFIRE_VERSION).sif docker://$(UNIFIRE_IMAGE); \
	elif command -v docker >/dev/null 2>&1; then \
	  echo "==> docker pull (amd64) $(UNIFIRE_LITE)"; \
	  docker pull --platform linux/amd64 $(UNIFIRE_LITE); \
	  echo "note: the full 60 GB image is not pulled on a laptop by default."; \
	  echo "      run this on Sherlock, or: docker pull --platform linux/amd64 $(UNIFIRE_IMAGE)"; \
	else \
	  echo "neither apptainer nor docker found"; exit 1; \
	fi

define UNIFIRE_RUN
	@mkdir -p $(2)
	@if command -v apptainer >/dev/null 2>&1; then \
	  apptainer run --bind $(dir $(1)):/volume --bind $(2):/out \
	    $(SIF_DIR)/unifire-$(UNIFIRE_VERSION)$(3).sif \
	    -i /volume/$(notdir $(1)) $(4) -o /out; \
	else \
	  docker run --rm --platform linux/amd64 \
	    --mount type=bind,source=$(dir $(1)),target=/volume \
	    --mount type=bind,source=$(2),target=/out \
	    $(UNIFIRE_REGISTRY):$(UNIFIRE_VERSION)$(3) \
	    -i /volume/$(notdir $(1)) $(4) -o /out; \
	fi
endef

run-unifire-mini: check-species prepare-fasta
	@mkdir -p $(WORKDIR)/$(SPECIES)-mini $(OUTDIR)/$(SPECIES)-mini
	@head -n $$(( $(MINI_N) * 2 )) $(PREPARED) > $(WORKDIR)/$(SPECIES)-mini/mini.fasta
	@echo "==> $(SPECIES): $$(grep -c '^>' $(WORKDIR)/$(SPECIES)-mini/mini.fasta) proteins through the full image"
	$(call UNIFIRE_RUN,$(WORKDIR)/$(SPECIES)-mini/mini.fasta,$(OUTDIR)/$(SPECIES)-mini,,)
	@$(MAKE) summarise SPECIES=$(SPECIES) RESULT_DIR=$(OUTDIR)/$(SPECIES)-mini \
	    SUMMARY_FASTA=$(WORKDIR)/$(SPECIES)-mini/mini.fasta

run-unifire: check-species queue-empty prepare-fasta
	@mkdir -p $(OUTDIR)/$(SPECIES) $(OUTDIR)/$(SPECIES)/logs
	@echo "==> $(SPECIES): full proteome, full image (InterProScan runs inside it)"
	sbatch --wait --job-name=unifire-$(SPECIES) \
	  --cpus-per-task=$(CPUS) --mem=$(MEM) --time=$(TIME) --partition=$(PARTITION) \
	  --output=$(OUTDIR)/$(SPECIES)/logs/%j.out \
	  --wrap='apptainer run --bind $(dir $(PREPARED)):/volume --bind $(OUTDIR)/$(SPECIES):/out \
	    $(SIF_DIR)/unifire-$(UNIFIRE_VERSION).sif -i /volume/$(notdir $(PREPARED)) -o /out'
	@$(MAKE) summarise SPECIES=$(SPECIES) RESULT_DIR=$(OUTDIR)/$(SPECIES)

# The cheap path: InterProScan has already run in the sibling proteome-interproscan tree,
# so the lite image (4 GB, no InterProScan inside) reuses that XML instead of spending the
# hours again. The XML must have come from the PREPARED fasta -- annotate the original and
# the taxid never reaches UniFIRE.
run-unifire-lite: check-species
	@if [ -z "$(IPRSCAN_XML)" ]; then \
	  echo "IPRSCAN_XML is required:"; \
	  echo "  make run-unifire-lite SPECIES=$(SPECIES) IPRSCAN_XML=\$$SCRATCH/proteome-interproscan/results/$(SPECIES)/$(SPECIES).xml"; \
	  exit 1; fi
	@if [ ! -s "$(IPRSCAN_XML)" ]; then echo "no XML at $(IPRSCAN_XML)"; exit 1; fi
	@mkdir -p $(OUTDIR)/$(SPECIES)-lite
	$(call UNIFIRE_RUN,$(IPRSCAN_XML),$(OUTDIR)/$(SPECIES)-lite,-lite,-t iprscanxml)
	@$(MAKE) summarise SPECIES=$(SPECIES) RESULT_DIR=$(OUTDIR)/$(SPECIES)-lite

RESULT_DIR ?= $(OUTDIR)/$(SPECIES)
# The denominator is the FASTA that was actually submitted. Defaulting it to the whole
# proteome would make the mini run report its 45_139 unsubmitted proteins as unannotated.
SUMMARY_FASTA ?= $(SRC_FASTA)
summarise: check-species
	$(PYTHON) bin/summarise_unifire.py \
	  --indir $(RESULT_DIR) --species $(SPECIES) --fasta "$(SUMMARY_FASTA)" \
	  --out $(RESULT_DIR)/$(SPECIES)_unifire.parquet \
	  --summary-out $(RESULT_DIR)/$(SPECIES)_unifire_summary.json

queue-empty:
	@if command -v squeue >/dev/null 2>&1; then \
	  n=$$(squeue -u $$USER -h 2>/dev/null | wc -l | tr -d ' '); \
	  if [ "$$n" -gt 0 ]; then \
	    echo "$$n job(s) already queued or running:"; squeue -u $$USER; \
	    echo; echo "wait, or override with SKIP_QUEUE_CHECK=1"; \
	    [ -n "$(SKIP_QUEUE_CHECK)" ] || exit 1; \
	  fi; \
	fi
