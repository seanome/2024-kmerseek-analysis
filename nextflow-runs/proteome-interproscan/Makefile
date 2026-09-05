# InterProScan over the non-vertebrate query proteomes.
#
# Not a Nextflow pipeline. It lives under nextflow-runs/ anyway so every benchmark run in
# this repo is found in one place, and so the Sherlock conventions (login-node downloads,
# SLURM for the work, $SCRATCH for outputs) are the same ones as its neighbours.
#
# What this is for. The proteome-annotate claim is that kmerseek transfers a real family
# label onto proteins where sequence search returns nothing. "Returns nothing" has to be
# measured against what conventional annotation actually delivers, and conventional
# annotation is not hmmscan against Pfam-A -- it is InterProScan, the union of 13+ member
# databases. This produces that number, and it produces the XML UniFire consumes.
#
# Two proteomes, neither of which has curated features:
#   botryllus   45_339 proteins, 16.9M aa, from the 2026 genome annotation
#   cgigas      57_874 proteins, UniProt UP000005408
#
# ---------------------------------------------------------------------------------------
# Usage
#   make download-interproscan     # LOGIN NODE. 6.6 GB tarball + unpack + index
#   make stage-cgigas              # LOGIN NODE. fetch UP000005408 from UniProt
#   make run-interproscan-mini SPECIES=botryllus   # 200 proteins, one node, ~minutes
#   make run-interproscan      SPECIES=botryllus   # full proteome, SLURM array
#   make merge SPECIES=botryllus                   # chunks -> one XML/TSV
# ---------------------------------------------------------------------------------------

SHELL := /bin/bash

# Resolved from https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/ on 2026-09-05, which is
# the only place the real version list lives. Override to pin an older one; do not edit a
# guessed value in.
IPS_VERSION   ?= 5.78-109.0
IPS_BASE_URL  ?= https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/$(IPS_VERSION)
IPS_TARBALL   := interproscan-$(IPS_VERSION)-64-bit.tar.gz

# $SCRATCH on Sherlock, a local dir off-cluster. Everything big lands here, never in the
# repo: the tarball alone is 6.6 GB and the unpacked install is far larger.
SCRATCH_DIR   ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
IPS_DIR       ?= $(SCRATCH_DIR)/interproscan/interproscan-$(IPS_VERSION)
IPS_SH        := $(IPS_DIR)/interproscan.sh
OUTDIR        ?= $(SCRATCH_DIR)/proteome-interproscan/results
CHUNK_DIR     ?= $(SCRATCH_DIR)/proteome-interproscan/chunks

# Where the query proteomes live. botryllus is already staged into the QfO tree by the
# region benchmark's `make stage-botryllus`; cgigas is fetched by `make stage-cgigas`.
QFO_DIR       ?= $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
BOTRYLLUS_FASTA ?= $(QFO_DIR)/Eukaryota/BOTSCH2026_30301.fasta
CGIGAS_FASTA    ?= $(SCRATCH_DIR)/proteomes/UP000005408_29159.fasta

PYTHON        ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# Proteins per SLURM array task. 45_339 botryllus proteins at ~2000/chunk is 23 tasks;
# InterProScan runs ~1-2 s/protein/core, so a chunk is roughly an hour on 16 cores.
CHUNK_SIZE    ?= 2000
CPUS          ?= 16
MEM           ?= 32G
TIME          ?= 12:00:00
PARTITION     ?= normal
MINI_N        ?= 200

# Which member databases. The default is every one InterProScan ships. PANTHER and
# SignalP/TMHMM are the slow ones; drop them with APPL=Pfam,SMART,... to cut runtime when
# only the domain calls matter.
APPL          ?=
APPL_ARG      := $(if $(APPL),-appl $(APPL),)

SPECIES       ?=
ifeq ($(SPECIES),botryllus)
  FASTA := $(BOTRYLLUS_FASTA)
  TAXID := 30301
else ifeq ($(SPECIES),cgigas)
  FASTA := $(CGIGAS_FASTA)
  TAXID := 29159
else
  FASTA :=
  TAXID :=
endif

.PHONY: help download-interproscan stage-cgigas check-species chunks \
        run-interproscan run-interproscan-mini merge queue-empty clean-chunks

help:
	@echo "InterProScan over the non-vertebrate query proteomes"
	@echo
	@echo "  download-interproscan   LOGIN NODE: fetch and unpack InterProScan $(IPS_VERSION) (6.6 GB)"
	@echo "  stage-cgigas            LOGIN NODE: fetch UP000005408 from UniProt"
	@echo "  run-interproscan-mini   SPECIES=<botryllus|cgigas>  $(MINI_N) proteins, one node"
	@echo "  run-interproscan        SPECIES=<botryllus|cgigas>  full proteome, SLURM array"
	@echo "  merge                   SPECIES=<...>  combine chunk XML/TSV"
	@echo
	@echo "  IPS_VERSION=$(IPS_VERSION)   IPS_DIR=$(IPS_DIR)"
	@echo "  OUTDIR=$(OUTDIR)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi
	@if [ -z "$(FASTA)" ]; then \
	  echo "unknown SPECIES '$(SPECIES)'. Known: botryllus, cgigas"; exit 1; fi
	@if [ ! -f "$(FASTA)" ]; then \
	  echo "no proteome for $(SPECIES) at $(FASTA)"; \
	  if [ "$(SPECIES)" = "cgigas" ]; then echo "  run: make stage-cgigas"; \
	  else echo "  stage it from the region benchmark: make stage-botryllus"; fi; exit 1; fi
	@echo "$(SPECIES): $$(grep -c '^>' $(FASTA)) proteins from $(FASTA)"

# ---- downloads. LOGIN NODE ONLY -------------------------------------------------------
# Sherlock compute nodes have no outbound internet -- established 2026-08-20 by the ProstT5
# weights job. Both targets below transfer from EBI and must run where there is a network.
# A bare backgrounded curl on a multi-GB EBI file stalls silently and then hangs for hours,
# so both use --continue-at with a speed floor that aborts and retries instead.
download-interproscan:
	@mkdir -p $(dir $(IPS_DIR))
	@if [ -x "$(IPS_SH)" ]; then echo "already installed at $(IPS_DIR)"; exit 0; fi
	@echo "==> InterProScan $(IPS_VERSION) (~6.6 GB) into $(dir $(IPS_DIR))"
	cd $(dir $(IPS_DIR)) && \
	  curl -L --fail --continue-at - --speed-limit 10240 --speed-time 60 --retry 10 \
	    -o $(IPS_TARBALL) $(IPS_BASE_URL)/$(IPS_TARBALL) && \
	  curl -L --fail --retry 5 -o $(IPS_TARBALL).md5 $(IPS_BASE_URL)/$(IPS_TARBALL).md5 && \
	  md5sum -c $(IPS_TARBALL).md5 && \
	  tar -pxzf $(IPS_TARBALL)
	@echo "==> building the member-database indexes (one-off, needs java)"
	cd $(IPS_DIR) && $(PYTHON) setup.py -f interproscan.properties || \
	  python3 setup.py -f interproscan.properties
	@echo "installed: $(IPS_SH)"

stage-cgigas:
	@mkdir -p $(dir $(CGIGAS_FASTA))
	@if [ -s "$(CGIGAS_FASTA)" ]; then echo "already staged: $(CGIGAS_FASTA)"; exit 0; fi
	@echo "==> UniProt UP000005408 (Magallana gigas, 57_874 proteins)"
	curl -L --fail --continue-at - --speed-limit 1024 --speed-time 60 --retry 10 \
	  -o $(CGIGAS_FASTA).tmp \
	  'https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000005408&format=fasta&compressed=false'
	@if [ ! -s "$(CGIGAS_FASTA).tmp" ]; then \
	  echo "download produced an empty file -- UniProt streams outages as a 200"; \
	  /bin/rm -f $(CGIGAS_FASTA).tmp; exit 1; fi
	@n=$$(grep -c '^>' $(CGIGAS_FASTA).tmp); \
	 if [ "$$n" -lt 50000 ]; then \
	   echo "only $$n records, expected ~57874 -- refusing to stage a truncated proteome"; \
	   /bin/rm -f $(CGIGAS_FASTA).tmp; exit 1; fi; \
	 /bin/mv -f $(CGIGAS_FASTA).tmp $(CGIGAS_FASTA); \
	 echo "staged $$n records to $(CGIGAS_FASTA)"

# ---- chunking + runs ------------------------------------------------------------------
chunks: check-species
	@mkdir -p $(CHUNK_DIR)/$(SPECIES)
	$(PYTHON) bin/split_fasta.py --in $(FASTA) \
	  --outdir $(CHUNK_DIR)/$(SPECIES) --chunk-size $(CHUNK_SIZE)

# -dp disables the precalculated-match lookup. It is not an optimisation: the lookup is an
# HTTPS call to EBI, compute nodes here have no outbound internet, and without this the run
# stalls on every chunk. Botryllus proteins are a 2026 gene set that is not in UniParc, so
# there would be nothing to look up regardless.
define IPS_CMD
$(IPS_SH) -i $(1) -f XML,TSV -o $(2) -T $(3) -cpu $(CPUS) -dp -goterms -pa $(APPL_ARG)
endef

run-interproscan-mini: check-species
	@mkdir -p $(OUTDIR)/$(SPECIES)-mini $(CHUNK_DIR)/$(SPECIES)-mini
	@if [ ! -x "$(IPS_SH)" ]; then echo "InterProScan not installed. make download-interproscan"; exit 1; fi
	$(PYTHON) bin/split_fasta.py --in $(FASTA) \
	  --outdir $(CHUNK_DIR)/$(SPECIES)-mini --chunk-size $(MINI_N) --max-chunks 1
	@echo "==> $(MINI_N) proteins, in the foreground"
	$(call IPS_CMD,$(CHUNK_DIR)/$(SPECIES)-mini/chunk_0000.fasta,$(OUTDIR)/$(SPECIES)-mini/$(SPECIES)-mini,$(OUTDIR)/$(SPECIES)-mini/tmp)
	@echo "wrote $(OUTDIR)/$(SPECIES)-mini/"

run-interproscan: check-species queue-empty chunks
	@mkdir -p $(OUTDIR)/$(SPECIES) $(OUTDIR)/$(SPECIES)/logs
	@if [ ! -x "$(IPS_SH)" ]; then echo "InterProScan not installed. make download-interproscan"; exit 1; fi
	@n=$$(ls $(CHUNK_DIR)/$(SPECIES)/chunk_*.fasta | wc -l | tr -d ' '); \
	 echo "==> $(SPECIES): $$n chunks x $(CPUS) cpus"; \
	 sbatch --wait --array=0-$$((n-1)) --job-name=ips-$(SPECIES) \
	   --cpus-per-task=$(CPUS) --mem=$(MEM) --time=$(TIME) --partition=$(PARTITION) \
	   --output=$(OUTDIR)/$(SPECIES)/logs/%A_%a.out \
	   --wrap='c=$$(printf "%04d" $$SLURM_ARRAY_TASK_ID); \
	     $(IPS_SH) -i $(CHUNK_DIR)/$(SPECIES)/chunk_$$c.fasta \
	       -f XML,TSV -o $(OUTDIR)/$(SPECIES)/chunk_$$c \
	       -T $(OUTDIR)/$(SPECIES)/tmp_$$c -cpu $(CPUS) -dp -goterms -pa $(APPL_ARG)'
	@$(MAKE) merge SPECIES=$(SPECIES)

merge: check-species
	$(PYTHON) bin/merge_interproscan.py \
	  --indir $(OUTDIR)/$(SPECIES) --out-prefix $(OUTDIR)/$(SPECIES)/$(SPECIES) \
	  --species $(SPECIES) --summary-out $(OUTDIR)/$(SPECIES)/$(SPECIES)_summary.json
	@echo "merged -> $(OUTDIR)/$(SPECIES)/$(SPECIES).xml (UniFire input)"

# The full target is gated on an empty queue so a 23-task array does not land on top of a
# sweep that is already running and push both past their time limits.
queue-empty:
	@if command -v squeue >/dev/null 2>&1; then \
	  n=$$(squeue -u $$USER -h 2>/dev/null | wc -l | tr -d ' '); \
	  if [ "$$n" -gt 0 ]; then \
	    echo "$$n job(s) already queued or running:"; squeue -u $$USER; \
	    echo; echo "wait for them, or override with: make $(MAKECMDGOALS) SKIP_QUEUE_CHECK=1"; \
	    [ -n "$(SKIP_QUEUE_CHECK)" ] || exit 1; \
	  fi; \
	fi

clean-chunks:
	/bin/rm -rf $(CHUNK_DIR)
