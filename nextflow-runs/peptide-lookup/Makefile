# Notebook 300: can kmerseek find the source protein of a short mass-spec peptide?
#
# Step 0 (the E-value floor) runs as four SLURM jobs chained by dependencies:
#   download -> composition of Swiss-Prot, UniRef90, UHGP-90 -> floor table.
# On Sherlock:
#   cd $SCRATCH/2024-kmerseek-analysis/nextflow-runs/peptide-lookup && make step0
# Then on the laptop:
#   make pull-step0

OUT ?= $(SCRATCH)/peptide-lookup-300
IMG ?= /scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-2026-10-18-nb344-0a22970.img
KA_STORE ?= $(SCRATCH)/invertebrate-dark-set/results-0.4/kmerseek_index
SHERLOCK_HOST ?= sherlock
REMOTE_OUT ?= /scratch/users/olgabot/peptide-lookup-300
NOTEBOOK_DATA := ../../notebooks/300_data

RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
BIN := $(RUNNING_DIR)/bin
JOBENV := OUT=$(OUT),IMG=$(IMG),BIN=$(BIN),KA_STORE=$(KA_STORE)

.PHONY: step0 floor pull-step0 pylib pxd069517 queries pull-queries test running

## Submit Step 0 as one chain. Prints the four job ids.
step0:
	@mkdir -p $(OUT)/logs
	@cd $(OUT)/logs && \
	dl=$$(sbatch --parsable --export=ALL,$(JOBENV) $(RUNNING_DIR)/sbatch/download.sbatch) && \
	sp=$$(sbatch --parsable --dependency=afterok:$$dl --job-name=nb300-comp-swissprot --export=ALL,$(JOBENV) \
	  $(RUNNING_DIR)/sbatch/composition.sbatch $(OUT)/db/uniprot_sprot.fasta.gz swissprot) && \
	ur=$$(sbatch --parsable --dependency=afterok:$$dl --job-name=nb300-comp-uniref90 --export=ALL,$(JOBENV) \
	  $(RUNNING_DIR)/sbatch/composition.sbatch $(OUT)/db/uniref90.fasta.gz uniref90) && \
	uh=$$(sbatch --parsable --dependency=afterok:$$dl --job-name=nb300-comp-uhgp90 --export=ALL,$(JOBENV) \
	  $(RUNNING_DIR)/sbatch/composition.sbatch $(OUT)/db/uhgp-90.faa.gz uhgp90) && \
	fl=$$(sbatch --parsable --dependency=afterok:$$sp:$$ur:$$uh --export=ALL,$(JOBENV) \
	  $(RUNNING_DIR)/sbatch/floor.sbatch) && \
	echo "download $$dl -> composition $$sp $$ur $$uh -> floor $$fl; logs in $(OUT)/logs"

## Rerun only the floor table, once the compositions exist.
floor:
	@mkdir -p $(OUT)/logs
	@cd $(OUT)/logs && sbatch --export=ALL,$(JOBENV) $(RUNNING_DIR)/sbatch/floor.sbatch

## openpyxl for reading the PXD069517 xlsx inside the kmerseek image (it has pip, not openpyxl).
pylib:
	apptainer exec --bind /scratch $(IMG) python3 -m pip install -q --break-system-packages \
	  --no-deps --target $(OUT)/pylib openpyxl==3.1.5 et_xmlfile==2.0.0

## The PXD069517 peptide table (18 MB). Step 1 fallback source, see bin/real_peptides.py.
PXD069517_URL := https://ftp.pride.ebi.ac.uk/pride/data/archive/2025/10/PXD069517
pxd069517:
	mkdir -p $(OUT)/step1/PXD069517
	cd $(OUT)/step1/PXD069517 && for f in MGDB_peptides_intensities_tax-funct_annotations.xlsx sample_code_legend.xlsx; do \
	  curl -fsS -o $$f $(PXD069517_URL)/$$f; done && date -u +%Y-%m-%dT%H:%M:%SZ > date_downloaded.txt

## Steps 1-3: query peptides, decoys and the search database, one SLURM job.
queries: pylib pxd069517
	@mkdir -p $(OUT)/logs
	@cd $(OUT)/logs && sbatch --export=ALL,$(JOBENV) $(RUNNING_DIR)/sbatch/queries.sbatch

## Copy the step 1-3 tables (not the database) to the notebook's data folder. Run on the laptop.
pull-queries:
	mkdir -p $(NOTEBOOK_DATA)/step1
	rsync -av --include '*.tsv' --include '*.txt' --include '*.fasta' --exclude 'search_db.fasta' \
	  --exclude 'uhgp90_lines_with_pool_peptide.tsv' --exclude '*/' \
	  $(SHERLOCK_HOST):$(REMOTE_OUT)/step1/ $(NOTEBOOK_DATA)/step1/
	rsync -av $(SHERLOCK_HOST):$(REMOTE_OUT)/step1/PXD069517/date_downloaded.txt $(NOTEBOOK_DATA)/step1/pxd069517_date_downloaded.txt

## Copy the Step 0 tables (a few MB) to the notebook's data folder. Run on the laptop.
pull-step0:
	mkdir -p $(NOTEBOOK_DATA)/step0
	rsync -av --exclude '*.n_headers' $(SHERLOCK_HOST):$(REMOTE_OUT)/step0/ $(NOTEBOOK_DATA)/step0/
	rsync -av $(SHERLOCK_HOST):$(REMOTE_OUT)/db/manifest.tsv $(NOTEBOOK_DATA)/step0/db_manifest.tsv
	gzip -9f $(NOTEBOOK_DATA)/step0/floor_estimates.tsv

test:
	python3 -m pytest -q tests

.PHONY: running
running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/peptide-lookup"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
