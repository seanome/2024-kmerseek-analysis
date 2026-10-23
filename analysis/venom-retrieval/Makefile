# Venom-peptide retrieval benchmark. CPU only, runs on the laptop in about four minutes.
#
# Nothing here runs kmerseek or Nextflow. Method C is kmerseek's alphabets and its k-mer
# sets at scaled=1, compared in Python, which is what a short-peptide pilot needs; a real
# kmerseek index and search belong on Sherlock, through the cys-frameworks pipeline next
# door in nextflow-runs/cys-frameworks.
#
#   make data      download ConoServer and the two Swiss-Prot searches (about 90 MB)
#   make all       manifest, splits, collisions, run, evaluate
#   make test      the unit tests, including the identical-encoding regression test
#   make clean     remove this config's results, leaving the downloads alone

SHELL := /bin/bash

PYTHON ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
HERE   := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO   := $(abspath $(HERE)/../..)
DATA   ?= $(HOME)/data/venom-retrieval
CONFIG ?= $(HERE)/config/pilot.json
OUT    ?= $(REPO)/tables/venom_retrieval/$(notdir $(basename $(CONFIG)))

RUN := cd $(HERE) && $(PYTHON) -m venom_retrieval

.PHONY: help data manifest splits collisions run evaluate all test clean

help:
	@echo ""
	@echo "  make data        download ConoServer and the Swiss-Prot searches into $(DATA)"
	@echo "  make all         the whole benchmark, writing to $(OUT)"
	@echo "  make manifest    build the peptide table only"
	@echo "  make splits      duplicates, neighbourhoods and the query sets only"
	@echo "  make collisions  the encoding-collision report only"
	@echo "  make run         rank the reference for every query with every method"
	@echo "  make evaluate    turn the rankings into per-method numbers"
	@echo "  make test        unit tests and docstring examples"
	@echo ""
	@echo "  CONFIG=$(CONFIG)"
	@echo "  OUT=$(OUT)"
	@echo ""

data:
	$(PYTHON) $(REPO)/scripts/download_venom_retrieval_data.py $(DATA)

manifest splits collisions run evaluate all:
	$(RUN) $@ --config $(CONFIG) --data $(DATA) --out $(OUT)

test:
	cd $(HERE) && $(PYTHON) -m pytest tests/ -q

clean:
	command rm -rf $(OUT)
