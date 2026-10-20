# Experiment 354: source protein of a de novo peptide call. See FROZEN_PLAN.md.
# Run from this folder.

PY ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python
INPUTS ?= fixtures/inputs.json
WORK ?= fixture_run

.PHONY: fixtures test validate run figures clean-run

## Write the labeled synthetic fixtures (software check only).
fixtures:
	$(PY) make_fixtures.py fixtures

test:
	$(PY) -m pytest -q test_peptide_source.py

## Check an inputs.json against the schema, print checksums.
validate:
	$(PY) peptide_source.py validate $(INPUTS)

## Prepare, build every index, search with every method, evaluate, draw.
run:
	$(PY) peptide_source.py all $(INPUTS) $(WORK)
	$(PY) make_figures.py $(WORK)

figures:
	$(PY) make_figures.py $(WORK)

## Delete one run folder before a rerun.
clean-run:
	command rm -rf $(WORK)
