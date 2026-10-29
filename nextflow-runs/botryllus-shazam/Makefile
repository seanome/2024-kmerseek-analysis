# Run on Sherlock. The controller runs inside a Slurm allocation.
SHELL := /bin/bash
.DEFAULT_GOAL := help
PIPELINE := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
CODE := $(abspath $(PIPELINE)/../..)
OUTROOT ?= $(SCRATCH)/botryllus-shazam-PEC-nextflow-20261027
PREFLIGHT ?= $(SCRATCH)/botryllus-production-preflight-49525580
REFERENCE_CHECK ?= $(SCRATCH)/botryllus-reference-check-49521127
FIXTURE ?= $(SCRATCH)/botryllus-nextflow-fixture-49525580
# Supply the exact controller ID for status/stop; never select unrelated jobs by name.
JOB_ID ?=
# Reviewed-failure file for resume after a FAILED task (read by production.sbatch).
FAILURE_REVIEW ?=

.PHONY: help run resume test check watch stop
help:
	@echo 'Sherlock: make run | resume | test | watch JOB_ID=<id> | stop JOB_ID=<id>'
	@echo 'run requires a new OUTROOT; resume retains its outputs and work cache.'

run:
	@test -n "$(SCRATCH)" && test ! -e "$(OUTROOT)"
	sbatch --output="$(SCRATCH)/botryllus-64-nextflow-%j.log" "$(PIPELINE)/production.sbatch" "$(CODE)" "$(OUTROOT)" "$(PREFLIGHT)" "$(REFERENCE_CHECK)" "$(FIXTURE)"

resume:
	@test -s "$(OUTROOT)/params.json" && test -s "$(OUTROOT)/preflight-source.txt"
	FAILURE_REVIEW="$(FAILURE_REVIEW)" sbatch --output="$(SCRATCH)/botryllus-64-nextflow-%j.log" "$(PIPELINE)/production.sbatch" "$(CODE)" "$(OUTROOT)"

test:
	sbatch --output="$(SCRATCH)/botryllus-nextflow-fixture-%j.log" "$(PIPELINE)/smoke.sbatch" "$(CODE)"

check:
	python3 -m unittest discover -s "$(CODE)/scripts/botryllus_shazam" -p 'test_*.py'

# One snapshot: Sherlock forbids unattended polling loops.
watch:
	@test -n "$(JOB_ID)"
	sacct -j "$(JOB_ID)" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
	tail -40 "$(SCRATCH)/botryllus-64-nextflow-$(JOB_ID).log"

stop:
	@test -n "$(JOB_ID)"
	scancel --signal=INT --full "$(JOB_ID)"
