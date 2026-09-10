# How much of an invertebrate proteome does conventional sequence search miss?
#
# The dark set is every protein that phmmer, jackhmmer AND mmseqs2 all fail to place into
# reviewed Swiss-Prot with the query's clade removed. It needs no answer key, which is the
# point: C. gigas has 16 reviewed entries and Botryllus has 0, so neither can be scored
# until the structural key exists -- but "no arm hit this" is a property of the searches
# alone. This runs while Chainsaw and InterProScan are still going, and the number it
# returns decides whether the expensive half is worth running.
#
#   make run-dark-set SPECIES=botryllus     full proteome
#   make run-dark-set-mini SPECIES=botryllus  200 proteins, one node, sanity only
#
# The mini is a plumbing check, NOT a small answer: a dark fraction measured on 200
# proteins is not the proteome's dark fraction.

SHELL := /bin/bash

MAC_PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
ifneq ($(wildcard $(MAC_PYTHON)),)
PYTHON ?= $(MAC_PYTHON)
else
PYTHON ?= python3
endif

# Pinned on the mac, from the module tree on Sherlock. `ml load biology nextflow` needs
# both words: nextflow lives under the biology hierarchy, so `ml load nextflow` resolves
# only in a shell that already has biology loaded.
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
ifneq ($(wildcard $(MAC_NEXTFLOW)),)
NEXTFLOW ?= $(MAC_NEXTFLOW)
PROFILE  ?= standard
else
NEXTFLOW ?= nextflow
PROFILE  ?= sherlock
endif

SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
OUTDIR      ?= $(SCRATCH_DIR)/invertebrate-dark-set/results
# One reference per excluded clade, shared by every species that asks for that clade.
REF_CACHE   ?= $(SCRATCH_DIR)/invertebrate-dark-set/reference
# Inputs are in different places on the laptop and on Sherlock, and hard-coding either
# produced a run that launched against files that were not there. On Sherlock the shared
# inputs are hard-linked into the region benchmark's data/ by its `sherlock-clone`, so that
# is the second place to look. Both are probed for a file that must exist; the first hit
# wins, and passing QFO_DIR= or SWISSPROT= on the command line still overrides.
PRIMARY_DATA ?= $(SCRATCH_DIR)/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data

QFO_CANDIDATES := $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143 \
                  $(PRIMARY_DATA)/qfo
QFO_DIR     ?= $(firstword $(foreach d,$(QFO_CANDIDATES),$(if $(wildcard $(d)/Eukaryota),$(d),)))

SWISSPROT_CANDIDATES := $(HOME)/data/uniprot/uniprot_sprot.dat.gz \
                        $(PRIMARY_DATA)/uniprot_sprot.dat.gz
SWISSPROT   ?= $(firstword $(foreach f,$(SWISSPROT_CANDIDATES),$(if $(wildcard $(f)),$(f),)))

BOTRYLLUS_FASTA ?= $(QFO_DIR)/Eukaryota/BOTSCH2026_30301.fasta

SHERLOCK_HOST ?= sherlock
REPO_URL      ?= https://github.com/seanome/2024-kmerseek-analysis/
BRANCH        ?= $(shell git rev-parse --abbrev-ref HEAD)
SSH_MUX       := ssh -o ControlMaster=auto -o ControlPath=$$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m
SHERLOCK_CLONE     ?= $$SCRATCH/2024-kmerseek-$(notdir $(BRANCH))
SHERLOCK_CLONE_DIR := $(SHERLOCK_CLONE)/nextflow-runs/invertebrate-dark-set

SPECIES ?=
NF_ARGS ?=
WITH_KMERSEEK ?=
KMERSEEK_ARGS := $(if $(WITH_KMERSEEK),--with_kmerseek true,)
MINI_N  ?= 200

.PHONY: help check-species check-inputs check-python sherlock-clone sync run-dark-set \
        run-dark-set-sbatch watch-dark-set stop-dark-set run-dark-set-mini queue-note

help:
	@echo "Dark set: proteins no sequence arm can place into the reference"
	@echo
	@echo "  sherlock-clone          push and clone this branch to \$$SCRATCH"
	@echo "  check-python            host python diagnostic; NOT needed for a run"
	@echo "  run-dark-set-mini       SPECIES=<botryllus|cgigas>  plumbing check, $(MINI_N) proteins"
	@echo "  run-dark-set            SPECIES=<...>  full proteome, driver in this shell"
	@echo "  run-dark-set-sbatch     SPECIES=<...>  driver on a compute node (USE THIS)"
	@echo "  watch-dark-set          SPECIES=<...>  follow a running driver's log"
	@echo "  stop-dark-set           SPECIES=<...>  cancel the driver and its tasks"
	@echo "                          add WITH_KMERSEEK=1 for the kmerseek arm"
	@echo
	@echo "  PROFILE=$(PROFILE)  OUTDIR=$(OUTDIR)"
	@echo "  REF_CACHE=$(REF_CACHE)  (one file per excluded clade, shared across species)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi

# NOT a prerequisite of the run targets, and deliberately so. Every python step in this
# pipeline (buildReference, computeDarkSet) carries `label python_scoring`, which the
# sherlock profile runs INSIDE params.kmerseek_image -- so the host python is never used and
# gating the run on it blocks a run that would have worked. It did exactly that on Sherlock,
# where PYTHON falls back to a bare python3 with no polars. Kept as a manual diagnostic for
# anyone running bin/ scripts by hand.
# Nothing is launched until the inputs actually resolve. A missing reference or query does
# not fail loudly downstream -- it produces an empty result that reads as a real one.
check-inputs:
	@ok=1; \
	 if [ -z "$(QFO_DIR)" ] || [ ! -d "$(QFO_DIR)/Eukaryota" ]; then \
	   echo "QfO tree not found. Looked for Eukaryota/ under:"; \
	   for d in $(QFO_CANDIDATES); do echo "    $$d"; done; \
	   echo "  override: make <target> QFO_DIR=/path/to/release"; ok=0; \
	 else echo "QFO_DIR   = $(QFO_DIR)"; fi; \
	 if [ -z "$(SWISSPROT)" ] || [ ! -f "$(SWISSPROT)" ]; then \
	   echo "uniprot_sprot.dat.gz not found. Looked at:"; \
	   for f in $(SWISSPROT_CANDIDATES); do echo "    $$f"; done; \
	   echo "  override: make <target> SWISSPROT=/path/to/uniprot_sprot.dat.gz"; ok=0; \
	 else echo "SWISSPROT = $(SWISSPROT)"; fi; \
	 [ "$$ok" = 1 ] || exit 1

check-python:
	@$(PYTHON) -c 'import polars, sys; print("host python OK:", sys.executable, "polars", polars.__version__)' \
	  || { echo; echo "host PYTHON=$(PYTHON) cannot import polars."; \
	       echo "This does NOT block a pipeline run -- the python steps run in"; \
	       echo "params.kmerseek_image. It only matters for running bin/ scripts by hand:"; \
	       echo "  make <target> PYTHON=\"apptainer exec \$$SCRATCH/containers/kmerseek.sif python3\""; \
	       exit 1; }

sherlock-clone:
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) '\
	  set -e; \
	  if [ ! -d $(SHERLOCK_CLONE)/.git ]; then \
	    git clone --branch $(BRANCH) $(REPO_URL) $(SHERLOCK_CLONE); \
	  else \
	    cd $(SHERLOCK_CLONE) && git fetch origin $(BRANCH) \
	      && git checkout $(BRANCH) && git pull --ff-only; \
	  fi; \
	  echo "ready: $(SHERLOCK_CLONE_DIR)"'
	@echo
	@echo "  ssh $(SHERLOCK_HOST)"
	@echo "  cd $(SHERLOCK_CLONE_DIR)"

sync:
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) 'cd $(SHERLOCK_CLONE_DIR) && git pull --ff-only'

# -resume, single dash. `--resume` is a pipeline param and silently does nothing.
# Never piped through tee: the live task table is the point, and tee makes nextflow
# reprint the whole table on every update.
# One definition, used by the foreground driver and the sbatch driver alike, so the two
# cannot drift into running different pipelines.
NF_RUN = $(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) --qfo_dir $(QFO_DIR) --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) --reference_cache $(REF_CACHE) $(KMERSEEK_ARGS) $(NF_ARGS)

run-dark-set: check-species check-inputs queue-note
	$(NF_RUN)

# ---- the driver as a batch job -------------------------------------------------------
# Sherlock's login nodes enforce a TEN CPU-MINUTE cap and SIGKILL past it. A Nextflow head
# process polling dozens of tasks burns that in well under an hour, and the kill looks
# exactly like an OOM -- which is how a 90-task run died here with squeue left empty,
# because Nextflow cancels its own jobs on shutdown. On 2026-09-09 a login node also
# rebooted and took a running head and its whole queue with it.
#
# So the driver goes to a compute node, through tools/nf-head, which is the same submitter
# the qfo-pfam-region-benchmark Makefile uses. It runs Nextflow inside a tmux session
# there, so the live task table survives: `make watch-dark-set` attaches to it and Ctrl-b d
# detaches without touching the run. That replaces the earlier --wrap driver, which had to
# write to a file and therefore lost the table.
#
# The partition is named here rather than inherited. The previous version passed
# `--partition=$(PARTITION)` against a variable this Makefile never defined, so it expanded
# to nothing, sbatch fell back to the cluster default of `normal`, and the driver landed on
# the public partition. That is not merely untidy: `normal` has no partition QOS, so it is
# checked against the normal QOS cap of cpu=512, and on 2026-09-10 the driver sat in
# QOSMaxCpuPerUserLimit behind an unrelated 622-CPU sweep of ours on hns. hns carries the
# `owner` partition QOS at cpu=99999 and would have started it immediately.
HEAD_ACCOUNT   ?= ayeletv
HEAD_PARTITION ?= hns
HEAD_QOS       ?= normal
HEAD_CPUS      ?= 2
HEAD_MEM       ?= 16G
# SLURM enforces --mem with a cgroup, and a JVM left to size its own heap would measure the
# node's RAM rather than the allocation and be killed inside it.
HEAD_XMX       ?= 12g
export NXF_OPTS := -Xms1g -Xmx$(HEAD_XMX)
HEAD_TIME      ?= 2-00:00:00
HEAD_SESSION   ?= darkset
# Per species, and deliberately not `nf-head`: the guard below, watch and stop all key on
# this name, and the qfo benchmark runs a head of its own at the same time.
HEAD_JOBNAME    = nfdrv-$(SPECIES)

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT = HEAD_ACCOUNT=$(HEAD_ACCOUNT) HEAD_PARTITION=$(HEAD_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              HEAD_HINT_WATCH="make watch-dark-set SPECIES=$(SPECIES)" \
              HEAD_HINT_STOP="make stop-dark-set SPECIES=$(SPECIES)" \
              $(MAKEFILE_DIR)/tools/nf-head

run-dark-set-sbatch: check-species check-inputs queue-note
	@if squeue -u $$USER -h -o '%j' 2>/dev/null | grep -qx '$(HEAD_JOBNAME)'; then \
	  echo "a driver for $(SPECIES) is already queued or running:"; \
	  squeue -u $$USER -o '%.12i %.2t %.10M %j' | grep '$(HEAD_JOBNAME)'; \
	  echo; echo "watch it with:  make watch-dark-set SPECIES=$(SPECIES)"; \
	  echo "or stop it with: make stop-dark-set SPECIES=$(SPECIES)"; \
	  exit 1; fi
	$(HEAD_SUBMIT) $(NF_RUN)

# Attaches to the driver's tmux and shows the live task table. Read-only, because a stray
# Ctrl-C in an attached session reaches Nextflow and shuts the run down; Ctrl-b d detaches
# and the run carries on. Stopping it deliberately is `make stop-dark-set`.
watch-dark-set: check-species
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 if [ -z "$$jid" ]; then \
	   pend=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T %r"); \
	   if [ -n "$$pend" ]; then \
	     echo "the driver for $(SPECIES) has not started yet:"; echo "$$pend" | sed 's/^/    /'; \
	   else \
	     echo "no driver for $(SPECIES)."; \
	     echo "start one with: make run-dark-set-sbatch SPECIES=$(SPECIES)"; \
	   fi; \
	   exit 1; \
	 fi; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

# Asks the run to stop the way Nextflow can handle. `scancel --signal=TERM --batch` signals
# the batch script only and does not cancel the job, so the trap in tools/nf-head sends
# Ctrl-C to Nextflow and it cancels its own tasks before exiting. Plain `scancel` kills the
# driver outright and leaves those tasks behind. Nothing finished is lost either way: the
# reference and indexes are under storeDir and completed tasks stay cached, so a later
# run-dark-set-sbatch resumes rather than restarts.
stop-dark-set: check-species
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | head -1); \
	 if [ -z "$$jid" ]; then echo "no driver running for $(SPECIES)"; exit 0; fi; \
	 echo "asking driver $$jid to shut down; its tasks are cancelled as it goes"; \
	 scancel --signal=TERM --batch $$jid

run-dark-set-mini: check-species check-inputs
	@mkdir -p $(OUTDIR)/_mini
	@# set -e plus an explicit count. Without both, a failed awk left an empty FASTA, printed
	@# "staged 0 proteins", and make went on to launch nextflow against it -- a whole run
	@# whose only possible answer was "every protein is dark".
	@set -e; \
	 if [ "$(SPECIES)" != "botryllus" ]; then \
	   echo "the mini stages botryllus only; for cgigas pass --query_fasta yourself"; exit 1; fi; \
	 src="$(BOTRYLLUS_FASTA)"; \
	 [ -f "$$src" ] || { echo "no botryllus proteome at $$src"; exit 1; }; \
	 awk -v m=$(MINI_N) '/^>/{n++} n>m{exit} {print}' "$$src" > $(OUTDIR)/_mini/$(SPECIES).mini.fasta; \
	 n=$$(grep -c '^>' $(OUTDIR)/_mini/$(SPECIES).mini.fasta || true); \
	 [ "$$n" -gt 0 ] || { echo "staged 0 proteins from $$src -- refusing to launch"; exit 1; }; \
	 echo "staged $$n proteins"
	$(NEXTFLOW) run main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) \
	  --query_fasta $(OUTDIR)/_mini/$(SPECIES).mini.fasta \
	  --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR)/mini --run_tag mini \
	  --reference_cache $(REF_CACHE) \
	  $(KMERSEEK_ARGS) $(NF_ARGS)

# Advisory, NOT a block. The original gate refused to launch while any job was queued,
# which on this account is never true -- the region benchmark keeps several hundred jobs
# pending for days, so the gate would have blocked this run indefinitely. It also could not
# have told the two apart: nextflow names jobs after the process, and both pipelines have
# nf-phmme, nf-jackh and nf-mmseq.
#
# Double-launching the same run is already prevented by nextflow's own session lock in the
# launch directory, which is the mechanism that actually works. So this reports the queue
# depth as context for how long a submission will sit, and gets out of the way.
queue-note:
	@if command -v squeue >/dev/null 2>&1; then \
	  n=$$(squeue -u $$USER -h 2>/dev/null | wc -l | tr -d ' '); \
	  r=$$(squeue -u $$USER -h -t RUNNING 2>/dev/null | wc -l | tr -d ' '); \
	  echo "queue: $$n job(s) total, $$r running -- this run's tasks queue behind them"; \
	fi
