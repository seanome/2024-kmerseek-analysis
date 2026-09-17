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
#   make run-ladder-sweep-sbatch            the nine midi-plus QfO species, every
#                                           alphabet x ksize, each against Swiss-Prot
#                                           minus its own clade -- the scale the
#                                           Botryllus number is read against
#
# The mini is a plumbing check, NOT a small answer: a dark fraction measured on 200
# proteins is not the proteome's dark fraction.

SHELL := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

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

# One registry label, or a comma-separated list; every species in the list runs against
# Swiss-Prot minus ITS OWN clade, with one dark set and one report each.
SPECIES ?=
NF_ARGS ?=
WITH_KMERSEEK ?=
KMERSEEK_ARGS := $(if $(WITH_KMERSEEK),--with_kmerseek true,)
MINI_N  ?= 200

# What the driver job, the trace files and (for the ladder) the launch directory are named
# after. Defaults to SPECIES, which is what every single-species target has always used;
# the ladder target sets it to `ladder` so that `make watch-dark-set RUN=ladder` and
# `make stop-dark-set RUN=ladder` find the right head without spelling out nine species.
RUN ?= $(SPECIES)

# Human plus the nine QfO species of the midi-plus ladder, in divergence order from
# human. The nine are the same list as qfo-pfam-region-benchmark's MIDI_TARGETS. Human
# subtracts Mammalia, the same clade as mouse, so the two share one reference and one set
# of indexes and human adds only its searches. Botryllus is deliberately not in the list,
# because its dark set is already published and this run is the scale to read it against.
# Add it back with SPECIES= on the command line once the sweep is wanted on it too.
LADDER_SPECIES ?= human,mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli

# The ladder's own launch directory. Nextflow keeps its session lock, history and
# .nextflow.log in the directory it is launched from, so a ladder head and a single-species
# head launched from the same directory would refuse to coexist -- and the second would
# also rotate the first's log out from under `make status-dark-set`. The single-species
# targets keep launching from here, as they always have.
LAUNCH_DIR ?= $(MAKEFILE_DIR)
LADDER_LAUNCH_DIR := $(MAKEFILE_DIR)/runs/ladder

# Ladder scale. ~1_650 index builds and ~14_800 searches on top of 243 sequence-arm
# chunks; the region benchmark ran a sweep of the same shape at these widths. The head
# tracks ~17_000 tasks and the Botryllus head reached 11.3 GB on 175, so the JVM gets
# room; 7 days is the hns walltime ceiling, and a head that dies at the wall resumes.
LADDER_MAX_FORKS  ?= 150
LADDER_QUEUE_SIZE ?= 400
LADDER_HEAD_TIME  ?= 7-00:00:00
LADDER_HEAD_MEM   ?= 32G
LADDER_HEAD_XMX   ?= 24g

.PHONY: help check-species check-run check-inputs check-python sherlock-clone sync run-dark-set \
        run-dark-set-sbatch watch-dark-set stop-dark-set status-dark-set run-dark-set-mini queue-note \
        multiqc-dark-set run-ladder-sweep-sbatch run-ladder-sweep ladder-plan sweep-dark-set

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
	@echo "  status-dark-set         SPECIES=<...>  did it end, what killed it, what it said"
	@echo "                          add HISTORY=1 for one line per past run"
	@echo "                          add WITH_KMERSEEK=1 for the kmerseek arm"
	@echo "  multiqc-dark-set        SPECIES=<...>  re-render the report on a finished run"
	@echo
	@echo "  run-ladder-sweep-sbatch  the nine midi-plus QfO species, every alphabet x ksize"
	@echo "                          (SPECIES= to override the list; LOWCOMP=false for one"
	@echo "                          mask setting instead of the pair; NF_ARGS= passes through)"
	@echo "  ladder-plan             what run-ladder-sweep-sbatch would submit, without submitting"
	@echo "  watch-dark-set RUN=ladder / stop-dark-set RUN=ladder / status-dark-set RUN=ladder"
	@echo
	@echo "  PROFILE=$(PROFILE)  OUTDIR=$(OUTDIR)"
	@echo "  REF_CACHE=$(REF_CACHE)  (one file per excluded clade, shared across species)"

check-species:
	@if [ -z "$(SPECIES)" ]; then \
	  echo "SPECIES is required: make $(MAKECMDGOALS) SPECIES=botryllus"; exit 1; fi

# watch, stop and status key on the driver's NAME, which is RUN: the species for a
# single-species run, `ladder` for the ladder. RUN defaults to SPECIES, so SPECIES=x
# still works for those.
check-run:
	@if [ -z "$(RUN)" ]; then \
	  echo "RUN (or SPECIES) is required: make $(MAKECMDGOALS) SPECIES=botryllus, or RUN=ladder"; exit 1; fi

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
# main.nf by absolute path, so the same definition works from the ladder's launch
# directory as from this one.
NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
	  --species $(SPECIES) --qfo_dir $(QFO_DIR) --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) --reference_cache $(REF_CACHE) $(KMERSEEK_ARGS) $(NF_ARGS)

run-dark-set: check-species check-inputs queue-note
	$(NF_RUN)

# ---- the report on its own -----------------------------------------------------------
# A full run builds the report itself, at the end. This re-renders it over a results
# directory that already exists, which is what you want after editing a section or after
# a later arm (length, disorder, kmerseek) has published its products into the same place.
# It reads $(OUTDIR)/$(SPECIES)/ and touches no search.
#
# --report_only true is not decoration: it flips the report processes from 'ignore' to
# 'terminate'. In a full run a failed plot must not throw away a finished proteome; here
# the report IS the work, so a failure has to be loud instead of leaving yesterday's HTML
# in place and exiting green.
multiqc-dark-set: check-species check-inputs
	$(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -entry darkReport -resume \
	  --species $(SPECIES) --qfo_dir $(QFO_DIR) --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) --reference_cache $(REF_CACHE) --report_only true $(NF_ARGS)
	@echo
	@echo "report: $(OUTDIR)/$(SPECIES)/$(SPECIES)_dark_set_multiqc.html"

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
HEAD_JOBNAME    = nfdrv-$(RUN)

HEAD_SUBMIT = HEAD_ACCOUNT=$(HEAD_ACCOUNT) HEAD_PARTITION=$(HEAD_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              HEAD_HINT_WATCH="make watch-dark-set RUN=$(RUN)" \
              HEAD_HINT_STOP="make stop-dark-set RUN=$(RUN)" \
              $(MAKEFILE_DIR)/tools/nf-head

run-dark-set-sbatch: check-species check-inputs queue-note
	@n=$$(squeue -u $$USER -h -o '%j' 2>/dev/null | grep -c '^nf-darkSet_' || true); \
	 if [ "$$n" != 0 ] && ! squeue -u $$USER -h -o '%j' 2>/dev/null | grep -q '^nfdrv-'; then \
	   echo "WARNING: $$n nf-darkSet_ task(s) are in the queue with no driver alive -- orphans of"; \
	   echo "         a stopped run. An orphaned index build finishing under this run collides on"; \
	   echo "         storeDir. Clear them first:  make sweep-dark-set"; exit 1; fi
	@if squeue -u $$USER -h -o '%j' 2>/dev/null | grep -qx '$(HEAD_JOBNAME)'; then \
	  echo "a driver named $(HEAD_JOBNAME) is already queued or running:"; \
	  squeue -u $$USER -o '%.12i %.2t %.10M %j' | grep '$(HEAD_JOBNAME)'; \
	  echo; echo "watch it with:  make watch-dark-set RUN=$(RUN)"; \
	  echo "or stop it with: make stop-dark-set RUN=$(RUN)"; \
	  exit 1; fi
	@mkdir -p $(LAUNCH_DIR)
	cd $(LAUNCH_DIR) && $(HEAD_SUBMIT) $(NF_RUN)

# ---- the ladder: human + nine QfO species, the whole alphabet x ksize matrix ----------
# Each species is the QUERY against reviewed Swiss-Prot minus its own clade, which is the
# construction Botryllus got, so its numbers land on the same axis. Species whose
# annotation is deep enough that a dark protein is a real miss rather than a gene-model
# artefact give the dark fraction a scale, and sweeping every alphabet and ksize on them
# says where on the k axis "dark reached" stops meaning "reports something for
# everything" -- on Botryllus, hp_thomas_dill2 k23 put a region on all 45_339 proteins.
#
# Scale, from the Botryllus measurements (2026-09-13): 183 alphabet x ksize combos, or
# 203 with --kmerseek_extra_encodings; 9 clade-specific references, so one index build
# per clade per combo (1_647 at 5-10 min and 90-150 GB each, ~10 GB on disk each -- the
# index store alone is ~16 TB, and $SCRATCH is purged 90 days after last write); 81 query
# chunks of 2_000 across the nine proteomes, so ~14_800 searches at 3-30 min and 24-137 GB
# each. Roughly 3_000 search task-hours, 2-3 days at LADDER_MAX_FORKS, and every number
# above DOUBLES with the mask pair. Search memory is sized per index from its k-mer
# spectrum (see kmerseekSearchMemory in main.nf), and a combo no hns node can hold --
# gbmr7 at k=9-10 against a 572k-sequence reference -- is skipped and named in the log
# rather than run into three OOM kills; the first ladder attempt (2026-09-14) lost 259 of
# 1_057 searches that way and ended on the third kill of one chunk. See the note on
# params.kmerseek_lowcomp in main.nf for what the mask pair guards against and why the
# ladder drops it. `make ladder-plan` prints the expansion without submitting anything.
# false by default: the ladder ran the mask OFF only (2026-09-16), and a relaunch that
# forgot to pass LOWCOMP=false doubled the sweep to 16_601 searches and 3_294 indexes.
# LOWCOMP=true,false runs the pair.
LOWCOMP ?= false
LADDER_LC_ARGS := $(if $(LOWCOMP),--kmerseek_lowcomp $(LOWCOMP),)
LADDER_NF_ARGS  = --kmerseek_sweep true --run_tag ladder \
                  --max_forks $(LADDER_MAX_FORKS) --queue_size $(LADDER_QUEUE_SIZE) \
                  $(LADDER_LC_ARGS) $(NF_ARGS)

run-ladder-sweep-sbatch:
	@$(MAKE) run-dark-set-sbatch SPECIES=$(if $(SPECIES),$(SPECIES),$(LADDER_SPECIES)) RUN=ladder \
	  WITH_KMERSEEK=1 LAUNCH_DIR=$(LADDER_LAUNCH_DIR) \
	  HEAD_TIME=$(LADDER_HEAD_TIME) HEAD_MEM=$(LADDER_HEAD_MEM) HEAD_XMX=$(LADDER_HEAD_XMX) \
	  NF_ARGS="$(LADDER_NF_ARGS)"

# The foreground version, for a shell that is already on a compute node. Not for a login
# node: see run-dark-set-sbatch.
run-ladder-sweep:
	@$(MAKE) run-dark-set SPECIES=$(if $(SPECIES),$(SPECIES),$(LADDER_SPECIES)) RUN=ladder \
	  WITH_KMERSEEK=1 NF_ARGS="$(LADDER_NF_ARGS)"

# What the ladder would expand to: species, clades, combo count and the per-combo memory
# asks, from the same code path as the run, with nothing submitted. Uses Nextflow's
# -preview, which resolves the workflow and prints its log lines without creating tasks.
ladder-plan: check-inputs
	$(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -preview \
	  --species $(if $(SPECIES),$(SPECIES),$(LADDER_SPECIES)) \
	  --qfo_dir $(QFO_DIR) --swissprot_dat $(SWISSPROT) \
	  --outdir $(OUTDIR) --reference_cache $(REF_CACHE) --with_kmerseek true \
	  $(LADDER_NF_ARGS)

# Attaches to the driver's tmux and shows the live task table. Read-only, because a stray
# Ctrl-C in an attached session reaches Nextflow and shuts the run down; Ctrl-b d detaches
# and the run carries on. Stopping it deliberately is `make stop-dark-set`.
watch-dark-set: check-run
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 if [ -z "$$jid" ]; then \
	   pend=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T %r"); \
	   if [ -n "$$pend" ]; then \
	     echo "the driver $(HEAD_JOBNAME) has not started yet:"; echo "$$pend" | sed 's/^/    /'; \
	   else \
	     echo "no driver named $(HEAD_JOBNAME)."; \
	     echo "start one with: make run-dark-set-sbatch SPECIES=$(RUN), or make run-ladder-sweep-sbatch"; \
	   fi; \
	   exit 1; \
	 fi; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

# Asks the run to stop the way Nextflow can handle. `scancel --signal=TERM --batch` signals
# the batch script only and does not cancel the job, so the trap in tools/nf-head sends
# Ctrl-C to Nextflow and it cancels its own tasks before exiting. Plain `scancel` kills the
# driver and leaves those tasks behind. Nothing finished is lost either way: the
# reference and indexes are under storeDir and completed tasks stay cached, so a later
# run-dark-set-sbatch resumes rather than restarts.
#
# Then it waits for the head to be gone and cancels whatever tasks it left behind. On
# 2026-09-16 Nextflow logged "Killing running tasks (296)" and exited 13 seconds later,
# and 222 of them were still queued or running: slurmctld throttles cancel RPCs
# (`scancel: _add_delay: adding delay in RPC send`), so a burst of 296 scancels from a
# process that is about to exit does not all land. Orphaned tasks are not just wasted
# node-hours -- an orphaned kmerseekIndex that finishes after the next run has already
# created the same task is the storeDir "Directory not empty" collision.
stop-dark-set: check-run
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | head -1); \
	 if [ -z "$$jid" ]; then echo "no driver named $(HEAD_JOBNAME) is running"; \
	 else \
	   echo "asking driver $$jid to shut down; its tasks are cancelled as it goes"; \
	   scancel --signal=TERM --batch $$jid; \
	   for i in $$(seq 1 30); do \
	     squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | grep -q . || break; sleep 5; \
	   done; \
	 fi; \
	 $(MAKE) --no-print-directory sweep-dark-set

# Cancel every task this pipeline's processes submitted, whatever run they belong to.
# Tasks are named nf-darkSet_<process>_(<tag>) by Nextflow, so the prefix is the only
# handle; a single-species run and the ladder share it. Run this only when no driver of
# this pipeline is alive, which is what stop-dark-set checks first. Cancelling a few
# hundred jobs takes a minute or two under slurmctld's RPC throttle; that is normal.
sweep-dark-set:
	@ids=$$(squeue -u $$USER -h -o "%i %j" | awk '$$2 ~ /^nf-darkSet_/ {print $$1}'); \
	 n=$$(echo "$$ids" | grep -c . || true); \
	 if [ "$$n" = 0 ]; then echo "no nf-darkSet_ tasks left in the queue"; exit 0; fi; \
	 if squeue -u $$USER -h -o '%j' | grep -q '^nfdrv-'; then \
	   echo "a driver is still running:"; squeue -u $$USER -o '%.12i %.2t %.10M %j' | grep nfdrv-; \
	   echo "its tasks are not orphans; stop it first (make stop-dark-set RUN=<run>)"; exit 1; fi; \
	 echo "cancelling $$n orphaned nf-darkSet_ task(s) -- slurmctld throttles this, give it a minute"; \
	 echo "$$ids" | xargs scancel 2>&1 | grep -v _add_delay || true; \
	 echo "left: $$(squeue -u $$USER -h -o '%j' | grep -c '^nf-darkSet_' || true)"

# What happened. Reads .nextflow.log and says whether the run ended, which task took it
# down and what that task printed, per-process counts, and the head jobs from sacct. This
# is the question a dead tmux session leaves behind, and the answer used to be grepping a
# 400 KB log for ERROR and then walking into the work dir. HISTORY=1 gives one line per
# rotated log instead; LOG=.nextflow.log.2 picks an older run.
# The script is scripts/nf-postmortem at the repo root, one copy shared with the qfo
# benchmark's `make postmortem` (PR #35), for the same reason nf-head lives there.
status-dark-set: check-run
	@cd $(if $(filter ladder,$(RUN)),$(LADDER_LAUNCH_DIR),$(LAUNCH_DIR)) && \
	  $(MAKEFILE_DIR)/../../scripts/nf-postmortem $(if $(HISTORY),--history,--jobname $(HEAD_JOBNAME) $(LOG))

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
	$(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
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
