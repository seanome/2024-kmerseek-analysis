# DisProt disorder benchmark: does kmerseek find homologs through disordered regions?
#
# Human DisProt proteins against nine QfO proteomes, scored against the Pfam pair truth
# and split by predicted disorder. The kmerseek arm sweeps every alphabet over its ksize
# range, the same matrix as qfo-pfam-region-benchmark and invertebrate-dark-set, and
# reuses the region benchmark's target indexes on Sherlock.
#
#   make run-stub                 wiring check, no containers, no data, seconds
#   make run-mini                 laptop: two combos, two targets, real data
#   make push-python-image        build + push the Python image (once per change)
#   make sherlock-clone           clone this branch under $SCRATCH on Sherlock
#   make sync-data                ship the Pfam pair truth and the frozen DisProt TSV
#   make prefetch-images          (on Sherlock) pull the four containers into the cache
#   make run-sweep-sbatch         (on Sherlock) the whole sweep, head as a batch job
#   make watch / make stop        attach to / shut down that head
#   make pull-report              copy the report and metrics back to the laptop
#
# Every `run-*` target blocks in the foreground: run it inside your own tmux and watch.
# Never wrapped in tmux here, and never piped through tee, which makes nextflow reprint
# the whole task table on every update.

SHELL := /bin/bash
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
REPO_ROOT    := $(abspath $(MAKEFILE_DIR)/../..)

MAC_PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
ifneq ($(wildcard $(MAC_PYTHON)),)
PYTHON  ?= $(MAC_PYTHON)
MULTIQC ?= $(dir $(MAC_PYTHON))multiqc
else
PYTHON  ?= python3
MULTIQC ?= multiqc
endif

# Pinned on the mac, from the module tree on Sherlock (`ml load biology nextflow`).
MAC_NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
ifneq ($(wildcard $(MAC_NEXTFLOW)),)
NEXTFLOW ?= $(MAC_NEXTFLOW)
PROFILE  ?= standard
ON_MAC   := 1
else
NEXTFLOW ?= nextflow
PROFILE  ?= sherlock
ON_MAC   :=
endif
# The nf-core-v2 JVM; nextflow cannot find it from a tool shell without this.
NEXTFLOW_JVM := /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm
ifneq ($(wildcard $(NEXTFLOW_JVM)/bin/java),)
export NXF_JAVA_HOME := $(NEXTFLOW_JVM)
endif

# ---- where things are, on each side ----------------------------------------------------
SCRATCH_DIR ?= $(if $(SCRATCH),$(SCRATCH),$(HOME)/data)
DATABASE    ?= disprot
OUTDIR      ?= $(SCRATCH_DIR)/disprot-benchmark/$(DATABASE)/results

# Inputs the pipeline needs that are not in git. On the laptop they are where the earlier
# runs left them; on Sherlock `make sync-data` puts them under $SCRATCH.
# results/ is gitignored, so a worktree has none; fall back to the main checkout's.
PFAM_PAIRS_CANDIDATES := $(REPO_ROOT)/results/pfam_benchmark/pairs \
                         /Users/olga/code/2024-kmerseek-analysis/results/pfam_benchmark/pairs
LOCAL_PFAM_PAIRS   := $(firstword $(foreach d,$(PFAM_PAIRS_CANDIDATES),$(if $(wildcard $(d)/.),$(d),)))
LOCAL_DISPROT_TSV  := $(HOME)/data/disprot-benchmark/disprot/results/disprot/disprot_human.tsv
REMOTE_DATA        := $(SCRATCH_DIR)/disprot-benchmark/data
ifeq ($(ON_MAC),1)
PFAM_PAIRS  ?= $(LOCAL_PFAM_PAIRS)
DISPROT_TSV ?= $(LOCAL_DISPROT_TSV)
QFO_DIR     ?= $(HOME)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
# The laptop has no shared index store; indexes go under OUTDIR.
DB_CACHE    ?= $(OUTDIR)
else
PFAM_PAIRS  ?= $(REMOTE_DATA)/pfam_pairs
DISPROT_TSV ?= $(REMOTE_DATA)/disprot_human.tsv
# Where qfo-pfam-region-benchmark's sherlock-clone puts the QfO release.
QFO_DIR     ?= $(SCRATCH_DIR)/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
# The region benchmark's results directory: its kmerseek_index/ holds one entry per
# (target, alphabet, ksize, mask) cell of this same matrix, so every index is a store hit.
DB_CACHE    ?= $(SCRATCH_DIR)/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/results
endif

# ---- images ----------------------------------------------------------------------------
# The tag is the build date and must match params.python_image in main.nf.
PYTHON_IMAGE_TAG    ?= 2026-09-18
PYTHON_IMAGE_REMOTE := docker.io/olgabot/disprot-benchmark-python:$(PYTHON_IMAGE_TAG)
# Sherlock compute nodes are amd64; an Apple Silicon build defaults to arm64.
DOCKER_PLATFORM := linux/amd64
# What prefetch-images pulls: every container main.nf names, read from the file rather
# than restated here, plus the one params-driven Python image.
CONTAINERS = $(PYTHON_IMAGE_REMOTE) \
             $(shell grep -h -o -E "params\.(kmerseek|metapredict)_image *= *'[^']+'" $(MAKEFILE_DIR)/nextflow.config | sed -E "s/.*'([^']+)'/\1/") \
             $(shell grep -h -o -E "container '[^']+'" $(MAKEFILE_DIR)/main.nf | sed -E "s/container '([^']+)'/\1/")

# ---- Sherlock ----------------------------------------------------------------------------
SHERLOCK_HOST ?= sherlock
REPO_URL      ?= https://github.com/seanome/2024-kmerseek-analysis.git
BRANCH        ?= $(shell git rev-parse --abbrev-ref HEAD)
SSH_MUX       := ssh -o ControlMaster=auto -o ControlPath=$$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m
RSYNC         := rsync -avz --progress -e "$(SSH_MUX)"
SHERLOCK_CLONE     ?= $$SCRATCH/2024-kmerseek-$(notdir $(BRANCH))
SHERLOCK_CLONE_DIR := $(SHERLOCK_CLONE)/nextflow-runs/disprot-benchmark
SLURM_ACCOUNT   ?= ayeletv
SLURM_PARTITION ?= hns
PREFETCH_MEM    ?= 48G
PREFETCH_TIME   ?= 2:00:00

# The head as a batch job, through scripts/nf-head (shared with the other Sherlock
# pipelines). Login nodes cap every process at ten CPU-minutes and SIGKILL past it, which
# a Nextflow head tracking ~5_000 tasks crosses in hours. Deliberately NOT in this
# pipeline's bin/: Nextflow stages bin/ into every task and hashes it, so a file there
# invalidates the resume cache.
HEAD_CPUS    ?= 2
HEAD_MEM     ?= 16G
HEAD_XMX     ?= 12g
export NXF_OPTS := -Xms1g -Xmx$(HEAD_XMX)
HEAD_QOS     ?= normal
HEAD_TIME    ?= 2-00:00:00
HEAD_SESSION ?= disprot
# Its own job name: watch/stop key on it, and the region benchmark and the dark set run
# heads of their own at the same time.
HEAD_JOBNAME ?= nfdrv-disprot
HEAD_SUBMIT = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(SLURM_PARTITION) \
              HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
              HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
              HEAD_SESSION=$(HEAD_SESSION) HEAD_JOBNAME=$(HEAD_JOBNAME) \
              HEAD_HINT_WATCH="make watch" HEAD_HINT_STOP="make stop" \
              $(REPO_ROOT)/scripts/nf-head

NF_ARGS ?=
# Per-process fan-out and the submission throttle, the region benchmark's values for a
# sweep of this shape.
MAX_FORKS  ?= 100
QUEUE_SIZE ?= 400

.PHONY: help run-stub run-mini run-sweep run-sweep-sbatch watch stop multiqc \
        build-python-image push-python-image sherlock-clone sync sync-data prefetch-images \
        pull-report check-inputs require-cluster require-mac

help:
	@sed -n '2,20p' $(MAKEFILE_LIST)

require-cluster:
	@command -v sbatch >/dev/null 2>&1 || { echo "run this on Sherlock"; exit 1; }
require-mac:
	@[ -n "$(ON_MAC)" ] || { echo "run this on the laptop"; exit 1; }

check-inputs:
	@[ -d "$(QFO_DIR)/Eukaryota" ] || { echo "QFO_DIR has no Eukaryota/: $(QFO_DIR)"; exit 1; }
	@[ -d "$(PFAM_PAIRS)" ] || { echo "no Pfam pair truth at $(PFAM_PAIRS) (make sync-data)"; exit 1; }
	@[ -s "$(DISPROT_TSV)" ] || { echo "no frozen DisProt TSV at $(DISPROT_TSV) (make sync-data)"; exit 1; }
	@echo "inputs:  qfo=$(QFO_DIR)"; echo "         pairs=$(PFAM_PAIRS)"; echo "         disprot=$(DISPROT_TSV)"
	@echo "outdir:  $(OUTDIR)"
	@n=$$(ls -d $(DB_CACHE)/kmerseek_index/*.kmerseek.rocksdb 2>/dev/null | wc -l | tr -d ' '); \
	 echo "indexes: $(DB_CACHE)/kmerseek_index ($$n stored; a full sweep names 1647 per mask setting)"; \
	 if [ -z "$(ON_MAC)" ] && [ "$$n" = 0 ]; then \
	   echo "         none found: every index will be BUILT. If the region benchmark's store is"; \
	   echo "         elsewhere, pass DB_CACHE=<its results dir> (the one holding kmerseek_index/)."; fi

# -resume, single dash. `--resume` is a pipeline param and silently does nothing.
# One definition for every real run so the foreground and the sbatch driver cannot drift.
NF_RUN = $(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile $(PROFILE) -resume \
	  --database $(DATABASE) --qfo_dir $(QFO_DIR) --pfam_pairs_dir $(PFAM_PAIRS) \
	  --disprot_tsv $(DISPROT_TSV) --outdir $(OUTDIR) --db_cache $(DB_CACHE) \
	  --max_forks $(MAX_FORKS) --queue_size $(QUEUE_SIZE) $(NF_ARGS)

# ---- checks that need no cluster ---------------------------------------------------------
# Every stub: block, local executor, no containers. Verifies that the sweep expands, the
# index/search rejoin works and the results regroup per species, in seconds.
# Nextflow checks that path inputs exist even under -stub-run, so the QfO tree is
# empty placeholder files under stub-results/.
STUB_DIR := $(MAKEFILE_DIR)/stub-results
run-stub:
	@mkdir -p $(STUB_DIR)/qfo/Eukaryota $(STUB_DIR)/qfo/Bacteria $(STUB_DIR)/pairs
	@for f in Eukaryota/UP000005640_9606 Eukaryota/UP000000589_10090 Bacteria/UP000000625_83333; do \
	   printf '>x\nM\n' > $(STUB_DIR)/qfo/$$f.fasta; done
	$(NEXTFLOW) run $(MAKEFILE_DIR)/main.nf -profile stub -stub-run \
	  --qfo_dir $(STUB_DIR)/qfo --pfam_pairs_dir $(STUB_DIR)/pairs --disprot_tsv $(MAKEFILE_DIR)/Makefile \
	  --outdir $(STUB_DIR)/results --run_tag stub \
	  --kmerseek_encodings hp_thomas_dill2,protein20 --target_species mouse,ecoli $(NF_ARGS)

# Two combos against two targets on the laptop, real data, Docker. Needs the Python image
# built locally (make build-python-image) and the kmerseek image pulled.
# Its own outdir (and so its own index store): the real results directory must not get
# a two-target aggregate written over its figures and report.
run-mini: OUTDIR := $(SCRATCH_DIR)/disprot-benchmark-mini/$(DATABASE)/results
run-mini: require-mac check-inputs
	$(NF_RUN) --run_tag mini --target_species mouse,ecoli \
	  --kmerseek_combos hp_thomas_dill2:26,protein20:10

# ---- the sweep on Sherlock -------------------------------------------------------------
# Foreground, for a shell that is already on a compute node. Not for a login node.
run-sweep: require-cluster check-inputs
	$(NF_RUN)

run-sweep-sbatch: require-cluster check-inputs
	@if squeue -u $$USER -h -o '%j' 2>/dev/null | grep -qx '$(HEAD_JOBNAME)'; then \
	  echo "a driver named $(HEAD_JOBNAME) is already queued or running:"; \
	  squeue -u $$USER -o '%.12i %.2t %.10M %j' | grep '$(HEAD_JOBNAME)'; \
	  echo; echo "watch it with:  make watch"; echo "or stop it with: make stop"; exit 1; fi
	cd $(MAKEFILE_DIR) && $(HEAD_SUBMIT) $(NF_RUN)

watch: require-cluster
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 if [ -z "$$jid" ]; then \
	   pend=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i %T %r"); \
	   if [ -n "$$pend" ]; then echo "the driver has not started yet:"; echo "$$pend" | sed 's/^/    /'; \
	   else echo "no driver named $(HEAD_JOBNAME); start one with: make run-sweep-sbatch"; fi; exit 1; fi; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(HEAD_SESSION)

# Ctrl-C the run so it cancels its own jobs; a killed head leaves its tasks running and
# an orphaned index build finishing under the next run collides on storeDir.
stop: require-cluster
	@jid=$$(squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | head -1); \
	 if [ -z "$$jid" ]; then echo "no driver named $(HEAD_JOBNAME) is running"; \
	 else echo "asking driver $$jid to shut down"; scancel --signal=TERM --batch $$jid; \
	   for i in $$(seq 1 30); do squeue -u $$USER -h -n $(HEAD_JOBNAME) -o "%i" | grep -q . || break; sleep 5; done; fi
	@left=$$(squeue -u $$USER -h -o '%j' | grep -c '^nf-' || true); \
	 [ "$$left" = 0 ] || echo "$$left nf-* task(s) still queued; cancel with: squeue -u $$USER -h -n ... | xargs scancel"

# Re-render MultiQC over an existing results directory, no searching.
multiqc: check-inputs
	rm -rf $(MAKEFILE_DIR)/mqc_input
	$(PYTHON) $(MAKEFILE_DIR)/bin/make_multiqc_input.py $(OUTDIR)/all_disprot_metrics.parquet $(MAKEFILE_DIR)/mqc_input
	$(MULTIQC) $(MAKEFILE_DIR)/mqc_input/ --config $(MAKEFILE_DIR)/mqc_input/multiqc_config.yaml \
	  --outdir $(OUTDIR) --filename multiqc_report.html --force --no-version-check --no-ansi

# ---- images ------------------------------------------------------------------------------
build-python-image: require-mac
	docker build --platform $(DOCKER_PLATFORM) -f $(MAKEFILE_DIR)/Dockerfile.python \
	  -t $(PYTHON_IMAGE_REMOTE) $(MAKEFILE_DIR)

push-python-image: build-python-image
	docker push $(PYTHON_IMAGE_REMOTE)

# Nextflow finds a pre-pulled image by an exact filename: the container ref with '/' and
# ':' turned into '-', plus .img. Pulled inside a job: mksquashfs sizes its cache off the
# node's RAM, not the login cgroup, and dies there with "Out of memory (cache_alloc)".
prefetch-images: require-cluster
	@mkdir -p $${SCRATCH}/apptainer-cache $${SCRATCH}/apptainer-tmp
	@set -e; \
	export APPTAINER_CACHEDIR=$${SCRATCH}/apptainer-cache APPTAINER_TMPDIR=$${SCRATCH}/apptainer-tmp; \
	for ref in $(CONTAINERS); do \
	  name=$$(echo "$$ref" | sed -e 's|^docker://||' -e 's|/|-|g' -e 's|:|-|g'); \
	  out="$${SCRATCH}/apptainer-cache/$${name}.img"; \
	  if [ -s "$$out" ]; then echo "have  $$name"; continue; fi; \
	  echo "pull  $$name"; rm -f "$$out" "$$out".pulling.* 2>/dev/null || true; \
	  srun --account=$(SLURM_ACCOUNT) --partition=$(SLURM_PARTITION) \
	       --mem=$(PREFETCH_MEM) --time=$(PREFETCH_TIME) --cpus-per-task=4 \
	       apptainer pull --name "$$out" "docker://$$ref" || { echo "FAILED: $$ref"; rm -f "$$out"; exit 1; }; \
	done; echo "all containers cached in $${SCRATCH}/apptainer-cache"

# ---- getting the code and the data to Sherlock ------------------------------------------
sherlock-clone: require-mac
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) '\
	  set -e; \
	  if [ ! -d $(SHERLOCK_CLONE)/.git ]; then \
	    git clone --branch $(BRANCH) $(REPO_URL) $(SHERLOCK_CLONE); \
	  else \
	    cd $(SHERLOCK_CLONE) && git fetch origin $(BRANCH) && git checkout $(BRANCH) && git pull --ff-only; \
	  fi; echo "ready: $(SHERLOCK_CLONE_DIR)"'
	@echo; echo "  ssh $(SHERLOCK_HOST)"; echo "  cd $(SHERLOCK_CLONE_DIR)"

# Code moves by git, never rsync.
sync: require-mac
	git push origin $(BRANCH)
	$(SSH_MUX) $(SHERLOCK_HOST) 'cd $(SHERLOCK_CLONE_DIR) && git pull --ff-only'

# The two inputs that are not in git: the Pfam pair truth (results/ is gitignored) and
# the frozen DisProt parse. Sherlock compute nodes have no outbound internet, so the
# pipeline cannot fetch DisProt there; shipping the TSV the laptop run published also
# pins the DisProt release.
sync-data: require-mac
	@[ -d "$(LOCAL_PFAM_PAIRS)" ] || { echo "no Pfam pairs at $(LOCAL_PFAM_PAIRS)"; exit 1; }
	@[ -s "$(LOCAL_DISPROT_TSV)" ] || { echo "no DisProt TSV at $(LOCAL_DISPROT_TSV); run the pipeline locally once"; exit 1; }
	$(SSH_MUX) $(SHERLOCK_HOST) 'mkdir -p $$SCRATCH/disprot-benchmark/data/pfam_pairs'
	$(RSYNC) $(LOCAL_PFAM_PAIRS)/ $(SHERLOCK_HOST):'$$SCRATCH/disprot-benchmark/data/pfam_pairs/'
	$(RSYNC) $(LOCAL_DISPROT_TSV) $(SHERLOCK_HOST):'$$SCRATCH/disprot-benchmark/data/disprot_human.tsv'

LOCAL_RESULTS ?= $(HOME)/data/disprot-benchmark/$(DATABASE)/sherlock
pull-report: require-mac
	mkdir -p $(LOCAL_RESULTS)/figures
	$(RSYNC) $(SHERLOCK_HOST):'$$SCRATCH/disprot-benchmark/$(DATABASE)/results/{multiqc_report.html,all_disprot_metrics.parquet,all_disprot_pr_curves.parquet,kmerseek_sweep_summary.parquet,disprot_benchmark_report.md}' $(LOCAL_RESULTS)/
	$(RSYNC) $(SHERLOCK_HOST):'$$SCRATCH/disprot-benchmark/$(DATABASE)/results/figures/' $(LOCAL_RESULTS)/figures/
	@echo "report: $(LOCAL_RESULTS)/multiqc_report.html"
