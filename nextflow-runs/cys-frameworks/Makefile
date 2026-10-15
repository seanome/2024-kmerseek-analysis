# Short disulfide-rich peptides by cysteine framework.
#
# Everything that runs kmerseek runs ON SHERLOCK: `pilot`, `run-phylum`, `run-genus`.
# (ssh sherlock; cd $SCRATCH/2024-kmerseek-analysis-cys-frameworks/nextflow-runs/cys-frameworks)
# On the Mac: `build-image`, `push-image`, `data`, `sync-data`, and `stub`, which checks the
# wiring with empty files and runs no kmerseek. Two pilot runs on the laptop crashed it
# (2026-10-14 and 2026-10-15): cys2 index tasks pass 60 GB each and four started at once.
#
# A pilot's numbers are not the experiment's numbers: its reference holds every toxin but
# only a sample of everything else, so precision comes out better than it is.

SHELL := /bin/bash

NF     := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
# Nextflow needs a JVM, and the one in the nf-core-v2 environment is under lib/jvm.
NXF_JAVA_HOME := /Users/olga/anaconda3/envs/nf-core-v2/lib/jvm
PYTHON := /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3

# kmerseek branch olgabot/cys-runs-040 at c39370c: v0.4.0 + PR 141 (an index entry keyed by
# its sequence) + cys2 (PR 147) + cys_hp3 (PR 148). The key fix has to be in, or a cys2
# index silently drops every target that shares a k-mer set with one already stored.
KMERSEEK_SHA   ?= c39370c4e8d3b1b89cee726945bb0aa1bf90a8a3
KMERSEEK_SRC   ?= /Users/olga/code/kmerseek-cys-runs
KMERSEEK_BIN   ?= $(KMERSEEK_SRC)/target/release

# IMAGE_TAG is the one place the image is named. Every run target passes
# --pipeline_image $(IMAGE_REMOTE), so the tag here is what runs even though
# nextflow.config also declares a default: two strings that must be edited together, where
# disagreeing produces results rather than an error, is the shape that costs a week.
IMAGE_TAG      ?= 2026-10-15-cys-frameworks-c39370c
IMAGE_LOCAL    := kmerseek-cys:latest
IMAGE_REMOTE   := docker.io/olgabot/kmerseek:$(IMAGE_TAG)

# $SCRATCH is set on Sherlock and not on the Mac, so the same Makefile finds the data in
# both places. sync-data copies the Mac's copy to the Sherlock one.
DATA      ?= $(if $(SCRATCH),$(SCRATCH)/data/cys-frameworks,$(HOME)/data/cys-frameworks)
# sherlock-dtn takes rsync and scp only, so it cannot expand $SCRATCH; this is its value.
SHERLOCK_DATA ?= /scratch/users/olgabot/data/cys-frameworks
SWISSPROT ?= $(DATA)/uniprot_sprot.fasta.gz
TABLES    := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../tables/cys_frameworks

.PHONY: help check-kmerseek build-image push-image data sync-data stub pilot run-phylum run-genus running require-sherlock require-cys2-memory watch stop

# --- The Nextflow head runs as a SLURM job -------------------------------------------------
# Sherlock kills any login-node process after ten CPU-minutes, and a login node that
# reboots takes a foreground head with it. scripts/nf-head (shared with deep-twilight and
# qfo-pfam-region-benchmark) submits the head as its own job, runs it in tmux on that
# node, and attaches you read-only to the live task table. Ctrl-b d detaches and leaves
# it running; `make watch RUN=pilot` attaches again; `make stop RUN=pilot` stops it so it
# cancels its own task jobs.
#
# hns, not normal: hns's partition QOS allows 7 days under `--qos=normal`, and the
# public normal partition stops at 2 days.
SLURM_ACCOUNT  ?= ayeletv
HEAD_PARTITION ?= hns
HEAD_QOS       ?= normal
HEAD_CPUS      ?= 2
HEAD_MEM       ?= 8G
# The pilot's slowest path is a cys2 task through all three attempts (4 + 8 + 12 h), plus
# time queued for an hns or bigmem node. The full runs are 3 or more references.
HEAD_TIME      ?= $(if $(filter pilot,$(MAKECMDGOALS)),2-00:00:00,7-00:00:00)
export NXF_OPTS := -Xms512m -Xmx6g

# One job name and tmux session per run folder, so watch and stop find the right head
# when the pilot and a full run are both up.
RUN          ?= pilot
HEAD_NAME     = nf-cys-$(1)
MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
HEAD_SUBMIT   = HEAD_ACCOUNT=$(SLURM_ACCOUNT) HEAD_PARTITION=$(HEAD_PARTITION) \
                HEAD_QOS=$(HEAD_QOS) HEAD_TIME=$(HEAD_TIME) \
                HEAD_CPUS=$(HEAD_CPUS) HEAD_MEM=$(HEAD_MEM) \
                HEAD_SESSION=$(call HEAD_NAME,$(1)) HEAD_JOBNAME=$(call HEAD_NAME,$(1)) \
                HEAD_HINT_WATCH="make watch RUN=$(1)" HEAD_HINT_STOP="make stop RUN=$(1)" \
                $(MAKEFILE_DIR)/../../scripts/nf-head

# Stops a target that runs kmerseek when it is started anywhere but Sherlock.
require-sherlock:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  This target runs kmerseek, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis-cys-frameworks/nextflow-runs/cys-frameworks"; \
	  echo "    make $(MAKECMDGOALS)"; \
	  echo ""; \
	  exit 1; \
	fi

help:
	@echo ""
	@echo "  make build-image  build the amd64 image from the pinned kmerseek commit"
	@echo "  make push-image   build it and push to $(IMAGE_REMOTE)"
	@echo "  make data         download ConoServer, Swiss-Prot and the NCBI taxonomy, then"
	@echo "                    build the query, reference and settings tables"
	@echo "  make sync-data    copy Swiss-Prot and reference_entries.csv to Sherlock"
	@echo "  make stub         check the wiring here with empty files; runs no kmerseek"
	@echo "  make pilot        ON SHERLOCK: 10 queries per framework, Cnidaria, a sampled"
	@echo "                    reference; proves the pipeline and measures memory"
	@echo "  make run-phylum CYS2_MEMORY_GB=<from the pilot>"
	@echo "                    ON SHERLOCK: 3 held-out references (one per phylum)"
	@echo "  make run-genus CYS2_MEMORY_GB=<from the pilot>"
	@echo "                    ON SHERLOCK: one per genus with >= 20 queries"
	@echo "  make watch RUN=pilot   ON SHERLOCK: attach to a run's live task table"
	@echo "  make stop RUN=pilot    ON SHERLOCK: stop a run; it cancels its own jobs"
	@echo "  make running      ON SHERLOCK: what is in the queue"
	@echo ""

# kmerseek is pinned by commit, and this checks the binary was built from that commit
# rather than trusting the path. A cys2 index built by anything older than PR 141 drops
# targets silently, which is the one failure that would not show up as an error.
check-kmerseek:
	@test -x $(KMERSEEK_BIN)/kmerseek || { \
	  echo "No kmerseek at $(KMERSEEK_BIN)/kmerseek."; \
	  echo "Build it:"; \
	  echo "  cd $(KMERSEEK_SRC)"; \
	  echo "  export SDKROOT=\$$(xcrun --show-sdk-path)"; \
	  echo "  export BINDGEN_EXTRA_CLANG_ARGS=\"-isysroot \$$SDKROOT\""; \
	  echo "  cargo build --release --bins"; \
	  exit 1; }
	@have=$$(git -C $(KMERSEEK_SRC) rev-parse HEAD); \
	 if [ "$$have" != "$(KMERSEEK_SHA)" ]; then \
	   echo "$(KMERSEEK_SRC) is at $$have, not the pinned $(KMERSEEK_SHA)."; \
	   echo "Check it out and rebuild, or pass KMERSEEK_SHA=$$have if that is deliberate."; \
	   exit 1; \
	 fi
	@echo "kmerseek $(KMERSEEK_SHA) at $(KMERSEEK_BIN)/kmerseek"

# The build context is the kmerseek worktree, not this repo, and the Dockerfile COPYs that
# worktree, so the image holds whatever commit is checked out there. check-kmerseek is a
# dependency so the tag cannot name a commit the image does not contain.
build-image: check-kmerseek
	docker build --platform linux/amd64 \
	  -t $(IMAGE_LOCAL) -t $(IMAGE_REMOTE) \
	  --build-arg GIT_SHA=$(KMERSEEK_SHA) \
	  -f $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/Dockerfile \
	  $(KMERSEEK_SRC)
	@echo ""
	@echo "kmerseek commit inside the image:"
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) cat /etc/kmerseek_git_sha
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) kmerseek --version
	@docker run --rm --platform linux/amd64 $(IMAGE_LOCAL) \
	  python3 -c "import polars; print('polars', polars.__version__)"

push-image: build-image
	docker push $(IMAGE_REMOTE)
	@echo ""
	@echo "Pin this digest in nextflow.config if you want the run to name an immutable image:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(IMAGE_REMOTE)

data:
	$(PYTHON) $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../scripts/download_cys_framework_data.py $(DATA)
	$(PYTHON) $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/../../scripts/build_cys_framework_data.py $(DATA) $(TABLES)

# The two input files, from the Mac to Sherlock's $SCRATCH (188 MB). The mkdir goes
# through `sherlock`, since sherlock-dtn runs no commands.
sync-data:
	@test -f $(SWISSPROT) && test -f $(DATA)/reference_entries.csv || { \
	  echo "No $(SWISSPROT) or $(DATA)/reference_entries.csv here. Run: make data"; exit 1; }
	ssh sherlock 'mkdir -p $(SHERLOCK_DATA)'
	rsync -av --partial $(SWISSPROT) $(DATA)/reference_entries.csv sherlock-dtn:$(SHERLOCK_DATA)/

# Every process has a stub: block that only touches its outputs, so this runs in seconds
# on the Mac and checks channels and file names. main.nf refuses a real run under
# `standard`, so this is the only thing `standard` can do.
stub:
	mkdir -p stub
	cd stub && \
	  NXF_JAVA_HOME=$(NXF_JAVA_HOME) \
	  $(NF) run ../main.nf -profile standard -stub-run \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level phylum \
	    --outdir $(CURDIR)/stub/results

# The pilot reference keeps every toxin and 20_000 other entries (27_570 for Cnidaria).
# Its numbers are not the experiment's; it proves the pipeline end to end on SLURM, and
# its trace's peak_rss is what sizes CYS2_MEMORY_GB for run-phylum and run-genus.
# cys2 tasks ask 180 GB on hns first (see kmerseekGb in main.nf), then bigmem.
pilot: require-sherlock
	mkdir -p pilot
	cd pilot && \
	  ml load biology nextflow && \
	  $(call HEAD_SUBMIT,pilot) nextflow run ../main.nf -profile sherlock -resume \
	    --pipeline_image $(IMAGE_REMOTE) \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level phylum \
	    --only Cnidaria \
	    --sample_non_toxins 20000 \
	    --queries_per_framework 10 \
	    --outdir $(CURDIR)/pilot/results

# The full runs index Swiss-Prot with one clade removed, about 575_700 entries, 21 times
# the pilot's reference. How cys2 memory grows with that is not known until the pilot
# reports peak_rss, so these targets make you set it rather than inherit the pilot's 180.
CYS2_MEMORY_GB ?=
require-cys2-memory:
	@test -n "$(CYS2_MEMORY_GB)" || { \
	  echo ""; \
	  echo "  Set CYS2_MEMORY_GB from the pilot's trace first: the largest cys2 peak_rss in"; \
	  echo "  pilot/cys_frameworks.<date>.trace.txt, scaled to a reference 21 times larger."; \
	  echo "  Example: make $(MAKECMDGOALS) CYS2_MEMORY_GB=700"; \
	  echo ""; \
	  exit 1; }

# On Sherlock, nextflow comes from the module and the data from $$SCRATCH, so DATA and NF
# are overridden there rather than hard-coded to Mac paths.
run-phylum: require-sherlock require-cys2-memory
	mkdir -p run-phylum
	cd run-phylum && \
	  ml load biology nextflow && \
	  $(call HEAD_SUBMIT,run-phylum) nextflow run ../main.nf -profile sherlock -resume \
	    --pipeline_image $(IMAGE_REMOTE) \
	    --cys2_memory_gb $(CYS2_MEMORY_GB) \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level phylum \
	    --outdir $(CURDIR)/run-phylum/results

run-genus: require-sherlock require-cys2-memory
	mkdir -p run-genus
	cd run-genus && \
	  ml load biology nextflow && \
	  $(call HEAD_SUBMIT,run-genus) nextflow run ../main.nf -profile sherlock -resume \
	    --pipeline_image $(IMAGE_REMOTE) \
	    --cys2_memory_gb $(CYS2_MEMORY_GB) \
	    --swissprot_fasta $(SWISSPROT) \
	    --reference_entries $(DATA)/reference_entries.csv \
	    --level genus \
	    --genus_min_queries 20 \
	    --outdir $(CURDIR)/run-genus/results

# The live task table of the head running in RUN's folder (pilot, run-phylum, run-genus).
watch: require-sherlock
	@jid=$$(squeue -u $$USER -h -n $(call HEAD_NAME,$(RUN)) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(call HEAD_NAME,$(RUN)) job is running"; squeue -u $$USER -h -n $(call HEAD_NAME,$(RUN)); exit 1; }; \
	 exec srun --jobid=$$jid --overlap --pty tmux -S /tmp/nf-head-$$jid attach -r -t $(call HEAD_NAME,$(RUN))

# Sends the head TERM; nf-head turns that into Ctrl-C, and Nextflow cancels its task jobs.
stop: require-sherlock
	@jid=$$(squeue -u $$USER -h -n $(call HEAD_NAME,$(RUN)) -o "%i %T" | awk '$$2=="RUNNING"{print $$1; exit}'); \
	 [ -n "$$jid" ] || { echo "no $(call HEAD_NAME,$(RUN)) job is running"; exit 1; }; \
	 scancel --signal=TERM --batch $$jid && echo "asked job $$jid to shut down"

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, then one row per job of the runs under
## this folder. Runs on Sherlock only, since it reads the queue.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/cys-frameworks"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
