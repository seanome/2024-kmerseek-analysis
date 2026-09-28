OUTDIR := $(HOME)/data/gencode/results-human-mouse-orthologs
NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
TODAY := $(shell date +%Y-%m-%d)
DOCKER_SETTINGS := $(HOME)/Library/Group Containers/group.com.docker/settings-store.json
DOCKER_MEMORY_MIB := 102400

# --- Sherlock targets below (see README-sherlock.md for the full walkthrough) ---
# build-image/push-image/sync-pipeline/sync-data run on your Mac; run/status run ON
# Sherlock (ssh sherlock; cd $SCRATCH/human-mouse-orthologs-analysis/nextflow-runs/human-mouse-gencode-orthologs).
IMAGE_TAG        ?= 2026-08-18
IMAGE_LOCAL      := kmerseek-python-sweep:latest
IMAGE_REMOTE     := docker.io/olgabot/kmerseek-python-sweep:$(IMAGE_TAG)
SHERLOCK_HOST    ?= sherlock
SHERLOCK_CHECKOUT := $$SCRATCH/human-mouse-orthologs-analysis
SHERLOCK_PIPELINE_DIR := $(SHERLOCK_CHECKOUT)/nextflow-runs/human-mouse-gencode-orthologs
LOCAL_DATA_DIR   := $(HOME)/data/gencode
DOCKER_PLATFORM  := linux/amd64

.PHONY: multiqc resume docker-memory set-docker-memory \
        build-image push-image sync-pipeline sync-data run status pull

# Regenerate the MultiQC summary directly from the existing kmer_sweep_summary.json,
# without re-running the pipeline.
multiqc:
	bin/make_multiqc_input.py \
	    $(OUTDIR)/kmer_sweep_summary.json \
	    /tmp/human_mouse_orthologs_mqc_input
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \
	    /tmp/human_mouse_orthologs_mqc_input/ \
	    --config /tmp/human_mouse_orthologs_mqc_input/multiqc_config.yaml \
	    --outdir $(OUTDIR) \
	    --filename multiqc_report.html \
	    --force \
	    --no-megaqc-upload

# -resume only re-executes tasks whose storeDir output is missing on disk (indices/,
# raw results.csv.zst) -- exactly the hp-kyte-doolittle / hp-thomas-dill /
# hp-thomas-dill-no-c / hp-pbotc-1st-ed gaps from the 201 investigation. Bump Docker
# Desktop's memory first (`make set-docker-memory`) or these will likely OOM-kill again,
# since `memory = '100 GB'` in nextflow.config is only a scheduling hint Docker doesn't
# enforce -- the real ceiling is Docker Desktop's own VM allocation.
resume:
	$(NEXTFLOW) run kmerseek_human_mouse_orthologs.nf -resume \
	    -with-report   kmerseek_orthologs.$(TODAY).report.html \
	    -with-timeline kmerseek_orthologs.$(TODAY).timeline.html \
	    -with-trace    kmerseek_orthologs.$(TODAY).trace.txt

# Docker Desktop's VM caps container memory regardless of nextflow.config's `memory =`
# directive -- check what it's actually set to before assuming a rerun has headroom.
docker-memory:
	docker info | grep "Total Memory"

# Bumps Docker Desktop's VM to $(DOCKER_MEMORY_MIB) MiB and restarts it -- kills any
# other running containers, so this is a separate, deliberate target rather than a
# dependency of `resume`. Confirm before running.
set-docker-memory:
	python3 -c "import json; p='$(DOCKER_SETTINGS)'; d=json.load(open(p)); d['MemoryMiB']=$(DOCKER_MEMORY_MIB); json.dump(d, open(p, 'w'), indent=2)"
	docker desktop restart

## ---- Sherlock: run these from your Mac ----

# The image is just polars/numpy/scipy/scikit-learn/pyarrow/requests (see
# docker/Dockerfile.python-sweep) -- no cargo build, no kmerseek binary, so this is fast
# regardless of what else is running locally.
build-image:
	docker build --platform $(DOCKER_PLATFORM) -t $(IMAGE_LOCAL) \
	  -f docker/Dockerfile.python-sweep .

push-image: build-image
	docker tag $(IMAGE_LOCAL) $(IMAGE_REMOTE)
	docker push $(IMAGE_REMOTE)
	@echo "If Apptainer already cached a stale image on Sherlock, delete it there before"
	@echo "resuming: rm \$$SCRATCH/apptainer-cache/olgabot-kmerseek-python-sweep-$(IMAGE_TAG).img"

# Pushes your local commits and fast-forwards the Sherlock checkout -- fails loudly instead
# of silently overwriting if the two have diverged, so commit on the Mac first. One-time
# checkout setup (sparse-checkout of both this pipeline dir AND notebooks/ortholog_analysis_
# utils.py -- the two .py deps this pipeline's bin/ scripts import) is in README-sherlock.md
# step 3; this target only fast-forwards an existing checkout.
sync-pipeline:
	git push origin $$(git branch --show-current)
	ssh $(SHERLOCK_HOST) 'cd $(SHERLOCK_CHECKOUT) && git pull --ff-only'

# Data (NOT code -- see README-sherlock.md) rsynced into this pipeline's own data/ subdir on
# Sherlock, matching KMERSEEK_DATA_DIR/KMERSEEK_EXTRA_DATA_DIRS/params.outdir/params.of_tsv in
# nextflow.config's sherlock profile. Excludes indices/ (74GB, not needed -- computeFamilyAuc/
# computeRbhF1/computeMetricLeaderboard never read it) and skips the human/mouse FASTAs
# entirely (only needed by the search chain, which params.skip_search=true never runs).
# --partial + --timeout so an interrupted 267GB transfer over ssh can resume instead of
# restarting from zero -- just rerun `make sync-data` again if it dies partway.
sync-data:
	ssh $(SHERLOCK_HOST) 'mkdir -p $(SHERLOCK_PIPELINE_DIR)/data/orthofinder'
	rsync -avz --partial --progress --timeout=120 -e ssh \
	  --exclude 'indices/' \
	  $(LOCAL_DATA_DIR)/results-human-mouse-orthologs/ \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/'
	rsync -avz --partial --progress --timeout=120 -e ssh \
	  $(LOCAL_DATA_DIR)/results-human-mouse-orthologs-hp-v040/ \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs-hp-v040/'
	rsync -avz --progress -e ssh \
	  "$(LOCAL_DATA_DIR)/data-for-orthofinder/OrthoFinder/Results_Mar03/Orthologues/Orthologues_gencode.v49.pc_translations/gencode.v49.pc_translations__v__gencode.vM38.pc_translations.tsv" \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/orthofinder/gencode.v49.pc_translations__v__gencode.vM38.pc_translations.tsv'

# Pulls just the three sweeps' outputs back -- not the (already-synced, unchanged) input
# results files -- into the SAME local paths notebooks 200/206 already read from.
pull:
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/206_family_auc/' \
	  $(OUTDIR)/206_family_auc/
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/206_family_auc_all_combos.csv' \
	  $(OUTDIR)/206_family_auc_all_combos.csv
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/200_rbh_f1/' \
	  $(OUTDIR)/200_rbh_f1/
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/200_rbh_f1_all_combos.csv' \
	  $(OUTDIR)/200_rbh_f1_all_combos.csv
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/200_metric_leaderboard/' \
	  $(OUTDIR)/200_metric_leaderboard/
	rsync -avz --progress -e ssh \
	  $(SHERLOCK_HOST):'$(SHERLOCK_PIPELINE_DIR)/data/results-human-mouse-orthologs/200_metric_leaderboard_all_combos.csv' \
	  $(OUTDIR)/200_metric_leaderboard_all_combos.csv

## ---- Sherlock: run this ON Sherlock (ssh sherlock; cd $SCRATCH/human-mouse-orthologs-analysis/nextflow-runs/human-mouse-gencode-orthologs) ----
##
## Nextflow's head process is lightweight but needs to survive an SSH disconnect -- run
## inside tmux, not bare on the login node. Each combo becomes its own sbatch job on `hns`
## (--account=ayeletv), up to 4 concurrent (maxForks, from the top-level process{} block --
## raise/lower depending on how `hns` is looking that day).
run:
	ml load nextflow && \
	nextflow run kmerseek_human_mouse_orthologs.nf -profile sherlock $(NF_ARGS) \
	    -with-report   kmerseek_orthologs_sherlock.$(TODAY).report.html \
	    -with-timeline kmerseek_orthologs_sherlock.$(TODAY).timeline.html \
	    -with-trace    kmerseek_orthologs_sherlock.$(TODAY).trace.txt \
	    2>&1 | tee nextflow_sherlock.$(TODAY).log

status:
	squeue -u $$USER --account=$${SLURM_ACCOUNT:-ayeletv}
