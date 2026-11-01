OUTDIR := $(HOME)/data/gencode/results-human-mouse-orthologs
NEXTFLOW := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow
TODAY := $(shell date +%Y-%m-%d)
DOCKER_SETTINGS := $(HOME)/Library/Group Containers/group.com.docker/settings-store.json
DOCKER_MEMORY_MIB := 102400

.PHONY: multiqc resume docker-memory set-docker-memory

# Regenerate the MultiQC summary directly from the existing kmer_sweep_summary.json,
# without re-running the pipeline.
multiqc:
	bin/make_multiqc_input.py \
	    $(OUTDIR)/kmer_sweep_summary.json \
	    /tmp/human_mouse_orthologs_mqc_input \
	    $(OUTDIR)/ortholog_stats.txt \
	    $(HOME)/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa \
	    $(HOME)/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa
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

# --- What is running --------------------------------------------------------------------
## Every Nextflow run with jobs in the SLURM queue, from any pipeline, then one row per job
## of the runs under this folder. scripts/nf-running does the work. Runs on Sherlock only,
## since it reads the queue. Read TIME as squeue's [DD-[HH:]]MM:SS: `4:03` is four minutes
## and `1-02:19:10` is a day. PENDING jobs show 0:00.
RUNNING_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

.PHONY: running
running:
	@if [ -z "$$SCRATCH" ] || [ ! -d "$$SCRATCH" ]; then \
	  echo ""; \
	  echo "  'running' reads the SLURM queue, so it runs ON SHERLOCK, not on your Mac."; \
	  echo ""; \
	  echo "  ssh sherlock"; \
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
