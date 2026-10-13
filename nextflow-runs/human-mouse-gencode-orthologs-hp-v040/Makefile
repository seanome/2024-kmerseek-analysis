OUTDIR := $(HOME)/data/gencode/results-human-mouse-orthologs-hp-v040

.PHONY: multiqc

# Regenerate the MultiQC summary directly from the existing kmer_sweep_summary.json,
# without re-running the pipeline.
multiqc:
	bin/make_multiqc_input.py \
	    $(OUTDIR)/kmer_sweep_summary.json \
	    /tmp/human_mouse_orthologs_hp_v040_mqc_input \
	    $(OUTDIR)/ortholog_stats.txt \
	    $(HOME)/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa \
	    $(HOME)/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \
	    /tmp/human_mouse_orthologs_hp_v040_mqc_input/ \
	    --config /tmp/human_mouse_orthologs_hp_v040_mqc_input/multiqc_config.yaml \
	    --outdir $(OUTDIR) \
	    --filename multiqc_report.html \
	    --force \
	    --no-megaqc-upload

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
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs-hp-v040"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
