NF := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

.PHONY: run-curated-mobidb run-comprehensive-disprot run-disprot-thomas-dill resume-disprot resume-disprot-thomas-dill resume-mobidb resume-disprot-foldseek rerun-disprot-missing-arms resume-mobidb-foldseek multiqc

run-curated-mobidb:
	$(NF) run main.nf --database mobidb --mobidb_source curated --skip_foldseek

run-comprehensive-disprot:
	$(NF) run main.nf --database disprot --skip_foldseek

run-disprot-thomas-dill:
	$(NF) run main.nf --database disprot --kmerseek_encoding hp-thomas-dill --kmerseek_k_values 26 --skip_foldseek

resume-disprot:
	$(NF) run main.nf -resume --database disprot --skip_foldseek

resume-disprot-thomas-dill:
	$(NF) run main.nf -resume --database disprot --kmerseek_encoding hp-thomas-dill --kmerseek_k_values 26 --skip_foldseek

resume-mobidb:
	$(NF) run main.nf -resume --database mobidb --mobidb_source curated --skip_foldseek

resume-disprot-foldseek:
	$(NF) run main.nf -resume --database disprot

# Repeat only the arms whose published result is missing or empty (Foldseek after the
# 2026-09-20 downloader fix), reading the ground truth, disorder scores and the other
# arms' results from $(HOME)/data/disprot-benchmark/disprot/results. Needed because the
# old work/ was pruned, so -resume alone reruns everything, kmerseek 0.3.1 included.
rerun-disprot-missing-arms:
	$(NF) run main.nf -resume --database disprot --reuse_results true

resume-mobidb-foldseek:
	$(NF) run main.nf -resume --database mobidb --mobidb_source curated

# Run MultiQC directly against existing results (skips the pipeline)
multiqc:
	bin/make_multiqc_input.py \
	    $(HOME)/data/disprot-benchmark/disprot/results/all_disprot_metrics.parquet \
	    /tmp/disprot_mqc_input \
	    $(HOME)/data/disprot-benchmark/disprot/results/disprot/benchmark_stats.txt
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \
	    /tmp/disprot_mqc_input/ \
	    --config /tmp/disprot_mqc_input/multiqc_config.yaml \
	    --outdir $(HOME)/data/disprot-benchmark/disprot/results \
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
	  echo "    cd \$$SCRATCH/2024-kmerseek-analysis/nextflow-runs/disprot-benchmark"; \
	  echo "    make running"; \
	  echo ""; \
	  exit 1; \
	fi
	@python3 $(RUNNING_DIR)/../../scripts/nf-running --jobs "$(RUNNING_DIR)"
