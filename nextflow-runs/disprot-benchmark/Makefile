NF := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

.PHONY: run-curated-mobidb run-comprehensive-disprot run-disprot-thomas-dill resume-disprot resume-disprot-thomas-dill resume-mobidb resume-disprot-foldseek resume-mobidb-foldseek multiqc

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

resume-mobidb-foldseek:
	$(NF) run main.nf -resume --database mobidb --mobidb_source curated

# Run MultiQC directly against existing results (skips the pipeline)
multiqc:
	bin/make_multiqc_input.py \
	    $(HOME)/data/disprot-benchmark/disprot/results/all_disprot_metrics.parquet \
	    /tmp/disprot_mqc_input
	/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \
	    /tmp/disprot_mqc_input/ \
	    --config /tmp/disprot_mqc_input/multiqc_config.yaml \
	    --outdir $(HOME)/data/disprot-benchmark/disprot/results \
	    --filename multiqc_report.html \
	    --force \
	    --no-megaqc-upload
