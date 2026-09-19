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
