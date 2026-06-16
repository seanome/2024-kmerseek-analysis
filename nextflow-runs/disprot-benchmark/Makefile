NF := /Users/olga/anaconda3/envs/nf-core-v2/bin/nextflow

.PHONY: run-curated-mobidb run-comprehensive-disprot run-disprot-thomas-dill resume-disprot resume-disprot-thomas-dill resume-mobidb resume-disprot-foldseek resume-mobidb-foldseek

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
