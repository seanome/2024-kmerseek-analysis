# Notebook 353. Every target runs on the laptop except kmerseek (Sherlock; see README.md).
PY     ?= /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python
BLAST  ?= /Users/olga/anaconda3/envs/orthofinder/bin
DATA   ?= /Users/olga/data/353_measurement_retrieval
WORK   ?= $(DATA)/work
QUEUE  ?= $(DATA)/analyst_queue.csv
SYN    ?= $(DATA)/synthetic_run
THREADS ?= 4
M = $(PY) mr353.py

.PHONY: test evidence queries controls search-b sherlock-inputs synthetic-run cost-probe

test:
	$(PY) -m pytest -q -p no:cacheprovider tests/

# Real evidence collection: ELM + Sanborn 2021 tiles, with UniProt sequence versions.
evidence:
	$(M) --work $(WORK) import-evidence --raw $(DATA)/raw
	mkdir -p results/evidence && cat $(WORK)/evidence/manifest.json > results/evidence/manifest.json

# The steps below need the analyst's queue at $(QUEUE). There is none yet.
queries:
	test -s $(QUEUE) || { echo "no analyst queue at $(QUEUE); see templates/analyst_queue.csv"; exit 2; }
	$(M) --work $(WORK) import-queries --queue $(QUEUE)
controls: queries
	$(M) --work $(WORK) make-controls
search-b: controls
	$(M) --work $(WORK) search-b --blast-bin $(BLAST) --threads $(THREADS)
sherlock-inputs: controls
	$(M) --work $(WORK) kmerseek-jobs
	@echo "copy to Sherlock: rsync -av $(WORK)/sherlock_in/ dtn.sherlock.stanford.edu:/scratch/users/olgabot/353_measurement_retrieval/in/"

# Software check on SYNTHETIC fixtures, end to end. The verdict must be INSUFFICIENT_EVIDENCE.
synthetic-run:
	command rm -rf $(SYN)
	$(PY) fixtures/make_synthetic.py --work $(SYN) --step evidence
	$(M) --work $(SYN) import-queries --queue $(SYN)/synthetic_queue.csv
	$(M) --work $(SYN) make-controls
	$(M) --work $(SYN) search-b --blast-bin $(BLAST) --threads $(THREADS)
	$(PY) fixtures/make_synthetic.py --work $(SYN) --step stub
	$(M) --work $(SYN) read-kmerseek --dir $(SYN)/kmerseek_stub
	$(M) --work $(SYN) cards
	$(PY) fixtures/make_synthetic.py --work $(SYN) --step dev-labels
	$(M) --work $(SYN) tune --dev-labels $(SYN)/dev_labels.csv
	$(M) --work $(SYN) freeze --queue $(SYN)/synthetic_queue.csv
	$(M) --work $(SYN) session-pack
	$(PY) fixtures/make_synthetic.py --work $(SYN) --step fill-session
	$(M) --work $(SYN) review-pack --session $(SYN)/session_filled.csv
	$(PY) fixtures/make_synthetic.py --work $(SYN) --step fill-review
	$(M) --work $(SYN) score --review $(SYN)/review_filled.csv --session $(SYN)/session_filled.csv
	$(M) --work $(SYN) viewer --out $(SYN)/viewer.html --banner "SYNTHETIC FIXTURES, not data."
	mkdir -p results/synthetic_fixture_run
	for f in results/verdict.json results/per_query_outcomes.csv tuning_table.csv tuning_choice.json \
	         evaluation_plan.lock.json compute_log.jsonl viewer.html review/session_order.csv \
	         review/blind_review.csv; do cat $(SYN)/$$f > results/synthetic_fixture_run/$$(basename $$f); done

# Machine cost of workflow B on the real evidence collection, with synthetic queries.
cost-probe:
	command rm -rf $(DATA)/work_cost_probe $(DATA)/syn_queue_only
	mkdir -p $(DATA)/work_cost_probe && cp -R $(WORK)/evidence $(DATA)/work_cost_probe/
	$(PY) fixtures/make_synthetic.py --work $(DATA)/syn_queue_only --step evidence
	$(M) --work $(DATA)/work_cost_probe import-queries --queue $(DATA)/syn_queue_only/synthetic_queue.csv
	$(M) --work $(DATA)/work_cost_probe make-controls
	$(M) --work $(DATA)/work_cost_probe search-b --blast-bin $(BLAST) --threads $(THREADS)
	$(M) --work $(DATA)/work_cost_probe cards
	mkdir -p results/cost_probe && cat $(DATA)/work_cost_probe/compute_log.jsonl > results/cost_probe/compute_log.jsonl
