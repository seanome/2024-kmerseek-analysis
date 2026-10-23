# Data sources for notebook 301

## Step 1

| File | Source |
|---|---|
| `step1_cases_nb244_swissprot.tsv` | the 87 rows of `tables/244_hero_candidates.csv` admitted under "all criteria", PR 78 at commit 9e6434f5 |
| `step1_queries_nb244_human_swissprot.fasta` | the human sequences in `tables/244_case_sequences.fasta`, same commit |
| `step1_decoys_nb244_swissprot.fasta` | written by `scripts/score_301_step1_scans.py` (seed 301) |

## Step 2

| File | Source |
|---|---|
| `step2_elm_true_positive_instances_under60.tsv` | `analysis/351_short_feature_boundaries/data/filtered/instances_kept.tsv` from closed PR 181, commit e08c5ca3. ELM instance table downloaded from http://elm.eu.org/instances.tsv?q=* on 2026-10-20 (ELM download format 1.4, 4,277 instances), filtered there to true positives under 60 residues whose coordinates fall inside the current UniProt sequence: 4,006 instances on 2,572 proteins. SHA-256 of this copy: dc3812d0af57f9db9581f05366644566c147d336ee67d401524561072ad55e6d |
| Swiss-Prot `uniprot_sprot.dat.gz` | downloaded by `scripts/301_step2.sbatch` from https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/ on the run date, with `reldate.txt` beside it. Not in git (too large); the release and SHA-256 are written to the run's `sprot_release.txt` |
| UniRef90 clusters | rest.uniprot.org, queried by `scripts/fetch_301_uniref90.py`; every answer is kept in `uniref90_responses.jsonl` |
