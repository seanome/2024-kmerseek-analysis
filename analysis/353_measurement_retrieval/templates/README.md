# Templates for notebook 353

Fill these in; `mr353.py` reads them. Leave `synthetic` as `no` for real entries.

| file | who fills it | when |
|---|---|---|
| `analyst_queue.csv` | the analyst, from their existing queue | before anything is searched. One row per query, in arrival order. `source_accession` and `source_start` (0-based) say where the query came from, if anywhere; they are used to drop the query's own record. |
| `workflow_C_description.md` | the analyst | before the evaluation |
| `development_labels.csv` | blind reviewers | on development queries only, before `tune` |
| `timed_session.csv` | the analyst | during the evaluation. `mr353.py review-pack` writes it pre-filled with the order of workflows for each query. Three times in seconds: searching, reading the results, checking the source. List the evidence ids (or citations, for workflow C) kept for the investigation. |
| `blind_review.csv` | blind reviewers | after the evaluation. `review-pack` writes it pre-filled, with no workflow column. `useful` is yes only for a measurement the analyst had not seen before and kept for the investigation; `reason` is required for every yes. `previously_seen_by_analyst` must be yes or no. |
