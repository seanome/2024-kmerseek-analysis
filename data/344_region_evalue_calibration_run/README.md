# Notebook 344 run records

Copied from the Sherlock run of `nextflow-runs/region-evalue-calibration` (clone
`/scratch/users/olgabot/2024-kmerseek-analysis-nb344`, commit `6e46f61`, image
`olgabot/kmerseek:2026-10-18-nb344-0a22970`, digest `sha256:9ba4a7f0...`), 2026-10-19.

- `index.log`: kmerseek version, commit `0a229700`, and the Karlin-Altschul fit the index stored.
- `<kind>.summary.tsv`: per search chunk, region rows reported, rows kept (any E-value at most 1),
  rows with no `region_ka_evalue`, decoys with at least one row.
- `measured.json`: the search memory and time `make measure` set from the pilot.
- `*.trace.txt`: the Nextflow traces of the pilot and the full run.

The region files themselves (150 MB) stay on Sherlock under `run/results/regions/`;
`make pull-results` copies them for the notebook.
