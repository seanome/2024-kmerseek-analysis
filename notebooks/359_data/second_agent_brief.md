# Notebook 359: brief for the second-agent check

Written 2026-10-20, before the run. After the Sherlock run, a fresh agent (one that has
not seen the task text or the notebook's prose) gets exactly this brief, the files listed
under "Inputs", and nothing else.

## Inputs

* `notebooks/359_data/run/`: `blank_exons.parquet`, `landings.parquet`, `scan.parquet`,
  `summary.csv`, `verdict.json`, `survivors.parquet`, `gate.json`, `build_summary.json`.
* Code: `notebooks/isoform_blank_exons_359.py`, `scripts/359_isoform_blank_exons/*.py`.
* The section "The rule, as written in the task, fixed now" of `preregistered.md`, and
  nothing else from that file.

## Task

Recompute from the raw tables, with your own code (not by calling `score_landings.py`):

1. The blank-set size, and whether any exon appears twice (the same gene, chromosome,
   strand and genomic start and end under two exon IDs, or the same exon reached through
   two isoforms).
2. For each setting: the region E-value threshold at which decoy landings are at most 1%
   of the blank set, the real and decoy landing counts there, and the net fraction.
   Confirm the real and decoy counts use the same threshold.
3. That landing coverage is computed on the exon's protein segment (residues of the call
   inside the segment over the segment's length), not on the call.
4. Rule 2: the share of landings on a named Swiss-Prot feature.
5. The scan's threshold, and that its decoy landings are at most 1% of the blank set; then
   the overlap of the scan's picks with kmerseek's landings.
6. The verdict under the rule.

Try to break each number. Write two lists for the PR body, "held" and "failed", each item
with the number you got and the number in `summary.csv` or `verdict.json`. Put your
scripts and their printed output in `notebooks/359_data/second_agent_check/`.
