# Extend every significant oyster anchor

Olga authorized all trimmed-read SPLASH anchors with adjusted P < 0.05 on 2026-10-23. The saved SPLASH scores contain 213,658 such anchors. These are sequence anchors, not distinct genes.

Job 49191753 submits 22 disjoint batches of at most 10,000 anchors, with two concurrent tasks. Each task requests eight CPUs, 32 GB and two hours. The last batch contains 3,658 anchors. The successful 300-anchor Compactors pilot took 40.30 seconds and 709,508 KiB maximum resident memory; a linear runtime estimate gives about 22 minutes per 10,000-anchor batch. Runtime and memory for the full selection have not yet been measured.

The input is the existing `splash-trimmed/splash.after_correction.scores.tsv`; statistical testing and trimming are not repeated. The driver verifies finite adjusted P values, unique anchors, a strict threshold and deterministic sequence ordering. Each batch records the score-file hash, selection and output hash. Existing validated outputs are reused only when their metadata and hashes match. An unfinished prior attempt stops for inspection.

Results go to `splash-trimmed-all/batch-NNN` beneath the analysis scratch directory. Original pilot outputs remain in `splash-trimmed`. Batch outputs still need aggregation, overlapping-fragment grouping and downstream annotation before interpreting genes or functions. Individual batches can contain overlapping DNA even though their input anchors are disjoint.

Local checks: Python compilation, Bash syntax, strict-threshold boundary (0.05 excluded), NaN rejection and deterministic sorting passed. An independent pre-submission partition check on Sherlock confirmed 213,658 unique selected anchors and 22 disjoint batches. Full Compactors execution is pending; submission is not a completed result.
