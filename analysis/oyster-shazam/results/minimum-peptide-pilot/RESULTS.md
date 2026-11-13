# Minimum 20-aa stop-and-X pilot (superseded)

> Superseded. This pilot splits at stop and X with a 20-aa minimum, which is not the current rule (split only at stops, keep X, at least 30 aa; see `HANDOFF-20261025.md`). Do not compare its numbers with the corrected pilot table in `results/migration-20261025/STATUS.md`.

1000 systematic DE contigs,1000 systematic first-batch SPLASH extensions; entire small public run

Both arms split all six frames at stop/X. Reference matching requires at least one shared sampled hash; it does not establish function. DNA was not changed.

| Group | Original sequences | With ≥1-aa stretch | With ≥20-aa stretch | Stretches kept / all | Amino acids kept / all |
|---|---:|---:|---:|---:|---:|
| splash | 1,000 | 1,000 | 989 | 4,151 / 17,591 (23.6%) | 129,194 / 233,826 (55.3%) |
| dekupl | 1,000 | 1,000 | 341 | 1,137 / 10,694 (10.6%) | 32,866 / 110,062 (29.9%) |
| reference | 1,881 | 1,881 | 1,823 | 6,812 / 26,368 (25.8%) | 193,413 / 344,777 (56.1%) |

| Group | Encoding | Scale | Hashes retained | Signature bytes retained | Matches kept / previous |
|---|---|---:|---:|---:|---:|
| splash | protein | 10 | 71.7% | 78.3% | 7 / 10 |
| dekupl | protein | 10 | 45.9% | 54.5% | 2 / 5 |
| reference | protein | 10 | 74.3% | 74.7% | — |
| splash | protein | 100 | 72.5% | 86.2% | 1 / 2 |
| dekupl | protein | 100 | 44.7% | 73.1% | 0 / 0 |
| reference | protein | 100 | 74.1% | 74.8% | — |
| splash | dayhoff | 10 | 86.4% | 90.4% | 18 / 23 |
| dekupl | dayhoff | 10 | 69.6% | 75.6% | 8 / 12 |
| reference | dayhoff | 10 | 86.4% | 86.6% | — |
| splash | dayhoff | 100 | 86.5% | 94.8% | 5 / 5 |
| dekupl | dayhoff | 100 | 70.9% | 88.8% | 1 / 2 |
| reference | dayhoff | 100 | 86.1% | 86.6% | — |
| splash | hp | 10 | 100.0% | 100.0% | 0 / 0 |
| dekupl | hp | 10 | 100.0% | 100.0% | 0 / 0 |
| reference | hp | 10 | 100.0% | 100.0% | — |
| splash | hp | 100 | 100.0% | 100.0% | 0 / 0 |
| dekupl | hp | 100 | 100.0% | 100.0% | 0 / 0 |
| reference | hp | 100 | 100.0% | 100.0% | — |

Exact per-segment Branchwater/API comparisons: 163,959 baseline; 36,300 at ≥20aa. Hash-subset checks passed for every original sequence and setting; HP30 unchanged. Signature bytes are gzip collections, excluding indexes and intermediate peptide ZIPs. Systematic samples are diagnostic and are not random uncertainty estimates.
