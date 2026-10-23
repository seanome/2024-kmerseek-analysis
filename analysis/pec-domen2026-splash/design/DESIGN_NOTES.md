# Frozen sample-to-contrast table (2026-10-23)

`samples.tsv` and the `contrast_*.cjs` / `contrast_*.design.tsv` files were built by
`scripts/freeze_design.py` from SI Dataset S1 (`pnas.2610968123.sd01.xlsx`, sheet "DEseq group
annotation", sha256 4e2f2164…a0072eba) and the ENA file report (sha256 d363ed32…1c687db6). They
were committed before any contrast was run, and are not edited afterwards. Log: `design_freeze.log`.

| contrast | samples | PEC | pairs | genotypes |
|---|---|---|---|---|
| young_2h | 12 | 6 | 6 | 6941c, 6990b, 6996m, 6996n |
| young_24h | 12 | 6 | 6 | 6941c, 6990b, 6996m, 6996n |
| old_24h | 4 | 2 | 2 | 944aXBYd196-4 |
| young_13_49d | 8 | 4 | 3 (+2 unpaired) | 6941c, 6990b, 6996m |
| young_132d | 6 | 3 | 0 (ages 2 or more days apart) | 6990b, 6996m, 6996n |
| old_41_93d | 5 | 3 | 2 (+1 unpaired) | 944aXBYd196-4 |

47 of the 64 runs are in a contrast, the same 47 the paper analysed. The other 17 are the
14 libraries with under 10,000 read pairs and 3 that the SI leaves out (JA132 is under 10,000; check
`samples.tsv`, column `contrast` = `not_in_SI`).

## Where the SI differs from the handoff's guess from sample names

The handoff guessed time points from the ENA aliases and got 4 + 4 pairs at 2 h and 24 h. The SI
puts 8 more samples, sampled 15 to 16 days after the first PEC session, into those groups:

| sample | handoff guess | SI group |
|---|---|---|
| JA123, JA124, JA125, JA126 | 13–19 d | young 2 h |
| JA127, JA128, JA129, JA130 | 13–19 d | young 24 h |

So the paper's 2 h and 24 h groups each pool two sampling days, 2 weeks apart, from the same
colonies. Two genotypes (6996n and 6941c) therefore appear twice in each group. These samples are
not independent colonies, which is why the primary anchor-count model blocks on pair.

The old 24 h group has 4 samples in the SI (2 pairs); the paper's text says n = 5.
