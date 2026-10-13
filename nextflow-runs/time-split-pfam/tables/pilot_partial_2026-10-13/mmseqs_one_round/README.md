# Pilot rescored with MMseqs2 as one round (`make mmseqs-one-round`)

Slurm job 47690191 at 3a12009. MMseqs2 searched the 248 tune-half and control queries with
`--num-iterations 1` (otherwise as in main.nf); kmerseek, phmmer and hmmscan outputs are the
same as in `../`. The sample test half was not searched.

| | 3 rounds (`../`) | 1 round (here) |
|---|---|---|
| tune decoy hits at E < 1e-5 | 84 | 0 |
| best tune decoy hit | E 1.1e-22 (DECOY_A0A2P4ZFP6) | E 0.026 |
| MMseqs2 tune cutoff | E 1.5e-22 | E 0.095 |
| real queries with an MMseqs2 hit | 102 | 107 |
| kmerseek settings with a unique find | 35 | 35 |
| P4 pick | mmseqs12 k9 extend, 3 finds | same |

The 3-round profile search gave shuffled decoys strong hits; one round does not. The unique
finds do not change, because all three come from the landing rule (README in `../`): with one
round MMseqs2 aligns A0A2K6L9H6 end to end with its orthologs (E 1.9e-178 over 1-295), so the
41-aa LysM domain is 14% of each call.
