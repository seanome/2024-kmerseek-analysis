# SCOPe40 hit lists

Every tool of the leaderboard's Table 1 searched all against all on ASTRAL SCOPe 2.08 at 40%
identity (15,177 domains), with each search's whole hit list kept, then scored per query by
`bin/score_hitlist.py`. The scorer follows Foldseek's benchmark script
(`bench.noselfhit.awk`, github.com/steineggerlab/foldseek-analysis) with LoL-align's
denominators (soedinglab/LoLalign-analysis, the same script except lines 26-27: the
superfamily and fold counts leave the query out). `sensitivity` is on LoL-align's scale and
`sensitivity_foldseek_script` on Foldseek's original one; `make test` checks both against the
two scripts on made-up hit lists. The leaderboard's SCOPe40 page reads the scored rows.

What runs where:

| step | tools | input |
|---|---|---|
| `make score-kmerseek` | kmerseek, 16 settings, real and shuffled queries | the pair tables the scope40-alphabet-ksize-v040 combination runs kept on Sherlock (no search) |
| `make run-sbatch` | Foldseek | the domain structures, `pdbstyle-2.08` (experimental, from the PDB) |
| | ProstT5 + Foldseek, MMseqs2 (one round and three), phmmer, jackhmmer (3 rounds), HH-suite (hhsearch, single-sequence profiles) | the domain sequences, and the same queries shuffled with their dipeptide counts kept (k = 2, seed 1) |

Not here: hmmscan (it scores a sequence against Pfam models, not one domain against another)
and the Kyte-Doolittle composition scan of notebook 231 (its ranking of targets is not defined
yet).

## Launch (Sherlock)

```bash
cd $SCRATCH/2024-kmerseek-scope40-hitlists/nextflow-runs/scope40-hitlists
make pdbstyle          # once, on the login node: 15,177 structures into $SCRATCH/scope/pdbstyle-2.08
make score-kmerseek    # 32 short array tasks, no search
make run-sbatch        # the searches; make watch to see the task table
```

Outputs: `$SCRATCH/scope/results-scope40-hitlists/{hits,scored,kmerseek}`. Each search gives
`<tool>.<setting>.<real|shuffled_k2_seed1>.hits.tsv.gz` (query, target, bit score, E-value) and
`scored/*.per_query.parquet` plus `*.hits_to_first_fp.parquet`.

## First asks, from measured runs

The closest measured task is one search of the QfO midi-plus run on Sherlock (trace
`run-midi-plus-all-qfo/qfo_pfam_region.2026-09-*.trace.txt`): 998 human proteins (570,697
residues) against the zebrafish proteome (25,698 proteins, 13,829,992 residues), 7.89e12
residue pairs. SCOPe40 against itself is 2,805,012 x 2,805,012 = 7.87e12 residue pairs, the
same. It has 15.2 times the queries, so a tool whose cost grows per query against the whole
database (hhsearch, jackhmmer) can take up to 15.2 x (2.8 / 13.8) = 3.1 times as long. The
prediction column is that range; the ask adds room above the top of it.

| task | measured analog (zebrafish task) | predicted for SCOPe40 | first ask |
|---|---|---|---|
| foldseekStructureDb | foldseekDb, 2 builds: 0.27 core-hours together, peak 0.7 GB | minutes | 8 cores, 8 GB, 2 h |
| foldseekSearch | 35 s on 16 cores, peak 1.5 GB | 1 to 2 min | 16 cores, 8 GB, 2 h |
| prostt5Db (x2: real, shuffled) | 3 builds: 5.3 core-hours and 0.66 GPU-hours together, peak 2.9 GB | about 0.2 GPU-hours each | 8 cores, 1 GPU, 32 GB, 2 h |
| prostt5Search (x2) | 25 s on 16 cores, peak 1.5 GB | 1 to 2 min | 16 cores, 8 GB, 2 h |
| mmseqsSearch, one round (x2) | 6 s on 16 cores, peak 1.1 GB | under 1 min | 16 cores, 8 GB, 2 h |
| mmseqsSearch, three rounds (x2) | 22 s on 16 cores, peak 1.1 GB | about 1 min | 16 cores, 8 GB, 2 h |
| hmmerSearch phmmer (x2) | 5 min 21 s on 8 cores, peak 4.9 GB | 5 to 17 min | 8 cores, 8 GB, 2 h |
| hmmerSearch jackhmmer -N 3 (x2) | 24 min 32 s on 8 cores, peak 5.2 GB | 25 to 76 min | 8 cores, 8 GB, 6 h |
| hhDb (x2) | hhblitsBuildDB, 7 builds: 30.9 core-hours at 8 cores, about 33 min each, peak 1.8 GB | about 10 min | 8 cores, 8 GB, 2 h |
| hhSearch (x2) | 1 h 49 min on 16 cores, peak 23.4 GB | 1.8 to 5.6 h | 16 cores, 48 GB, 12 h |
| scoreHits (x13) | not measured (new); largest hit list about 15,177 x 2,000 = 30 M rows | under 10 min | 2 cores, 16 GB, 1 h |
| score_existing_kmerseek (x32) | not measured (new); largest table 8.6 GB of parquet (dayhoff6 k6) | under 30 min | 4 cores, 128 GB, 2 h |

First usable result: the kmerseek rows, within about an hour of `make score-kmerseek`. The
whole search run is set by hhsearch, about 2 to 6 hours once its databases are built; every
other tool finishes inside the first hour. To get results sooner, run without HH-suite first:
`make run-sbatch NF_ARGS="--tools foldseek,prostt5,mmseqs2_seqseq,mmseqs2_iterative,hmmer3_phmmer,hmmer3_jackhmmer"`,
then the same with `--tools hhblits`.
