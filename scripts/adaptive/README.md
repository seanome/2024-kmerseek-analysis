# Keep or eject a nanopore read by translated reduced-alphabet k-mers

`decide.py` is the whole rule: translate the first `--chunk` bases in six frames, encode them in `--alphabet` (protein, dayhoff or hp, tables copied from sourmash), and keep the read if at least `--min-kmers` of its k-mers of size `--k` are in the k-mer set of the `--reference` proteins. Standard library only.

```bash
python3 decide.py --reference family.fasta --reads reads.fastq --chunk 400 --alphabet protein --k 7 --min-kmers 2 > decisions.tsv
```

Output: one row per read with the k-mer count, the frame with the most hits, keep (1/0) and the decision time in seconds.
The other scripts here build notebook 292: `subsample_fastq.py` (reads), `sprot_table.py`, `build_references.py` and `family_candidates.py` (references), `run_truth_diamond.sh` and `truth_table.py` (truth), `score_chunks.py` (every setting, real and shuffled reads), `make_chunks.py` and `run_comparators.sh` (minimap2 and DIAMOND).
Run order and the exact commands are in `run_nb292.sh`.
