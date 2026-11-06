# Why KaMRaT contigs are short, and what could lengthen them (notebook 361)

Asked 2026-11-06. Job 51337178 (merge settings), `scripts/kamrat_merge_lengths.py`,
`scripts/kamrat_kmer_neighbours.py`, `scripts/plot_kamrat_merge_settings.py`,
`slurm/kamrat_merge_settings.sbatch`. Numbers: `lengths.*.tsv`, `young_132d.neighbours.tsv`,
`merge_settings_and_forks.tsv`.

![Merge settings and forks](merge_settings_and_forks.png)

**a.** Share of all contigs with 60+ bases, the shortest that can hold a peptide of 20 residues. Each
mark is one contrast. Circles sit behind the squares. **b.** Contigs of one k-mer in young 132 d, by the
k-mers one base further along in the table KaMRaT indexed, either strand.

- Selection is not the cause. Before the p < 0.05 cut, 43 to 49 % of contigs are one k-mer and 5 to 11 %
  have 60+ bases, in all six contrasts.
- Merge settings are not the cause. Across six settings in the three small contrasts, the share with
  60+ bases moves by about 1 point. Removing the correlation check adds almost nothing. The six
  settings use the same k-mer table, so k stays 31.
- Missing k-mers are rarely the cause. Only 1.6 % of one-k-mer contigs in young 132 d have no next
  k-mer on either side. 61.7 % have a fork, two or more possible next k-mers on one side. A merge stops
  at a fork. These forks sit in k-mers with high counts: a median top count of 112 in any sample when
  both sides fork, against 8 when one side has no next k-mer.
- The other 36.7 % have a single next k-mer and still did not merge. The likely reason is that the next
  k-mer forks, but this was not checked.

This was measured in one contrast. Whether the forks come from repeats, sequence variants or
sequencing errors was not tested.

## Options, none tested yet

| option | why it could help | cost and risk |
|---|---|---|
| Replace each kept k-mer with the reads holding it | A read is up to 150 bases, crosses forks and always passes 60 bases | One pass over the reads. Reads add sequence around the k-mer, and may join different copies of a repeat |
| Assemble those reads (rnaSPAdes or Trinity) | Longer than one read | Slower, and low-count genes may not assemble |
| Count k-mers at k=61 | Every k-mer can hold a peptide, and longer k-mers fork less | Recounts all reads, and shallow samples lose more k-mers because a read has fewer of them |
| Drop k-mers far below their neighbour's count | Removes forks caused by sequencing errors | Cheap, but only helps if errors cause the forks, and may lose variants |
