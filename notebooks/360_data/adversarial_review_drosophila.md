Written by Claude Code (independent checker agent), 2026-11-01.

# Notebook 360 kill test, Drosophila: independent recount

I recounted the Drosophila result from the raw tables (DESeq2 tables, DIAMOND
blastx hits, kmerseek region files, the search-database Pfam table and the
answer key). I did not read the pipeline's `landings.py`, `standard_calls.py`,
the notebook builders or any notebook before my own numbers were done. I read
`results/summary.tsv` only after all my family sets were built.

## Verdict under the frozen rule: FAIL

- R = `hsdm17.k7`. It recovers 54 of the 100 changed families on the real
  labels at region E-value <= 1, with 44 families outside the changed set. The
  runner-up is `sdm12.k8` (50 recovered, 36 outside).
- The standard path (Trinity, DIAMOND blastx, best-hit Pfam, DESeq2 padj < 0.05)
  recovers 85, with 12 outside.
- Part 1: 54 >= 85 is false.
- Part 2: R's count over the 9 label swaps is 0 in every swap, so the median is
  0 and the bar is 5 x 0 = 0. 54 >= 0 is true, but see check 1: this part cannot fail.
- One part fails, so the result is FAIL. My numbers agree exactly with the
  pipeline summary.

## Recovered and outside families per setting (real labels)

"Recovered" = changed families reported. "Outside" = reported families that
are not among the 100 changed ones. Rows = rows in the published region file.
Finite E = rows whose `region_evalue` is a finite number (the rest are `inf`).

| setting | status | rows | finite E | rec. E<=1 | out. E<=1 | rec. E<=10 | out. E<=10 |
|---|---|---:|---:|---:|---:|---:|---:|
| hsdm17.k7 (R) | fitted | 161,472 | 79,670 | **54** | 44 | 60 | 329 |
| sdm12.k8 | fitted | 244,683 | 59,651 | 50 | 36 | 57 | 462 |
| uniprot18.k7 | fitted | 68,617 | 62,440 | 41 | 21 | 45 | 188 |
| hp_lehninger_hpc3.k19 | fitted | 7,999,212 | 311,665 | 31 | 1,611 | 65 | 5,258 |
| wass14.k8 | fitted | 54,455 | 27,286 | 24 | 2 | 31 | 11 |
| funcgroups8.k10 | fitted | 3,502,214 | 6,402 | 17 | 1 | 19 | 4 |
| hp_kyte_doolittle2.k30 | fitted | 933,543 | 49,899 | 15 | 4 | 24 | 109 |
| hp_thomas_dill2.k30 | fitted | 113,105 | 11,206 | 14 | 2 | 20 | 4 |
| mmseqs12.k8 | fitted | 519,889 | 959 | 14 | 0 | 20 | 3 |
| polarity4.k15 | fitted | 7,653,613 | 17,438 | 14 | 0 | 18 | 44 |
| gbmr4.k15 | fitted | 25,801,049 | 51,269 | 10 | 66 | 23 | 494 |
| hp_lehninger2.k30 | fitted | 7,906 | 6,831 | 3 | 0 | 9 | 0 |
| hp_lehninger_c_nonpolar2.k30 | fitted | 14,013 | 9,812 | 3 | 0 | 4 | 0 |
| hp_pbotc_1st_ed2.k30 | fitted | 2,260 | 1,264 | 3 | 0 | 9 | 1 |
| gbmr7.k11 | fitted | 610,126,369 | 72,473 | 0 | 15 | 27 | 1,662 |
| dayhoff6.k12 | fitted | 2,709,340 | 1 | 0 | 0 | 0 | 0 |
| hp_thomas_dill_no_c2.k30 | unfitted | 0 | 0 | 0 | 0 | 0 | 0 |
| wwmj5.k13 | unfitted | 0 | 0 | 0 | 0 | 0 | 0 |
| protein20.k7 (not eligible for R) | fitted | 24,162 | 19,678 | 30 | 2 | 38 | 10 |
| **standard path** | | | | **85** | 12 | | |

"Unfitted" means kmerseek refused the search because the index had no
Karlin-Altschul fit (the statistics that turn a score into an E-value) for that
setting's penalty, so it gave no regions.

## Checks

1. **DESeq2 significant features per labelling. HELD.** Real labels: 173,056 of
   341,828 k-mers and 113 of 228 Trinity transcripts have padj < 0.05. Every swap
   (swap1 to swap9) has 0 significant k-mers and 0 significant transcripts; the
   smallest padj in any swap table is 0.994. No padj is NA. The peptide map holds
   only `real` peptides, so the kmerseek path has no queries at all for the
   swaps. Consequence: part 2 compares R with 5 x 0 = 0 and passes for any
   count, including 0. Part 2 gives no protection here; the verdict rests on
   part 1 alone.
2. **Standard path rebuilt. HELD.** 113 significant transcripts, 88 with a
   DIAMOND hit (weakest hit E = 9.2e-6, so the hits respect a 1e-5 cut-off).
   Taking every Pfam domain of each best hit gives 97 families: 85 recovered,
   12 outside (1 of them an unchanged-key family). Same as the pipeline.
3. **kmerseek families rebuilt for all 19 settings; R picked by the rule. HELD.**
   All reduced settings were streamed in full, including gbmr7.k11 (610 million
   rows). R = hsdm17.k7 (54). Verdict FAIL.
4. **Comparison with `results/summary.tsv` (drosophila). HELD, no mismatches.**
   Recovered and outside counts agree for all 19 settings at E <= 1 and E <= 10,
   for the standard path (85, 12) and for all swap rows (0).
5. **Attempts to break the result.**
   - Looser cut-off, E <= 10: the best reduced setting is hp_lehninger_hpc3.k19
     with 65 recovered (5,258 outside). Still below 85. **HELD.**
   - E <= 100 (pipeline summary, not recomputed by me): hp_lehninger_hpc3.k19
     recovers 95 of 100, above 85, but reports 11,804 families in total, 57% of
     the 20,720 families in the search database. At that cut-off it lands on
     most families, changed or not. This does not support the claim. **HELD.**
   - Does taking all domains of the best hit inflate the standard path? Keeping
     only domains that the DIAMOND alignment covers by at least half their length
     still recovers 85 (outside drops from 12 to 7). **HELD.**
   - Answer-key families missing from the search database: none of the 100
     changed and none of the 100 unchanged families is missing. **HELD.**
   - Every kmerseek region's target has at least one Pfam domain in the table
     (0 rows with an unannotated target), so no landings are lost to a failed
     join. **HELD.**
   - Simulation scope: reads come only from the 200 answer-key transcripts
     (`transcripts.fa` has 200; `true_counts.tsv` has 200 rows). Neither path
     faces background genes. So "outside" counts here come from wrong
     landings and wrong best hits only, not from genes outside the key. This
     makes the test easier for both paths; it does not favour either.
   - E-values: every fitted setting's log reports `Karlin-Altschul: K 0.0000`,
     and the searches ran with `--ka-queries 0`. In most settings most region
     E-values are `inf` (dayhoff6.k12: 1 finite of 2,709,340; gbmr7.k11: 72,473
     of 610 million). The E <= 1 rule therefore throws away almost every region
     of the gbmr, polarity and dayhoff settings. Whether those E-values are right
     is a kmerseek question, not checked here. It could cut either way for the
     claim: with no E-value filter, several settings reach 99 to 100 recovered,
     but with roughly 20,000 outside families each, so the E-value is doing all
     of the work of telling real landings from noise. **Not broken, but a weak
     point.**
   - The pipeline's keep filter (`awk ... ($e+0) <= 100`) runs in GNU Awk
     4.0.2, which reads the string `inf` as 0. So `inf` rows are all kept (this
     is why the gbmr7.k11 file is 39 GB) while finite rows above 100 are dropped.
     It does not change any count at E <= 1 or 10, because the landing step
     excludes `inf` (my counts, which treat `inf` as failing every cut-off,
     match). **No effect on the verdict.**
6. **gbmr7.k11 lands 0 key families. HELD, explained.** Of 610,126,369 rows,
   72,473 have a finite E-value, the smallest is 0.038, and only 98 rows have
   E <= 1. Those 98 land on 15 families, none of them changed. The other rows
   have `region_ka_bits = 0.0` and `region_evalue = inf` (all of the first
   200,000 rows checked). Regions are not short (mean target interval 44.6
   residues). Ignoring E-values, gbmr7.k11 lands on 99 changed families and
   20,416 others, almost every family in the database. So the 0 comes from the
   E-values, not from regions missing their domains.

## Commands run

Script: `scripts/nb360/adversarial_check_drosophila.py` (standard library,
Python 3.6). Copied to
`/scratch/users/olgabot/360-de-kmer-landing/adversarial_drosophila/` and run as
one Slurm job (50728849, 59 min, 8 CPUs) with this `run_all.sh`:

```bash
A=/scratch/users/olgabot/360-de-kmer-landing/adversarial_drosophila
D=/scratch/users/olgabot/360-de-kmer-landing/run/results/drosophila
K=/scratch/users/olgabot/360-de-kmer-landing/key/drosophila
# one JSON of landed families per setting, 8 settings at a time
ls -S $D/kmerseek/*.regions.csv.zst | xargs -P 8 -I{} sh -c \
  'b=$(basename {} .regions.csv.zst); python3 $A/adversarial_check_drosophila.py setting \
     --regions {} --pfam $K/search_db.pfam.tsv \
     --peptide-map $D/kmer_path/drosophila.peptide_map.tsv --out $A/settings/$b.json'
python3 $A/adversarial_check_drosophila.py summarize --answer-key $K/answer_key.tsv \
  --pfam $K/search_db.pfam.tsv --deseq2-dir $D/deseq2 \
  --blastx $D/standard/drosophila.blastx.tsv --setting-dir $A/settings > $A/summary_out.txt
```

Submitted with:

```bash
ssh sherlock 'cd /scratch/users/olgabot/360-de-kmer-landing/adversarial_drosophila && \
  sbatch --parsable --account=ayeletv -p hns,normal -c 8 --mem 32G -t 6:00:00 -o run_all.%j.log run_all.sh'
```

A second count of significant DESeq2 features with awk (same numbers):

```bash
for f in $D/deseq2/*.deseq2.tsv.gz; do zcat $f | awk -F'\t' 'NR>1{n++; if($5=="NA")na++; else if($5+0<0.05)s++} END{print n, na+0, s+0}'; done
```

Comparison with the pipeline:

```bash
awk -F'\t' '$1=="drosophila" && ($5=="real" || $2!="kmerseek") && ($6=="1.0" || $6=="10.0" || $2!="kmerseek")' \
  /scratch/users/olgabot/360-de-kmer-landing/run/results/summary.tsv | cut -f2-10
```

Full output: `/scratch/users/olgabot/360-de-kmer-landing/adversarial_drosophila/summary_out.txt`;
per-setting family lists: `.../adversarial_drosophila/settings/*.json`.

DIAMOND column order (the file has no header) was inferred from the values as
qseqid, sseqid, pident, length, evalue, bitscore, qstart, qend, sstart, send,
and afterwards confirmed against the `--outfmt 6` line in `main.nf`.
