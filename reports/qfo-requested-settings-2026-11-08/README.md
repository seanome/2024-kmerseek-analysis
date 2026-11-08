# QfO Pfam domain benchmark: protein20 k5-10, dayhoff6 k9-21, H/P k19-40

Asked 2026-11-08: how did these kmerseek settings do on the QfO data, and could the results
go to the Quest for Orthologs benchmark service?

![Median Pfam domain Fmax against bits per k-mer](fmax_by_k.png)

The x axis is bits per k-mer, k × log2(alphabet size): 4.3 bits per protein20 letter, 2.6 per
dayhoff6 letter, 1.6 for the 3-letter hp_lehninger_hpc3 and 1 for the two-letter H/P alphabets.
On that scale protein20 k5-10 (22-43 bits), dayhoff6 k9-17 (23-44 bits) and H/P k25-40 cover
the same range, which is why H/P k25, 30, 35 and 40 were asked for as a follow-up.

## Answer

Every setting asked about scores far below the alignment tools. Median Pfam domain Fmax
(the best F1 over all score cut-offs) is 0.056-0.063 for protein20 and dayhoff6 and
0.019-0.053 for the H/P alphabets. MMseqs2 gets 0.31 and phmmer 0.32 on the same 9 species.
The best settings reach about 1/5 of MMseqs2 (dayhoff6 k9: 0.063, ratio 0.20). Changing k
barely moves protein20 (0.056-0.057) or dayhoff6 (0.058-0.063). For H/P, Fmax rises from
k19 to k25 and then stays flat: higher k buys precision and loses recall.

At equal bits, H/P is no better than protein20 or dayhoff6: at 25 and 30 bits the best H/P
alphabet (hp_thomas_dill2) scores 0.052 and 0.054, against 0.057 for protein20 at 26 and 30 bits.

Not run: dayhoff6 k19 and k21 (the sweep stopped at k17), H/P k31, k35 and k40 (no alphabet).
H/P k30 exists for hp_thomas_dill2, hp_thomas_dill_no_c2 and hp_kyte_doolittle2; k33 only for
hp_thomas_dill2. Getting H/P k35 and k40 needs new searches.

| alphabet | k | species with calls | median Fmax | median recall (reachable) | median precision at Fmax | Fmax / MMseqs2 |
|---|---|---|---|---|---|---|
| protein20 | 5-10 | 6 of 9 | 0.056-0.057 | 0.027-0.028 | 0.66-0.67 | 0.17 |
| dayhoff6 | 9 | 9 | 0.063 | 0.039 | 0.41 | 0.20 |
| dayhoff6 | 11, 13 | 9 | 0.060, 0.058 | 0.029, 0.026 | 0.56, 0.61 | 0.19 |
| dayhoff6 | 15, 17 | 8 | 0.059 | 0.026, 0.025 | 0.64, 0.69 | 0.18 |
| hp_thomas_dill2 | 19 / 25 / 30 / 33 | 9 | 0.029 / 0.052 / 0.054 / 0.051 | 0.097 / 0.058 / 0.036 / 0.030 | 0.02 / 0.09 / 0.34 / 0.46 | 0.09 / 0.17 / 0.17 / 0.15 |
| hp_pbotc_1st_ed2 | 19 / 25 / 29 | 9 | 0.024 / 0.053 / 0.047 | 0.085 / 0.047 / 0.029 | 0.02 / 0.14 / 0.42 | 0.08 / 0.17 / 0.15 |

Every row is in [kmerseek_requested_settings.csv](kmerseek_requested_settings.csv); the other
tools are in [other_tools.csv](other_tools.csv).

## What the numbers are

- Run: `qfo-pfam-region-benchmark`, midi-plus set: 998 human queries (chr6 plus the MHC-related
  genes) searched against 9 QfO 2020_04 proteomes (arabidopsis, chicken, ciona, ecoli, fly,
  mouse, worm, yeast, zebrafish). Searches from 2026-09-01, kmerseek image
  `2026-08-24-reduced-alphabets` (before 0.4), regions ranked by `region_poisson_score`.
- Each kmerseek region becomes a Pfam domain call by copying the Pfam family annotated on the
  matching target interval onto the human interval. A call counts only if it lands on a real
  human domain instance of that family. Truth set: Pfam; low-complexity masking off; no dedup.
- Medians are over species, using only species where the search gave at least one call.
  Recall is out of the domains the target proteome could have given (`recall_reachable`).
- 20 searches gave zero calls, listed in [empty_searches.csv](empty_searches.csv): protein20 at
  every k for arabidopsis, ecoli and yeast, and dayhoff6 k15 and k17 for ecoli. Their region
  files are not empty (yeast protein20 k7: 246 KB), so the zero is lost somewhere between the
  regions and the calls. The cause has not been found. Counting those zeros into the median
  would drop protein20 to 0.014.
- Two extra copies exist for protein20 k6-8 and hp_thomas_dill2 k27 and k33: the 2026-09-13
  ladder run rescored them on 15 species and overwrote the 9-species files for those settings.
  This table uses only the 9 species shared by every setting.

## Can it go to the QfO benchmark service?

No, not as it is. The service (https://orthology.benchmarkservice.org) scores **ortholog
pairs**: an OrthoXML file or a two-column list of UniProt accessions, predicted on one of its
reference proteome releases (2020_04 or 2022_02). It then tests those pairs against species
trees, SwissTree gene trees, and GO and EC conservation. These results are domain calls on
998 human proteins, not ortholog pairs, so there is nothing to upload.

The proteomes are already right: the pipeline uses `QfO_release_2020_04`, which the service
still accepts. A submission would need:

1. kmerseek searches all-vs-all over the 78 proteomes of 2020_04, not 998 human queries.
2. A rule that turns hits into orthologs, for example reciprocal best hits by a whole-protein
   score.
3. Output as OrthoXML or tab-separated UniProt accession pairs, submitted through OpenEBench.

The docs do not say whether a subset of species or proteins is accepted, or how partial
coverage would change the scores. Given an Fmax of 1/5 of MMseqs2 at finding domains, an
orthology submission would very likely land well below the published methods.

## Rebuild

```
python summarize.py <metrics dir> <all_domain_metrics.parquet>
```

`<metrics dir>` is
`sherlock:/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results/metrics/`
(the `pfam.kmerseek.*_lcFalse.*.metrics.parquet` files). `all_domain_metrics.parquet` is in
the same `results/` directory (md5 3080c0bf94816472dfe277ff7ee31b8c).
