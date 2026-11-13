# Reference-free routes on the paper's three Fig. 4 RNA-seq comparisons (notebook 361)

Sherlock jobs 51086109 (run), 51086121 (lists), 51086122 (score), 2026-11-04. Scoring:
`scripts/human_compare.py`. Numbers: `human_compare_fig4.tsv`. All skeletal muscle.

The three series' lists are not made the same way, so the pooled table below mixes them:

| series | genes on the lists | design and cutoff of the list |
|---|---|---|
| GSE99963 | 3,980 | GEO counts, `~ subject + condition`, padj < 0.05 |
| GSE144304 | 4,957 | GEO counts, `~ condition`, uncorrected p < 0.01 |
| GSE120862 | 10,469 | GEO normalised counts, `~ subject + condition`, padj < 0.05 |

Groups are those that reproduced the paper's printed scores in PR 212: GSE99963 before vs after
10 days, both arms pooled (14 + 14); GSE144304 young vs fit old adults, frail left out (26 + 30);
GSE120862 all untrained vs all trained (42 + 42). Reference lists are rebuilt with PyDESeq2 from each
study's GEO counts, because no published list exists for these exact groups, so they are not
independent of the reads.

| route | list genes reached in the right direction, of 19,406 | lists beating random, of 6 |
|---|---|---|
| KaMRaT t-test, p < 0.05 | 9,960 (51 %) | 6 |
| KaMRaT t-test, corrected | 8,942 (46 %) | 6 |
| KaMRaT DESeq2, p < 0.05 | 8,867 (46 %) | 6 |
| KaMRaT DESeq2, corrected | 6,347 (33 %) | 6 |
| SPLASH, p < 0.05 | 3,914 (20 %) | 0 |
| SPLASH, corrected | 1,607 (8 %) | 1 |

GSE120862 is 54 % of the pooled genes, so here is each series alone (`human_by_series.tsv`,
`scripts/human_by_series.py`), share of list genes reached in the right direction:

| route | GSE99963 | GSE144304 | GSE120862 |
|---|---|---|---|
| KaMRaT t-test, p < 0.05 | 48 % | 59 % | 49 % |
| KaMRaT t-test, corrected | 12 % | 55 % | 55 % |
| KaMRaT DESeq2, p < 0.05 | 47 % | 47 % | 44 % |
| KaMRaT DESeq2, corrected | 21 % | 28 % | 40 % |
| SPLASH, p < 0.05 | 15 % | 25 % | 20 % |
| SPLASH, corrected | 5 % | 12 % | 8 % |

Beating random means 5 % or fewer of 1,000 random gene sets, matched on baseMean, did as well. SPLASH
has no direction, so its genes count for both lists.

Limit: GSE120862's down list is reached in the wrong direction often (947 of 4,967 genes reached only
going the other way by the p < 0.05 t-test, against 14 to 106 in the other lists).
