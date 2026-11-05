# Reference-free routes on the paper's three Fig. 4 RNA-seq comparisons (notebook 361)

Sherlock jobs 51086109 (run), 51086121 (lists), 51086122 (score), 2026-11-04. Scoring:
`scripts/human_compare.py`. Numbers: `human_compare_fig4.tsv`. All skeletal muscle.

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

Beating random means 5 % or fewer of 1,000 random gene sets, matched on baseMean, did as well. SPLASH
has no direction, so its genes count for both lists.

Limit: GSE120862's down list is reached in the wrong direction often (947 of 4,967 genes reached only
going the other way by the p < 0.05 t-test, against 14 to 106 in the other lists).
