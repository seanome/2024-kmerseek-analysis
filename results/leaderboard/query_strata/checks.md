# Query strata checks

## Protein-level bins, recounted from the raw outputs

| axis | bin | queries (recount) | queries (query_strata.tsv) | per-query match |
|---|---|---:|---:|---|
| membrane | alpha-helical TM | 288 | 288 | yes |
| membrane | neither | 710 | 710 | yes |
| low_complexity | 5-20% | 480 | 480 | yes |
| low_complexity | <5% | 379 | 379 | yes |
| low_complexity | >20% | 139 | 139 | yes |
| protein_length | 100-300 | 269 | 269 | yes |
| protein_length | 300-1000 | 587 | 587 | yes |
| protein_length | <100 | 31 | 31 | yes |
| protein_length | >1000 | 111 | 111 | yes |
| busco | complete | 592 | 592 | yes |
| busco | not a BUSCO gene | 406 | 406 | yes |

## Nearest-protein identity, recounted from every MMseqs2 hit

| target | no hit | <30% | 30-50% | 50-90% | >=90% | per-query match |
|---|---:|---:|---:|---:|---:|---|
| arabidopsis | 513 | 135 | 265 | 83 | 2 | yes |
| btaurus | 12 | 4 | 40 | 543 | 399 | yes |
| chicken | 65 | 38 | 196 | 590 | 109 | yes |
| ciona | 246 | 129 | 400 | 217 | 6 | yes |
| clupus | 12 | 5 | 60 | 502 | 419 | yes |
| ecoli | 882 | 59 | 49 | 8 | 0 | yes |
| fly | 265 | 158 | 386 | 178 | 11 | yes |
| mdomestica | 24 | 8 | 99 | 690 | 177 | yes |
| mouse | 7 | 7 | 59 | 582 | 343 | yes |
| ptroglodytes | 4 | 1 | 7 | 61 | 925 | yes |
| rnorvegicus | 14 | 5 | 73 | 577 | 329 | yes |
| worm | 323 | 211 | 349 | 110 | 5 | yes |
| xtropicalis | 98 | 73 | 255 | 517 | 55 | yes |
| yeast | 584 | 149 | 211 | 53 | 1 | yes |
| zebrafish | 57 | 64 | 302 | 534 | 41 | yes |

## Long table: bins add up to the whole set

results.parquet has no query-side rows yet: the midi-plus run has not been rescored with the strata covariates.

Mismatches: 0.
