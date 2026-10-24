# Pilot tune half with the exact arm scored on region_tfidf

`make score-partial MMSEQS2_DIR=mmseqs2-one-round SCORE_OUT=scores-partial-tfidf` (Slurm job
47780503 at b215e87, 5 min, peak 9.7 GiB): domains placed through the hit, MMseqs2 one round,
exact arm on `region_tfidf`. 205 kmerseek searches had finished. Test half unread.

`compare_tfidf.py` and `per_setting_found.py` compare it with the Poisson-scored exact arm
(`../pilot_partial_2026-10-14_projected/`) on the 186 settings both scored, over all 115 tune-half
instances; `found_per_setting.svg` (from `build_found_svg.py`) draws `per_setting_found.csv`.

| | exact on Poisson score | exact on tf-idf |
|---|---|---|
| exact settings that find anything | 5 / 107 | 107 / 107 |
| best exact setting (instances, of which P4) | 10 (6) | 14 (9) |
| instances found by any exact setting | 11 | 22 |
| best extend setting | 24 (13) | 24 (13) |
| MMseqs2 / hmmscan / phmmer | 54 / 63 / 68 | same |
| instances a kmerseek setting finds and no other tool finds | 0 | 0 |
