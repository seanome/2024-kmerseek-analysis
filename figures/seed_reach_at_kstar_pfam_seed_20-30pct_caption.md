**Seed reach at k* | At its own seed length k*, no alphabet shares a seed along the alignment in more than 5.6% of Pfam seed pairs at 20–30% identity.**

One row per kmerseek alphabet, in the order of Supplementary Figure 1; k* is from its panel c. Pairs are two sequences from the same Pfam-A 38.2 seed alignment, 20–30% identity over at least 50 aligned columns (37,085 pairs from 15,276 families). A run is a stretch of consecutive aligned columns with no gap in either sequence and the same class in both: an exact seed the two sequences share along their alignment. A seed shared elsewhere, off the alignment, is not counted. Dot: share of pairs whose longest run is at least k*; line: Wilson 95% interval. The highest is mmseqs12, 5.55% (2,060 pairs); protein20 4.87%; the 2–3 letter alphabets 0.47% (hp_kyte_doolittle2) to 0.85% (hp_pbotc_1st_ed2); gbmr7 0.21%.

The share depends on rounding k* up to a whole letter. At one letter less, it is 1.3 to 2.3 times higher, and the highest is protein20, 11.19%.

Cohen's κ for the class of aligned residues ranks the alphabets partly in the opposite order (Spearman ρ = −0.50 against this share): κ is 0.20 for protein20 and 0.39–0.46 for the 2–3 letter alphabets, which keep classes better but carry fewer bits per letter. κ values are in `tables/suppfig1_values.csv`.

On SCOPe 2.08 (40% set) domain pairs in the same superfamily, aligned by structure with USalign (TM-score ≥ 0.5, 5,712 pairs), the alphabets come out in nearly the same order (Spearman ρ = 0.98), each share moves by at most 0.9 percentage point, and the highest is mmseqs12, 5.18%.
