# Figure legend: the k-sizes to test for each kmerseek alphabet

Figure files: `274_ksizes_to_test_per_alphabet_human_swissprot.{pdf,svg,png}` (89 mm wide).
Made by notebook `notebooks/274_ksizes_to_test_per_alphabet_human_swissprot.ipynb`; every
number below is in `tables/274_ksizes_to_test_per_alphabet_human_swissprot.csv`.

**The k-sizes to test for each kmerseek alphabet, from k_min to k_max.** One row per alphabet,
grouped by the number of letters (20; 12 to 18; 4 to 8; 2 to 3), with a small gap between
groups. The x axis is the seed length k, in letters of that alphabet. Each grey bar runs from
k_min to k_max and is cut into one cell per k-size the benchmark would run; the number in the
"# k-sizes" column at the right counts the cells. A dotted line leads from each alphabet name to
the start of its bar.
The ranges run from 4 k-sizes (protein20, k = 5 to 8) to 23 (gbmr7, k = 13 to 35), 229 in
total over the 19 alphabets.

Only the magenta diamond is measured directly. The other three marks come from Equation 4b, which
gives the shortest seed length at which a seed expects at most a set number of chance
matches in a database:

k = ⌈(log2 N − log2 allowed) / B⌉

where N is the number of residues in the database, "allowed" is the number of chance matches
accepted per seed, B is the bits of information one matching letter carries, and ⌈ ⌉ rounds
up to a whole number. Each extra letter divides the expected chance matches by 2^B.

B comes from one of two sources. From composition: B = −log2(sum over letter classes of q²),
where q is the share of Swiss-Prot release 2026_03 residues that fall in that class; the sum
is the chance that two residues drawn at random are the same letter. Measured in the human
proteome: from how fast the number of proteins sharing a seed falls between the two seed
lengths counted by kmerseek. The measured value is lower than the composition value for all
19 alphabets.

- **Magenta diamond, k_main (measured).** The shortest k at which seeds of the human proteome
  are shared by fewer than 100 proteins on average (each seed weighted by how often it occurs),
  counted by kmerseek 0.4.0 in the QfO 2020_04 human proteome (20,600 proteins, 11,395,293
  residues) in notebook 250. k_main is the longer of the two measured lengths kept per
  alphabet (the other, k_small, is under 1,000 proteins), and k is a whole number, so the
  length at which the average crosses 100 proteins can sit below it.
- **Magenta ring, Equation 4b for the human proteome.** N = 11,395,293 residues, 100 chance
  matches allowed, B from composition: the same question as the diamond, answered by the
  formula. Where the ring and the diamond fall on the same k, the ring is drawn larger and
  the diamond sits inside it. The low end of the range, k_min, is the smaller of the diamond and the ring.
- **Teal dot, k\*.** Equation 4b for Swiss-Prot 2026_03 (209,017,843 residues), 1 chance
  match allowed, B from composition.
- **Purple dot, k_max.** Equation 4b for Swiss-Prot 2026_03, 1 chance match allowed, B
  measured in the human proteome. This is the high end of the range. It is an estimate: the
  formula with a measured input. Where k\* and k_max are equal (wass14, k = 9), the teal dot is
  drawn smaller, inside the purple one.

The two-letter hydrophobic-polar alphabets need 14 to 22 k-sizes each, and the three-letter
hp_lehninger_hpc3 needs 13. The four widest ranges, gbmr7 (23), hp_thomas_dill_no_c2 (22),
hp_kyte_doolittle2 (21) and hp_thomas_dill2 (19), belong to the four alphabets with the lowest
measured bits per letter, 0.71 to 0.80.

Sources. Seed counts: `tables/250_two_k_per_alphabet.csv` (notebook 250). Swiss-Prot 2026_03
composition: section 6.1 of https://web.expasy.org/docs/relnotes/relstat.html (575,748
entries). Letter classes: `src/rust/alphabets.rs` of seanome/kmerseek at v0.4.0; dayhoff6 from
sourmash, as listed in `notebooks/hp_conservation_utils.py`.
