**kmerseek's seven H/P alphabets put 15 of the 20 amino acids in the same class and differ only on C, G, P, W and Y.**

Each row is one of the seven hydrophobic/polar (H/P) alphabets in kmerseek v0.4.0, labelled with its kmerseek name. Each column is one amino acid, in one-letter code. Each square shows the class that alphabet gives the residue: amber with **h** for hydrophobic, blue with **p** for polar. Grey with **c** is cysteine in a class of its own, which only `hp_lehninger_hpc3` has. The letter inside each square makes the figure readable in greyscale.

The columns form three groups. `AFILMV` is hydrophobic in all seven alphabets and `DEHKNQRST` is polar in all seven. The middle group, `CGPWY`, holds the five residues the alphabets disagree on. Counting alphabets that put the residue in h: C 4 (2 put it in p, 1 in its own class), G 3, P 4, W 6, Y 6. `hp_kyte_doolittle2` is the only alphabet with W and Y in p.

Rows are sorted by how many of C, G, P, W and Y the alphabet puts in h, most first. Ties keep the order the alphabets are listed in kmerseek's source. The right-hand column gives the number of the 20 residues each alphabet puts in h, from 11 (`hp_lehninger_c_nonpolar2`) to 7 (`hp_kyte_doolittle2`).

Source: the class tables in `src/rust/alphabets.rs` of seanome/kmerseek at tag v0.4.0 (commit `0aed5ca`), parsed by `scripts/plot_hp_borderline_residues.py`. Every square drawn is a row of `tables/hp_borderline_residues.csv`.
