## Summary and conclusions

**Claims the data supports**

1. In this tarantula venom gland run, no k-mer setting picks out Conotoxin-family (PF02950)
   reads from the other reads at N = 400. With the genus held out, the rule's best setting
   (hp k = 30, m = 1, the highest sensitivity at a false keep of 5% or less) keeps 117 of
   21_270 family reads (0.55%) and 0.41% of non-family reads. Every setting with a false
   keep of 5% or less keeps under 4% of family reads, at any N (SF2 and the settings table).
   The decision rule (at least 50% of family reads at a false keep of 5% or less) is not
   met, and the idea stops here for this sample.
2. The settings that keep many family reads keep many other toxin reads too. Protein k = 7,
   m = 2 keeps 39.1% of family reads and 25.0% of non-family reads. It keeps 0.11% of
   shuffled reads, so the matches are real sequence, not chance. They are not specific to
   the family.
3. The shared k-mers are mostly signal peptides. At protein k = 10, the most common
   reference k-mer, IIAGLALLSV (residues 6-15 of P0DM66, inside its signal peptide), is in
   the first 400 bases of 5_403 family reads and 16_397 non-family reads. Tarantula toxin
   precursors from different families start with similar signal peptides, and the start of
   a forward-strand cDNA read is where the signal peptide is.
4. The brief's Pfam truth splits one toxin subfamily: B1P1F1 carries PF02950, its paralog
   B1P1F2 in the same UniProt subfamily does not. With a truth based on UniProt subfamilies
   instead (69_778 family reads), the conclusion is the same (F1 b).
5. DNA mapping to the coding sequences of other tarantula genera keeps 28 of 21_270 family
   reads (0.13%) at N = 400, and none once tarantulas are held out. DNA mapping does fail
   across genera here, as the background predicted. The translated rule does not do better
   in a way that matters, because it does not separate the family from the other toxins.
6. With all tarantula proteins removed from the reference (cone snail and sea anemone
   proteins only), DNA mapping keeps no family read, and DIAMOND blastx on the 400-base
   chunk keeps 267 of 21_270 (1.3%) at a false keep of 2.1%. No k-mer setting, at any N,
   keeps family reads at more than twice its rate on non-family reads, except settings that
   keep 3 family reads or fewer. Settings that keep many family reads keep even more of the
   rest (dayhoff k = 10, m = 1, N = 400: 52.5% of family reads, 64.5% of non-family reads).
7. The decision is fast: the median time per read is 0.15 to 0.21 ms in Python, more than
   3_000 times below the 800 ms chunk (SF3).

**Claims the data does not support**

- That a translated reduced-alphabet rule "reaches where DNA mapping does not". It keeps
  more family reads than DNA mapping only in settings that keep as many unrelated reads.
- Anything about the genus hold-out as a test of distance: Swiss-Prot has no *Selenobrachys*
  entry, so the genus hold-out removes nothing, and its reference holds 8 proteins from
  three other tarantula genera.
- Anything about other families, tissues or newer chemistry. One run, one family, R9 flow
  cell, basecaller unknown. 93% of family reads (19_772 of 21_270) have the same best hit,
  so the reads test few distinct genes, and the 95% intervals over reads are too narrow.
- Why minimap2 maps 4.6% of non-family reads but 0.13% of family reads to the family's coding
  sequences at N = 400. This was not looked into.

**Figures**: F1 `figures/292/f1_sensitivity_by_holdout_N400_PF02950.png`; F2
`figures/292/f2_by_chunk_genus_holdout_PF02950.png`; shared k-mers
`figures/292/fig_shared_kmers_protein_k10_N400_PF02950.png`; SF1 to SF4 and the family and
truth figures in `figures/292/`.

**Next smallest experiment**: build the k-mer set from the mature toxin only (drop the
Swiss-Prot SIGNAL and PROPEP residues of each reference protein) and rescore the same reads
at N = 400 and 800, genus hold-out. This removes the shared signal peptides that cause the
false keeps. If no setting then keeps half the family reads at a false keep of 5% or less,
the read start does not carry enough family-specific sequence for any rule, translated or
not. Cost: one run of `score_chunks.py` with two new reference files, about one hour on the
laptop.
