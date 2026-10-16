## Summary and conclusions

**Claims the data supports**

1. The decision rule is not met. With the genus held out, the rule's best setting (hp
   k = 30, m = 1: the highest sensitivity among settings that keep 5% or fewer of
   non-family reads) keeps 117 of 21_270 family reads (0.55%) and 0.41% of non-family
   reads. Every setting that keeps 5% or fewer of non-family reads keeps under 4% of family
   reads, at every N.
2. In this sample, no rule separates Conotoxin-family (PF02950) reads from the other reads
   at the read start, the slow comparator included. DIAMOND blastx of the 400-base chunk
   keeps 62.8% of family reads and 56.7% of non-family reads with the genus held out. The
   test therefore cannot tell a good fast rule from a bad one with this family and these
   labels. That is the main result, more than any statement about k-mers.
3. The reason is shared precursor sequence. At protein k = 10 the reference k-mer found in
   the most read starts, IIAGLALLSV, is residues 6-15 of the signal peptide of P0DM66; it
   is in the first 400 bases of 5_403 family reads and 16_397 non-family reads. Summed over
   all 61 shared k-mers, those overlapping a signal peptide account for 35_537 non-family and
   5_483 family read matches; those only in the mature toxin, for 22 and 3. Pfam's
   PF02950 match on these tarantula proteins covers the signal peptide and propeptide too
   (B1P1F1 residues 3-74 of 79), so the family label itself rests partly on sequence that
   tarantula toxins of other families share.
4. The labels are unstable. The Pfam truth splits one UniProt subfamily: B1P1F1 carries
   PF02950, its paralog B1P1F2 does not. Relabelling by UniProt subfamily moves 48_508
   reads into the family (69_778 in all) and leaves the conclusion unchanged (F1 b).
5. There is a weak protein-level signal at the genus hold-out: protein k = 9 or 10, m = 1,
   keeps family reads at about 1.7 times its non-family rate (k = 10: 25.9% against
   14.9%). Protein k = 7, m = 2 keeps 39.1% against 25.0%. These rates are far above the
   shuffled rates (0.11% or less), so the matches are real sequence; they are not specific
   to the family.
6. With all tarantula proteins removed (cone snail and sea anemone proteins only), the DNA
   rule keeps no family read and DIAMOND on the chunk keeps 267 of 21_270 (1.3%) at a false
   keep of 2.1%. No k-mer setting, at any N, keeps family reads at more than twice its
   non-family rate, except settings that keep 3 family reads or fewer. Settings that keep
   many family reads keep more of the rest (dayhoff k = 10, m = 1, N = 400: 52.5% against
   64.5%).
7. The decision is fast: the median time per read is 0.15 to 0.21 ms in Python, more than
   3_000 times below the 800 ms chunk (SF3).

**Claims the data does not support**

- That a translated reduced-alphabet rule "reaches where DNA mapping does not". It keeps
  more family reads than DNA mapping only in settings that keep as many unrelated reads.
- That DNA mapping fails across genera. minimap2 keeps 0.13% of family reads but 4.6% of
  non-family reads at N = 400; that pattern points at the labels (paralogs on both sides),
  and was not looked into further.
- Anything about distance from the reference. Swiss-Prot has no *Selenobrachys* entry, so
  the genus hold-out removes nothing; its reference holds 8 proteins from three other
  tarantula genera.
- Anything about other families, tissues or newer chemistry. One run, one family, R9 flow
  cell, basecaller unknown. 93% of family reads (19_772 of 21_270) have the same best hit,
  so the reads test few distinct genes, and the 95% intervals over reads are too narrow.
- That the read start shows the same part of the transcript in every read. This is PCR
  cDNA. Forward-strand reads (9_467 of the 21_270 family reads) start at the 5' end, and the
  protein's N-terminus, signal peptide included, is in their first 400 bases in 98.7% of
  them. Reverse-strand reads (11_803) start with adapter, poly(T) and 3' UTR; their first
  400 bases overlap the coding alignment in 86.2% of them, but from the C-terminal end, and
  the protein's N-terminus is in the first 400 bases in only 0.16%. The signal-peptide
  matches therefore come from forward-strand reads, and a rule built on the mature toxin
  would be tested mostly on reverse-strand reads.
- That every family/non-family label is right. The truth step keeps one DIAMOND hit per
  read (`--max-target-seqs 1`), so between paralogs of equal length such as B1P1F1 (has
  PF02950) and B1P1F2 (no Pfam line, both 79 residues) DIAMOND's choice decides the label,
  and that choice is not visible in the output.
- That the pore-time numbers describe a real run. The model charges a read shorter than N
  the extra 500 bases when it is ejected, although such a read ends before any decision.

**Figures**: F1 `figures/292/f1_sensitivity_by_holdout_N400_PF02950.png`; F2
`figures/292/f2_by_chunk_genus_holdout_PF02950.png`; shared k-mers
`figures/292/fig_shared_kmers_protein_k10_N400_PF02950.png`; SF1 to SF4 and the family and
truth figures in `figures/292/`.

**Next smallest experiment**: before scoring any fast rule, find a target the slow
comparator can separate. For each candidate family in
[`292_data/toxin_family_candidates.csv`](292_data/toxin_family_candidates.csv), and for a
few non-toxin families with many reads, run DIAMOND blastx of the 400-base chunks against
that family's Swiss-Prot proteins (the step that already ran for PF02950) and keep only
families where it keeps at least half the family reads at a false keep of 5% or less, with
family membership checked against UniProt subfamilies as well as Pfam. Then rescore the
k-mer rule on those families only. Cost: about 10 minutes of DIAMOND per family on the
laptop; no new download. If no family passes, this sample cannot test the idea, and the
next step is a different sample, for example a tissue where the target family does not
share its signal peptide with most other transcripts.
