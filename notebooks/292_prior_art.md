# Prior art for notebook 292: linked word pairs over a reduced protein alphabet

Searched 2026-10-14. Question: has anyone published strobemers (two or more short exact
words linked by a hash rule, Sahlin 2021) over amino-acid or reduced-alphabet protein
sequences? Answer: no paper or code found. This was the first check (Gate 0) of the task: had
someone already done it, notebook 292 would have stopped here. No one had, so it runs the test.

## What was searched

- PubMed: `strobemers` (9 hits, all DNA or RNA reads: read mapping, k-mer extraction,
  sketching reviews, seed theory); `(strobemer OR strobemers OR randstrobe OR randstrobes)
  AND (protein OR amino acid OR peptide)` (0 hits); `reduced amino acid alphabet spaced
  seed protein homology search` (1 hit, Takabatake 2021 below); `hydrophobic polar binary
  pattern k-mer seed remote homology detection` (0 hits).
- Web (covers bioRxiv, arXiv and Google Scholar-indexed pages; Google Scholar itself was not
  queried directly): strobemers with protein or reduced amino-acid alphabets; randstrobes on
  amino acids; "paired k-mers" or "k-mer pairs" for protein remote homology; arXiv strobemer
  protein seeds 2024-2026; bioRxiv strobemer protein homology 2025. No hit applied
  strobemers to proteins. The closest protein seed work is DIAMOND's spaced seeds over an
  11-letter alphabet, the two-step reduced-alphabet seeds of Takabatake 2021, and subset or
  spaced seeds for protein alignment (Roytberg and coauthors, arXiv 0810.5434 and 0901.3198,
  not entered below because no DOI was returned by a search tool this session).
- GitHub: repositories matching "strobemer" (10, listed by `gh search repos`). All work on
  DNA. TravisWheelerLab/strobemers, the only one from a protein-search lab, estimates edit
  distance between 10,000-nt sequences (README read 2026-10-14). No code search hit for
  randstrobes on amino acids.

## Strobemers (all on DNA)

Sahlin, K. "Effective Sequence Similarity Detection with Strobemers." *Genome Research* 31,
no. 11 (2021): 2080–94. https://doi.org/10.1101/gr.275648.121

**Note:** Defines minstrobes, randstrobes and hybridstrobes on nucleotides. The abstract
names the alternative strobemers replace: pairs or groups of small k-mers that are matched
first and paired in a second step.

Maier, B. D., and K. Sahlin. "Entropy Predicts Sensitivity of Pseudorandom Seeds." *Genome
Research* 33, no. 7 (2023): 1162–74. https://doi.org/10.1101/gr.277645.123

Karami, M., A. Soltani Mohammadi, M. Martin, B. Ekim, W. Shen, L. Guo, M. Xu, G. E.
Pibiri, R. Patro, and K. Sahlin. "Designing Efficient Randstrobes for Sequence Similarity
Analyses." *Bioinformatics* 40, no. 4 (2024). https://doi.org/10.1093/bioinformatics/btae187

Sahlin, K. "Strobealign: Flexible Seed Size Enables Ultra-Fast and Accurate Read
Alignment." *Genome Biology* 23, no. 1 (2022): 260. https://doi.org/10.1186/s13059-022-02831-7

Tolstoganov, I., M. Martin, N. Buchin, and K. Sahlin. "Multi-Context Seeds Enable Fast and
High-Accuracy Read Mapping." *Genome Biology* 27, no. 1 (2026).
https://doi.org/10.1186/s13059-026-04017-x

Leinonen, M., and L. Salmela. "SAKE: Strobemer-Assisted k-mer Extraction." *PLoS One* 18,
no. 11 (2023): e0294415. https://doi.org/10.1371/journal.pone.0294415

Li, X., K. Chen, and M. Shao. "Efficient Seeding for Error-Prone Sequences with
SubseqHash2." *Bioinformatics* 41, no. 8 (2025). https://doi.org/10.1093/bioinformatics/btaf418

Ndiaye, M., S. Prieto-Baños, L. M. Fitzgerald, A. Yazdizadeh Kharrazi, S. Oreshkov, C.
Dessimoz, F. J. Sedlazeck, N. Glover, and S. Majidian. "When Less Is More: Sketching with
Minimizers in Genomics." *Genome Biology* 25, no. 1 (2024): 270.
https://doi.org/10.1186/s13059-024-03414-4

## Protein seeds: reduced alphabets, spaced seeds, two hits on one diagonal

Buchfink, B., C. Xie, and D. H. Huson. "Fast and Sensitive Protein Alignment Using
DIAMOND." *Nature Methods* 12, no. 1 (2015): 59–60. https://doi.org/10.1038/nmeth.3176

**Note:** Spaced seeds (several shapes of weight 12 in the default mode) over an 11-letter
reduced alphabet. One seed, with skipped positions inside it; no second word and no stored
distance.

Takabatake, K., K. Izawa, M. Akikawa, K. Yanagisawa, M. Ohue, and Y. Akiyama. "Improved
Large-Scale Homology Search by Two-Step Seed Search Using Multiple Reduced Amino Acid
Alphabets." *Genes* 12, no. 9 (2021). https://doi.org/10.3390/genes12091455

**Note:** One contiguous seed whose centre uses a finer alphabet than its ends. Closest
published protein seed to a two-part key, but the parts are adjacent and searched in two
steps.

Altschul, S. F., T. L. Madden, A. A. Schäffer, J. Zhang, Z. Zhang, W. Miller, and D. J.
Lipman. "Gapped BLAST and PSI-BLAST: A New Generation of Protein Database Search Programs."
*Nucleic Acids Research* 25, no. 17 (1997): 3389–3402. https://doi.org/10.1093/nar/25.17.3389

**Note:** The two-hit rule: extension starts only when two word hits fall on one diagonal
within a set distance. The pairing happens after the lookup, on the hits, not inside the
index key.

Steinegger, M., and J. Söding. "MMseqs2 Enables Sensitive Protein Sequence Searching for
the Analysis of Massive Data Sets." *Nature Biotechnology* 35, no. 11 (2017): 1026–28.
https://doi.org/10.1038/nbt.3988

**Note:** The prefilter looks up similar k-mers and keeps a target when two of them fall on
one diagonal. Like BLAST, the pairing is done on hits after the lookup.

## What is new in notebook 292

Two short exact words in a 2-letter hydrophobic/polar alphabet, joined into one index key
with or without the residue count between them, scored on aligned Pfam seed pairs against
notebook 232's single-seed baseline. Nothing above does this for proteins or for any
reduced protein alphabet.
