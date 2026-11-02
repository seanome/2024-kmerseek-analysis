# How much of each reading frame matches Swiss-Prot, for Botryllus reads (notebook 361)

Asked 2026-11-02: show the distribution of the share of k-mers matching Swiss-Prot for each frame,
for Botryllus reads.

![Share of each frame's k-mers in Swiss-Prot](frame_shares.png)

Each line is one group of frames. Its height at x is the share of frames in the group with at least
x of their k-mers in Swiss-Prot (UniProt's reviewed proteins). The groups are the true frame, the
two other frames on the same strand (1 or 2 nucleotides off), and the three frames on the opposite
strand. Light grey vertical lines mark the search's two cutoffs, 0.1 and 0.5. Rows are orpheum's
two default k-mer types: protein 7-mers and Dayhoff 12-mers. Dayhoff is a 6-letter alphabet that
groups amino acids by chemistry.

Reads, all with a known true frame:
- **simulated**: 39,854 reads of 150 nt cut from 1,993 new-genome Botryllus CDS (coding sequences);
- **real**: 243,938 reads from one PEC and one control run (JA113, JA123; PEC is pulsatile
  electrical current). Each read's best DIAMOND blastx hit to a new-genome protein has 95 % or
  more identity and covers 90 % or more of the read.

| reads | k-mers | frames | median share | at least 0.1 | at least 0.5 |
|---|---|---|---|---|---|
| simulated | protein 7 | true | 0.227 | 88 % | 6 % |
| simulated | protein 7 | same strand, 1 nt off | 0.116 | 56 % | 0.1 % |
| simulated | protein 7 | same strand, 2 nt off | 0.093 | 46 % | 0.1 % |
| simulated | protein 7 | opposite strand | 0.136 | 62 % | 0.5 % |
| real | protein 7 | true | 0.25 | 82 % | 18 % |
| real | protein 7 | same strand, 1 nt off | 0.167 | 70 % | 4 % |
| real | protein 7 | same strand, 2 nt off | 0.10 | 51 % | 1 % |
| real | protein 7 | opposite strand | 0.14 | 63 % | 3 % |
| simulated | Dayhoff 12 | true | 0.487 | 97 % | 49 % |
| simulated | Dayhoff 12 | same strand, 1 nt off | 0.184 | 72 % | 9 % |
| simulated | Dayhoff 12 | same strand, 2 nt off | 0.132 | 61 % | 5 % |
| simulated | Dayhoff 12 | opposite strand | 0.205 | 74 % | 12 % |
| real | Dayhoff 12 | true | 0.50 | 85 % | 51 % |
| real | Dayhoff 12 | same strand, 1 nt off | 0.278 | 74 % | 29 % |
| real | Dayhoff 12 | same strand, 2 nt off | 0.118 | 55 % | 10 % |
| real | Dayhoff 12 | opposite strand | 0.205 | 66 % | 22 % |

- The true frame is shifted right of the wrong frames in every panel. But the curves overlap a
  lot, so no single cutoff separates them. At 0.1, half or more of the wrong frames pass. At 0.5,
  only 6 % (simulated) and 18 % (real) of true frames pass with protein 7-mers.
- Wrong frames match Swiss-Prot often. A random 7-mer is in Swiss-Prot by chance more often than
  one might expect, and with Dayhoff's 6 letters a 12-mer is even less specific. Real reads have
  more conserved, highly expressed genes, so all their frames score higher.
- Wrong frames mostly contain a stop codon (61 % to 91 % of them, against 3 % to 5 % of true
  frames: reads that cover the CDS's own stop). That is why orpheum only allows stop-free frames.
  These curves include every frame, stops or not.

Code: `scripts/frame_shares.py` ran on Sherlock on `sra_search/frame_test/scores.{protein.k7,dayhoff.k12}.tsv`
(from `pec-tools frame-scores`) and `real_blastx.tsv`. `scripts/plot_frame_shares.py` draws the figure.
Numbers: `frame_shares.*_summary.tsv`, `frame_shares.*_hist.tsv` (0.02 bins).
