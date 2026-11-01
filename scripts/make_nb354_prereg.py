"""The pre-registered first cells of notebook 354, written before any result was computed.

scripts/make_nb354.py imports PREREGISTRATION and puts it at the top of the notebook
unchanged. Its git history shows it was committed before the development and test runs.
"""

PREREGISTRATION = r"""
# 354. Kill test: does a reduced-alphabet kmerseek index find the source protein of a short de novo peptide call?

**Question.** A mass spectrometer reads a peptide's sequence from its fragment masses
(de novo sequencing), with errors. Can a kmerseek index in a reduced alphabet, where
residues the spectrometer confuses share one letter, find the protein the peptide came from
better than the standard route: an exact index in the full 20-letter alphabet plus
blastp-short? The decision: if it fails, kmerseek is not pitched for proteomics source
lookup, and notebook 300's peptide work does not go on to a database-scale run.

**Prior, stated before any run: unfavourable.** Merging residues throws away the
differences that usually tell one source protein from another.

## Data (frozen under `data/kill_tests/fragment_source/`, with checksums)

- **Database:** UniProt human reference proteome UP000005640, release 2026_03 (file dated
  15 Sep 2026, downloaded 20 Oct 2026), 20,652 proteins, plus the cRAP contaminant list
  (The GPM, file dated 4 Mar 2019), 82 of its 116 entries; the other 34 are already in the
  human proteome with the same sequence. 20,734 proteins in all.
- **Peptide calls:** one completed analysis, PRIDE PXD005280 (Blank-Landeshammer et al.,
  *J Proteome Res* 16, no. 9 (2017): 3209-18, https://doi.org/10.1021/acs.jproteome.7b00198,
  as PRIDE lists it), run HeLa 1 (`qExHF01_02580`, a Q Exactive HF going by the file names). The calls are PEAKS
  de novo sequences, file `PEAKS_qExHF01_02580_de novo peptides.csv` inside `PEAKS.zip`,
  with PEAKS's per-residue local confidence and its ALC score (average local confidence)
  kept. PEAKS exported only calls with ALC >= 50%; nothing else is removed. Wrong and
  ambiguous calls stay in.
- **Accepted source:** the database search of the same spectrum, file
  `HeLa_1_qExHF01_02580.msf` (Proteome Discoverer 1.4: Mascot, MS Amanda and SEQUEST against
  a human UniProt file `U507_human_fw_2015_07_22.fasta` of 20,163 proteins; Percolator). A spectrum counts when every engine with a
  rank-1 match at Percolator confidence "High" (q < 0.01) gives the same peptide. Its
  accepted sources are all the proteins that peptide maps to, kept if the frozen database
  has them. This search never sees the de novo call. Both analyses still use the same spectrum,
  so their errors need not be statistically independent.
- 22,332 de novo calls have an accepted source in the database (joined to their PD spectrum
  on charge, precursor mass within 10 ppm and retention time within 0.02 min, because the two
  programs number spectra differently). **500**
  are drawn at random (seed 354).
- **Split by source protein:** calls that share an accepted protein form a group; groups go
  to development until it holds half the calls (seed 3540). 250 development, 250 test, no
  protein in both.
- **Source-absent set:** 50 of the 250 test calls (20%, seed 3541). Their 59 accepted
  proteins are removed from the database these 50 are searched against. The right answer
  for them is "abstain".
- **Shuffled controls:** 10 composition shuffles of each development call (2,500, used to
  set each method's confidence threshold) and, for each test call, 5 composition shuffles
  and 5 shuffles that keep every adjacent residue pair (seed 3542).

## The two workflows

Both get the same uncertainty information: I and L have the same mass, and Q and K differ
by 0.036 Da, so both pairs count as the same residue. PEAKS writes every I/L as L. The
per-residue local confidence is kept in the frozen table but given to neither workflow.

- **K, kmerseek** (seanome/kmerseek 4d7e27b, PR 156): index the database in the mass-spec
  alphabet `massspec18` (I = L, Q = K, the other 16 residues apart), search each call, take
  every protein kmerseek reports as a candidate (`--min-shared-kmers 1
  --max-query-pvalue 1 --extend-mismatch-penalty 0`, scaled 1). One k from {4, 5, 6, 7, 8,
  9}, chosen on development. A call shorter than k gets no candidates and counts as an
  abstention. At most one coarser alphabet that also merges I and L
  (`hsdm17` or `sdm12`) with its own k, chosen on development, goes to test only if it
  recovers more development calls than massspec18.
- **C, comparison:** an exact 5-mer index of the database in the full alphabet (I written as
  L, Q as K) that finds every protein holding the whole peptide, plus blastp-short (BLAST+
  2.16.0, `-task blastp-short -evalue 1000 -max_target_seqs 500`) against the same
  database. Its candidates are the union of the two.
- **The check, the same for both:** each candidate is aligned with the call in the full
  alphabet (Smith-Waterman, PAM30, gap open 9, extend 1, as blastp-short; parasail 1.3.4).
  Its score ranks the candidates; ties go by accession.
- **Confidence threshold:** for each method, the smallest score that at most 1% of the
  2,500 shuffled development calls reach with their best candidate. A call whose best
  candidate scores below it is an abstention. The candidate list is every candidate at or
  above it.
- **Runtime, end to end:** index build, search and check, on one CPU, for the 250 test
  calls against the full database; wall and CPU time.

## Pass rule (fixed before running; all must hold)

On the 200 source-present and 50 source-absent test calls, for a kmerseek setting frozen on
development:

1. At least 95% of source-present calls have an accepted source among the top five
   candidates at or above the threshold.
2. At most 1% of source-absent calls get a confident assignment (best candidate at or above
   the threshold). With 50 calls, that means none.
3. The median candidate-list length over source-present calls is at most five.
4. And one of:
   a. recovery (rule 1's fraction) at least 10 percentage points above workflow C's best
      recovery at a threshold whose false-assignment rate is no higher than kmerseek's; or
   b. at least fivefold lower end-to-end wall time than workflow C, with recovery no more
      than 2 points below C's ("matched recovery").

PASS if one frozen kmerseek setting meets all four. Otherwise FAIL. Also a fail, whatever
the four say: the reduced alphabet makes candidate lists longer without recovering more
real calls, or its speed gain disappears once the candidates are checked in the full
alphabet.

## How 354 differs from notebook 300

Notebook 300 (branch `olgabot/nb300-peptide-lookup`, not merged) asks whether kmerseek can
find the source of gut metaproteomics peptides in UHGP-90 with a decoy-measured error rate;
its Step 0 computes the smallest E-value one exact window can reach. 354 copies its
`massspec18` alphabet table and its counting estimate of chance matches (expected count =
starts x product of residue-class frequencies), but uses a human proteome plus
contaminants, real de novo calls whose sources were accepted by an independent database
search, and a 20% source-absent set that must be abstained on.
"""
