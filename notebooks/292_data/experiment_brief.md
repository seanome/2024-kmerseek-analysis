# Experiment A brief (Olga, 2026-10-14), as given to the session that built notebook 292

Experiment A: adaptive sampling across species, in silico. Two days, timeboxed below.

Repos: kmerseek (seanome/kmerseek) for the alphabet tables only; analysis notebook in
seanome/2024-kmerseek-analysis; the decision script lives in a new folder
seanome/2024-kmerseek-analysis/scripts/adaptive/ with no kmerseek dependency beyond the
alphabet tables. The point is the part list: a hash set, a translator and a rule.

Shared rules
- Before any kmerseek call, run `kmerseek --help` and use only the flags it prints. List the
  alphabets kmerseek ships and pick from that list; copy the alphabet tables into the script.
- One combination of alphabet, k-size and rule threshold is a "setting"; prose never says arm.
- Every rule gets a chance control: the same rule on 10 base-shuffled copies of every read,
  fixed seed. Report keeps above the shuffled rate, never raw keeps.
- Whenever a read and a protein are shown, print the translated read in its frame, the
  protein residues with coordinates, the match line, and "x of y identical".
- Invoke nextflow-notebook-workflow before any environment step, writing-style before prose,
  clear-figures before every figure. Plain words. No word coined in this session.
- Record every download: URL, accession, date, size, md5, basecaller and chemistry if stated.
- Finish with: claims the data supports, claims it does not, the figures, and the next
  smallest experiment. Then the adversarial pass (below). Push nothing, open no PR.

Background. Oxford Nanopore sequencers can eject a read part-way and sequence another one.
readfish decides from the first chunk of signal, by default about 0.8 s in MinKNOW 6 or
later (https://pypi.org/project/readfish/), by mapping that chunk to a DNA reference with
minimap2. DNA mapping stops working once the sample is far from any reference genome. A
translated read in a reduced alphabet keeps matching further: in the QfO benchmark,
reduced-alphabet k-mers held parity with phmmer inside mammals, crossover near 166 Mya.

Question: for a read from an organism with no genome, can a translated reduced-alphabet
k-mer rule on the first few hundred bases decide "this read is from protein family X"
when DNA mapping to the nearest genus cannot? The test that kills it: if DNA mapping with
the same held-out reference keeps as many family reads, the cross-species claim is dead.

Step 0. Reads. Search SRA for Oxford Nanopore cDNA or direct-RNA runs from a venom gland
or from a tunicate, published 2021 or later (platform OXFORD_NANOPORE, library source
TRANSCRIPTOMIC, release date 2021 or later; use the SRA Run Selector or `esearch`, record
the exact query). Prefer a venomous marine animal; any venomous animal is acceptable. Pick
the run with the newest chemistry and the most reads; subsample 200,000 reads with a fixed
seed. If no 2021-or-later run exists, say so, use the newest there is, and state its
basecaller and chemistry beside every number. Record read length distribution (SF1).

Step 1. The target family and the held-out references. Choose a protein family that (a) is
expressed in the tissue, (b) has at least 50 Swiss-Prot entries outside the sample's genus,
and (c) has a Pfam family. For a venom gland: toxin families with a Pfam entry (list the
candidates and their Swiss-Prot counts; pick the largest). Build three references, each
with the Swiss-Prot entries of the family from (i) all genera, (ii) all but the sample's
genus, (iii) all but the sample's family (taxonomic family). Record counts. For the DNA
comparator, fetch the coding sequences of the same entries from ENA by cross-reference;
record how many had one.

Step 2. Truth per read. DIAMOND blastx of every full-length read against Swiss-Prot
(reference i, plus the whole of Swiss-Prot as background), E-value 1e-5, best hit. A read
is a family read when its best hit carries the target Pfam family (Pfam-A.regions.uniprot).
Report family reads, non-family reads, unassigned reads. If the sample's species has a
published transcriptome assembly with protein annotations, add it as a second truth.

Step 3. The chunk. For each read, take the first N bases as sequenced: N = 200, 400, 800.
Say in the notebook that readfish's default chunk is about 0.8 s of signal, which is a few
hundred bases at current speeds, and that 400 is the working value.

Step 4. The k-mer rule. Translate the chunk in six frames. Encode in three alphabets from
kmerseek's list: the 20-letter one, a 6-letter one (Dayhoff-like) and the 2-letter hp.
k per alphabet: 20-letter k = 7, 9, 11; 6-letter k = 10, 13, 16; hp k = 20, 30, 42 (the
last of each is sourmash's default for that alphabet, say so). Index = the set of encoded
k-mers of reference (ii) or (iii). Rule: keep the read if at least m of its chunk k-mers
are in the index, m = 1, 2, 3. Score every chunk against every setting; one row per
read x setting x N, every quantity in its own named column. Time the decision per read in
Python and report it; it must be far below 0.8 s.

Step 5. Comparators on the same chunks and the same held-out references.
  DNA rule, as readfish does it: minimap2 (map-ont) of the chunk against the coding
  sequences from step 1, held out the same way; keep if any primary alignment.
  Slow upper bound: DIAMOND blastx of the chunk alone against reference (ii) or (iii),
  E-value 1e-3; keep if any hit.

Step 6. Chance control. 10 base-shuffled copies of every read, same chunks, same rules.

Step 7. Metrics, per setting, per N, per hold-out level: sensitivity = family reads kept /
family reads; false keep = non-family reads kept / non-family reads, with the shuffled rate
beside it; enrichment = family fraction among kept reads / family fraction among all reads;
pore time: model a kept read as its full length and an ejected read as N bases plus 500
bases of lost time (state this as an assumption, named pore_time_model, not a
measurement), and report enrichment at equal pore time. Pick the best setting on hold-out
(ii) by sensitivity at false keep <= 5%.

Step 8. Hero example. One family read kept by the k-mer rule at hold-out (iii) and not kept
by the DNA rule, with the chunk, its translation, the k-mers that fired, the reference
residues, and the DIAMOND alignment of the full read.

Step 9. Figures. F1 sensitivity against hold-out level, one line per rule, N = 400, shuffled
rate dashed. F2 sensitivity and false keep against N. SF1 read lengths; SF2 settings table;
SF3 decision time; SF4 shuffled distributions.

Step 10. Verdict sentences, both branches in code, filled from the F1 function.
  Alive: "The translated rule reaches where DNA mapping does not." Dead: "DNA mapping keeps
  as many; the cross-species claim is dead at this hold-out."

Step 11. Writing and checks: notebook, scripts/adaptive/decide.py with --chunk, --alphabet,
--k, --min-kmers and a ten-line README, prose check, adversarial pass, biologist pass.

Decision: continue if, at the genus hold-out, the best k-mer setting keeps at least half of
the family reads at a false keep of 5% or less while the DNA rule keeps under 10%.

(Condensed from the brief pasted in chat; steps 6 to 11 are shortened, wording of every
rule and threshold kept.)
