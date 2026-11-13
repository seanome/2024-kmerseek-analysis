# RNA-seq kill test: fixed rules

Recorded 2026-10-22 (America/Los_Angeles), before any biological run.

You are implementing a kill test. Read everything before writing code. The pass/fail rules below are fixed now and may not be changed after any result is seen. If a rule turns out to be wrong, say so in the report and still report the result under it.

WHAT IS BEING TESTED

Claim 1 (speed): a reference-free k-mer route from RNA-seq reads to a ranked, family-tagged, GO-enriched list runs in under a tenth of the wall-clock time and CPU-hours of the standard route as a lab with no reference genome would have to run it, and in under half the time of the standard route with a reference.
Claim 2 (concordance): the k-mer route's top 300 units hit at least 70 of the standard route's top 100 differentially expressed genes, and its top 20 enriched GO terms overlap the standard route's top 20 in at least 10.
Claim 3 (tagging): on hits that sit inside a known Pfam domain, the k-mer family tag is right for at least 80 of every 100; on hits outside any protein, a family is wrongly given to at most 5 of every 100; the tag step uses under a tenth of the CPU-hours of a Pfam profile search (hmmscan) on the same sequences.
Claim 4 (alphabet): the reduced amino-acid alphabet recovers at least 5 more of every 100 in-domain hits than the full 20-letter alphabet at the same false-family rate, each alphabet at its best k from a sweep. If not, the reduced alphabet is dropped and claims 1 to 3 are re-reported in 20 letters.
Claim 5 (not noise): over every balanced swap of condition labels, the true labels yield more significant units than every swap, and the median swap yields fewer than 5 significant units for every 100 the true labels yield.

Every threshold above (tenth, half, 300, 70, 20, 10, 80, 5, 5 points) is chosen, not measured. Record them in a file named decision_rules.md before the first run, with the date, and never edit that file afterwards.

THE TWO ROUTES

Standard route, with reference (control A, the answer key): nf-core/rnaseq 3.27.0 (https://nf-co.re/rnaseq/3.27.0) with STAR and Salmon on the species' reference genome and annotation, then nf-core/differentialabundance 2.0.0 (https://nf-co.re/differentialabundance/2.0.0) with DESeq2 and GO enrichment. Record the top 100 genes by adjusted p-value with |log2 fold change| ≥ 1, the top 20 enriched GO terms, and the per-gene Pfam domains from the annotation.

Standard route, no reference (control B, what the customer faces): the same reads, with the genome hidden. Assemble a transcriptome (Trinity), predict coding regions (TransDecoder), annotate (DIAMOND blastp against the same protein database the k-mer route uses, plus hmmscan against Pfam-A), quantify (Salmon against the assembly), test (DESeq2 via nf-core/differentialabundance with the assembly as the "reference"), enrich (GO terms via pfam2go and the DIAMOND hits). Record the same outputs and every timing.

K-mer route (the thing under test), three layers after trimming (fastp) and a per-sample sketch and contaminant screen (sourmash sketch + gather against a small contaminant set: human, mouse if not the test species, E. coli, yeast, mycoplasma, PhiX):
  Layer 1, SPLASH2 (refresh-bio/splash): anchors and targets per sample, the SPLASH test between the two conditions, extend significant anchors to compactors. Use SPLASH2's documented defaults; record every parameter.
  Layer 2, kmerseek (seanome/kmerseek): infer strandedness from stop-codon counts per strand on a sample of 100,000 reads (the strand whose three frames carry fewer stops is the transcript strand; report the ratio); translate each compactor in three frames; drop any frame containing a stop codon; cut reduced-alphabet protein k-mers; look them up in the family index; count only k-mers that belong to the family and to none of its nearest competing families; assign the family with the most such k-mers if it passes a threshold set on the development data (see below), else "no family". Attach GO terms from pfam2go for the family and from the matched proteins' UniProt GO annotations. Run enrichment (clusterProfiler enricher with a custom term-to-unit mapping, or gprofiler2 with a custom background) with tagged units as the background. Rank units by SPLASH effect size and reproducibility across replicates.
  Output: one line per unit: sequence, family or "no family", matched proteins, GO terms, event class from SPLASH, per-sample counts, supporting reads.

The family index: build it from Pfam-A seed and full alignments' member sequences (or UniProt proteins carrying Pfam domains) in the chosen reduced alphabet and in 20 letters. REMOVE every sequence from the test species' held-out lineage (defined below) before building. Never let the test species' own proteins or its lineage's proteins into the index.

DATA

Test species: mouse. Reason: it has the best annotation and the most public RNA-seq, and its distance to the nearest non-held-out relatives after removing all vertebrates from the index (tunicates, amphioxus, echinoderms) crosses the same node a tunicate such as Botryllus must cross to reach a well-annotated relative. Held-out lineage: all vertebrates (NCBI taxon 7742). Second run at a closer distance: put back all vertebrates except mammals (hold out taxon 40674) and report what the shorter distance buys.

Development data, for setting the family-call threshold and nothing else: one public two-condition stranded RNA-seq experiment from zebrafish. Do not touch the mouse test data until every threshold is set and written to decision_rules.md.

Test data: choose one public mouse bulk RNA-seq experiment meeting all of: stranded paired-end library, two conditions, at least four biological replicates per condition, at least 20 million read pairs per sample, published differential expression result available as a sanity check. Record the SRA/GEO accession, the selection date, and every sample used. If several qualify, take the one with the most replicates. Do not choose by looking at results.

Synthetic control (control C): 2,000 sequences cut from known mouse proteins at lengths 18, 50, 100 and 200 residues (500 each), back-translated with mouse codon usage, plus 2,000 sequences of the same lengths cut from mouse 3' untranslated regions, plus 500 sequences from known proteins with a single-base substitution placed at random. Run layer 2 on all of them. This gives the false-family rate on non-protein sequence and the loss from a stop created by error, both with a known answer. Classify every mouse compactor from the real run into three classes by mapping to the hidden reference AFTER the k-mer route has finished: inside a Pfam domain, inside a protein but outside any domain, outside any protein. Score claim 3 on the first and third classes only; report the second class separately.

Label swaps (control D): every balanced reassignment of the condition labels (with four against four that is 18 swaps). Run SPLASH2 on each. This is control D for claim 5.

Do not use shuffled sequences or shuffled databases as controls anywhere.

MEASUREMENT

Same machine, same reads, same declared input size for every route. Record wall-clock time, CPU-hours and peak memory per stage with /usr/bin/time -v and the workflow's own trace, on one ordinary cluster node (no GPU) and separately on a 16 GB laptop-class allocation if the cluster offers one. Include reference and index construction in each route's time, and report it also as a separate line. Report staff effort as a separate line: count the number of manual interventions each route needed, do not estimate hours.

Compute the top-300 units to top-100 genes overlap by mapping units to the hidden reference only after the k-mer route is done, and count a gene as hit if any of its exons overlaps a unit by at least 20 bases. Report also the overlap at 100, 200, 500 and 1,000 units, as a curve.

CONVENTIONS

Nextflow with one conda environment per process, pinned versions, no containers that fetch at runtime. No GPU. Every parameter in a params file. Every run under -resume with its trace, timeline and report kept. Notebooks for figures, code cells collapsed, one figure per claim, each with a legend before the marks, one mark per meaning, units on every axis. Plain words in every README, docstring, commit message and figure caption; define each technical term where it first appears; no coined terms. Never write a number into prose that the code did not print; read it from the output file.

REPORT

A single report (report.md with figures) in this order:
1. decision_rules.md reproduced verbatim.
2. Data chosen, with accessions and dates.
3. One table per claim: the rule, the measured value, PASS or FAIL. Measured values only; a value the run did not produce is "not measured".
4. The overlap curve (claim 2), the timing table per stage per route (claim 1), the three-class tagging table (claim 3), the alphabet sweep (claim 4), the swap distribution with the true labels marked (claim 5).
5. What the second run at the closer distance changed.
6. Everything that surprised you, and every place you had to make a choice the rules did not cover, with the choice stated.
7. An adversarial pass: a separate agent that has not seen the code is given the report, the code and the outputs and told to disprove each PASS. Its verdicts (HOLDS / WRONG / UNVERIFIABLE with the deciding line or number) go in the report unedited, followed by what you changed in response.

DO NOT

Do not tune anything on the mouse data. Do not change a threshold after a result. Do not present a formula value as a measurement. Do not drop a failed claim from the report. Do not let any vertebrate protein into the index for the main run. Do not use the test species' annotation anywhere in the k-mer route; it is used only to score, after the route has finished.