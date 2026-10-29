# RNA-seq kill test: pre-run report

## 1. Fixed decision rules, verbatim

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

## 2. Data chosen

No development or test experiment has been selected. No sequencing reads were
downloaded or analysed. No published DE output was inspected. No biological run began.
No calibration file exists. Mouse preflight is blocked until calibration is frozen.

Public metadata searches found possible stranded zebrafish studies. These are
candidates, not selected inputs. The study described in
[The Genome-Wide Impact of Nipblb Loss-of-Function on Zebrafish Gene Expression](https://pmc.ncbi.nlm.nih.gov/articles/PMC7766774/)
describes stranded paired-end libraries and triplicate conditions. Full sample/run
metadata still need verification. The neuroblastoma sample record
[GSM7016830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7016830)
also documents stranded paired-end sequencing, but its study includes several
genotypes and has not been accepted as a two-condition development experiment.

Searches also returned mouse papers before development; no experiment was selected,
no sample-level test manifest was read, and no test result was used. Future mouse
selection must follow the committed calibration. Selection dates and complete sample
accessions remain not measured; a search date is not a selection date.

## 3. Fixed claim decisions

### Claim 1: speed

| Rule | Measured value | Decision |
| --- | --- | --- |
| Wall time and CPU-hours below 0.1 of the no-reference route | not measured | not measured |
| Wall time and CPU-hours below 0.5 of the reference route | not measured | not measured |

### Claim 2: concordance

| Rule | Measured value | Decision |
| --- | --- | --- |
| Top 300 units hit at least 70 of the top 100 genes | not measured | not measured |
| Top 20 GO terms overlap by at least 10 | not measured | not measured |

### Claim 3: family tagging

| Rule | Measured value | Decision |
| --- | --- | --- |
| In-domain correct-family fraction at least 0.80 | not measured | not measured |
| Outside-protein false-family fraction at most 0.05 | not measured | not measured |
| Tagging CPU-hours below 0.1 of hmmscan on the same sequences | not measured | not measured |

### Claim 4: amino-acid alphabet

| Rule | Measured value | Decision |
| --- | --- | --- |
| Reduced-alphabet recall gain at least 0.05 at equal false-family rate, each at its best k | not measured | not measured |

The alphabet has not been retained or dropped. A measured failure requires claims
1 to 3 to be reported again using protein20. Missing evidence does not select a winner.

### Claim 5: label swaps

| Rule | Measured value | Decision |
| --- | --- | --- |
| True labels produce more significant units than every balanced swap | not measured | not measured |
| Median swap count below 0.05 of true count | not measured | not measured |

## 4. Required figures and stage measurements

The overlap curve, route timing table, three-class tagging table, alphabet comparison
and swap distribution are not measured. No plot is drawn without measured outputs.
Index-build cost, peak memory, 16 GB allocation outcomes and manual-intervention counts
are not measured. Software unit tests are not biological measurements.

## 5. Closer-distance run

Not measured. Neither the vertebrate-excluded nor mammal-excluded family index exists
for this experiment. No gain from the closer distance has been measured.

## 6. Problems and choices identified before running

The supplied decision rules are unchanged. The separately frozen
[implementation choices](implementation_choices.md) contain the calibration procedure,
ranking ties, lineage exclusions, condition test and missing-value treatment.
Olga approved freezing that procedure first and putting the learned numeric thresholds
in a separate immutable calibration file before mouse access.

The literal swap rule includes an exchange of all condition names. A symmetric
two-condition test gives that exchange the same significant-unit count, so the strict
rule cannot pass in that case. This is a property of the rule, not a measured result.
The primary result will keep the literal comparison even if it fails for this reason.
The statement that four against four gives 18 swaps is inconsistent with enumeration:

```json
{
  "source": "exhaustive combinatorial enumeration; not a biological measurement",
  "balanced_assignments": 70,
  "alternative_labelled_assignments": 69,
  "distinct_splits_ignoring_condition_names": 35,
  "alternative_splits_ignoring_condition_names": 34
}
```

These counts were printed by the code and checked independently by binary enumeration.
They describe possible assignments, not observed SPLASH results.

SPLASH's default test does not take condition labels. Source inspection found the
fixed-weight condition test and its Cjs_0_pval output in version 2.11.9. The optional
GLM output filters coefficients; it must not be treated as the requested significant
unit count. The condition-test binary and corrected column remain untested.

Sources: [pinned driver](https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/splash.py),
[pinned output writer](https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/satc_merge/satc_merge.cpp),
[GLM documentation](https://github.com/refresh-bio/SPLASH/wiki/GLM_supervised).

The implementation is incomplete. Missing biological stages and launch gates are
listed in [README.md](README.md). No Nextflow workflow, pinned conda installation,
cluster resource prediction, biological controls or figure notebook has been tested.

## 7. Independent adversarial review

Review of the pre-run snapshot, 2026-10-22. I ran all 12 existing unit tests; all passed. I also ran independent enumeration and two additional counterexamples. No biological PASS appears in this report.

- UNVERIFIABLE — Claim 1, speed. Neither route timing nor CPU-hours were measured. The entries in report.md:99–100 correctly say “not measured.”
- UNVERIFIABLE — Claim 2, concordance. No ranked units, answer-key genes or enrichment results exist. Report.md:106–107 correctly withholds a verdict.
- UNVERIFIABLE — Claim 3, tagging. No domain-labelled predictions, non-protein predictions or hmmscan timings exist. Report.md:113–115 correctly withholds a verdict.
- UNVERIFIABLE — Claim 4, alphabet advantage. No calibration or alphabet comparison was run. Report.md:121 correctly says “not measured”; neither retaining nor dropping the reduced alphabet is supported.
- UNVERIFIABLE — Claim 5, biological label-swap result. Enumerating assignments does not measure SPLASH results. Report.md:130–131 correctly withholds a verdict.

Technical claims:

- HOLDS — Current rule files match their saved fingerprints, and report.md contains decision_rules.md verbatim. I ran check_rules() and independently tested exact inclusion. This establishes consistency of the supplied snapshot.
- UNVERIFIABLE — The rules were never changed historically. A file and a neighbouring fingerprint cannot establish their history or when either was written.
- HOLDS — The preflight function checks calibration before opening the mouse sample manifest. The existing mocked test exercises the missing-calibration path and passes.
- UNVERIFIABLE — No mouse manifest was opened elsewhere before calibration. The supplied files provide no comprehensive access record. The preflight ordering cannot establish this historical claim.
- HOLDS — Eight samples divided four against four produce 70 labelled assignments, 69 alternatives, 35 splits ignoring exchanged condition names, and 34 alternative splits. I independently enumerated all 256 binary strings and obtained the identical set of 70 balanced assignments.
- HOLDS — Including the exchanged condition names makes a tied significant-unit count fail the strict primary swap rule. The executable complement-count test passes. This remains a mathematical consequence, not a measured biological failure.
- HOLDS — Missing measurements do not become zero or PASS. Calling score_claims({}) returns “not measured” for all five claims and for alphabet selection. The recorded evidence agrees. Zero true significant units also avoid division by zero and fail the strict comparison.
- WRONG — The scoring helpers reject all boolean values masquerading as measurements. Executable counterexample: score_claims({"4": {"recall_gain_at_equal_false_family_rate": True}}) returns PASS. The recall-gain branch converts True to 1.0 without the boolean rejection used by finite_number(); see protocol.py:273–274 in the formatted snapshot. Reject booleans before this conversion.
- HOLDS — The tested numerical boundaries follow the stated strict and inclusive comparisons: equality fails the speed limits and the median-swap ratio limit; equality passes the gene, GO, accuracy and recall-gain thresholds. The existing boundary tests pass.
- WRONG — The index-member audit alone establishes complete held-out-lineage exclusion. Executable counterexample: a row with taxid 9606, lineage_taxids "1;9606", and an otherwise valid record passes exclusion of taxon 7742. The audit checks the supplied ancestor list without establishing its completeness; see protocol.py:151–156 in the formatted snapshot. README.md:24–26 acknowledges this limitation. A taxonomy-backed check and correspondence between audited records and built index sequences are still required.
- HOLDS — The overlap helper counts each answer-key gene once and rejects a 19-base overlap in the supplied test. It consumes reported overlap lengths; it does not independently establish exon coordinates or that an overlap came from the hidden-reference scoring stage.
- UNVERIFIABLE — The pinned SPLASH binary produces the required corrected condition-test column. No binary execution or output demonstrating this was supplied; report.md:175 explicitly records that it remains untested.
- WRONG — The report writer guarantees it will never overwrite biological measurements. Its protection checks only whether the existing report contains “The implementation is incomplete.” A report containing that sentence and newly added biological measurements would still be replaced; see write_status.py:163–170. The comment promises stronger protection than the condition provides.

The requested overall implementation is incomplete. README.md:73–100 lists missing data acquisition, biological workflows, calibration, synthetic controls, actual label-swap runs, timing instrumentation, cluster launch checks and figures. The supplied work supports a preliminary protocol-checking utility and an honest pre-run report. It does not perform the RNA-seq kill test.

### Changes in response

The original review above is reproduced without edits. Its line references describe the reviewed pre-run snapshot, before these fixes.

- Reject boolean recall gains before conversion to a number. The regression test sends True and requires an error.
- Require an NCBI nodes.dmp snapshot and derive each sequence species’ ancestors from its parent links. Reject unknown taxa, cycles and records that disagree. The regression test rejects a human record whose supplied lineage omits vertebrates. Correspondence between the audited members and a built index remains unimplemented.
- Require the existing report to match its last generated SHA-256 before changing any output. A regression test appends a measurement while retaining the incomplete-implementation sentence. Regeneration now fails and leaves the report unchanged.

These fixes do not supply biological evidence. Every scientific claim remains not measured. The biological workflows, calibration, data selection and figures remain unimplemented.

### Independent follow-up, unedited

Follow-up review, 2026-10-22:

- HOLDS — Both True and False are rejected as recall-gain measurements. I reran the original boolean counterexample.
- HOLDS — The truncated human lineage is rejected using ancestry derived from the supplied taxonomy. A nonvertebrate control remains accepted. This checks the audit logic; the taxonomy’s authenticity and its correspondence to actual index sequences remain unverified.
- HOLDS — Adding a biological measurement to a generated report blocks regeneration. I checked every file in the temporary directory before and after rejection; none changed. Regeneration of an unchanged report still succeeds.
- HOLDS — All 15 unit tests pass. I found no new regression in these three fixes.
- UNVERIFIABLE — All five biological claims remain unmeasured. The overall RNA-seq kill-test implementation remains incomplete.
