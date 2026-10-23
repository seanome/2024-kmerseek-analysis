# Botryllus electrical-stimulation analysis and unknown-gene investigation

Frozen 2026-10-23 before new full-depth or functional-search results.

## Study and inputs

The primary experiment is Domen et al. 2026, DOI 10.1073/pnas.2610968123,
PRJNA1219309. Recover all 64 public paired runs, including libraries excluded
from the published tests. Keep the published inclusion list as a separate field.
The 2018 sorted-cell and 2021 developmental studies are historical material.
No new inference for those studies is authorized by this protocol.

Use the paper, Dataset S1, and BioSample attributes to resolve each run.
The six published comparison groups are provenance, not permission to pool times.
Split the published 13/49-day and 41/93-day groups for new treatment tests.
Check collection ages and genotype for each proposed pair; do not infer independence
from different vial numbers. Young 2-hour and 24-hour cohorts each contain repeated
genotypes. The old cohort contains one genotype. Its inference cannot generalize
to a population of genotypes.

Treatment inference remains blocked until the experimental unit, repeated collections,
and estimable design are documented. Preserve upstream DE-kupl counts with masking
set to `nomask`. Its condition-only inferential model is not the primary analysis for
matched collections or repeated genotypes. Any external count-matrix model must carry
its own name, design, tests, and multiple-testing population.

## Completion and preprocessing

Archive MD5, compressed byte count, gzip exit status, synchronized identifiers,
paired-record counts, and total bases must agree before an input completion marker
is written. Existing final files with different checksums cause a failure and remain
untouched. Partial downloads have separate filenames. A directory lock prevents
concurrent writes; a stale lock requires job inspection before removal.

The pilot used cutadapt 5.2, adapter AGATCGGAAGAGC on both mates, quality 20,
minimum length 62 nt, and rejection of a pair if either mate fails.
Full-depth preprocessing settings must be recorded separately after comparing the
pilot with nf-core/rnaseq 3.27.0. No custom subset is called a complete nf-core run.
SPLASH and DE-kupl must consume the same validated trimmed pairs and sample IDs.

## Sequence evidence and unknown-gene selection

Preserve the input proteome byte for byte. Assign stable record IDs BSP000001 onward
by original FASTA order. Keep duplicate original IDs and all reference mappings.
The pilot's explicit historical GFF crosswalk does not establish the paper's exact
reference version or a mapping to the new proteome.

Standard baseline A: DIAMOND 2.1.23 ultra-sensitive protein search against a checksummed
reviewed Swiss-Prot snapshot, E-value at most 1e-5, retain up to 25 targets per query;
HMMER 3.4 against the checksummed existing Pfam-A database using family gathering
thresholds (`--cut_ga`). Record domain coordinates, coverage, conflicting families,
and database version evidence. These searches cover every protein, including unchanged
proteins. eggNOG-mapper is an additional orthology layer only after its environment
and database are verified. A missing layer is `not measured`.

Layer B adds remote homology or structure evidence for reliable prioritized sequences.
Layer C adds experimental association and gene-model checks. Report changes in claims,
contradictions, and costs separately. More labels do not measure annotation accuracy.
No structure search is interpreted confidently for short or unreliable fragments.

For deep review select up to 15 nonredundant unresolved proteins after baseline A.
Require length at least 100 aa, no internal stop, and at most 1 percent ambiguous
residues. Prefer validated experimental sequence support, then the number of distinct
supporting anchors, then peptide coverage, then stable record ID. Exact duplicate
sequences share one review card while retaining every record in exports. If fewer
than 10 pass, report the shortage instead of lowering thresholds. A second descriptive
queue can contain reliable proteins without experimental support, ranked by stable ID.
Do not rank by an appealing functional story.

Include five annotated controls and five proteins without detected experimental
association, matched as closely as possible by length. Lack of selected pilot support
does not establish unchanged expression. Such controls are labelled `association not
measured` until full-depth abundance analysis supports an unchanged-control definition.
Use deterministic residue shuffles as retrieval/search diagnostics, not an empirical
false-discovery rate. Freeze the specific candidate IDs before deep review.

Classify unknown status separately: missing supplied annotation; no informative match
in searched databases; recognizable uncharacterized family; possible incomplete,
split, or fused model; ambiguous paralog; possible contamination/artifact; or no detected
relatives under the completed searches. A no-hit result never establishes a new gene.

Each claim has an evidence source and confidence description. Molecular function,
localization, and experimental association are different fields. Model changes remain
proposals in separate files, with sequence crosswalks and read/junction evidence where
available. Do not edit supplied models. Blinded cards hide method labels and treatment
direction until reviewers assess sequence support, ambiguity, and model reliability.
No review by Ayelet is claimed or requested without further user instruction.

## Discovery and interpretation rules

SPLASH composition results are unsupervised across-sample findings. Preserve all tested
anchors and full statistics. As amended by Olga on 2026-10-23, extend every anchor with
`pval_opt_corrected < 0.05` in each separately analyzed cohort. There is no anchor cap.
Sort by corrected p-value and then anchor for deterministic processing.
The older top-300 pilot remains a historical subset and is not overwritten.
Keep parent/child and terminal Compactors records separate. Full discovery counts are
never replaced by extension-subset counts. A new treatment association requires an
explicit valid model, not an interpretation of unsupervised significance.

For a valid external differential-abundance analysis, prefilter counts at least 5 in
at least the smallest valid biological group, use BH-adjusted p at most 0.05 and absolute
log2 fold change greater than 1. Report filtering and multiplicity per contrast.
Keep published nominal-p lists separate from published adjusted-p results.
No combined p-value across methods. Matching sequence findings on the same reads are
corroboration, not independent validation. Contigs and k-mers are not gene counts.

Use actual sourmash protein k7, Dayhoff k12, HP k30; scaled10 primary, 100 sparse,
5 sensitivity. Preserve original pilot thresholds as historical retrieval rules:
at least two shared hashes and raw query containment at least 0.1, top five hits per
query/database/setting. Report empty sketches and eligible denominators. Confirm
promising matches by alignment and domain evidence. Six-frame translations split at
stops/ambiguity and at least 20 aa are peptide fragments, not complete coding sequences.

SPLASH+ requires a consistent genome/annotation, STAR index, splice sites, exon
boundaries, and gene coordinates. Compactors alone does not complete SPLASH+.
Do not infer alternative splicing without junction or transcript-structure evidence.

## Evaluation and proposed experiments

Record observed, predicted, and hypothetical statements separately. Review rubric:
sequence reliability; family/domain specificity; competing assignments; support for
experimental association; model ambiguity; and a falsifiable follow-up hypothesis.
Report each on a stated ordinal scale with evidence, never an invented probability.
Evaluate held-out sequences by family/species only after leakage checks. Candidate
ranking is not an estimate of overall annotation accuracy.

Propose localization, perturbation, or biochemical experiments only when the relevant
protocol is supported as feasible in Botryllus. Label feasibility that has not been
verified. No sorted-population or bulk RNA-seq table is passed to SAMap as single cells.
