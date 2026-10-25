# Botryllus Shazam Nextflow migration

The DSL2 workflow consumes a validated paired-read manifest. It separates canonical
31-mer counting, Kallisto quantification, pooling, normalization, each blocked model,
SPLASH, native mergeTags, paired-read recruitment, Compactors, translation, target
preparation and each database/alphabet/scale search. It does not invoke Snakemake or
submit the historical sbatch chain. Counting explicitly uses 63-bit in-memory and
seven-byte output counters: independent recounts exposed an error with the pinned
Jellyfish default counter width. This differs from the installed Snakefile flags.

The preserved nf-core run failed after all 64 Trim Galore tasks completed. Use
`recover_trimmed.sbatch` to validate the existing outputs before supplying its JSON
manifest. Recovery checks gzip integrity, synchronized identifiers, sequence/quality
lengths, counts and SHA-256. It creates shared links only after every input passes.
It does not alter existing FASTQ files or infer strandedness.

The historical plan and results remain under `reports/botryllus-shazam/nextflow-plan`.
Real-tool fixtures and independent count/statistical checks passed on Sherlock,
including a 125-task cache resume. See the migration STATUS report for the exact
positive/empty source coverage. Production execution and app publication remain
separate acceptance gates; no full-depth result is claimed here.
This workflow starts from validated trimmed reads. Fresh download/trimming and
archive-wide download orchestration remain outside it.

## Inputs

All parameters are required: `inputs`, `samples`, `design`, `reference`, `gtf`,
`targets`, `paper`, `annotations`, and a new `outdir`.

- `inputs`: recovered JSON records with run, pairs, two file paths and SHA-256 values.
- `samples` and `design`: frozen sample TSV and contrast JSON. The workflow checks
  the sample fingerprint and requires the same library IDs in both input populations.
- `reference` and `gtf`: the original transcript FASTA and GTF. Ambiguous transcript
  mappings and multiple transcripts per gene fail, as in the existing analysis.
- `targets`: TSV with `database`, `kind`, `mode`, `k`, `scaled`, `paths`. Kind is
  `protein` or `reads`; paths are separated by semicolons. Each row is one explicit
  target batch, with one signature per reference protein or external run. Use
  DNA31, protein7, Dayhoff12 and HP30. Swiss-Prot uses scaled5; external runs use
  scaled10 and scaled100. Only listed targets are searched; an incomplete manifest
  must not be described as the whole tunicate archive.
- `paper`: TSV with `gene_id` plus membership fields. Every listed gene survives
  the final join, including genes without measured sequence evidence.
- `annotations`: TSV keyed by `record_id`, containing available reference names,
  species, Pfam/GO labels or SRA metadata. Missing annotations remain unknown.

Target preparation writes new signature collections with protocol and checksum
records. It preserves internal X, splits translation only at stops, and retains
stretches of at least 30 amino acids. Every eligible query sketch is checked against
Sourmash's API. SBT databases are rejected. Empty query sketches are reported as
unsearchable rather than biological negatives. Protein encodings combine eligible
six-frame fragments per source DNA context. A match row names that DNA context, not
one individual peptide; fragment sequences, membership and coordinates remain saved.

Native contigs retain the tool's representative seed and adjusted P. Neither is a
new contig-level statistical test. The final gene link is exact seed containment;
full-context alignment and direct query Pfam annotation remain unmeasured.

## Execution and checks

Use pinned Nextflow 25.10.4 and the `sherlock` profile for production. Absolute pinned
Python and DE-kupl paths are resolved from `GROUP_HOME`. Diagnostics have unique
filenames. Keep the work directory to resume with `-resume` and use a distinct output
directory for this protocol. Process scripts and imported helpers are staged inputs.
The results directory retains per-library validation, selected seeds and their tests,
native contigs, recruited pairs and hit coordinates, query origins, sketch coverage,
raw comparisons and filtered nonzero-overlap matches. Keep the work directory too:
it retains the full SPLASH intermediates and supports resuming.

`smoke.sbatch` generates synthetic inputs inside a Slurm allocation and exercises the
actual tools through the full workflow. Its local executor runs inside that allocation,
not on a login node or laptop. `smoke.config` is only for this small software fixture.

Before production, measure first memory and time requests against the recovery and
fixture traces and earlier failures. Default resource requests are provisional.
Do not treat a preview, a stub or a submitted job as completed validation.

`prepare-launch.sh` writes a new resource report and resolved configuration from supplied
trace files. Pass all relevant historical and current traces. It does not submit work.
Failed tasks remain a launch blocker until their causes are reviewed. Runtime and
memory estimates from synthetic fixtures are not production measurements.

Optional `genome`/`minimap2` and `pfam_hmm`/`hmmscan` parameters enable separate context
alignment and direct peptide-domain searches. Pfam requires the HMM plus its four
pressed index files. Missing assets produce explicit `not measured` records. Raw PAF
and domain tables are retained; their interpretation is separate from exact seed links.

## Coverage of the proposed metro map

| Station | Migration state |
|---|---|
| Existing trimmed reads | Recovered and fully validated for all 64 experimental libraries |
| Native counts, pooled models and both sequence sources | Separate DSL2 processes; tested with synthetic inputs |
| Translation, target preparation and searches | One versioned protocol; explicit target batches |
| Gene/paper evidence | Waits for both sources and gene models; preserves missing evidence |
| Genome alignment and direct query Pfam | Optional processes; reference-backed execution not yet tested |
| Fresh downloads and trimming | Existing preparation remains outside this workflow |
| All-tunicate expansion and app publication | Not run; require validated production outputs |

The original proposed map is preserved at
`reports/botryllus-shazam/nextflow-plan/metro-plan.svg`.
