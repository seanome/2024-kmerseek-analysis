# Clam Context — runnable research prototype

A local pipeline from **SPLASH sequence events → extended DNA → candidate peptides → sourmash protein / Dayhoff / HP search**, with an offline interactive HTML report.

This is a code prototype for the discussion with Max. No clam data have been supplied or analyzed. The included demo uses synthetic sequences and fabricated score/count tables; its sourmash results are actually computed. The native integration test separately runs real SPLASH and Compactors on generated FASTQs.

## Quick start

Python 3.10+; tested here with Python 3.12 and sourmash 4.9.4.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m clam_context demo --out my-demo
```

Open `my-demo/results/report.html` in a browser. A precomputed example is included at `examples/demo/results/report.html`. The report is self-contained and does not send data to a server. Its UniProt links only open when clicked.

Switch between Protein, Dayhoff, and HP, and between scaled 10 and 100. Click a sequence event to inspect peptide coordinates, shared k-mers, alternative matches, and retained query hashes. JSON, TSV hits and peptide FASTA are generated alongside the report. No login or web deployment is needed.

## Defaults chosen for this experiment

| Encoding | k in amino acids | Approximate alphabet information | Default index scaled | Search scales |
|---|---:|---:|---:|---|
| Protein | 7 | 7 × log2(20) = 30.25 bits | 10 | 10, 100 |
| Dayhoff | 12 | 12 × log2(6) = 31.02 bits | 10 | 10, 100 |
| HP | 30 | 30 × log2(2) = 30 bits | 10 | 10, 100 |

These are nearest whole-number matches under uniform alphabet assumptions, not an entropy estimate for a real protein. Sourmash uses 64-bit hashes for all encodings. Reduced alphabets can create more accidental matches; equal approximate information does not calibrate their confidence scores.

The index retains roughly 1/10 of hashes. Searches at 100 use the corresponding subset, without a second index build. It is impossible to search at scaled 1 from an index built at 10: rebuild with `--scaled 1` first. A 60-aa fragment has 54, 49, or 31 k-mer windows respectively, before duplicates; expected retained hashes at scaled 10 are only about 5.4, 4.9, and 3.1, and tenfold fewer at 100. Actual counts are shown, including zeros.

Default reporting filters are **at least 2 shared retained hashes and query containment >=0.1**, with up to 5 hits per event per encoding/scale. These are exploratory display thresholds, NOT validated significance cutoffs. Adjust using `--min-shared` and `--min-containment`. The same shared-count cutoff at scaled 100 is stricter in underlying sequence evidence. Do not interpret comparisons as sensitivity benchmarks.

## Build a protein database

Supply a local protein FASTA, such as a downloaded Swiss-Prot release or a curated subset. This prototype does not silently download a database or claim that synthetic entries are real proteins. Preserve UniProt FASTA headers to get accession links in the report.

```bash
python -m clam_context index \
  --fasta /data/uniprot_sprot.fasta.gz \
  --out proteins.sqlite \
  --label 'Swiss-Prot RELEASE — local download' \
  --protein-k 7 --dayhoff-k 12 --hp-k 30 --scaled 10
```

One sketch per protein per encoding. The SQLite inverted index contains actual sourmash hashes, protein sequences, descriptions, and reproducibility metadata. Candidate retrieval uses retained hash postings; scores are calculated with sourmash `MinHash.contained_by`. Shared encoded words are then string-verified for inspectable evidence. It is not an alignment search or MMseqs2 wrapper.

The SQLite format is specific to this prototype, not a sourmash SBT or prebuilt Branchwater database. It avoids scanning every reference on each query, but has not been benchmarked on full Swiss-Prot or billion-protein collections. Start with a manageable protein collection. FASTA identifiers must be unique. Sequences, descriptions, hashes, k and scale settings, and source checksums are stored locally. Building a lower scaled index increases disk/build costs even if a later query uses 100.

## Run real FASTQs through SPLASH and Compactors

Install the official [SPLASH 2.11.9 Linux release](https://github.com/refresh-bio/SPLASH/releases/tag/v2.11.9) separately. Keep its executables together. GNU `time` at `/usr/bin/time` is required on Linux (`apt install time` on a conventional Debian/Ubuntu system). This project does not bundle SPLASH binaries.

Fill in `examples/samples.template.tsv`. Required columns: `sample`, `read1`. Optional: `read2`, `condition`. Relative read paths are relative to the sample sheet. Input files are FASTQ or FASTQ.gz. Read mates are pooled **within each sample** for SPLASH; no paired-read information is inferred. Input staging writes another compressed copy; budget disk space accordingly. Existing output folders must be empty.

```bash
python -m clam_context splash \
  --samples samples.tsv --splash-bin /tools/splash-2.11.9 \
  --out native-run --top 100 --threads 4

python -m clam_context analyze \
  --scores native-run/splash.after_correction.scores.tsv \
  --compactors native-run/compactors.tsv \
  --counts native-run/counts.tsv \
  --database proteins.sqlite --out clam-results \
  --scales 10 100
```

The native runner uses SPLASH's unsupervised sample-dependent variation test, selects the top corrected-p-value anchors, and invokes Compactors for those anchors. It normalizes sample/anchor/target counts for the viewer, preserves condition labels, and records commands, file checksums and runtime in `native-run/run.json`. Conditions are descriptive metadata; this runner does NOT perform a replicated treatment-effect test. SPLASH defaults and the top-anchor cap affect which events are extended. All Compactor branches/intermediate sequences are preserved and may be redundant.

SPLASH logs are in `native-run/splash.log`, `native-run/logs/`, and `native-run/compactors.log`. These are the first places to check after a failure. Native working directories must not contain whitespace due to upstream command construction. This runner targets bulk sequencing; 10x/Visium workflows are not implemented.

## Import existing outputs instead

```bash
python -m clam_context analyze \
  --scores scores.tsv --compactors compactors.tsv \
  --database proteins.sqlite --out results \
  --q-column pval_opt_corrected --sequence-column compactor \
  --effect-column effect_size_bin --scales 10 100
```

Scores must have `anchor` (nucleotide sequence) and an already multiple-testing-corrected p-value column. Raw p-values are not converted to q-values here. Compactors must have `anchor` and an extended nucleotide sequence column. For older SPLASH-plus tables, use `--sequence-column compactor_valid` when appropriate. A selected anchor with no extension remains in the report.

Optional counts TSV requires `anchor`, `sample`, `target`, `count`; `condition` is optional. Counts must be nonnegative integers. Native samples/conditions are retained in the input and `run.json`, and conditions are also included in report JSON. The plot shows target proportions, not normalized expression.

Translation considers six frames, splits at stops and ambiguous residues, and keeps fragments of at least 20 aa. `--min-aa` and `--translation-table` are configurable. Coordinates are 0-based, half-open intervals on the original extension; frame 1/2/3 is relative to the selected strand. No ORF predictor, start-codon requirement, or full-protein reconstruction is claimed. A candidate may match elsewhere in an extension; its displayed coordinates must be checked against the anchor before assigning the anchor a coding function. Genetic code 1 is the default; mitochondrial or other special codes require explicit selection.

## Interpretation boundaries

- SPLASH reports sample-dependent sequence variation, not automatically differentially expressed genes.
- Protein/Dayhoff/HP containment scores are separate measures. They are not sequence identity, alignment scores, E-values, posterior probabilities or functional confidence.
- Peptides with too few hashes are explicitly marked; absence of a hit may be sampling failure, insufficient context, divergence, or a database limitation.
- Reduced-alphabet matches can join different amino-acid strings. Highlighted words include all exact *encoded* matches; retained hash counts describe only the sampled search evidence.
- Multiple anchors, extensions or frames can match the same protein; these are not independent gene discoveries. Ties are ordered deterministically and the output is truncated per encoding/scale.
- Descriptions are copied from reference FASTA headers, not transferred as confirmed functions. No Pfam validation, GO enrichment, cross-experiment upregulation inference, automatic low-complexity masking, or calibrated false-discovery estimate for protein hits is implemented.
- The protein index is hashed for provenance on each analysis; this adds I/O. The per-encoding timing comparison is a single run with a shared, potentially warm cache. It is not a controlled performance benchmark.

## Tests

```bash
python -m pytest -q
SPLASH_BIN=/tools/splash-2.11.9 python -m pytest -q
```

Core tests cover translation coordinates on both strands, stop/ambiguity splitting, protein matches, no-upsample enforcement, database identifier conflicts, unmatched/short controls, and scaled subset consistency. Dayhoff/HP equivalence classes used for highlighting are checked exhaustively against sourmash's actual hash behavior.

The optional native integration test generates 6 FASTQs with sample-dependent variants, runs actual SPLASH and Compactors, normalizes the actual outputs, then searches a deliberately constructed positive-control protein collection. It verifies the handoff, not biological accuracy. See `VALIDATION.md` for results from this build.

## Implementation sources

- [SPLASH](https://github.com/refresh-bio/SPLASH) and bundled Compactors
- [Sourmash sketch documentation](https://sourmash.readthedocs.io/en/latest/sourmash-sketch.html)
- [Sourmash API](https://sourmash.readthedocs.io/en/latest/api.html)

The next scientific step is to run Max's real data with an identified protein database release, inspect candidate evidence with him, and calibrate the encoding/scale thresholds with suitable controls.
