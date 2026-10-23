# Validation and execution limits

- Independent supplied-pilot audit: all 45,339 original protein sequence hashes,
  1,632 translation-coordinate occurrences, and 3,984 real Botryllus match rows.
- App packaging rechecks protein hashes, 987 peptide records, 339 DNA records,
  and the audited real-match row count before writing browser data.
- Original HTML report and original uploaded protein FASTA bytes preserved.
- Strict adjusted-P selection tests: 0.049999 and zero included; 0.05 excluded;
  nonfinite values and duplicate anchors rejected. Complete pilot selection has
  13,669 records, independently counted from its TSV lines.
- Interrupted-transfer tests: curl 18 and 28/56 recover, retry exhaustion fails,
  nonretryable curl 22 fails. No success marker is emitted on those failures.
  Recovery array task 3 also completed on the real preserved partial inputs.
- Incomplete full-depth population rejected before discovery output is created.
- Black, Python compilation, JavaScript syntax and all Slurm Bash syntax checked.
- Every generated downloadable asset fingerprint and HTTP response checked locally.
- Local browser: peptide ID lookup, DNA membership coordinates, protein ID lookup,
  original protein hash display, empty search result, and separate file-type table.
- Metro SVGs rendered with nf-metro 2.1.0 strict geometry checks. The full canvas is
  retained after moving its legend above the geometry; SVG XML parsing checked.
- Representative full-run Sourmash indexing completed on Sherlock: four encodings,
  three historical scales, signatures, real SBT indexes, and nonempty self-searches.
- New bulk protein pilot completed on Sherlock, but its raw matches still require
  independent Sourmash/hash validation. Completion is not biological validation.
- Full-depth discovery and abundance results remain unmeasured. The submitted
  dependent full-depth job has not yet provided a biological result.
- Hosted deployment succeeded; owner-private access requires the user's sign-in.
  Local browser QA does not establish that the hosted sign-in was completed.

- Swiss-Prot size comparison: all nine ZIP manifests independently contain 575,748
  signatures with the expected alphabet and scaled value; empty-reference counts
  are retained separately. Original FASTA and gzip sizes are distinct measurements.
- New overview rendered in Seqera style with strict nf-metro geometry checks; the
  final Results heading is moved above its box after rendering and visually checked.
- Extension EXT000331 independently joins to 8 distinct peptides and 60 real retained
  match rows; browser displays those counts and 247 nt. Rows span settings/databases,
  not independent proteins. Compactors support is shown with its source field name.
- Storage benchmark 49212645 remains in progress; merged reference sizes and new
  RocksDB correctness are not yet claimed as measured results.
