# PEC reproduction handoff (built 2026-10-23 in a 2-CPU cloud container; nothing here has run on real reads yet)

- samples_all64.tsv — all 64 PRJNA1219309 libraries with ENA URLs/md5 and the design INFERRED from sample aliases
  (treatment, genotype, age, clock time, timepoint_inferred, usable). Check against SI Dataset S1 before trusting timepoints.
- ena_runs.tsv — raw ENA filereport.
- download_prefix.py — capped prefix downloader (N_PAIRS env var). On Sherlock download FULL files instead (see prompt).
- run_contrast.sh — fastp -> SPLASH 2.11.0 (predefined Cjs, correction on Cjs_0_pval) -> compactors -> Rust translate -> sourmash multisearch.
  Paths are container paths; adapt ENV, proteome and SwissProt locations.
- compactors_config.json — {"num_threads": 2, "reads_buffer_gb": 3}; raise on Sherlock.
- rust/ — pec-tools (translate: six-frame stop-free peptides with coordinates, oracle-checked vs Biopython on 4,500 peptides;
  simulate: paired-read simulator for the known-answer control). cargo test passes (4 tests).
- control/ — known-answer control design: 3,000 botznik transcripts; 150 up x4, 150 down x0.25, 100 isoform switches
  (120-nt internal deletion, 90/10 -> 10/90), 2,600 unchanged; 6 control + 6 PEC samples; seed 11; 1M pairs each.
- sketch_swissprot.sh — manysketch command for SwissProt (NCBI blast/db/FASTA/swissprot.gz, 2024-02-06, md5 1f885c2bba9d35231f920b71f7fa3b36, 482,697 entries).
