# Test data

Test series, chosen by searching GEO's platform records with NCBI E-utilities and opening them. Read on
2026-11-03.

| field | value |
|---|---|
| series | GSE2183, "Gene expression profiling of mussels treated with two chemical mixtures" |
| samples | 4: GSM39421, GSM39422, GSM39423, GSM39424 |
| platform | GPL1799, Mussel MytArray 1.0, spotted DNA/cDNA |
| organism | Mytilus galloprovincialis, a marine mussel |
| probes | 3840 rows in the platform table, of which 3374 carry a GenBank accession |
| columns | GB_ACC and no SEQUENCE column, so sequences must come from GenBank |
| channels | 2 per sample. ch1 is control digestive gland, ch2 treated digestive gland |

`data/GSE2183/` holds the platform table, the series record, GSM39421's table, the probe sequences the
tool fetched, `probe_sequence_source.tsv` and `provenance.json`. The last lists a checksum for each of
the 7 downloaded files. The other 4 files (the matrix and three GSM tables) are not kept here, and
`provenance.json` gives their URLs.

`data/known_answer_probes.fasta` holds 50 of those fetched sequences. Each is 300 nt or longer, has only
A, C, G and T, and differs from the others.

## Known-answer checks

- `test_sketch_holds_exactly_the_kmers_of_the_20_bright_probes`: no real sample has exactly 20 probes
  above threshold, so the test writes a sample table by hand. The first 20 probes have intensity 5000
  and the other 30 have 100, so the median is 100 and only the 20 pass. The sequences are real. The test
  cuts the 20 sequences into k-mers with its own small loop and requires the sketch to hold exactly
  that set.
- `test_weight_sum_equals_probe_weight_times_kmers_on_a_real_array`: for GSM39421 ch2, the sum of
  `intensity_weight` read from the written TSV must equal the sum over probes of probe weight times
  number of k-mers. Both the weights and the k-mer counts are recomputed in the test from the raw GEO
  table and the FASTA file, so double counting in the code cannot cancel out.
