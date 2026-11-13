---
name: polars-gzip-streaming
description: Use when streaming Polars CSV batches into gzip, or when a gzip file written by Polars fails decompression.
---

# Stream CSV batches into gzip

Observed on 2026-10-23 while repairing notebook 360 landings. Passing a gzip text
handle directly to Polars DataFrame.write_csv produced a corrupt gzip stream in the
project environment. The 50,003-row regression test reproduced the decompression error.

1. Keep each DataFrame bounded; the notebook 360 scorer uses 25,000 input rows.
2. Serialize one batch with `frame.write_csv(separator="\t", include_header=first)`.
3. Pass that returned text to the gzip handle's `.write()` method. This keeps compression
   in Python's gzip wrapper instead of exposing its underlying file descriptor to Polars.
4. Reopen the completed gzip with `gzip.open(..., "rt")`, parse all rows and check the
   row count. A successful writer return alone does not prove the archive is valid.
5. Test at least two batches, including header handling, and empty input.

Reference regression: `scripts/nb360/test_landings_streaming.py`.
