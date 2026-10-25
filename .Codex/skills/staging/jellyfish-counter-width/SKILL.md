---
name: jellyfish-counter-width
description: Check native k-mer count accuracy when changing Botryllus Jellyfish counting, especially after updating counter width, hash size or the pinned Jellyfish version.
---

Compare native counts to an independent read-window count before accepting the
workflow. Include repeated canonical 31-mers around 128 and 256 occurrences and a
shared-sequence fixture with enough distinct keys to exercise the count table.

Pinned Jellyfish 2.2.10 reported 6 for a sequence observed 134 times in the Botryllus
fixture. Removing count-time filtering did not fix it. Recounting the same reads
with `-c 32` returned 134. Do not infer correctness from a successful count exit or
from reproducing the installed DE-kupl Snakefile flags.

Eight-byte output returned zeros in boundary-test jobs 49516138 and 49516219.
Diagnostic job 49516266 compared 12 combinations. Eight-byte output returned zeros
at every tested memory width; 32, 63 and 64 memory bits with seven output bytes
returned the correct 134. The migration uses 63 memory bits and seven output bytes.
Test the actual COUNT_KMERS script with `test_nextflow_counts.py`, then independently recount
the full synthetic workflow output with `verify_nextflow_fixture.py`. A wider
counter changes memory use; measure production resources again before launch.
Do not reuse affected count/model outputs after changing counter settings.

Added 2026-10-25 during PR 203. Evidence: fixture job 49515476, S11T,
`AATACATGAAGAACCACAATTGGCAGATTCG`; diagnostic job 49515995 compared default
and 32-bit counters on the same reads. All test and historical output directories
were retained. The internal Jellyfish cause was not established beyond this
reproduced dependence on counter width.
