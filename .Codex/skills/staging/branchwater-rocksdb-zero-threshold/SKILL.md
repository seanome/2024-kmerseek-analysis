---
name: branchwater-rocksdb-zero-threshold
description: Use when a Branchwater 0.9.15 RocksDB manysearch is intended to return every nonzero shared-hash pair, or when scaled100 unexpectedly returns pairs absent at scaled10.
---

# Explicit threshold for all shared-hash comparisons

Recorded2026-10-23 after the oyster public-search audit found eight HP pairs present at scaled100 but absent at10.

1. Preserve the original query sketches, index, command, output tables and hashes.
2. Use the same failing query and same RocksDB index at both scales. Compare default threshold with explicit `-t 0`, retaining `-A -N --ignore-abundance` and the same k and alphabet. Run on Sherlock compute nodes; follow its agent and scheduler rules.
3. Do not assume `-A` removes the index's threshold. In the reproduced Branchwater0.9.15 case, scaled10 default omitted a one-hash pair with containment0.004830917874396135. Explicit `-t 0` recovered it. Both commands returned the one-hash pair at scaled100, containment0.038461538461538464. See `analysis/oyster-shazam/threshold-reproduction.json` and `check_threshold.py`.
4. When the frozen rule is any nonzero shared hash, specify `-t 0 -A`. This implements that rule; do not change a study's intentional positive containment cutoff.
5. Rerun into separate output directories. Never overwrite the original tables or relabel them as corrected. Check actual intersection hashes are positive, query/run IDs are unique, and scaled100's pairs are a subset of scaled10's under otherwise identical nested sketches.
6. Independently check input/result hashes, empty-sketch counts, reference denominators, exact lost pair IDs, and self containment1.0 before releasing dependent archive workers.
7. Until verified, report net differences and the audit discrepancy. Do not call the net difference a complete loss count or infer biological absence from missing/empty sketches.

This is a version-specific observed application issue, not a scheduler or filesystem failure. Preserve the reproduction before considering another version.
