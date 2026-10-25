# Dataset-specific sketch scales

Olga updated these settings on 2026-10-23:

| Reference collection | Scales | Encodings |
|---|---|---|
| Public SRA RNA-seq | 10 and 100 only | DNA, protein, Dayhoff, HP |
| Swiss-Prot proteins | 5 | Protein, Dayhoff, HP |

Scaled100 is the preferred smaller archive index, but compare its coverage against scaled10 using the same runs, preprocessing, k values and queries. Report empty sketches separately from searched queries without a match. Preserve losses and ambiguous results; do not tune rules to make scaled100 appear successful. Public archive processing is trimming and sourmash only, with no SPLASH or DE-kupl. Existing completed pilot outputs at other scales remain historical measurements.

Swiss-Prot protein FASTA does not provide a DNA reference. Keep its protein retrieval separate from optional downstream alignment/domain annotation. The settings were sent to the Botryllus Shazam chat at Olga's explicit request.

The existing `archive100.py` and queued scale100 worker plan predate the two-scale public-SRA instruction. Update and validate them before the full worker rollout. They must not be described as a completed scaled10-versus-scaled100 comparison.

## Shared oyster and Botryllus extension rule

For both focal experiments, extend every anchor with adjusted P strictly less than 0.05. There is no top-N or top-300 cap. Retain every resulting extension and explicitly count significant anchors without extension output. Input depth, completion state and output counts are separate fields; none changes this rule. Historical 300-anchor pilots remain labelled as historical. Public archive runs still receive no SPLASH discovery or extension.

## Stop-free translation policy, confirmed 2026-10-24

Keep stop-free stretches from all six standard-code reading frames. Exclude individual words crossing a stop codon or ambiguous residue; do not reject a whole frame because another stretch contains a stop. Reference-proteome support is a separate annotation or optional comparison, not an eligibility filter for the primary search. Preserve unmatched sequences as unclassified and searchable. Historical stop-inclusive outputs remain separately labelled. Both translated query and reference sketches must use the same declared stop policy.
