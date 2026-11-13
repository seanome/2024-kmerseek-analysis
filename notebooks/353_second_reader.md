## 8. Second reader's check: held and failed claims

A second agent that had not seen this work read the notebook, the scripts, the frozen data
and the kmerseek source at 0a22970, recomputed every number in the summary a second way from
the committed tables, and tried to disprove each claim (2026-10-20). Its list, and what was
changed in response:

| claim | verdict | what the reader found | change made |
|---|---|---|---|
| Rule written before any result | HELD | pre-registration 14:00:12; first job 14:09:46; `notebooks/353_prereg.md` unchanged since 641e7262 | none |
| Setting frozen before the held-out half ran | HELD | freeze commit 736d3f80 at 15:45:00; last development job ended 15:43:35; held-out prep started 15:47:13; later script changes are formatting only | none |
| Verdict "not run" follows the rule | HELD | the rule says so without the 100 real queries; `stage1/` holds only empty templates | none |
| Checksums in MANIFEST.tsv | HELD | all 6 match | none |
| Split by source protein | HELD | recomputed, 0 mismatches | none |
| Compute: 17.42 CPU-hours, 2.36 h first start to last end, 11.22 h summed | HELD | recomputed from sacct.tsv; the 13:54 start is not recorded in a file | none |
| mmseqs12 k = 4 picked | HELD | 0.8268, next best mmseqs12 k = 6 at 0.8215; none excluded (highest shuffle rate 0.037 vs cut 0.285) | none |
| 5,656 held-out records; 11,376 targets | PARTLY | 5,657 held-out records, one (4 residues) gives no window; the index holds 11,192 targets (184 skipped as repeated k-mer sets) | summary corrected |
| Point 1, within-class substitutions cost nothing | HELD | 0.9668 at every within-class level (15 aa); 30 aa unchanged is 0.9836, which rounds to 0.984; "cost what they cost any exact-word search" was never measured | 0.984 used; the comparison removed and marked as not measured |
| Point 2, found = placed | HELD | 0 differences in 203,616 queries at all four cut-offs; coordinates 0-based end-exclusive checked on every example row | none |
| Point 3, 4 of 22,624 found; self-match filter | HELD | `compare()` returns None when the two `md5sum` fields are equal, but that field is a sum of k-mer hashes, not an MD5; the 4 found are one record holding a U | wording corrected; U record named |
| Point 4, identical records | PARTLY | 92 is the E ≤ 10 count, 93 at E ≤ 0.1; the collapse is by k-mer set (329 records), and 92 of the 93 come from such records | rewritten with the k-mer-set count, both cut-offs, cells added in section 3 |
| Point 5, 0 of 18 strata calibrated; ratios | PARTLY | 0 of 18 and every ratio recomputed; the "thirds" are 5,417 / 4,089 / 7,462 windows because of ties | bin sizes stated and printed |
| Point 5, dipeptide excess lands on the window's own record | FAILED | in the 300-shuffle sample, of 45 regions at E ≤ 0.1: 12 own record, 28 other records of the same protein, 5 other proteins; pooled hit share 0.099 vs 0.095 predicted | sentence corrected to "records of the window's own protein"; sample split and hit share printed in section 4 |
| Point 6, 0.37 to 0.52 other-protein regions per query | PARTLY | right as a range over substitution levels; by length 0.08-0.24 (15), 0.35-0.48 (30), 0.69-0.84 (53); many are tile-to-tile | by-length values (0.16, 0.42, 0.77 pooled over levels) added; tiles of related factors named |
| Closing: 92 to 98% found with up to one cross-class change | HELD | 0.921 to 0.984 | none |

Other problems it raised:

1. The summary pointed to a second-reader section that did not exist yet: this is it.
2. `n_chunks` was fixed by hand on Sherlock and the repo did not say so: now stated in
   section 7.
3. All ten development settings lose the 53-residue, no-cross-class cells to the self-match
   filter, which lowers every score by the same cells; the choice stands.
4. The unit tests pass.
