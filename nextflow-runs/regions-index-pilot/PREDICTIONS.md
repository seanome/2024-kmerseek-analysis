# Regions-index pilot: predictions, written before any search

Written 2026-10-06 at commit 87cb6cc6342cc9fbeb83ff19e6c59eb9005b236e (branch
olgabot/regions-index-pilot). At that commit the pipeline builds the queries, truth and
indexes only. No kmerseek or MMseqs2 search had been written or run. Index sizes are in
`BUILD_REPORT.md`.

## How a result is scored

- **Call:** a kmerseek region (query interval `region_start`..`region_end`, ranked by
  `region_evalue`) or an MMseqs2 alignment (`qstart`..`qend`, ranked by its E-value).
  Never ranked by containment.
- **Correct call:** the hit entry's feature type matches a feature of that type on the
  query protein, and at least half the call lies inside that feature (landing fraction
  ≥ 0.5). Landing fraction = residues of the call inside the feature ÷ call length.
  Coverage fraction = residues of the feature inside the call ÷ feature length; reported,
  not used for the decision. A feature-name match is a secondary metric.
  For the whole-protein index, the hit entry's feature types are the features of the
  target protein that the target side of the call overlaps.
- **Threshold:** for each tool × index × kmerseek setting, calls are ranked by E-value,
  and the threshold is the loosest one at which decoy calls ÷ target calls ≤ 5%.
- **Recall:** the share of query features with at least one correct call at or below
  that threshold.
- **Feature kinds:** folded domain, short (under 60 aa), disordered, motif,
  composition-driven, as defined in `bin/build_query_truth.py` and `BUILD_REPORT.md`.

## Predictions

1. **E-values fall by about log2(n_whole / n_regions) bits.** That is arithmetic, not a
   result. Measured n: 175_144_459 and 43_379_517 target residues, 350_288_918 and
   86_759_034 with decoys. Both give a ratio of 4.04, so 2.01 bits, about a 4-fold lower
   E-value for the same score.
2. **Main test.** At 5% decoy error, kmerseek's recall on short features and on
   disordered features rises from the whole-protein index to the regions index, by more
   than MMseqs2's recall rises.
   Test: for each tool and each of the two kinds, a paired McNemar test over the
   features (found or not with the whole-protein index, found or not with the regions
   index). The prediction holds for a kind when kmerseek's change has p < 0.05 and its
   recall gain, in percentage points, is larger than MMseqs2's. It is checked for each
   kmerseek setting separately.
3. **Folded domains change little for either tool:** a recall change of at most 5
   percentage points, or McNemar p ≥ 0.05.
4. **kmerseek beats a composition-only classifier on disordered features.** The
   classifier runs no search. It cuts each query into 30-residue windows every 10
   residues, and labels each window with the regions-index entry (target or decoy)
   nearest to it by amino-acid composition (Euclidean distance between the 20
   residue frequencies). A window is a call scored by minus that distance, held to the
   same 5% decoy rule and the same correctness rule. kmerseek beats it when its recall on
   disordered features is higher, McNemar p < 0.05.

## What would sink the idea

- No change in recall at a fixed decoy error rate.
- The same gain for kmerseek and MMseqs2.
- Composition alone matching kmerseek on disordered features.

## Expectation from the arithmetic, before searching

The regions index lowers the bar by 2.01 bits. At the 0.157 bits per aligned H/P
position given in the design (assumed, not measured here), that is about 13 aligned
positions. A hit still needs about 180 aligned positions to reach E = 1 against the
regions index. So the arithmetic expects prediction 2 to fail for the H/P setting on
features under 60 aa, and leaves room for a gain only for settings with more bits per
position (protein20).

## Not yet fixed

Whether the headline uses experimental-evidence features only, as designed, or all
evidence: the experimental set holds 0 disordered features, 6 folded domains and 37
motifs (`BUILD_REPORT.md`). The choice, and the kmerseek settings, will be recorded here
in a dated line before any search runs.

**2026-10-06, Olga's decisions, before any search:**
- The headline uses all evidence. Experimental-evidence features (ECO:0000269) are
  reported as a secondary split.
- kmerseek settings, all with the extension at the alphabet's own mismatch penalty,
  X-drop = 4 × penalty, mask off, and the dark-set search filters:
  polarity4 k11 scaled 10 (C 0.54); hp_pbotc_1st_ed2 k19 scaled 1 (C 1.59);
  hp_lehninger2 k19 scaled 1 (C 1.51); protein20 k5 scaled 1 (C 0.14).
  k_max stays 19, so the regions index as built serves all four.

**2026-10-06, later the same day, still before any search:** Olga moved both H/P settings
from k = 19 to k = 21, to carry about the same bits per k-mer as protein20 k = 5
(20.9 bits). Settings now: polarity4 k11 scaled 10; hp_pbotc_1st_ed2 k21 scaled 1;
hp_lehninger2 k21 scaled 1; protein20 k5 scaled 1. k_max became 21 and the regions index
was rebuilt with 20-residue flanks. Prediction 1's numbers change to: 175_144_459 against
45_736_962 target residues (350_288_918 against 91_473_924 with decoys), a ratio of 3.83,
so 1.94 bits. The expectation above changes to 1.94 bits, about 12 aligned H/P positions
at the assumed 0.157 bits each; a hit still needs about 180 positions against the regions
index. Everything else above is unchanged.

**2026-10-07, still before any search:** polarity4 moves to scaled 1, so all four
settings are at scaled 1. The decoys stay shuffled within 10-residue windows, and every
result is repeated with decoys shuffled within 20-residue windows as a check. The 10-residue
result is the headline; the check passes when the 5% thresholds and the recall of each
feature kind stay within 5 percentage points between the two window sizes.

**2026-10-07, a second landing rule, added after the 20-query local test.** The test had
scored MMseqs2 and the composition classifier only; no kmerseek result had been seen. On
the whole-protein index a long MMseqs2 alignment spans several domains, lands under half in
any one query feature, and counts as wrong under the rule above, while on the regions
index the same alignment is confined to one feature by the entry's length. On 20 queries
that rule gave MMseqs2 21 of 100 features with the whole-protein index and 35 with the
regions index. A second rule cuts each call down to the query residues aligned to the
target feature's own residues and judges landing on that piece: 62 and 48. The sign of the
index difference depends on the rule, so both are reported for every result. The rule
above stays the one these predictions were written for; the choice of headline rule is
Olga's and will be recorded here before the full run's results are read.

**2026-10-13, Olga's decisions, before the full run:**
- The headline landing rule is the aligned piece. The whole-call rule is reported next to
  it.
- Prediction 4 compares kmerseek with the composition classifier at the same number of
  calls: the classifier gets as many target calls as kmerseek makes at its 5% threshold on
  the regions index (same decoy window), its best-scoring first. A window-shuffled decoy
  has the same composition as its source, so the classifier cannot be held to the decoy
  rule itself.
- Two more readings, both reported:
  1. One Karlin-Altschul fit, from the whole-protein index, used for both indexes, so the
     indexes differ only in residue count n. With this, prediction 1 holds by construction.
     It needs kmerseek to search with another index's fitted slope as well as its K
     (open: kmerseek v0.4.0 takes only --ka-k), so it is not in this run yet.
  2. The whole-protein index with only the calls whose target side lies inside an
     annotated feature (widened by the regions index's flank): same index, same E-values,
     only annotated targets. Compared with both indexes it splits the regions index's
     effect into "only annotated targets" and "smaller index". In the tables its index is
     "wholeannot".

**2026-10-13, later, Olga's decisions, before the full run:**
- Every alphabet with a measured kappa (19), each at the k that carries about the 20.9 bits
  of a protein20 5-mer (bits per letter from real amino-acid frequencies), kept inside the
  alphabet's tested k range (notebook 274): hp_lehninger2, hp_lehninger_c_nonpolar2 and
  hp_pbotc_1st_ed2 at 21; hp_thomas_dill2, hp_thomas_dill_no_c2 and hp_kyte_doolittle2 at
  22; hp_lehninger_hpc3 19; gbmr4 14; wwmj5 10; dayhoff6 9; funcgroups8 8; sdm12 7;
  mmseqs12, wass14 and hsdm17 6; uniprot18 and protein20 5. Two exceptions: polarity4 at
  11, to match the dark set (its bits-matched k is 12), and gbmr7 at 16 (its bits-matched k
  is below its tested floor of 13, and at 13 the dark set's searches needed 379-500 GB
  without decoys). All at scaled 1, extension at the alphabet's own mismatch penalty, mask
  off.
- k_max becomes 22, so the regions index is rebuilt with 21-residue flanks.
- The shared-fit reading runs with kmerseek 0.4.1-rc1: v0.4.0 plus `search
  --ka-r-database` (seanome/kmerseek PR 144, ported onto v0.4.0 on branch
  olgabot/v0.4.1-ka-r-database). Every setting runs on that image. Its index label is
  "regionsshared".
