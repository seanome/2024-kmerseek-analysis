# Held-out AUC plan

Written on 2026-11-06, before any held-out AUC was computed. KaMRaT's counts are already known; these numbers are not.

**Question.** If k-mers are chosen without one person, can they tell that person's pre samples from post?

**Comparisons.** GSE83578 pre vs 30 min (8 people), GSE28998 (7 people) and GSE111555 with the corrected labels
(16 people). These are the three where pre and post arrays share scan days.

**Steps, repeated once per person.**

1. Set aside every sample from that person.
2. Run KaMRaT's t-test on the other people: Welch's two-sided test on log2(intensity + 1).
3. Keep the 1000 k-mers with the lowest p, whatever their false discovery rate. Seven people may leave none below 5 %.
4. Standardise each k-mer with the mean and standard deviation of the other people only.
5. Score each set-aside sample: the mean of its standardised k-mers, with k-mers lower after exercise counted negative.

**Headline number.** One AUC over all the set-aside scores: the chance a post sample scores above a pre sample.
1000 k-mers is the headline. 100 and 10000 are reported beside it and do not change the verdict.

**Second number.** The share of people whose post score is above their own pre score, which does not need scores
from different people to be comparable.

**Chance.** Swap pre and post labels within each person and repeat every step, including choosing the k-mers. Use 200
swaps. Where fewer than 200 arrangements exist, as with GSE28998's 128, use every one. An arrangement and its exact
opposite give the same AUC, because the k-mers' direction is learned, so two of GSE28998's arrangements always tie
and its lowest possible p is 2/128. p = (1 + swaps at or above the true value) / (1 + swaps).

**Verdict for each comparison.**

- Works in new people: p below 0.05 for the headline AUC.
- Does not work in new people: p of 0.05 or more.
- As good as the published signature: also within 0.15 of the published AUC's distance from no separation, written
  as the larger of AUC and 1 - AUC, because these k-mers learn their direction. That is 0.93 for GSE83578 30 min, 0.80
  for GSE28998 and 0.809 for GSE111555, the corrected-label value.

**Contrast.** The same AUC with k-mers chosen from everyone, to show how much choosing them on the same people adds.

**What this cannot show.** It says the k-mers carry over within one series. It does not test another series, and a
batch difference shared by most people would carry over too.
