# Cross-series plan

Written on 2026-11-08, before any cross-series AUC was computed.

**Question.** Do k-mers chosen on one series tell pre from post in another series on the same array?

**Series.** Only the U133 Plus 2 array (GPL570) has two exercise series here: GSE28998 (biceps, 7 people before and
after 12 weeks of resistance training) and GSE27536 (vastus lateralis, 27 people, 12 healthy and 15 with COPD, before
and after 8 weeks of endurance training). GSE111555 is the only HTA 2.0 series, so it has no partner. Both directions
are run:

- choose on GSE28998, score GSE27536;
- choose on GSE27536, score GSE28998.

GSE27536's pre and post arrays were mostly scanned on different days, so k-mers chosen on it may follow the scan day.

**Steps.**

1. Keep the k-mers present in both series' tables and not constant across the samples of either series. (The
   constant-k-mer rule was added after the first run, whose random sets in GSE28998 gave no AUC because a constant
   k-mer cannot be standardised; no other rule changed.)
2. On the training series, run KaMRaT's t-test (Welch, log2(intensity + 1)) on all its people, and keep the 1000 k-mers
   with the lowest p and the direction each moved.
3. Standardise each kept k-mer with the mean and standard deviation of all samples of the scored series. This uses no
   labels, and removes a shift of the whole series.
4. Score each sample of the scored series: the mean of its standardised k-mers, k-mers lower after exercise counted
   negative.

**Headline number.** The AUC over the scored series' samples: the chance a post sample scores above a pre sample.
The direction is fixed by the training series, so an AUC below 0.5 means the k-mers moved the other way. 1000 k-mers is
the headline; 100 and 10000 are reported beside it.

**Second number.** How many people in the scored series have a post score above their own pre score.

**Chance, two kinds.**

- Labels: swap pre and post within each person of the scored series, keeping the k-mers. 200 swaps for GSE27536; all
  127 other arrangements for GSE28998, so its lowest possible p is 1/128.
- K-mers: 200 random sets of 1000 k-mers from the training series, each with its own direction there, with as
  many going up as among the chosen k-mers. This asks whether
  any k-mers would do as well, which a shift shared by most k-mers in the scored series (a scan-day effect) would cause.

p = (1 + chance values at or above the true value) / (1 + chance values), one-sided.

**Verdict for each direction.**

- Carries over: p below 0.05 against both kinds of chance.
- Does not carry over: p of 0.05 or more against either.

**What this cannot show.** The two series differ in muscle, kind of training, its length, and in health of the people,
so a failure does not tell which of those is why. One pair on one array is one test.
