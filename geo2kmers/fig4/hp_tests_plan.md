# Held-out and cross-series tests on HP k-mers: plan

Written on 2026-11-08, before any held-out or cross-series AUC was computed on HP k-mers.

**Question.** Do 30-letter hydrophobic-polar (HP) k-mers from the proteins behind each probe set pass the same two
tests the 25-letter probe k-mers passed: telling a held-out person's pre samples from post (`held_out_plan.md`), and
telling pre from post in another series on the same array (`cross_series_plan.md`)?

**Tables.** The HP k-mer tables of `hp_table_plan.md`, one per array and alphabet. Each alphabet is its own test, and
both are reported; neither is picked afterwards as the better one.

- `hp_lehninger2`, kmerseek's default `hp`: hydrophobic `AFGILMPVWY`.
- `hp_thomas_dill2`: hydrophobic `ACFILMVWY`.

**Values.** The series matrix values GEO holds (the authors' RMA summary, already log2), used as they are. The probe
k-mer tests took log2(intensity + 1) of raw intensities; RMA values are already on a log scale, so they are not logged
again. A k-mer held by several probe sets has their mean, as in the tables. A k-mer with no value in some sample of a
series is dropped from that series.

**Comparisons.**

- Held out: GSE28998 (7 people) and GSE111555 with the corrected labels (16 people). GSE83578 is on the U219 array,
  which has no protein table, so it is left out.
- Cross-series: choose on GSE28998 and score GSE27536, and the reverse, both on U133 Plus 2.

**Steps.** Exactly those of `held_out_plan.md` and `cross_series_plan.md`: Welch's t-test, the 1000 k-mers with the
lowest p, standardised and averaged with their direction, 100 and 10000 reported beside 1000. Two additions:

1. K-mers held by the same probe sets have the same values in every sample, so they are tested once and counted as
   many times as there are such k-mers. This gives the same scores as testing each k-mer, and is what lets 10 million
   k-mers fit in memory. All k-mers of one probe set tie in p; ties go to the probe set listed first in the table.
   When the 1000th k-mer falls among k-mers that tie, only as many of them are used as are needed to reach 1000.
2. K-mers constant across all samples of a series are dropped, as the cross-series plan already does.

**Chance.** As before: labels swapped within each person (200 swaps; GSE28998 has all 127 other arrangements), and for
the cross-series test 200 random sets of k-mers with as many going up in training as among the chosen ones. Random
k-mers are drawn from all k-mers, each equally likely, so a probe set with many k-mers is drawn more often, as it is
among the chosen ones.

**Verdict, for each alphabet.**

- Held out: works in new people when p for the 1000 k-mers is below 0.05 against swapped labels.
- Cross-series: carries over when p is below 0.05 against both kinds of chance.

**Reported beside each result.** How many probe sets the chosen k-mers come from. A 500-residue protein gives about
470 k-mers, so 1000 k-mers can come from only a few proteins, and the test then rests on a few genes.

**What this cannot show.** An HP k-mer takes its value from the probe sets whose proteins hold it. It cannot tell two
proteins of one probe set apart, and it cannot find a change the probe sets did not measure. So this asks whether
moving from probe sequences to protein HP k-mers keeps the signal, not whether HP k-mers find new signal. The probe
k-mer tests read raw intensities and these read the authors' RMA values, so a difference between them may come from
that, not from the k-mers.

**Added after the run (2026-11-08).** Chosen on GSE27536 and scored on GSE28998, the chosen HP k-mers gave AUC 0.98
(p 0.008 against swapped labels) and random k-mers that went up in GSE27536 gave a median of 0.96 (p 0.38). Under the
verdict rule above that is a fail. Olga judged this a bad test rather than a failure of HP k-mers, and it is reported
as not informative, with the rule's verdict shown beside it. The reasons: GSE28998 compares trained and untrained arms
of 7 people, so AUCs near 1.0 leave no room between chosen and random k-mers; and the two series' training changes
agree gene by gene (Pearson r 0.27, `results/fig4/series_agreement/`), so random k-mers that went up in one series
are not a no-signal control for the other. No other verdict changed.
