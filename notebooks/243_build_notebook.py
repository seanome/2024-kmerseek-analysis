#!/usr/bin/env python3
"""Write notebooks/243_alphabet_logistic_regression.ipynb; execute it with nbconvert.
Numbers in figure conclusions are computed from the tables in the notebook."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "243_alphabet_logistic_regression.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {
    "cell_type": "code",
    "metadata": {"jupyter": {"source_hidden": True}},
    "execution_count": None,
    "outputs": [],
    "source": s.strip("\n"),
}

cells = [
    md(r"""
# 243: Does a learned combination of alphabets find domains better than one alphabet?

Notebook 242 combined alphabets by hand (rank products, then the best subset for BCL2) and
found nothing that held up: what helped BCL2 helped the control just as much, and the best
subset was picked on the answer. The principled version learns the combination from labels
that are not BCL2 or CD47.

**Data.** The 998 human proteins with a Pfam domain in the midi-plus truth set, searched
all against all at every alphabet-ksize pair of the notebook 241 sweep (one alphabet at one
seed length k, 16 to 44 bits per seed; `243_pfam998_search.py`).

**Unit: a matched region, pooled across alphabet-ksize pairs.** Every region between the same query and
target that lies on the same diagonal (within 10 residues) and overlaps on the query side
is one candidate, whichever alphabets and k found it. kmerseek regions are ungapped, so the
same aligned stretch lies on the same diagonal in every alphabet.

**Label.** Correct when the candidate's query span and target span each overlap a domain of
the same Pfam family by at least 20% of the span (PR #44's rule).

**Split.** The truth set's own split by Pfam family: the model is fitted on the "selection"
families and scored on the "heldout" families, which it never sees. The Bcl-2 family is
carried by one protein in the set (BAK1, held out), so no Bcl-2 pair is in training and
Ced9 → BCL2 is a test the model was not fitted to.

**Features**, chosen to mean the same in a 998-protein and a 19_732-protein database:
for each of the 19 alphabets, whether it found the candidate at any k, the highest seed
information (bits) at which it still did, the best mean IDF divided by ln(number of
proteins in the database), the best bit score, and the most shared k-mers; plus the
candidate's length, both proteins' lengths, the number of alphabets and of alphabet-ksize
pairs that found it,
and the membrane-helix-like hydrophobic windows in each span (notebook 242 found that
proteins with many such windows rank high for unrelated queries).

**The test.** Three nested models: length and hydrophobic make-up only; that plus
hp_lehninger2's own values; that plus all 19 alphabets. Combining alphabets helps only if
the third beats the second on families the model has not seen.
"""),
    code(r"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path.cwd()))
import alphabet_logreg_utils as lr

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(60)
FIG = lr.FIG

import json
plan = json.loads((lr.SWEEP / "plan.json").read_text())
done = {f.stem for f in (lr.PF998 / "regions").glob("*.parquet")}
missing = [p for p in plan if f"{p['alphabet']}.k{p['ksize']}" not in done]
print(f"{len(done)} of {len(plan)} alphabet-ksize pairs searched on the 998 proteins"
      + (f"; missing {len(missing)}, from {min(p['bits'] for p in missing):.1f} to {max(p['bits'] for p in missing):.1f} bits"
         if missing else ""))

# Every arm must have been searched the way the same arm was searched against the human
# proteome (extended or exact); otherwise a feature means different things in training and
# when the model is applied. The human setting is read from arms.csv: `fitted` is true
# exactly for the arms whose human regions carry mismatches, i.e. were extended (checked
# 2026-10-17 on all 119 arms with regions). The marker files the old check read
# (search/*.nofit, ka_survival/) were deleted with the 241 indexes.
human_extended = {
    f"{a}.k{k}": bool(x)
    for a, k, x in pl.read_csv(lr.SWEEP / "arms.csv").select("alphabet", "ksize", "fitted").iter_rows()
}
# Known, not yet fixed: a stale .nofit marker sent this arm's 998-protein search down the
# exact path in September, while its human search was extended. Its features in training
# (exact) and in section 3 (extended) differ; re-searching it needs Sherlock.
KNOWN_MISMATCH = {"hp_lehninger_hpc3.k21"}
bad, known = [], []
for f in sorted((lr.PF998 / "regions").glob("*.parquet")):
    d = pl.read_parquet(f, columns=None).head(1)
    if "extended" not in d.columns:
        bad.append((f.stem, "searched before the settings were matched"))
    elif d.height and bool(d["extended"][0]) != human_extended[f.stem]:
        (known if f.stem in KNOWN_MISMATCH else bad).append(
            (f.stem, "extended here" if d["extended"][0] else "exact here, extended in the human search"))
assert not bad, f"alphabet-ksize pairs searched differently from the human search, rerun them: {bad}"
print(f"{len(human_extended) - len(known)} of {len(human_extended)} alphabet-ksize pairs match their human search setting (extended or exact)")
print(f"known mismatch, kept in the model: {known}")

T = lr.training_table()
counts = pl.read_csv(lr.PF998 / "candidate_counts.csv", schema_overrides={"split": pl.Utf8})
print("every candidate, by split (null = touches families of both splits, left out):")
print(counts.sort("split", "correct"))
print("\nkept for the model (every correct one; 50 incorrect per correct one in training; 3 M incorrect held-out,")
print("each kept incorrect candidate weighted by how many it stands for):")
print(T.group_by("split", "correct").agg(pl.len().alias("kept"), pl.col("weight").sum().round(0).alias("stands_for")).sort("split", "correct"))
train = T.filter(pl.col("split") == "selection")
test = T.filter(pl.col("split") == "heldout")
"""),
    md(r"""
## 1. On families it has never seen, does combining alphabets help?

Precision-recall on the held-out families for the three nested models, with three single
features beside them. Average precision is the area under each curve; chance is the share
of candidates that are correct. The held-out incorrect candidates are a random 3 M, each
weighted by how many it stands for, so the curves estimate those over every candidate.

The regularisation strength C is picked by cross-validation on the training families from
1e-4 to 10, extended down while the smallest value is still the best. The interval on each
average precision comes from resampling the held-out families: candidates are grouped by
the Pfam family group of the query protein (proteins that share a family are one group),
the groups are drawn with replacement 1_000 times, and the 2.5th and 97.5th percentiles
are shown. Candidates from one family are not independent, so this interval is wider, and
more honest, than one from resampling single candidates.
"""),
    code(r"""
models, infos, curves, heldout_scores = {}, {}, {}, {}
y = test["correct"].to_numpy()
w = test["weight"].to_numpy()
for name, feats in lr.FEATURE_SETS.items():
    m, info = lr.fit_model(train, feats)
    models[name], infos[name] = m, info
    p = m.predict_proba(test.select(feats).to_numpy())[:, 1]
    heldout_scores[name] = p
    prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
    curves[name] = (rec, prec, average_precision_score(y, p, sample_weight=w))
for name, col in [("number of alphabets that find it", "n_alphabets"),
                  ("hp_lehninger2 bit score", "hp_lehninger2__bit_score"),
                  ("candidate length", "log_span_length")]:
    s = test[col].to_numpy()
    prec, rec, _ = precision_recall_curve(y, s, sample_weight=w)
    curves[name] = (rec, prec, average_precision_score(y, s, sample_weight=w))
def recall_at(rec, prec, target):
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0
ap = pl.DataFrame({"ranker": list(curves),
                   "heldout_average_precision": [round(v[2], 4) for v in curves.values()],
                   "recall_at_precision_0.5": [round(recall_at(v[0], v[1], 0.5), 4) for v in curves.values()],
                   "recall_at_precision_0.1": [round(recall_at(v[0], v[1], 0.1), 4) for v in curves.values()]})
full, one, base = (curves[k][2] for k in ["+ all 19 alphabets", "+ hp_lehninger2's own values", "length and hydrophobic make-up only"])

# Interval: resample held-out Pfam family groups (family_groups of the query protein).
fg = lr.family_groups(pl.read_parquet(lr.TRUTH))
groups = np.array([fg.get(q, -1) for q in test["q"]])
boot = lr.bootstrap_ap(y, w, heldout_scores, groups, n_boot=1000, seed=0)
assert not any(np.isnan(v).any() for v in boot.values()), "a resample gave no average precision"
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}
ratio = boot["+ all 19 alphabets"] / boot["+ hp_lehninger2's own values"]
diff = boot["+ all 19 alphabets"] - boot["+ hp_lehninger2's own values"]
print(f"held-out family groups: {len(np.unique(groups))}, of which {len(np.unique(groups[y]))} have a correct candidate; "
      f"the largest holds {np.bincount(np.unique(groups[y], return_inverse=True)[1]).max() / y.sum():.0%} of the correct ones")
interval = pl.DataFrame({
    "model": list(boot),
    "average_precision": [round(curves[k][2], 4) for k in boot],
    "low_2.5pct": [round(ci[k][0], 4) for k in boot],
    "high_97.5pct": [round(ci[k][1], 4) for k in boot],
})
print(interval)
print(f"all 19 / hp_lehninger2 only: {full / one:.2f}, 95% interval {np.percentile(ratio, 2.5):.2f} to {np.percentile(ratio, 97.5):.2f}; "
      f"difference {full - one:+.4f}, interval {np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f}; "
      f"share of resamples where all 19 is not better: {(diff <= 0).mean():.3f}")
lr.fig_heldout(
    curves, float((w * y).sum() / w.sum()), FIG / "243_heldout_precision_recall.png", intervals=ci,
    hypothesis="A model that sees all 19 alphabets finds correct domain matches on unseen Pfam families better than one that sees a single alphabet.",
    conclusion=(f"Average precision on held-out families: all 19 alphabets {full:.3f}, hp_lehninger2 alone {one:.3f}, "
                f"length and hydrophobic make-up alone {base:.3f}; chance {(w * y).sum() / w.sum():.4f}. "
                + ("Adding the other 18 alphabets raises it by a factor of " + f"{full / one:.1f} "
                   f"(95% interval over resampled held-out families {np.percentile(ratio, 2.5):.1f} to {np.percentile(ratio, 97.5):.1f})."
                   if full > one else "Adding the other 18 alphabets does not raise it.")
                + f" At half the kept candidates correct, the full model finds {recall_at(*curves['+ all 19 alphabets'][:2], 0.5):.1%} of the correct ones."),
)
print(ap)
print("\nC picked by cross-validation (mean average precision over the 5 folds at each C tried):")
with pl.Config(tbl_cols=-1, tbl_width_chars=400):
  print(pl.DataFrame([dict(model=k, C=v["C"], smallest_C_tried=min(v["cv_ap"]), n_pos=v["n_pos"], n_neg=v["n_neg"],
                         **{f"cv_ap_C={c:g}": round(a, 4) for c, a in sorted(v["cv_ap"].items())}) for k, v in infos.items()],
                   infer_schema_length=None).select(pl.all().exclude("n_pos", "n_neg"))
        .select(["model", "C", "smallest_C_tried"] + [f"cv_ap_C={c:g}" for c in sorted({c for v in infos.values() for c in v["cv_ap"]})]))
print({k: {"n_pos": v["n_pos"], "n_neg": v["n_neg"]} for k, v in infos.items()})
"""),
    md(r"""
## 2. What the full model weighs

Standardised coefficients: how much one standard deviation of each feature moves the
log-odds that a candidate is correct, with every other feature held fixed. Alphabets
overlap heavily, so a single coefficient is not a clean measure of one alphabet's worth;
section 1's nested comparison is.
"""),
    code(r"""
m_full = models["+ all 19 alphabets"]
cf = dict(zip(lr.FEATURES, m_full.named_steps["logisticregression"].coef_[0]))
order = sorted(cf, key=cf.get, reverse=True)
top = pl.DataFrame({"feature": [lr.feature_label(f) for f in order[:8]], "coefficient": [round(float(cf[f]), 3) for f in order[:8]]})
bottom = pl.DataFrame({"feature": [lr.feature_label(f) for f in order[::-1][:5]], "coefficient": [round(float(cf[f]), 3) for f in order[::-1][:5]]})
coef = lr.fig_coefficients(
    m_full, lr.FEATURES, FIG / "243_model_coefficients.png",
    hypothesis="The model leans on a few alphabets, not on length.",
    conclusion=("The two largest weights are on length: " + ", ".join(f"{f} {c:+.2f}" for f, c in bottom.rows()[:2])
                + ". The largest positive weights: " + ", ".join(f"{f} {c:+.2f}" for f, c in top.rows()[:3]) + "."),
)
print(top); print(bottom)
"""),
    code(r"""
# Numbers the conclusions quote that the tables above do not show: precision once 10% of the
# held-out correct candidates are found (the best precision at recall >= 0.1, read off the
# curves of section 1), against chance; and the weights of the two membrane-helix-like
# window features in the full model.
chance = float((w * y).sum() / w.sum())
print(f"chance: {chance:.5f} of held-out candidates are correct (weighted)")
print(pl.DataFrame({
    "ranker": list(curves),
    "precision_at_10pct_recall": [round(float(v[1][v[0] >= 0.10].max()), 4) for v in curves.values()],
}).with_columns((pl.col("precision_at_10pct_recall") / chance).round(1).alias("times_chance")))
print(pl.DataFrame({
    "feature": [lr.feature_label(f) for f in ("hydrophobic_windows_query", "hydrophobic_windows_target")],
    "coefficient": [round(float(cf[f]), 3) for f in ("hydrophobic_windows_query", "hydrophobic_windows_target")],
}))
"""),
    md(r"""
## 3. Applied to the three test cases

The model is applied to notebook 241's searches of Ced9, P66 and BHF against the 19_732
human proteins, pooled into candidates in the same way. A target's score is its best
candidate's probability. The single features are ranked the same way for comparison.
"""),
    code(r"""
full_model = models["+ all 19 alphabets"]
rows, tops = [], {}
for q in ["Ced9", "P66", "BHF"]:
    Q = lr.query_table(q)
    ts = lr.target_scores(Q, full_model.predict_proba(Q.select(lr.FEATURES).to_numpy())[:, 1])
    tops[q] = (ts.with_columns(pl.len().over("prob").alias("n_tied")).sort("prob", descending=True).head(10)
                 .select("gene", "prob", "n_tied", "n_alphabets", pl.col("hydrophobic_windows_target").exp().sub(1).round(0).cast(pl.Int64)
                         .alias("membrane_helix_like_windows_in_match")))
    for partner in ["BCL2", "CD47"]:
        idx = ts["gene"].to_list().index(partner) if partner in ts["gene"].to_list() else None
        if idx is None:
            continue
        for ranker, col in [("model", "prob"), ("number of alphabets", "n_alphabets"),
                            ("hp_lehninger2 bit score", "hp_lehninger2__bit_score"), ("candidate length", "log_span_length")]:
            v = ts[col].to_numpy()
            rows.append(dict(query=q, partner=partner, ranker=ranker, rank=lr.midrank(v, idx),
                             n_tied=int((v == v[idx]).sum()), n_targets=ts.height, partner_value=float(v[idx])))
app = pl.DataFrame(rows)
b = app.filter((pl.col("query") == "Ced9") & (pl.col("partner") == "BCL2") & (pl.col("ranker") == "model"))["rank"][0]
c = app.filter((pl.col("query") == "P66") & (pl.col("partner") == "CD47") & (pl.col("ranker") == "model"))["rank"][0]
k = app.filter((pl.col("query") == "Ced9") & (pl.col("partner") == "CD47") & (pl.col("ranker") == "model"))["rank"][0]
lr.fig_application(
    app, FIG / "243_model_on_three_cases.png",
    hypothesis="The trained model puts BCL2 near the top for Ced9 without having been fitted to it.",
    conclusion=(f"The model ranks BCL2 {b:_.0f} for Ced9 and the control, CD47 for Ced9, {k:_.0f}: it does not put the "
                f"known homolog above the control. CD47 for P66 is {c:_.0f}. The best single alphabet-ksize pair of notebook 241 put BCL2 at 213."),
)
print(app)
for q, t in tops.items():
    print(f"\n{q}: the 10 targets the model ranks highest"); print(t)
"""),
    md(r"""
## 4. Conclusions

1. **On Pfam families the model never saw, adding the other 18 alphabets may help, but the
   gain is within the spread between families.** Average precision is 0.048 with all 19
   alphabets (95% interval over resampled held-out families 0.029 to 0.072), 0.029 with
   hp_lehninger2 alone (0.008 to 0.056) and 0.011 with length and hydrophobic make-up alone
   (0.001 to 0.036); chance is 0.0009. The ratio of the first two is 1.66, with an interval
   of 0.74 to 4.26, and in 8.7% of the resamples the 19-alphabet model is not better. The
   held-out correct matches sit in only 51 family groups, and one group holds 31% of them,
   so the comparison rests on few independent families.
2. **The regularisation strength barely matters.** With the C grid extended down to 1e-5,
   the full model picks C = 1e-4 (cross-validated average precision 0.063, against 0.061
   at C = 0.01 to 10 and at 1e-5); the hp_lehninger2 model picks 0.01 and the length-only
   model 0.001, both inside the grid. The earlier choice of C = 0.01, the old grid's edge,
   did not hide a better model.
3. **The gain is largest at the top of the list.** With half the kept matches correct,
   the full model finds 3.8% of the correct matches and the hp_lehninger2 model 2.2%. Once
   10% of the correct matches are found, the full model's precision is 0.048: 51 times
   chance (0.00093), but about 20 of every 21 matches it keeps are wrong. At that point the
   hp_lehninger2 model (0.022) does no better than length and hydrophobic make-up alone
   (0.024), and the hp_lehninger2 bit score alone is at chance (0.0009).
4. **The largest weights are on protein length, not on any alphabet.** The query's and
   the target's length weigh -0.44 and -0.43 per standard deviation: a match between two
   short proteins is more often a shared Pfam family. The largest positive weights are
   sdm12's bit score (+0.19), the length of the matched stretch (+0.17) and mmseqs12's
   bit score (+0.14). The bit score is the feature computed with the per-region lambda of
   kmerseek PR 101, which was closed unmerged; it is 0 for most sdm12 regions, so these
   weights wait on the re-run with kmerseek main.
5. **Applied to the three cases, the model does not separate BCL2 from the controls.** It
   ranks BCL2 4_258 for Ced9, and ranks CD47, which has no known link to Ced9, higher, at
   1_891. It ranks CD47 2_139 for P66. BHF, a query with no known link to either, gets
   BCL2 at 1_492 and CD47 at 2_943, so the model ranks BCL2 higher for BHF than for Ced9.
   The four highest-ranked proteins for Ced9 are olfactory receptors (OR4F3, OR4F16,
   OR4F21, OR4F29), tied at the same probability and each matched over a stretch with 8
   membrane-helix-like windows. The membrane pull that notebook 242 found is still there,
   although the windows are a feature (their weights are +0.053 on the query side and
   +0.044 on the target side).
6. **Together with notebook 242.** A random human protein of Ced9's length ranks BCL2 as
   well as Ced9 does in about one case in five, even with the best alphabets and k picked
   for each query. The trained model does no better. Neither of the two ways of combining
   alphabets tested here lifts BCL2 above what random queries give.
7. **One alphabet-ksize pair was searched differently in training.** hp_lehninger_hpc3 at
   k=21 was searched without extension on the 998 proteins but with extension against the
   human proteome, so its features mean different things in sections 1-2 and in section 3
   (the check in the first code cell names it). It is one of 152 pairs and stays in the
   model until it is re-searched.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
