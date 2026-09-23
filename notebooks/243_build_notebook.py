#!/usr/bin/env python3
"""Write notebooks/243_alphabet_logistic_regression.ipynb; execute it with nbconvert.
Numbers in figure conclusions are computed from the tables in the notebook."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "243_alphabet_logistic_regression.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "execution_count": None, "outputs": [], "source": s.strip("\n")}

cells = [
md(r"""
# 243: Does a learned combination of alphabets find domains better than one alphabet?

Notebook 242 combined alphabets by hand (rank products, then the best subset for BCL2) and
found nothing that held up: what helped BCL2 helped the control just as much, and the best
subset was picked on the answer. The principled version learns the combination from labels
that are not BCL2 or CD47.

**Data.** The 998 human proteins with a Pfam domain in the midi-plus truth set, searched
all against all with every arm of the notebook 241 sweep (one alphabet at one k, 16 to 44
bits per seed; `243_pfam998_search.py`).

**Unit: a matched region, pooled across arms.** Every region between the same query and
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
for each of the 19 alphabets, whether any of its arms found the candidate, the highest seed
information (bits) at which it still did, the best mean IDF divided by ln(number of
proteins in the database), the best bit score, and the most shared k-mers; plus the
candidate's length, both proteins' lengths, the number of alphabets and arms that found it,
and the membrane-helix-like hydrophobic windows in each span (the sticky-target effect of
notebook 242).

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
print(f"{len(done)} of {len(plan)} arms searched on the 998 proteins"
      + (f"; missing {len(missing)}, from {min(p['bits'] for p in missing):.1f} to {max(p['bits'] for p in missing):.1f} bits"
         if missing else ""))

# Every arm must have been searched the way the same arm was searched against the human
# proteome (extended or exact); otherwise a feature means different things in training and
# when the model is applied.
spec = __import__("importlib.util").util.spec_from_file_location("s243", "243_pfam998_search.py")
s243 = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(s243)
bad = []
for f in sorted((lr.PF998 / "regions").glob("*.parquet")):
    d = pl.read_parquet(f, columns=None).head(1)
    if "extended" not in d.columns:
        bad.append((f.stem, "searched before the settings were matched"))
    elif d.height and bool(d["extended"][0]) != s243.human_setting(f.stem)[0]:
        bad.append((f.stem, "extended here" if d["extended"][0] else "exact here"))
assert not bad, f"arms searched differently from the human search, rerun them: {bad}"
print("every searched arm matches its human search setting (extended or exact)")

T = lr.training_table()
print(T.group_by("split", "correct").agg(pl.len().alias("candidates")).sort("split", "correct"))
dropped = T.filter(pl.col("split").is_null()).height
print(f"\n{dropped:_} candidates touch families of both splits (or none, on proteins split both ways) and are left out")
train = T.filter(pl.col("split") == "selection")
test = T.filter(pl.col("split") == "heldout")
"""),
md(r"""
## 1. On families it has never seen, does combining alphabets help?

Precision-recall on the held-out families for the three nested models, with three single
features beside them. Average precision is the area under each curve; chance is the share
of candidates that are correct.
"""),
code(r"""
models, infos, curves = {}, {}, {}
y = test["correct"].to_numpy()
for name, feats in lr.FEATURE_SETS.items():
    m, info = lr.fit_model(train, feats)
    models[name], infos[name] = m, info
    p = m.predict_proba(test.select(feats).to_numpy())[:, 1]
    prec, rec, _ = precision_recall_curve(y, p)
    curves[name] = (rec, prec, average_precision_score(y, p))
for name, col in [("number of alphabets that find it", "n_alphabets"),
                  ("hp_lehninger2 bit score", "hp_lehninger2__bit_score"),
                  ("candidate length", "log_span_length")]:
    s = test[col].to_numpy()
    prec, rec, _ = precision_recall_curve(y, s)
    curves[name] = (rec, prec, average_precision_score(y, s))
ap = pl.DataFrame({"ranker": list(curves), "heldout_average_precision": [round(v[2], 4) for v in curves.values()]})
full, one, base = (curves[k][2] for k in ["+ all 19 alphabets", "+ hp_lehninger2's own values", "length and hydrophobic make-up only"])
lr.fig_heldout(
    curves, float(y.mean()), FIG / "243_heldout_precision_recall.png",
    hypothesis="A model that sees all 19 alphabets finds correct domain matches on unseen Pfam families better than one that sees a single alphabet.",
    conclusion=(f"Average precision on held-out families: all 19 alphabets {full:.3f}, hp_lehninger2 alone {one:.3f}, "
                f"length and hydrophobic make-up alone {base:.3f}; chance {y.mean():.4f}. "
                + ("Adding the other 18 alphabets raises it by a factor of " + f"{full / one:.1f}." if full > one
                   else "Adding the other 18 alphabets does not raise it.")),
)
print(ap)
print({k: {kk: v[kk] for kk in ("C", "n_pos", "n_neg")} for k, v in infos.items()})
"""),
md(r"""
## 2. What the full model weighs

Standardised coefficients: how much one standard deviation of each feature moves the
log-odds that a candidate is correct, with every other feature held fixed. Alphabets
overlap heavily, so a single coefficient is not a clean measure of one alphabet's worth;
section 1's nested comparison is.
"""),
code(r"""
coef = lr.fig_coefficients(
    models["+ all 19 alphabets"], lr.FEATURES, FIG / "243_model_coefficients.png",
    hypothesis="The model leans on a few alphabets, not on length.",
    conclusion="(filled below)",
)
top = coef.head(8)
bottom = coef.sort("coefficient").head(5)
lr.fig_coefficients(
    models["+ all 19 alphabets"], lr.FEATURES, FIG / "243_model_coefficients.png",
    hypothesis="The model leans on a few alphabets, not on length.",
    conclusion=("Largest positive weights: " + ", ".join(f"{f} {c:+.2f}" for f, c in top.rows()[:5])
                + ". Largest negative: " + ", ".join(f"{f} {c:+.2f}" for f, c in bottom.rows()[:3]) + "."),
)
print(top); print(bottom)
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
    tops[q] = ts.sort("prob", descending=True).head(10).select("gene", "prob", "n_alphabets")
    for partner in ["BCL2", "CD47"]:
        idx = ts["gene"].to_list().index(partner) if partner in ts["gene"].to_list() else None
        if idx is None:
            continue
        for ranker, col in [("model", "prob"), ("number of alphabets", "n_alphabets"),
                            ("hp_lehninger2 bit score", "hp_lehninger2__bit_score"), ("candidate length", "log_span_length")]:
            rows.append(dict(query=q, partner=partner, ranker=ranker, rank=lr.midrank(ts[col].to_numpy(), idx),
                             n_targets=ts.height, partner_value=float(ts[col][idx])))
app = pl.DataFrame(rows)
b = app.filter((pl.col("query") == "Ced9") & (pl.col("partner") == "BCL2") & (pl.col("ranker") == "model"))["rank"][0]
c = app.filter((pl.col("query") == "P66") & (pl.col("partner") == "CD47") & (pl.col("ranker") == "model"))["rank"][0]
k = app.filter((pl.col("query") == "Ced9") & (pl.col("partner") == "CD47") & (pl.col("ranker") == "model"))["rank"][0]
lr.fig_application(
    app, FIG / "243_model_on_three_cases.png",
    hypothesis="The trained model puts BCL2 near the top for Ced9 without having been fitted to it.",
    conclusion=f"The model ranks BCL2 {b:_.0f} for Ced9, CD47 {c:_.0f} for P66, and the control, CD47 for Ced9, {k:_.0f}.",
)
print(app)
for q, t in tops.items():
    print(f"\n{q}: the 10 targets the model ranks highest"); print(t)
"""),
md(r"""
## 4. Conclusions

(written once every arm of `243_pfam998_search.py` has been searched; the first cell says
how many are in)
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
