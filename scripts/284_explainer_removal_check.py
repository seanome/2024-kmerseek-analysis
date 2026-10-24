"""Notebook 284: how far the page's "remove an alphabet" shortcut is from a fresh count.

On the explainer page, removing an alphabet clears its bit in each stored polyglot match. A fresh count
drops that alphabet's regions first and then joins the rest into polyglot matches, so a polyglot match that only
the removed alphabet's region held together splits in two. This script makes the fresh count
with alphabet_agreement_utils for a few removals, so tests/284_explainer_compute.test.mjs can
compare it with the page's shortcut, and the page can say how big the gap is.

Output: notebooks/284_explainer_removal_check.csv
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import alphabet_agreement_utils as au  # noqa: E402

DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
CASES = {"Ced9": "BCL2", "P66": "CD47"}
REMOVALS = {
    "protein20": ["protein20"],
    "wwmj5": ["wwmj5"],
    "hp_lehninger2": ["hp_lehninger2"],
    "all seven H/P": au.HP,
    "everything but the seven H/P": [a for a in [
        "gbmr4", "polarity4", "wwmj5", "dayhoff6", "gbmr7", "funcgroups8", "sdm12",
        "mmseqs12", "wass14", "hsdm17", "uniprot18", "protein20"]],
}
CUTS = [None, 1e4, 1e2]

rows = []
for q, partner in CASES.items():
    for label, removed in REMOVALS.items():
        d = au.prepare(pl.scan_parquet(DATA / "regions.parquet")
                       .filter((pl.col("query_name") == q) & ~pl.col("alphabet").is_in(removed)))
        for cut in CUTS:
            s = au.polyglot_matches(d, cut)
            for c in au.COUNTS:
                b = au.best_per_target(s, c)
                p = b.filter(pl.col("gene") == partner)
                v = int(p[c].max()) if p.height else 0
                rows.append(dict(query=q, removed=label, removed_alphabets=",".join(removed),
                                 e_cut=-1.0 if cut is None else cut, count=c,
                                 proteins_with_a_call=b.height, partner_value=v,
                                 proteins_with_more=int(b.filter(pl.col(c) > v).height),
                                 proteome_max=int(b[c].max()) if b.height else 0))
                print(rows[-1], flush=True)
pl.DataFrame(rows).write_csv(ROOT / "notebooks/284_explainer_removal_check.csv")
