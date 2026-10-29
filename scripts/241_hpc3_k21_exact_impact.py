"""Notebooks 242 and 284 on notebook 241's regions.parquet before and after the exact rerun of
hp_lehninger_hpc3 k=21 (PR 194). Run on Sherlock (2026-10-20) with alphabet_ensemble_utils.py
from olgabot/alphabet-ranking-three-cases and alphabet_agreement_utils.py from
olgabot/alphabet-agreement-explainer beside it; matplotlib is stubbed because only the
table functions are used.

Usage: 241_hpc3_k21_exact_impact.py <241 dir> <old tables subfolder> <GENCODE fasta>
Output: notebooks/241_results/hpc3_k21_exact_{242_ensemble_ranks,284_agreement}_old_vs_new.csv
(242 and 284 are the notebook numbers; the 241 table beside them was written separately).
"""

import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from unittest import mock

for m in (
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "matplotlib.patches",
    "matplotlib.lines",
    "matplotlib.colors",
    "pubfig",
    "alphabet_ranking_utils",
):
    sys.modules.setdefault(m, mock.MagicMock())
import alphabet_ensemble_utils as ae
import alphabet_agreement_utils as au

D = Path(sys.argv[1])
OLD = D / sys.argv[2]
ae.HUMAN_FASTA = Path(sys.argv[3])
out = {}
for lab, d in (("old", OLD), ("new", D)):
    ae.DATA = d
    rows = []
    for m in ae.METRICS:
        s, _ = ae.ensemble(m)
        for q, g in ae.PARTNER.items():
            p = s.filter((pl.col("query_name") == q) & (pl.col("gene") == g))
            rows.append(
                dict(
                    metric=m,
                    query=q,
                    partner=g,
                    by_arm=p["ens_rank_by_arm"].min() if p.height else None,
                    by_alphabet=p["ens_rank_by_alphabet"].min() if p.height else None,
                )
            )
        b = s.filter(pl.col("query_name") == "BHF").sort("ens_rank_by_arm").head(1)
        rows.append(
            dict(
                metric=m,
                query="BHF",
                partner=b["gene"][0],
                by_arm=b["ens_rank_by_arm"][0],
                by_alphabet=None,
            )
        )
    out[("242", lab)] = pl.DataFrame(rows)
    reg = au.prepare(
        pl.scan_parquet(d / "regions.parquet").filter(
            pl.col("query_name").is_in(["Ced9", "P66"])
        )
    )
    rows = []
    for e in au.E_CUTS:
        s = au.polyglot_matches(reg, e)
        for q, g in (("Ced9", "BCL2"), ("P66", "CD47")):
            ps = au.partner_summary(s, q, g)
            for c, v in ps.items():
                rows.append(dict(e_cut=str(e), query=q, partner_gene=g, count=c, **v))
        if e is None:
            for q in ("Ced9", "P66"):
                b = au.best_per_target(
                    s.filter(pl.col("query_name") == q), "n_alphabets"
                )
                rows.append(
                    dict(
                        e_cut="None",
                        query=q,
                        partner_gene="targets with >= 4 alphabets",
                        count="n_alphabets",
                        partner=None,
                        n_more=b.filter(pl.col("n_alphabets") >= 4).height,
                        max=int(b["n_alphabets"].max()),
                        n_targets=b.height,
                    )
                )
    out[("284", lab)] = pl.DataFrame(rows, infer_schema_length=None)
with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
    for nb, keys in (
        ("242", ["metric", "query"]),
        ("284", ["e_cut", "query", "partner_gene", "count"]),
    ):
        j = out[(nb, "old")].join(
            out[(nb, "new")], on=keys, suffix="_new", how="full", coalesce=True
        )
        vals = [c for c in out[(nb, "old")].columns if c not in keys]
        changed = j.filter(
            pl.any_horizontal(
                [
                    pl.col(c).cast(pl.Utf8).ne_missing(pl.col(c + "_new").cast(pl.Utf8))
                    for c in vals
                ]
            )
        )
        print(f"== {nb}: {j.height} rows, {changed.height} changed")
        print(changed)
        name = {"242": "242_ensemble_ranks", "284": "284_agreement"}[nb]
        j.write_csv(
            Path(__file__).parent.parent
            / "notebooks"
            / "241_results"
            / f"hpc3_k21_exact_{name}_old_vs_new.csv"
        )
