#!/usr/bin/env python3
"""Write the JSON behind the notebook 256 explorer page.

Every number comes from bhf_shuffle_null_utils (best_score_p, talk_in_top, talk_summary,
bhf_best_regions, with_human_region); the page only draws them.

Usage: 256_explorer_data.py OUT.json [256_explorer_template.html OUT.html]

With a template, also writes the page with the JSON put in place of /*DATA*/null.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bhf_shuffle_null_utils as u  # noqa: E402
from importlib import import_module  # noqa: E402

CLUSTERS = import_module("241_alphabet_ranking_driver").CLUSTERS
TALK_GENES = import_module("256_bhf_dipeptide_shuffle_null").TALK_GENES


def r4(v):
    """4 significant digits; None for a missing or infinite value."""
    if v is None or not math.isfinite(v):
        return None
    return float(f"{v:.4g}")


def main(out: Path):
    df = u.load()
    copies = [q for q in u.queries() if q != "BHF"]
    grid = u.full_grid(df)
    bp = u.best_score_p(df)
    reg = u.with_human_region(u.bhf_best_regions(bp))
    hum = u.human_sequences()
    talk = u.talk_in_top(df)
    talk_s = u.talk_summary(talk)
    bp_s = {r["metric"]: r for r in u.best_score_summary(bp).iter_rows(named=True)}

    # BHF's top 10 per (metric, pair), with the number of copies that have each protein
    # in their own top 10 at the same (metric, pair).
    b10 = (grid.filter(pl.col("is_bhf")).select("metric", "alphabet", "k", "top10")
               .explode("top10").rename({"top10": "gene"}).drop_nulls("gene")
               .with_columns(pl.int_range(pl.len()).over("metric", "alphabet", "k").alias("rank0")))
    c10 = (grid.filter(~pl.col("is_bhf")).select("metric", "alphabet", "k", "query_name", "top10")
               .explode("top10").rename({"top10": "gene"}).drop_nulls("gene")
               .unique(["metric", "alphabet", "k", "query_name", "gene"])
               .group_by("metric", "alphabet", "k", "gene").agg(pl.len().alias("n_copies")))
    top10 = b10.join(c10, on=["metric", "alphabet", "k", "gene"], how="left").with_columns(
        pl.col("n_copies").fill_null(0)).sort("metric", "alphabet", "k", "rank0")
    top10_d = {}
    for (m, a, k), s in top10.group_by(["metric", "alphabet", "k"]):
        top10_d[(m, a, k)] = [[g, n] for g, n in zip(s["gene"], s["n_copies"])]

    n_hit = {(r["metric"], r["alphabet"], r["k"]): r["n_hit"]
             for r in grid.filter(pl.col("is_bhf")).select("metric", "alphabet", "k", "n_hit").iter_rows(named=True)}
    regs = {(x["metric"], x["alphabet"], x["k"]): x for x in reg.iter_rows(named=True)}
    copy_order = {q: i for i, q in enumerate(copies)}

    metrics = []
    for m, lower in u.METRICS.items():
        s = bp.filter(pl.col("metric") == m).sort("alphabet", "k")
        vals = grid.filter((pl.col("metric") == m) & ~pl.col("is_bhf")).select("alphabet", "k", "query_name", "top_value")
        pairs = []
        for x in s.iter_rows(named=True):
            v = vals.filter((pl.col("alphabet") == x["alphabet"]) & (pl.col("k") == x["k"]))
            cv = [None] * len(copies)
            for q, t in zip(v["query_name"], v["top_value"]):
                cv[copy_order[q]] = r4(t)
            rg = regs.get((m, x["alphabet"], x["k"]))
            region = None
            if rg is not None:
                region = dict(gene=rg["gene"], start=rg["start"], end=rg["end"], bhf_seq=rg["seq"],
                              human_start=rg["target_start"], human_end=rg["target_end"],
                              human_seq=rg["human_seq"], human_length=len(hum.get(rg["gene"], "")) or None)
            pairs.append(dict(alphabet=x["alphabet"], k=x["k"], bhf=r4(x["bhf"]), bhf_top_gene=x["bhf_top_gene"],
                              bhf_n_tied=x["bhf_n_tied"], bhf_n_hit=n_hit[(m, x["alphabet"], x["k"])], n_as_good=x["n_as_good"], n_copies=x["n_copies"],
                              n_copies_no_hit=x["n_copies_no_hit"], p=round(x["p"], 6), copies=cv,
                              region=region, top10=top10_d.get((m, x["alphabet"], x["k"]), [])))
        t = talk_s.filter(pl.col("metric") == m)
        talk_d = None
        if t.height:
            r = t.row(0, named=True)
            ct = talk.filter((pl.col("metric") == m) & ~pl.col("is_bhf")).sort("query_name")
            talk_d = dict(n_pairs=r["n_alphabet_ksize_pairs"], bhf_n_top=r["bhf_n_top"], bhf_pct=r["bhf_pct"],
                          copies_median_pct=r["copies_median_pct"], copies_p2_5=r["copies_p2_5"],
                          copies_p97_5=r["copies_p97_5"], n_copies=r["n_copies"], p=r["p"],
                          copies_pct=[round(v, 2) for v in ct["pct"]])
        bs = bp_s[m]
        summary = dict(n_pairs=bs["n_alphabet_ksize_pairs"], n_pairs_p_le_0_05=bs["n_pairs_p_le_0_05"],
                       expected_by_chance=bs["expected_by_chance"])
        metrics.append(dict(name=m, lower_is_better=lower, summary=summary, pairs=pairs, talk=talk_d))

    data = dict(bhf_seq=u.bhf_sequence(), bhf_length=len(u.bhf_sequence()), n_copies=len(copies),
                n_human=u.N_HUMAN, top=u.TOP, clusters=CLUSTERS, talk_genes=TALK_GENES, metrics=metrics)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out.read_text()


if __name__ == "__main__":
    js = main(Path(sys.argv[1]))
    if len(sys.argv) == 4:
        page = Path(sys.argv[2]).read_text()
        assert page.count("/*DATA*/null") == 1
        Path(sys.argv[3]).write_text(page.replace("/*DATA*/null", js))
        print(f"wrote {sys.argv[3]} ({Path(sys.argv[3]).stat().st_size / 1e6:.2f} MB)")
