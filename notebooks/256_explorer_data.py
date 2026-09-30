#!/usr/bin/env python3
"""Write the JSON behind the notebook 256 explorer page.

Every number comes from bhf_shuffle_null_utils (best_score_p, talk_in_top, talk_summary,
bhf_best_regions, with_human_region) and, for notebook 258's view, bhf_feature_utils
(feature_table, load_regions, label_matches, class_shares, class_test); the page only
draws them.

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
import bhf_feature_utils as f  # noqa: E402
import bhf_shuffle_null_utils as u  # noqa: E402
from importlib import import_module  # noqa: E402

CLUSTERS = import_module("241_alphabet_ranking_driver").CLUSTERS
TALK_GENES = import_module("256_bhf_dipeptide_shuffle_null").TALK_GENES


# Metrics drawn on a log10 axis: the E-value and p-values (smaller is better) and the
# Poisson score (larger is better), which spans two orders of magnitude.
LOG10_AXIS = ["E-value", "Poisson p-value", "protein Poisson p-value", "Poisson score"]
# notebook 258's measures: three classes counted outside disordered regions, and disorder
MEASURES = {"pct_zinc_finger_ordered": "zinc finger", "pct_membrane_ordered": "membrane",
            "pct_extracellular_ordered": "extracellular", "pct_disordered": "disordered"}


def encoded(seq: str, alphabet: str) -> str | None:
    """Each residue as the number of its class in `alphabet` (as bhf_feature_utils.encode,
    for every alphabet of notebook 241); None when a class number would take two digits."""
    cl = CLUSTERS[alphabet]
    if len(cl) > 10:
        return None
    m = {a: str(i) for i, c in enumerate(cl) for a in c}
    return "".join(m.get(a, "?") for a in seq)


def residue_match(a: str, b: str, alphabet: str) -> dict:
    """Match line and counts for an ungapped pair: | same residue, + same class only."""
    ea, eb = encoded(a, alphabet), encoded(b, alphabet)
    grp = {x: i for i, c in enumerate(CLUSTERS[alphabet]) for x in c}
    line = "".join("|" if x == y else "+" if grp.get(x) is not None and grp.get(x) == grp.get(y) else " "
                   for x, y in zip(a, b))
    return dict(match_line=line, n_identical=line.count("|"), n_same_class=line.count("|") + line.count("+"),
                bhf_encoded=ea, human_encoded=eb, n_classes=len(CLUSTERS[alphabet]))


def class_test_block(t: pl.DataFrame, sh: pl.DataFrame, by: dict | None = None) -> list[dict]:
    """class_test rows for the four measures, with every copy's share for the strip."""
    out = []
    for meas, name in MEASURES.items():
        r = t.filter(pl.col("measure") == meas)
        c = sh.filter(~pl.col("is_bhf"))
        for k, v in (by or {}).items():
            r, c = r.filter(pl.col(k) == v), c.filter(pl.col(k) == v)
        r = r.row(0, named=True)
        out.append(dict(cls=name, bhf_pct=r["bhf_pct"], copies_median_pct=r["copies_median_pct"],
                        copies_p2_5=r["copies_p2_5"], copies_p97_5=r["copies_p97_5"], n_copies=r["n_copies"],
                        p=r["p"], copies_pct=[round(v, 2) for v in c.sort("query_name")[meas]]))
    return out


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
    regs_by_gene = {(x["metric"], x["alphabet"], x["k"], x["gene"]): x for x in reg.iter_rows(named=True)}
    copy_order = {q: i for i, q in enumerate(copies)}

    # notebook 258: UniProt classes of the top-10 matches (the --keep-regions repeat search)
    feats, fstats = f.feature_table()
    lab = f.label_matches(f.load_regions(), feats)
    sh, sh_m, sh_a = f.class_shares(lab), f.class_shares(lab, ["metric"]), f.class_shares(lab, ["alphabet"])
    te, te_m, te_a = f.class_test(sh), f.class_test(sh_m, ["metric"]), f.class_test(sh_a, ["alphabet"])
    bl = lab.filter(pl.col("is_bhf"))
    bhf = u.bhf_sequence()
    # each top-10 protein's region in the --keep-regions repeat search (0-based, end exclusive)
    top_region = {}
    for x in bl.iter_rows(named=True):
        a, b, ta, tb = x["region_start"], x["region_end"], x["target_start"], x["target_end"]
        hs = hum.get(x["gene"], "")[ta:tb]
        if len(hs) != b - a:
            continue
        top_region[(x["metric"], x["alphabet"], x["k"], x["gene"])] = dict(
            start=a, end=b, human_start=ta, human_end=tb, human_seq=hs, human_length=len(hum.get(x["gene"], "")) or None,
            **residue_match(bhf[a:b], hs, x["alphabet"]))
    bhf_cls = {}
    for x in bl.iter_rows(named=True):
        on = [c for c in f.CLASSES if x[f"on_{c.replace(' ', '_')}"]]
        bhf_cls[(x["metric"], x["alphabet"], x["k"], x["gene"])] = (
            ", ".join(on) if on else "none of the four") if x["annotated"] else "no reviewed UniProt entry"
    n_top10_no_class = n_top10_no_region = 0

    # rank 1 shows notebook 241's region (as notebook 256 does); count where the repeat search's differs
    n_top1_region_differs = sum(1 for (m, al, k, g), x in regs_by_gene.items()
                                if (m, al, k, g) in top_region and (top_region[(m, al, k, g)]["start"], top_region[(m, al, k, g)]["end"]) != (x["start"], x["end"]))

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
                              human_seq=rg["human_seq"], human_length=len(hum.get(rg["gene"], "")) or None,
                              **residue_match(rg["seq"], rg["human_seq"], x["alphabet"]))
            t10 = top10_d.get((m, x["alphabet"], x["k"]), [])
            t10 = [[g, n, bhf_cls.get((m, x["alphabet"], x["k"], g)), top_region.get((m, x["alphabet"], x["k"], g))]
                   for g, n in t10]
            n_top10_no_class += sum(c is None for _, _, c, _ in t10)
            n_top10_no_region += sum(r is None for _, _, _, r in t10)
            pairs.append(dict(alphabet=x["alphabet"], k=x["k"], bhf=r4(x["bhf"]), bhf_top_gene=x["bhf_top_gene"],
                              bhf_n_tied=x["bhf_n_tied"], bhf_n_hit=n_hit[(m, x["alphabet"], x["k"])], n_as_good=x["n_as_good"], n_copies=x["n_copies"],
                              n_copies_no_hit=x["n_copies_no_hit"], p=round(x["p"], 6), copies=cv,
                              region=region, top10=t10))
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
        classes = class_test_block(te_m, sh_m, {"metric": m})
        metrics.append(dict(name=m, lower_is_better=lower, log10_axis=m in LOG10_AXIS, summary=summary,
                            pairs=pairs, talk=talk_d, classes=classes))

    by_alphabet = {name: [dict(alphabet=r["alphabet"], bhf_pct=r["bhf_pct"], copies_median_pct=r["copies_median_pct"],
                               copies_p2_5=r["copies_p2_5"], copies_p97_5=r["copies_p97_5"], p=r["p"])
                          for r in te_a.filter(pl.col("measure") == meas).sort("p", "alphabet").iter_rows(named=True)]
                   for meas, name in MEASURES.items()}
    features = dict(
        n_matched=fstats["n_matched"], n_gencode=fstats["n_gencode"],
        n_matches=lab.height, n_matches_annotated=int(lab["annotated"].sum()),
        n_top10_no_class=n_top10_no_class, n_top10_no_region=n_top10_no_region,
        n_top1_region_differs=n_top1_region_differs,
        pooled=class_test_block(te, sh),
        by_alphabet=by_alphabet)

    n_pairs_searched = len(json.loads((u.RUN / "arms.json").read_text()))
    data = dict(bhf_seq=u.bhf_sequence(), bhf_length=len(u.bhf_sequence()), n_copies=len(copies),
                n_pairs_searched=n_pairs_searched, n_alphabets=len(CLUSTERS),
                n_human=u.N_HUMAN, top=u.TOP, clusters=CLUSTERS, talk_genes=TALK_GENES, metrics=metrics,
                features=features)
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
