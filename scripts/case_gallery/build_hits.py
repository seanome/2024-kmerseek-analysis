"""Per query and alphabet/k of notebook 241: kmerseek's human hits ranked three ways.

A protein's score under a statistic is its best region (lowest E-value, highest mean IDF,
highest tf-idf), and its rank is 1 + the number of proteins with a strictly better score,
the rule notebook 241's collect script uses. Infinite or missing values are not ranked.
Rows kept: the top 20 under each statistic, the known partner (BCL2 for Ced9, CD47 for
P66) and, for BHF, the nine genes the Figure 1 draft named. Coordinates 0-based half-open.
Writes tables/case_gallery/hits241.json and hits241_summary.json.
"""
import json, math
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tables" / "case_gallery"
D = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")  # regions.parquet, 260 MB, not in git
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
TOP = 20
PARTNER = {"Ced9": "BCL2", "P66": "CD47", "BHF": None}
FIG1 = ["ZNF292", "RSF1", "TSHZ1", "TSHZ2", "TSHZ3", "RNMT", "SFI1", "TRAPPC10", "NDNF"]
MET = [("E", "region_evalue", True), ("idf", "region_mean_idf", False), ("tfidf", "region_tfidf", False)]


def fasta(p):
    out, name = {}, None
    for l in Path(p).read_text().splitlines():
        if l.startswith(">"):
            name = l[1:].strip(); out[name] = []
        else:
            out[name].append(l.strip())
    return {k: "".join(v) for k, v in out.items()}


def fin(x):
    return x is not None and math.isfinite(x)


def main():
    seqs, q = fasta(HUMAN), fasta(D / "queries.fa")
    reg = pl.read_parquet(D / "regions.parquet", columns=[
        "alphabet", "ksize_arm", "query_name", "target_name", "gene", "region_start", "region_end",
        "target_start", "target_end", "region_evalue", "region_mean_idf", "region_tfidf"])
    reg = reg.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
                            for _, c, _ in MET])
    out, summ = {}, []
    for (a, k, qq), g in reg.group_by(["alphabet", "ksize_arm", "query_name"]):
        per = g.group_by("target_name", "gene").agg(
            pl.col("region_evalue").min().alias("E"), pl.col("region_mean_idf").max().alias("idf"),
            pl.col("region_tfidf").max().alias("tfidf"))
        n, keep = {}, set()
        for key, _, low in MET:
            n[key] = per[key].is_not_null().sum()
            # rank "min": 1 + the number of proteins strictly better; nulls stay null
            per = per.with_columns(pl.col(key).rank("min", descending=not low).cast(pl.Int64).alias("rk_" + key))
            keep |= set(per.filter(pl.col("rk_" + key).is_not_null()).sort(["rk_" + key, "gene"]).head(TOP)["target_name"])
        wanted = ([PARTNER[qq]] if PARTNER[qq] else []) + (FIG1 if qq == "BHF" else [])
        keep |= set(per.filter(pl.col("gene").is_in(wanted))["target_name"])
        rows = []
        for r in per.filter(pl.col("target_name").is_in(keep)).iter_rows(named=True):
            t = seqs[r["target_name"]]
            rg = []
            for x in g.filter(pl.col("target_name") == r["target_name"]).sort("region_start").iter_rows(named=True):
                a0, a1, b0, b1 = int(x["region_start"]), int(x["region_end"]), int(x["target_start"]), int(x["target_end"])
                rg.append([a0, a1, b0, b1, x["region_evalue"], round(x["region_mean_idf"], 1),
                           None if x["region_tfidf"] is None else round(x["region_tfidf"], 1), q[qq][a0:a1], t[b0:b1]])
            rows.append({"gene": r["gene"], "tl": len(t), "E": r["E"], "idf": round(r["idf"], 1),
                         "tfidf": None if r["tfidf"] is None else round(r["tfidf"], 1),
                         "rk": {m: r["rk_" + m] for m, _, _ in MET}, "rg": rg})
            if r["gene"] in wanted:
                summ.append({"query": qq, "alphabet": a, "k": int(k), "gene": r["gene"],
                             **{"rank_" + m: r["rk_" + m] for m, _, _ in MET}, **{"n_" + m: n[m] for m, _, _ in MET}})
        out[f"{qq}|{a}|{k}"] = {"n": n, "rows": rows}
    (T / "hits241.json").write_text(json.dumps(out, separators=(",", ":")))
    (T / "hits241_summary.json").write_text(json.dumps(summ, separators=(",", ":")))
    print(len(out), "arm x query;", (T / "hits241.json").stat().st_size, "bytes;", len(summ), "tracked-gene rows")


if __name__ == "__main__":
    main()
