"""Notebook 241's searches: do different alphabets agree on the same spot of the true partner?

A spot is a stretch of query and target on one diagonal (target start - query start is the
same, so the two stretches are in register). Two regions from different alphabets that share
a diagonal and overlap on the query are joined into one spot. For each target protein we keep
the spot backed by the most alphabets, then ask where BCL2 (query Ced9) and CD47 (query P66)
rank among the 19_693 human proteins.

Input: notebook 241's regions.parquet ($SCRATCH/241-alphabet-ranking/regions.parquet,
2026-09-23, 8_295_664 rows), GENCODE v49 canonical translations, notebook 241's queries.fa.
Output: notebooks/284_alphabet_agreement_*.csv and notebooks/284_alphabet_agreement_data.json,
which the explainer page reads.
"""
import json
from pathlib import Path

import polars as pl

DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
GENCODE = Path.home() / "data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
OUT = Path(__file__).resolve().parents[1] / "notebooks"
HP = ["hp_kyte_doolittle2", "hp_lehninger2", "hp_lehninger_c_nonpolar2", "hp_lehninger_hpc3",
      "hp_pbotc_1st_ed2", "hp_thomas_dill2", "hp_thomas_dill_no_c2"]
CASES = [("Ced9", "BCL2", "true partner"), ("Ced9", "TKT", "most alphabets agree"),
         ("P66", "CD47", "true partner"), ("P66", "CD55", "most alphabets agree")]


def read_fasta(path, key=lambda h: h):
    seqs, name = {}, None
    for line in open(path):
        if line.startswith(">"):
            name = key(line[1:].strip())
            seqs[name] = []
        else:
            seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def spots(d):
    """Join regions on one diagonal whose query stretches overlap; count alphabets per spot."""
    d = d.sort("query_name", "target_name", "diag", "qs")
    d = d.with_columns(prev_end=pl.col("qe").cum_max().shift(1).over("query_name", "target_name", "diag"))
    d = d.with_columns(spot=(pl.col("prev_end").is_null() | (pl.col("qs") >= pl.col("prev_end")))
                       .cast(pl.Int32).cum_sum().over("query_name", "target_name", "diag"))
    return d.group_by("query_name", "gene", "target_name", "diag", "spot").agg(
        n_alphabets=pl.col("alphabet").n_unique(), n_families=pl.col("family").n_unique(),
        n_below_e10=pl.col("alphabet").filter(pl.col("region_evalue") < 10).n_unique(),
        qs=pl.col("qs").min(), qe=pl.col("qe").max())


def main():
    regions = (pl.scan_parquet(DATA / "regions.parquet")
               .filter(pl.col("query_name").is_in(["Ced9", "P66"]))
               .with_columns(qs=pl.col("region_start").cast(pl.Int64), qe=pl.col("region_end").cast(pl.Int64),
                             diag=(pl.col("target_start") - pl.col("region_start")).cast(pl.Int64),
                             family=pl.when(pl.col("alphabet").is_in(HP)).then(pl.lit("H/P"))
                             .otherwise(pl.col("alphabet")))
               .collect())
    s = spots(regions)
    best = (s.sort("n_alphabets", "n_families", descending=True)
            .group_by("query_name", "gene", "target_name", maintain_order=True).first())
    best = best.with_columns(rank=pl.col("n_alphabets").rank("min", descending=True).over("query_name"))
    best.drop("spot").sort("query_name", "rank", "gene").write_csv(OUT / "284_alphabet_agreement_best_spot_per_target.csv")

    queries = read_fasta(DATA / "queries.fa")
    human = read_fasta(GENCODE, key=lambda h: h.split("|")[0])
    data = {"histogram": {}, "cases": [], "e10": {}}
    for q in ["Ced9", "P66"]:
        b = best.filter(pl.col("query_name") == q)
        h = b.group_by("n_alphabets").len().sort("n_alphabets")
        data["histogram"][q] = {"n_targets": b.height, "counts": dict(zip(h["n_alphabets"], h["len"]))}
        data["e10"][q] = {"max_alphabets_below_e10": int(b["n_below_e10"].max())}
    for q, gene, why in CASES:
        row = best.filter((pl.col("query_name") == q) & (pl.col("gene") == gene)).row(0, named=True)
        r = regions.filter((pl.col("query_name") == q) & (pl.col("gene") == gene) & (pl.col("diag") == row["diag"])
                           & (pl.col("qe") > row["qs"]) & (pl.col("qs") < row["qe"]))
        per_alph = (r.group_by("alphabet").agg(qs=pl.col("qs").min(), qe=pl.col("qe").max(),
                                               ks=pl.col("ksize_arm").unique().sort(),
                                               best_e=pl.col("region_evalue").min()).sort("alphabet"))
        tseq = human[row["target_name"].split("|")[0]]
        qs, qe, dg = row["qs"], row["qe"], row["diag"]
        qa, ta = queries[q][qs:qe], tseq[qs + dg:qe + dg]
        run = best_run = 0
        for a, c in zip(qa, ta):
            run = run + 1 if a == c else 0
            best_run = max(best_run, run)
        data["cases"].append(dict(query=q, gene=gene, why=why, rank=int(row["rank"]), n_alphabets=row["n_alphabets"],
                                  n_families=row["n_families"], n_below_e10=row["n_below_e10"],
                                  qs=qs, qe=qe, ts=qs + dg, te=qe + dg, query_len=len(queries[q]),
                                  target_len=len(tseq), query_res=qa, target_res=ta,
                                  identical=sum(a == c for a, c in zip(qa, ta)), longest_identical_run=best_run,
                                  alphabets=[{k: (None if (isinstance(v, float) and v == float("inf")) else v)
                                              for k, v in d.items()} for d in per_alph.to_dicts()]))
        print(q, gene, "rank", row["rank"], "of", best.filter(pl.col("query_name") == q).height,
              "alphabets", row["n_alphabets"], "H/P counted once", row["n_families"], "below E 10", row["n_below_e10"],
              f"query {qs}-{qe} target {qs + dg}-{qe + dg}", f"{data['cases'][-1]['identical']}/{len(qa)} identical")
    print("max alphabets agreeing with E < 10:", data["e10"])
    for q in data["histogram"]:
        c = data["histogram"][q]["counts"]
        print(q, "targets with a spot backed by >= 4 alphabets:", sum(v for k, v in c.items() if k >= 4))
    # How identical are the spots that many alphabets agree on? Longest run of identical residues.
    data["identity_by_agreement"] = {}
    for q in ["Ced9", "P66"]:
        rows = []
        for r in s.filter(pl.col("query_name") == q).iter_rows(named=True):
            t = human.get(r["target_name"].split("|")[0], "")
            qa, ta = queries[q][r["qs"]:r["qe"]], t[r["qs"] + r["diag"]:r["qe"] + r["diag"]]
            run = longest = 0
            for a, c in zip(qa, ta):
                run = run + 1 if a == c else 0
                longest = max(longest, run)
            rows.append((r["n_alphabets"], longest))
        df = pl.DataFrame(rows, schema=["n_alphabets", "longest_identical_run"], orient="row")
        t = (df.group_by("n_alphabets").agg(median=pl.col("longest_identical_run").median(),
                                             q25=pl.col("longest_identical_run").quantile(0.25),
                                             q75=pl.col("longest_identical_run").quantile(0.75),
                                             n_spots=pl.len()).sort("n_alphabets"))
        t.write_csv(OUT / f"284_alphabet_agreement_identity_{q}.csv")
        print(q, "longest identical run by number of alphabets agreeing"); print(t)
        data["identity_by_agreement"][q] = t.to_dicts()
    json.dump(data, open(OUT / "284_alphabet_agreement_data.json", "w"), indent=1)


if __name__ == "__main__":
    main()
