#!/usr/bin/env python3
"""How far the paper's Botryllus list proteins reach into human reads, by alphabet (notebook 361).

Usage: sra_alphabet_reach.py <search_all> <search_shuffled> <query_map.tsv> <runs.tsv> <out_dir>

For each protein and species, the score is its highest containment in any run with hashes at that
setting. Containment = share of the protein's sketched k-mers found in the run. A protein passes when
it shares at least 2 hashes with a run and its score is above the 99th percentile of the shuffled
proteins' scores at the same setting and species. So about 1 % of shuffled proteins pass by design.

Writes:
  reach_by_setting.tsv  per setting, species and query set: proteins with hashes, proteins passing
  gene_passes.tsv       one row per list or random gene, setting and species: passed or not
  threshold.tsv         the 99th percentile used at each setting and species
"""
import sys
from pathlib import Path

import polars as pl

real, shuffled, qmap, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
SETTINGS = [("protein", 6), ("protein", 7), ("dayhoff", 12), ("dayhoff", 13), ("hp", 27), ("hp", 30)]
q = pl.read_csv(qmap, separator="\t", infer_schema_length=0).select("query", "gene", "set").unique()
species = (pl.read_csv(runs, separator="\t", infer_schema_length=0)
           .select(pl.col("run").alias("match_name"), pl.when(pl.col("organism").str.contains("Homo")).then(pl.lit("human"))
                   .otherwise(pl.lit("Botryllus")).alias("species")))


def hashes(path, m, k):
    d = pl.read_csv(path, infer_schema_length=0).filter((pl.col("moltype") == m) & (pl.col("ksize") == str(k)))
    return d.with_columns(pl.col("n_hashes").cast(pl.Int64)).filter(pl.col("n_hashes") > 0).select("name", "n_hashes")


rows, genes, cuts = [], [], []
for m, k in SETTINGS:
    query_ok = pl.concat([hashes(real / "queries.describe.csv", m, k), hashes(shuffled / "queries.describe.csv", m, k)])
    run_ok = hashes(real / "runs.ge0.1.describe.csv", m, k)["name"]
    d = pl.concat([pl.read_csv(p / f"ms.ge0.1.{m}.k{k}.csv", columns=["query_name", "match_name", "containment", "intersect_hashes"],
                               schema_overrides={"query_name": pl.String, "match_name": pl.String}) for p in (real, shuffled)])
    d = d.join(query_ok.rename({"name": "query_name"}), on="query_name").filter(pl.col("match_name").is_in(run_ok)).join(species, on="match_name")
    best = d.group_by("query_name", "species").agg(pl.col("containment").max().alias("score"), pl.col("intersect_hashes").max().alias("hashes"))
    best = best.with_columns(pl.when(pl.col("query_name").str.starts_with("shuf_")).then(pl.lit("shuffled")).otherwise(None).alias("shuf"))
    best = best.join(q, left_on="query_name", right_on="query", how="left").with_columns(pl.coalesce("shuf", "set").alias("set"))
    for (sp,), b in best.group_by(["species"]):
        cut = b.filter(pl.col("set") == "shuffled")["score"].quantile(0.99, "higher")
        cuts.append(dict(alphabet=m, k=k, species=sp, cut=cut, runs=d.filter(pl.col("species") == sp)["match_name"].n_unique()))
        b = b.with_columns(((pl.col("score") > cut) & (pl.col("hashes") >= 2)).alias("passed"))
        for s, x in b.group_by("set"):
            rows.append(dict(alphabet=m, k=k, species=sp, set=s[0], proteins=x.height, passed=int(x["passed"].sum()),
                             genes=x["gene"].n_unique() if s[0] != "shuffled" else None,
                             genes_passed=x.filter(pl.col("passed"))["gene"].n_unique() if s[0] != "shuffled" else None))
        g = b.filter(pl.col("set") != "shuffled").group_by("gene", "set").agg(pl.col("passed").any())
        genes.append(g.with_columns(pl.lit(m).alias("alphabet"), pl.lit(k).alias("k"), pl.lit(sp).alias("species")))
    print(m, k, flush=True)
pl.DataFrame(rows).sort("alphabet", "k", "species", "set").write_csv(out / "reach_by_setting.tsv", separator="\t")
pl.concat(genes).write_csv(out / "gene_passes.tsv", separator="\t")
pl.DataFrame(cuts).write_csv(out / "threshold.tsv", separator="\t")
print(pl.DataFrame(rows).sort("alphabet", "k", "species", "set"))
