#!/usr/bin/env python3
"""Score one SCOPe40 all-against-all search per query, the way Foldseek's benchmark does.

Input is one search's hits: a TSV (query, target, score columns...) or a kmerseek pair table
(.parquet, one row per protein pair, from scope40-alphabet-ksize-v040 --pairs-parquet).
Output is one row per (query, level) for the queries Foldseek scores, plus each query's hits
down to and including its first false positive (the only hits the score depends on).

The rules are those of bench.noselfhit.awk in github.com/steineggerlab/foldseek-analysis
(scopbenchmark/scripts), which the Foldseek paper's numbers come from:
  - the self-hit is dropped (awk line 17);
  - a target not in the domain list is a false positive (line 16);
  - walking the hits in score order, a hit in a different fold is the first false positive
    and nothing after it counts (line 18); before it, a same-family hit is a family TP
    (line 19), same superfamily but different family a superfamily TP (line 20), same fold
    but different superfamily a fold TP (line 21);
  - queries: domains with another member at all three levels (line 24);
  - sensitivity: family TPs / (family size - 1); superfamily TPs / (superfamily size -
    (family size - 1)); fold TPs / (fold size - (superfamily size - 1)) (lines 25-27). The
    last two keep the query itself in the count, so they are one more than the domains that
    can be found; sensitivity_possible divides by the domains that can be found.
A query with no hits scores 0 and is kept.

Ties: the awk takes the hits in file order. Here hits are sorted by the score, then with
false positives first among equal scores (the order least favourable to the tool), then by
target id, so the result does not depend on the order a tool wrote ties in.

Usage:
  score_hitlist.py --domains scope_domains.tsv --hits hits.tsv.gz --score bits --higher-first \
      --tool foldseek --setting default --out-prefix foldseek.default
  score_hitlist.py --domains scope_domains.tsv --hits x.pairs.parquet --score region_evalue \
      --tool kmerseek --setting hsdm17.k4.c_max --out-prefix kmerseek.hsdm17.k4.c_max
scope_domains.tsv: domain_id, scop_id (from prepare_scope.py, or written by --fasta).
"""

import argparse
import sys
from pathlib import Path

import polars as pl

LEVELS = ("family", "superfamily", "fold")


def read_domains(path: Path | None, fasta: Path | None) -> pl.DataFrame:
    if fasta is not None:
        rows = [line[1:].split()[:2] for line in fasta.open() if line.startswith(">")]
        dom = pl.DataFrame(rows, schema=["domain_id", "scop_id"], orient="row")
    else:
        dom = pl.read_csv(path, separator="\t").select("domain_id", "scop_id")
    p = pl.col("scop_id").str.split(".")
    return dom.with_columns(
        p.list.get(0).alias("scope_class"),
        p.list.slice(0, 2).list.join(".").alias("fold"),
        p.list.slice(0, 3).list.join(".").alias("superfamily"),
        pl.col("scop_id").alias("family"),
    )


def queries(dom: pl.DataFrame) -> pl.DataFrame:
    d = dom.with_columns(
        pl.len().over("family").alias("famcnt"),
        pl.len().over("superfamily").alias("sfamcnt"),
        pl.len().over("fold").alias("foldcnt"),
    )
    return d.filter((pl.col("famcnt") > 1) & (pl.col("sfamcnt") > pl.col("famcnt"))
                    & (pl.col("foldcnt") > pl.col("sfamcnt")))


def read_hits(path: Path, score: str, columns: list[str] | None) -> pl.LazyFrame:
    if path.suffix == ".parquet":
        lf = pl.scan_parquet(path).select(
            pl.col("query_domain").cast(pl.Utf8).alias("query"),
            pl.col("target_domain").cast(pl.Utf8).alias("target"), pl.col(score).alias("score"))
    else:
        names = columns or ["query", "target", score]
        lf = pl.scan_csv(path, separator="\t", has_header=False, new_columns=names,
                         schema_overrides={score: pl.Float64}).select("query", "target", pl.col(score).alias("score"))
    # One row per pair: a tool can report a pair more than once (several alignments).
    return lf


def label_hits(hits: pl.LazyFrame, dom: pl.DataFrame, higher_first: bool) -> pl.DataFrame:
    q = dom.select(pl.col("domain_id").alias("query"), pl.col("family").alias("q_fam"),
                   pl.col("superfamily").alias("q_sf"), pl.col("fold").alias("q_fold"))
    t = dom.select(pl.col("domain_id").alias("target"), pl.col("family").alias("t_fam"),
                   pl.col("superfamily").alias("t_sf"), pl.col("fold").alias("t_fold"))
    h = (hits.filter(pl.col("query") != pl.col("target"))
         .join(q.lazy(), on="query", how="inner")
         .join(t.lazy(), on="target", how="left")
         .with_columns(
             pl.when(pl.col("t_fold").is_null() | (pl.col("t_fold") != pl.col("q_fold"))).then(pl.lit("fp"))
             .when(pl.col("t_fam") == pl.col("q_fam")).then(pl.lit("family"))
             .when(pl.col("t_sf") == pl.col("q_sf")).then(pl.lit("superfamily"))
             .otherwise(pl.lit("fold")).alias("label"))
         # A pair reported more than once keeps its best score.
         .group_by("query", "target", "label")
         .agg(pl.col("score").max() if higher_first else pl.col("score").min())
         .with_columns((pl.col("label") == "fp").alias("is_fp"))
         .sort(["query", "score", "is_fp", "target"], descending=[False, higher_first, True, False],
               nulls_last=True)
         .with_columns(pl.int_range(1, pl.len() + 1).over("query").alias("rank"))
         .with_columns(pl.col("is_fp").cast(pl.Int32).cum_sum().over("query").alias("n_fp_so_far")))
    return h.collect(engine="streaming")


def score(labelled: pl.DataFrame, qset: pl.DataFrame, tool: str, setting: str) -> pl.DataFrame:
    above = labelled.filter((pl.col("n_fp_so_far") == 0) | ((pl.col("n_fp_so_far") == 1) & pl.col("is_fp")))
    per_q = above.group_by("query").agg(
        *[(pl.col("label") == lvl).sum().alias(f"tp_{lvl}") for lvl in LEVELS],
        pl.col("rank").filter(pl.col("is_fp")).first().alias("first_fp_index"))
    n_hits = labelled.group_by("query").agg(pl.len().alias("n_hits"))
    base = (qset.select(pl.col("domain_id").alias("query"), pl.col("scop_id").alias("scope_sccs"), "scope_class",
                        "fold", "superfamily", "family", "famcnt", "sfamcnt", "foldcnt")
            .join(per_q, on="query", how="left").join(n_hits, on="query", how="left")
            .with_columns(pl.col("n_hits").fill_null(0)))
    den_script = {"family": pl.col("famcnt") - 1, "superfamily": pl.col("sfamcnt") - (pl.col("famcnt") - 1),
                  "fold": pl.col("foldcnt") - (pl.col("sfamcnt") - 1)}
    den_poss = {"family": pl.col("famcnt") - 1, "superfamily": pl.col("sfamcnt") - pl.col("famcnt"),
                "fold": pl.col("foldcnt") - pl.col("sfamcnt")}
    out = []
    for lvl in LEVELS:
        tp = pl.col(f"tp_{lvl}").fill_null(0)
        out.append(base.select(
            pl.lit(tool).alias("tool"), pl.lit(setting).alias("setting"), pl.lit(lvl).alias("level"),
            "query", "scope_sccs", "scope_class", "fold", "superfamily", "family",
            den_script[lvl].alias("n_tp_total"), den_poss[lvl].alias("n_tp_possible"),
            tp.alias("n_tp_above_cut"), (tp / den_script[lvl]).alias("sensitivity"),
            (tp / den_poss[lvl]).alias("sensitivity_possible"),
            (pl.col("n_hits") > 0).alias("has_hits"), "first_fp_index", "n_hits"))
    return pl.concat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--domains", type=Path)
    g.add_argument("--fasta", type=Path)
    ap.add_argument("--hits", type=Path, required=True)
    ap.add_argument("--columns", help="comma list of the TSV's columns (it has no header)")
    ap.add_argument("--score", required=True)
    d = ap.add_mutually_exclusive_group(required=True)
    d.add_argument("--higher-first", action="store_true")
    d.add_argument("--lower-first", action="store_true")
    ap.add_argument("--tool", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    dom = read_domains(a.domains, a.fasta)
    qset = queries(dom)
    cols = a.columns.split(",") if a.columns else None
    lab = label_hits(read_hits(a.hits, a.score, cols), dom, a.higher_first)
    res = score(lab, qset, a.tool, a.setting)
    res.write_parquet(f"{a.out_prefix}.per_query.parquet")
    (lab.filter((pl.col("n_fp_so_far") == 0) | ((pl.col("n_fp_so_far") == 1) & pl.col("is_fp")))
     .select("query", "rank", "target", "score", "label").write_parquet(f"{a.out_prefix}.hits_to_first_fp.parquet"))
    s = res.group_by("level").agg(pl.col("sensitivity").mean().alias("auc"), pl.col("has_hits").mean().alias("coverage"),
                                  pl.len().alias("n")).sort("level")
    print(f"{a.tool} {a.setting}: {lab.height:,} hits after the self-hit is dropped", file=sys.stderr)
    print(s, file=sys.stderr)


if __name__ == "__main__":
    main()
