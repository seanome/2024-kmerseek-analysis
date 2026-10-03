#!/usr/bin/env python3
"""Family, superfamily and fold sensitivity AUC of one SCOPe40 search, for every score column.

For each query, hits are ranked by one score column; sensitivity at a level is the share of
that level's true positives ranked above the first false positive, with the FoldSeek/TEA
convention (eval_tsv_to_rocx in notebooks/scope_kmerseek_utils.py, exclude_gray_zone=True):
  family       TP = same family
  superfamily  TP = same superfamily, different family
  fold         TP = same fold, different superfamily
  FP at every level = a different fold; same fold, different superfamily is ignored.
The AUC is the area under the sorted per-query sensitivities (sensitivity_stats). Hits that
tie on the score are ordered with cross-fold hits first (auc_*) and last (auc_*_optimistic);
any other tie order gives an AUC between the two. Each is reported two ways, as in
notebooks 066 and 075:
  _all  over the queries with at least one same-family hit (n_queries_all)
  _ref  over the 5_713 FoldSeek SCOPe40 queries; one kmerseek returns nothing for scores 0
kmerseek writes one row per matched region, so a pair can have several rows. A pair's score
is its best region's: the smallest value for a probability or a frequency, the largest for
everything else. Whole-query columns are the same on every row of a pair.
"""

import argparse
import importlib.util
import sys

import polars as pl

# Column -> True when a smaller value ranks first. Every column of kmerseek 0.4.0's output
# that scores a pair; counts of expected k-mers, coordinates, sizes and settings are not
# scores and are left out. region_poisson_score is -log10(region_tail_probability), so the
# two must give the same AUC: a check on the scoring, not a second result.
SCORES = {
    "query_poisson_pvalue": True,
    "region_tail_probability": True,
    "region_evalue": True,
    "mean_matched_kmer_freq": True,  # lower: the shared k-mers are rarer in the database
    "sum_matched_kmer_freq": True,
    "containment": False,
    "max_containment": False,
    "containment_target_in_query": False,
    "f_weighted_target_in_query": False,
    "jaccard": False,
    "n_intersecting_hashes": False,
    "query_tfidf": False,
    "query_enrichment": False,
    "region_poisson_score": False,
    "region_enrichment": False,
    "region_tfidf": False,
    "region_mean_idf": False,
    "region_n_shared_kmers": False,
    "region_length": False,
    "region_ka_bits": False,
    "region_n_chained": False,
}
LEVELS = {"FAM": "family", "SFAM": "superfamily", "FOLD": "fold"}


def load_utils(path):
    """The notebooks' scope_kmerseek_utils, with scipy's trapezoid swapped for numpy's when
    scipy is absent (the kmerseek image has numpy 2.5 and polars, not scipy)."""
    try:
        import scipy.integrate  # noqa: F401
    except ImportError:
        import types

        import numpy as np

        scipy = types.ModuleType("scipy")
        integrate = types.ModuleType("scipy.integrate")
        integrate.trapezoid = np.trapezoid
        scipy.integrate = integrate
        sys.modules["scipy"], sys.modules["scipy.integrate"] = scipy, integrate
    spec = importlib.util.spec_from_file_location("scope_kmerseek_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print-directions", action="store_true",
                   help="print each score column and min/max for collapse_pairs.awk, then exit")
    for a in ("domains", "ref-rocx", "utils", "alphabet", "ksize", "setting", "out"):
        p.add_argument(f"--{a}")
    p.add_argument("--penalty", default="")
    p.add_argument("--pairs", help="one row per protein pair, from collapse_pairs.awk")
    p.add_argument("--raw-rows", type=int, help="region rows kmerseek wrote before collapsing")
    p.add_argument("--nofit", action="store_true", help="kmerseek refused the search: no fit")
    p.add_argument("--pairs-parquet", help="also write the merged pair table here")
    a = p.parse_args()
    if a.print_directions:
        print("column,best")
        for c, smaller_first in SCORES.items():
            print(f"{c},{'min' if smaller_first else 'max'}")
        return
    missing = [x for x in ("domains", "ref_rocx", "utils", "alphabet", "ksize", "setting", "out")
               if getattr(a, x) is None]
    if missing or not (a.nofit or a.pairs):
        sys.exit(f"missing arguments: {missing or '--pairs or --nofit'}")
    base = {"alphabet": a.alphabet, "ksize": int(a.ksize), "setting": a.setting, "penalty": a.penalty}

    def write(rows):
        pl.DataFrame(rows).write_csv(a.out, separator="\t")

    if a.nofit:
        write([{**base, "status": "nofit"}])
        return
    su = load_utils(a.utils)

    ref = set(pl.read_csv(a.ref_rocx, separator="\t")["NAME"])
    dom = pl.read_csv(a.domains, separator="\t")
    missing_ref = ref - set(dom["domain_id"])
    if missing_ref:
        sys.exit(f"{len(missing_ref)} FoldSeek reference queries are not in the FASTA")

    with open(a.pairs) as fh:
        header = fh.readline().strip().split(",")
    if header[:2] != ["query_name", "target_name"]:
        sys.exit(f"{a.pairs} does not start with query_name,target_name: {header[:3]}")
    cols = ["query_name", "target_name"] + [c for c in SCORES if c in header]
    n_rows = a.raw_rows
    # Domain names as an Enum over the SCOPe40 domains: a name outside the FASTA fails the
    # cast loudly, and the 25.6 million pairs of funcgroups8 k7 hold codes, not strings.
    domain = pl.Enum(dom["domain_id"].to_list())
    names = [pl.col(c).cast(domain) for c in ("query_name", "target_name")]
    # collapse_pairs.awk merged each pair's adjacent rows; this merges any it could not.
    pairs = (
        pl.scan_csv(a.pairs, schema_overrides={c: pl.Float64 for c in cols[2:]}, null_values=[""])
        .select(cols)
        .with_columns(names)
        .group_by("query_name", "target_name")
        .agg([pl.col(c).min() if SCORES[c] else pl.col(c).max() for c in cols[2:]])
        .collect(engine="streaming")
        .rename({"query_name": "query_domain", "target_name": "target_domain"})
    )
    if a.pairs_parquet:
        pairs.write_parquet(a.pairs_parquet)

    if pairs.height == 0:
        write([{**base, "status": "empty", "n_rows": n_rows}])
        return

    # SCOP levels as integer codes; only the four same-level flags are kept per pair.
    levels = ("class", "fold", "superfamily", "family")
    codes = dom.select(
        pl.col("domain_id").cast(domain),
        pl.col("scop_id").cast(pl.Categorical),
        *[pl.col(f"scop_{lvl}").cast(pl.Categorical).to_physical().alias(lvl) for lvl in levels],
    )
    q = codes.rename({"domain_id": "query_domain", "scop_id": "q_scop_id", **{l: f"q_{l}" for l in levels}})
    t = codes.drop("scop_id").rename({"domain_id": "target_domain", **{l: f"t_{l}" for l in levels}})
    df = (
        pairs.filter(pl.col("query_domain") != pl.col("target_domain"))
        .join(q, on="query_domain", how="left")
        .join(t, on="target_domain", how="left")
        .with_columns([(pl.col(f"q_{l}") == pl.col(f"t_{l}")).alias(f"same_{l}") for l in levels])
        .drop([f"{side}_{l}" for side in "qt" for l in levels])
    )
    del pairs
    n_pairs = df.height
    n_queries_hit = df["query_domain"].n_unique()
    keep = ["query_domain", "target_domain", "q_scop_id"] + [f"same_{l}" for l in levels]

    rows = []
    for col in cols[2:]:
        info = {**base, "score": col, "smaller_first": SCORES[col], "n_rows": n_rows,
                "n_pairs": n_pairs, "n_queries_hit": n_queries_hit}
        s = df[col]
        finite = s.filter(s.is_finite()) if s.dtype == pl.Float64 else s.drop_nulls()
        if finite.len() == 0 or finite.n_unique() == 1:
            rows.append({**info, "status": "constant", "n_finite": finite.len()})
            continue
        # A missing or infinite score ranks last (nulls_last in eval_tsv_to_rocx).
        ranked = df.select(keep + [col]).with_columns(
            pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None).alias(col)
        )
        rocx = {}
        for tie in ("pessimistic", "optimistic"):
            r = su.eval_tsv_to_rocx(ranked, score_col=col, ascending=SCORES[col],
                                    exclude_gray_zone=True, tie_break=tie)
            r = r.with_columns(pl.col("NAME").cast(pl.Utf8), pl.col("SCOP").cast(pl.Utf8))
            rocx[tie] = (r, su.rocx_restrict(r, ref))
        r_all, r_ref = rocx["pessimistic"]
        o_all, o_ref = rocx["optimistic"]

        def auc(r, short):
            return su.sensitivity_stats(r, short)[2] if r.height else 0.0

        for short, level in LEVELS.items():
            rows.append({**info, "status": "ok", "n_finite": finite.len(), "level": level,
                         "auc_all": auc(r_all, short), "auc_all_optimistic": auc(o_all, short),
                         "n_queries_all": r_all.height,
                         "auc_ref": auc(r_ref, short), "auc_ref_optimistic": auc(o_ref, short),
                         "n_queries_ref": r_ref.height,
                         "n_ref_with_hits": r_all.filter(pl.col("NAME").is_in(ref)).height})
    write(rows)


if __name__ == "__main__":
    main()
