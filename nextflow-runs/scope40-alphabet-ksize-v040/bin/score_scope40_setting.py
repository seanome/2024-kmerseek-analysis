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
A query's sensitivity divides by the true positives kmerseek reported for it (as in
eval_tsv_to_rocx). FoldSeek's .rocx divides by every true positive in the database
(d1b9ha_: SFAM 0.992366 = 129 / (143 - 13)), so auc_ref_db and auc_ref_db_optimistic divide
by the database counts too, excluding the query: family members, superfamily members outside
the family, fold members outside the superfamily, class members outside the fold.
With --class-level, a fourth level, class: true positive = same SCOP class, different fold;
false positive = a different class; a same-fold hit is ignored. It has no FoldSeek or MMseqs2
counterpart (the TEA .rocx files stop at fold).
With --pair-identity, each score column is also scored on a set of aligned SCOPe40 pairs
split by sequence identity (assets/scope40_pair_identity.tsv.gz, notebook 230's USalign
pairs, each in both directions). Per score column, pair class and identity band it writes
the number of pairs, how many kmerseek reported, and how many it ranked above the query's
first different-fold hit (a tie counts against kmerseek, as in auc_ref).
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
DB_COUNT = {"FAM": "db_family", "SFAM": "db_superfamily", "FOLD": "db_fold", "CLASS": "db_class"}


def database_counts(dom):
    """Per domain: true positives in the database at each level, the query itself left out."""
    n = {lvl: pl.len().over(f"scop_{lvl}") for lvl in ("family", "superfamily", "fold", "class")}
    return dom.select(
        pl.col("domain_id").alias("NAME"),
        (n["family"] - 1).alias("db_family"),
        (n["superfamily"] - n["family"]).alias("db_superfamily"),
        (n["fold"] - n["superfamily"]).alias("db_fold"),
        (n["class"] - n["fold"]).alias("db_class"),
    )


def class_rocx(ranked, col, smaller_first, tie):
    """Per query, class-level sensitivity to the first different-class hit.

    TP = same class, different fold; FP = different class; same-fold hits are skipped. Columns
    follow eval_tsv_to_rocx: CLASS divides by the class TPs kmerseek reported, CLASSTP is the
    number of TPs ranked above the first FP (all of them when there is no FP).
    """
    r = (ranked.filter(~pl.col("same_fold") | ~pl.col("same_class"))
         .sort(["query_domain", col, "same_class", "target_domain"],
               descending=[False, not smaller_first, tie == "optimistic", False],
               nulls_last=True, maintain_order=True))
    tp = pl.col("same_class") & ~pl.col("same_fold")
    before = (~pl.col("same_class")).cum_sum() == 0
    st = r.group_by("query_domain", maintain_order=True).agg(
        tp.sum().alias("n"), (~pl.col("same_class")).any().alias("has_fp"), (tp & before).sum().alias("tp_bfp"))
    return st.select(
        pl.col("query_domain").cast(pl.Utf8).alias("NAME"),
        pl.when(pl.col("n") == 0).then(0.0).when(~pl.col("has_fp")).then(1.0)
        .otherwise(pl.col("tp_bfp") / pl.col("n")).alias("CLASS"),
        pl.col("tp_bfp").cast(pl.Float64).alias("CLASSTP"),
    )


def with_db(r, counts, short):
    """Add <short>_db: the TPs ranked above the first FP over the database count."""
    tp = pl.col("CLASSTP") if short == "CLASS" else pl.col(short) * pl.col(f"{short}CNT")
    return (r.join(counts, on="NAME", how="left")
            .with_columns(pl.when(pl.col(DB_COUNT[short]) > 0)
                          .then((tp / pl.col(DB_COUNT[short])).clip(0.0, 1.0)).otherwise(0.0)
                          .fill_null(0.0).alias(f"{short}_db")))
# Identity bands, as in the identity-axis tables of the midi run: lower edge included.
BAND_EDGES = [0.0, 0.2, 0.3, 0.4, 0.6]
BAND_NAMES = ["0-20%", "20-30%", "30-40%", "40-60%", "60-100%"]


def read_identity(path, domain):
    """The identity pairs, bands assigned, restricted to domains in this run's FASTA."""
    ident = pl.read_csv(path, separator="\t")
    ident = ident.filter(pl.col("query_domain").is_in(domain.categories.to_list())
                         & pl.col("target_domain").is_in(domain.categories.to_list()))
    return ident.with_columns(
        pl.col("query_domain").cast(domain), pl.col("target_domain").cast(domain),
        pl.col("seqid_ali").cut(BAND_EDGES[1:], labels=BAND_NAMES, left_closed=True)
        .cast(pl.Utf8).alias("identity_band"),
    )


def band_rows(df, ident, col, smaller_first):
    """Per pair class and identity band: pairs, reported, ranked above the first false positive.

    A pair is above the first false positive when its score is finite and strictly better than
    the best finite score of any false-positive hit of the same query, or the query has none:
    a different-fold hit, or for a same-class different-fold pair a different-class hit.
    Equal scores count against kmerseek.
    """
    better = (lambda a, b: a < b) if smaller_first else (lambda a, b: a > b)
    s = pl.col(col)
    best = s.min() if smaller_first else s.max()
    fp = (df.filter(~pl.col("same_fold"), s.is_finite()).group_by("query_domain")
          .agg(best.alias("fp_fold"))
          .join(df.filter(~pl.col("same_class"), s.is_finite()).group_by("query_domain")
                .agg(best.alias("fp_class")), on="query_domain", how="full", coalesce=True))
    hits = df.select("query_domain", "target_domain", s.alias("score"))
    j = (ident.join(hits, on=["query_domain", "target_domain"], how="left")
         .join(fp, on="query_domain", how="left")
         # Class pairs (different fold) are cross-fold hits themselves: their false positive
         # is the first hit of a different class.
         .with_columns(pl.when(pl.col("category") == "diff_fold_same_class").then(pl.col("fp_class"))
                       .otherwise(pl.col("fp_fold")).alias("fp_score"))
         .with_columns(
             pl.col("score").is_not_null().alias("reported"),
             (pl.col("score").is_finite().fill_null(False)
              & (pl.col("fp_score").is_null() | better(pl.col("score"), pl.col("fp_score"))).fill_null(False))
             .alias("above_first_fp")))
    return (j.group_by("category", "identity_band")
            .agg(pl.len().alias("n_pairs"), pl.col("reported").sum().alias("n_reported"),
                 pl.col("above_first_fp").sum().alias("n_above_first_fp"),
                 pl.col("query_domain").n_unique().alias("n_queries"))
            .sort("category", "identity_band").to_dicts())


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
    p.add_argument("--pair-identity", help="aligned pairs with seqid_ali, to score by identity band")
    p.add_argument("--bands-out", help="where the identity-band table goes (with --pair-identity)")
    p.add_argument("--class-level", action="store_true", help="also score the class level")
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

    if bool(a.pair_identity) != bool(a.bands_out):
        sys.exit("--pair-identity and --bands-out go together")

    def write(rows, band=None):
        pl.DataFrame(rows).write_csv(a.out, separator="\t")
        if a.bands_out:
            pl.DataFrame(band if band else [{**rows[0]}]).write_csv(a.bands_out, separator="\t")

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

    ident = read_identity(a.pair_identity, domain) if a.pair_identity else None
    counts = database_counts(dom)
    rows, band = [], []
    for col in cols[2:]:
        info = {**base, "score": col, "smaller_first": SCORES[col], "n_rows": n_rows,
                "n_pairs": n_pairs, "n_queries_hit": n_queries_hit}
        s = df[col]
        finite = s.filter(s.is_finite()) if s.dtype == pl.Float64 else s.drop_nulls()
        if finite.len() == 0 or finite.n_unique() == 1:
            rows.append({**info, "status": "constant", "n_finite": finite.len()})
            if ident is not None:
                band.append({**info, "status": "constant"})
            continue
        if ident is not None:
            band += [{**info, "status": "ok", **r} for r in band_rows(df, ident, col, SCORES[col])]
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

        levels = dict(LEVELS)
        if a.class_level:
            levels["CLASS"] = "class"
            cls = {}
            for tie, (r, _) in rocx.items():
                c = class_rocx(ranked, col, SCORES[col], tie).join(r.select("NAME"), on="NAME", how="semi")
                # rocx_restrict gives a query kmerseek missed a CLASS of 0; its count joins back as 0.
                cr = (su.rocx_restrict(c.select("NAME", pl.lit("").alias("SCOP"), "CLASS"), ref)
                      .join(c.select("NAME", "CLASSTP"), on="NAME", how="left").fill_null(0.0))
                cls[tie] = (c, cr)
        for short, level in levels.items():
            if short == "CLASS":
                (ra, rr), (oa, orr) = cls["pessimistic"], cls["optimistic"]
            else:
                ra, rr, oa, orr = r_all, r_ref, o_all, o_ref
            rows.append({**info, "status": "ok", "n_finite": finite.len(), "level": level,
                         "auc_all": auc(ra, short), "auc_all_optimistic": auc(oa, short),
                         "n_queries_all": ra.height,
                         "auc_ref": auc(rr, short), "auc_ref_optimistic": auc(orr, short),
                         "n_queries_ref": rr.height,
                         "n_ref_with_hits": r_all.filter(pl.col("NAME").is_in(ref)).height,
                         "auc_ref_db": auc(with_db(rr, counts, short), f"{short}_db"),
                         "auc_ref_db_optimistic": auc(with_db(orr, counts, short), f"{short}_db")})
    write(rows, band)


if __name__ == "__main__":
    main()
