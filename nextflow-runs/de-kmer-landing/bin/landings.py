#!/usr/bin/env python3
"""Where one kmerseek setting's regions land on Pfam domains, and the score per labelling.

K-mer path steps 6 and 7 and "Score" in notebooks/360_data/preregistered.md. A region
lands on a Pfam domain of its target when at least half of [target_start, target_end)
lies inside the domain (both 0-based, end exclusive) and region_evalue <= 1. The changed
domains of a labelling are the families its peptides land on.

    landings.py --regions regions.csv --domains search_db.pfam.tsv \
        --peptide-map peptide_map.tsv --key answer_key.tsv --labellings labellings.tsv \
        --organism ciona --setting dayhoff6.k9 --status fitted --out-prefix ciona.dayhoff6.k9

Writes <out-prefix>.landings.tsv (one row per region x domain landed on, scores kept)
and <out-prefix>.summary.tsv: one row per labelling and E-value cutoff. The cutoff 1 row
is the preregistered one (`primary` true); the others are reported, not used in the rule.
"""

from __future__ import annotations

import argparse

import polars as pl

MIN_INSIDE = 0.5
PRIMARY_EVALUE = 1.0
EVALUE_CUTOFFS = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--regions", "--domains", "--peptide-map", "--key", "--labellings", "--organism", "--setting", "--status", "--out-prefix"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()

    key = pl.read_csv(args.key, separator="\t")
    key_fams = set(key.filter(pl.col("status") != "unchanged")["pfam_acc"].to_list())
    assert len(key_fams) == 100, len(key_fams)
    labellings = pl.read_csv(args.labellings, separator="\t")["labelling"].to_list()
    pmap = pl.read_csv(args.peptide_map, separator="\t").select("peptide_id", "labelling").unique()
    n_queries = pmap.group_by("labelling").agg(pl.col("peptide_id").n_unique().alias("n_queries"))

    cols = ["query_name", "target_name", "target_start", "target_end", "region_start", "region_end",
            "region_evalue", "region_ka_bits", "region_poisson_score", "region_n_shared_kmers"]
    regions = pl.read_csv(args.regions, columns=None, infer_schema_length=0)
    missing = [c for c in cols if c not in regions.columns]
    if missing and regions.height:
        raise SystemExit(f"columns missing from kmerseek output: {missing}")
    if regions.height:
        regions = regions.select(cols).with_columns(
            pl.col("target_start", "target_end", "region_start", "region_end", "region_n_shared_kmers").cast(pl.Int64),
            pl.col("region_evalue", "region_ka_bits", "region_poisson_score").cast(pl.Float64),
        )
    else:
        ints = ("target_start", "target_end", "region_start", "region_end", "region_n_shared_kmers")
        regions = pl.DataFrame(schema={c: pl.Utf8 if c in ("query_name", "target_name") else pl.Int64 if c in ints else pl.Float64 for c in cols})

    dom = pl.read_csv(args.domains, separator="\t").rename({"target": "target_name", "start0": "dom_start", "end": "dom_end"})
    hit = regions.join(dom, on="target_name", how="inner").with_columns(
        (pl.min_horizontal("target_end", "dom_end") - pl.max_horizontal("target_start", "dom_start")).clip(lower_bound=0).alias("overlap"),
        (pl.col("target_end") - pl.col("target_start")).alias("region_target_length"),
    ).with_columns((pl.col("overlap") / pl.col("region_target_length")).alias("inside"))
    landed = hit.filter(pl.col("inside") >= MIN_INSIDE)
    landed = landed.join(pmap.rename({"peptide_id": "query_name"}), on="query_name", how="inner").with_columns(
        pl.col("pfam_acc").is_in(list(key_fams)).alias("in_key"),
        pl.lit(args.organism).alias("organism"),
        pl.lit(args.setting).alias("setting"),
    )
    landed.sort("labelling", "query_name", "target_name", "target_start", "pfam_acc").write_csv(
        f"{args.out_prefix}.landings.tsv", separator="\t"
    )

    rows = []
    for cut in EVALUE_CUTOFFS:
        sub = landed.filter(pl.col("region_evalue") <= cut)
        for lab in labellings:
            fams = set(sub.filter(pl.col("labelling") == lab)["pfam_acc"].to_list())
            nq = n_queries.filter(pl.col("labelling") == lab)["n_queries"]
            rows.append(dict(
                organism=args.organism, path="kmerseek", setting=args.setting, status=args.status,
                labelling=lab, evalue_cutoff=cut, primary=cut == PRIMARY_EVALUE,
                recovered=len(fams & key_fams), recovery_fraction=len(fams & key_fams) / 100,
                outside=len(fams - key_fams), n_changed_domains=len(fams),
                n_queries=int(nq[0]) if nq.len() else 0,
                n_landing_rows=sub.filter(pl.col("labelling") == lab).height,
            ))
    pl.DataFrame(rows).write_csv(f"{args.out_prefix}.summary.tsv", separator="\t")


if __name__ == "__main__":
    main()
