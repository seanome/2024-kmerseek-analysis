#!/usr/bin/env python3
"""Assemble the regions index from the clustered long cut-outs and the deduplicated short ones.

Inputs: regions.parquet (extract_regions.py), MMseqs2 easy-cluster's <prefix>_cluster.tsv
(representative, member) over regions_long.fasta, and regions_short.fasta.

Outputs:
  regions_index.fasta     the cluster representatives plus every distinct short cut-out
  cluster_members.parquet one row per cut-out: member, rep (the index entry it is
                          represented by), how (mmseqs_cluster or exact_duplicate),
                          member_type, rep_type, cluster_size, cluster_n_types,
                          types_agree (every member of the cluster has the rep's type)
  cluster_summary.tsv     per feature type: cut-outs, index entries, clusters whose
                          members all share the type, and those that mix types
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--cluster-tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    reg = pl.read_parquet(args.regions)
    types = reg.select(pl.col("name"), pl.col("feature_type"))

    clu = pl.read_csv(args.cluster_tsv, separator="\t", has_header=False,
                      new_columns=["rep", "member"], quote_char=None)
    n_long = reg.filter(pl.col("is_long")).height
    if clu.height != n_long or clu["member"].n_unique() != n_long:
        sys.exit(f"cluster table has {clu.height} rows for {n_long} long cut-outs")
    long_m = clu.with_columns(pl.lit("mmseqs_cluster").alias("how"))
    short_m = (reg.filter(~pl.col("is_long"))
               .select(pl.col("short_rep").alias("rep"), pl.col("name").alias("member"))
               .with_columns(pl.lit("exact_duplicate").alias("how")))
    members = pl.concat([long_m, short_m])
    members = (members
               .join(types.rename({"name": "member", "feature_type": "member_type"}),
                     on="member", how="left")
               .join(types.rename({"name": "rep", "feature_type": "rep_type"}),
                     on="rep", how="left"))
    if members["member_type"].null_count() or members["rep_type"].null_count():
        sys.exit("a cluster member or representative is not in regions.parquet")
    members = members.with_columns(
        pl.len().over("rep").alias("cluster_size"),
        pl.col("member_type").n_unique().over("rep").alias("cluster_n_types"),
    ).with_columns((pl.col("cluster_n_types") == 1).alias("types_agree"))
    members.write_parquet(args.out_dir / "cluster_members.parquet")

    reps = set(members["rep"].unique())
    seq_of = dict(reg.filter(pl.col("name").is_in(list(reps)))
                  .select("name", "sequence").iter_rows())
    if len(seq_of) != len(reps):
        sys.exit("a representative has no sequence")
    with open(args.out_dir / "regions_index.fasta", "w") as fh:
        for name in sorted(reps):
            fh.write(f">{name}\n{seq_of[name]}\n")

    per_rep = members.unique("rep").select("rep", "rep_type", "how", "cluster_size",
                                           "types_agree")
    summary = (members.group_by("member_type").agg(pl.len().alias("cut_outs"))
               .rename({"member_type": "feature_type"})
               .join(per_rep.group_by("rep_type").agg(
                   pl.len().alias("index_entries"),
                   (pl.col("how") == "mmseqs_cluster").sum().alias("from_clusters"),
                   (pl.col("how") == "exact_duplicate").sum().alias("from_short_dedup"),
                   ((pl.col("cluster_size") > 1) & pl.col("types_agree")).sum()
                   .alias("multi_member_same_type"),
                   (~pl.col("types_agree")).sum().alias("mixed_type"),
               ).rename({"rep_type": "feature_type"}), on="feature_type", how="left")
               .sort("feature_type"))
    summary.write_csv(args.out_dir / "cluster_summary.tsv", separator="\t")
    print(f"[regions_index] cut_outs={members.height} index_entries={len(reps)} "
          f"residues={sum(len(s) for s in seq_of.values())} "
          f"mixed_type_clusters={int((~per_rep['types_agree']).sum())}", file=sys.stderr)
    with pl.Config(tbl_rows=20, tbl_width_chars=160):
        print(summary, file=sys.stderr)


if __name__ == "__main__":
    main()
