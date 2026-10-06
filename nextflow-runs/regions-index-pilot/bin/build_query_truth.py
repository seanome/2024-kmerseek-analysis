#!/usr/bin/env python3
"""Queries and their truth table: reviewed human chromosome 6 proteins and their own features.

Inputs: the feature and sequence tables from parse_swissprot_features.py, the chromosome 6
accession list (assets/chr6_queries.<release>.tsv) and DisProt consensus disorder
(assets/disprot_human_disorder.tsv).

Outputs:
  queries.fasta        one record per query, header = accession alone
  truth.parquet        one row per query feature (columns below)
  truth_summary.tsv    feature counts per kind and evidence class

Each truth row carries:
  feature_kind   folded_domain     DOMAIN, REPEAT or ZN_FING of 60 aa or more
                 motif             MOTIF
                 disordered        REGION "Disordered" whose evidence is experimental, or
                                   with at least half its residues inside DisProt consensus
                                   disorder
                 composition       TRANSMEM, INTRAMEM, COILED, COMPBIAS
                 other             everything else: other REGIONs, a "Disordered" REGION that
                                   fails the rule above, DOMAIN/REPEAT/ZN_FING under 60 aa
  is_short       feature under 60 aa (any kind; reported as its own row, overlapping the kinds)
  experimental   any ECO:0000269 on the feature
  disprot_frac   share of the feature's residues inside DisProt consensus disorder
  gene_group     MHC (HLA-*), histone (H1-*, H2AC*, H2BC*, H3C*, H4C*), olfactory receptor
                 (OR<digits><letters><digits>), or none
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

COMPOSITION = ["TRANSMEM", "INTRAMEM", "COILED", "COMPBIAS"]
FOLDED = ["DOMAIN", "REPEAT", "ZN_FING"]
SHORT_AA = 60
DISPROT_MIN_FRAC = 0.5

HISTONE_RE = re.compile(r"^(H1-|H2AC\d|H2BC\d|H3C\d|H4C\d)")
OR_RE = re.compile(r"^OR\d+[A-Z]+\d+")


def gene_group(gene: str | None) -> str:
    g = gene or ""
    if g.startswith("HLA-"):
        return "MHC"
    if HISTONE_RE.match(g):
        return "histone"
    if OR_RE.match(g):
        return "olfactory_receptor"
    return "none"


def disprot_fraction(truth: pl.DataFrame, disprot: pl.DataFrame,
                     lengths: dict[str, int]) -> list[float]:
    """Share of each feature's residues that DisProt calls disordered."""
    masks: dict[str, np.ndarray] = {}
    for acc, start, end in disprot.select("accession", "start", "end").iter_rows():
        if acc not in lengths:
            continue
        m = masks.setdefault(acc, np.zeros(lengths[acc] + 1, dtype=bool))
        m[start:min(end, lengths[acc]) + 1] = True
    out = []
    for acc, start, end in truth.select("accession", "start", "end").iter_rows():
        m = masks.get(acc)
        out.append(0.0 if m is None else float(m[start:end + 1].mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--chr6", type=Path, required=True)
    ap.add_argument("--disprot", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chr6 = pl.read_csv(args.chr6, separator="\t", schema_overrides={"gene": pl.Utf8})
    seqs = pl.read_parquet(args.sequences).join(chr6.select("accession", "gene"),
                                                on="accession", how="inner")
    missing = set(chr6["accession"]) - set(seqs["accession"])
    if missing:
        sys.exit(f"{len(missing)} chromosome 6 accessions are not in the flat file, e.g. "
                 f"{sorted(missing)[:5]}: the query list and the flat file are from "
                 f"different releases")
    if not seqs["is_mammal"].all():
        sys.exit("a human query is not flagged Mammalia; the lineage parse is wrong")

    with open(args.out_dir / "queries.fasta", "w") as fh:
        for acc, seq in seqs.sort("accession").select("accession", "sequence").iter_rows():
            fh.write(f">{acc}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    truth = (pl.read_parquet(args.features)
             .join(chr6.select("accession", "gene"), on="accession", how="inner")
             .drop("is_mammal")
             .sort("accession", "start", "end", "feature_type"))
    lengths = dict(seqs.select("accession", "length").iter_rows())
    disprot = pl.read_csv(args.disprot, separator="\t")
    truth = truth.with_columns(
        pl.Series("disprot_frac", disprot_fraction(truth, disprot, lengths)),
        pl.col("gene").map_elements(gene_group, return_dtype=pl.Utf8).alias("gene_group"),
        (pl.col("length") < SHORT_AA).alias("is_short"),
    )
    is_dis = (pl.col("feature_type") == "REGION") & (pl.col("description") == "Disordered")
    truth = truth.with_columns(
        pl.when(pl.col("feature_type").is_in(COMPOSITION)).then(pl.lit("composition"))
        .when(pl.col("feature_type") == "MOTIF").then(pl.lit("motif"))
        .when(is_dis & (pl.col("experimental")
                        | (pl.col("disprot_frac") >= DISPROT_MIN_FRAC)))
        .then(pl.lit("disordered"))
        .when(pl.col("feature_type").is_in(FOLDED) & (pl.col("length") >= SHORT_AA))
        .then(pl.lit("folded_domain"))
        .otherwise(pl.lit("other"))
        .alias("feature_kind"),
        (is_dis & ~pl.col("experimental")
         & (pl.col("disprot_frac") < DISPROT_MIN_FRAC)).alias("disordered_unconfirmed"),
    ).with_row_index("truth_id")
    truth.write_parquet(args.out_dir / "truth.parquet")

    rows = []
    groups = {k: pl.col("feature_kind") == k for k in
              ["folded_domain", "motif", "disordered", "composition", "other"]}
    groups["short (<60 aa, any kind)"] = pl.col("is_short")
    for name, cond in groups.items():
        sub = truth.filter(cond)
        rows.append({"feature_kind": name, "all_evidence": sub.height,
                     "experimental": int(sub["experimental"].sum()),
                     "proteins": sub["accession"].n_unique(),
                     "median_length_aa": float(sub["length"].median() or 0)})
    summary = pl.DataFrame(rows)
    summary.write_csv(args.out_dir / "truth_summary.tsv", separator="\t")
    print(f"[truth] queries={seqs.height} features={truth.height} "
          f"proteins_with_features={truth['accession'].n_unique()} "
          f"disordered_unconfirmed={int(truth['disordered_unconfirmed'].sum())}",
          file=sys.stderr)
    with pl.Config(tbl_rows=20, tbl_width_chars=150):
        print(summary, file=sys.stderr)
        print(truth.group_by("gene_group").agg(pl.col("accession").n_unique().alias("proteins"),
                                                pl.len().alias("features")).sort("gene_group"),
              file=sys.stderr)


if __name__ == "__main__":
    main()
