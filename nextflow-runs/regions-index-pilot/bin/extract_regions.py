#!/usr/bin/env python3
"""Cut every annotated region out of the whole-protein index's entries.

The regions index holds the same Swiss-Prot entries as the whole-protein index (reviewed,
Mammalia removed: the accession list build_clade_excluded_reference.py wrote), but only the
parts of them that carry a feature of these types: DOMAIN, REGION, MOTIF, REPEAT, ZN_FING,
COMPBIAS, COILED, TRANSMEM, INTRAMEM.

Each feature is cut out with (k_max - 1) residues on each side, clipped at the protein
ends, so that a k-mer of the largest k-size can start at the feature's first residue and
end at its last. Each entry is named

    accession|feature type|description|start-end

with start-end the feature's own 1-based inclusive coordinates in the source protein (not
the cut-out's). In the description, whitespace becomes `_`, and `|` and `,` become `/` and
`;`, because MMseqs2 reports the first whitespace token and kmerseek writes the full header
into a CSV.

Outputs:
  regions_long.fasta     features of --min-cluster-aa or more, for MMseqs2 clustering
  regions_short.fasta    features under --min-cluster-aa, one record per distinct cut-out
                         sequence (exact duplicates removed); the kept record is the first
                         by name
  regions.parquet        every cut-out: name, accession, feature_type, description, start,
                         end, feature_length, cut_start, cut_end, cut_length, is_long,
                         short_rep (for a short one, the name of the record it collapsed
                         onto), sequence
"""

import argparse
import re
import sys
from pathlib import Path

import polars as pl

FEATURE_TYPES = [
    "DOMAIN",
    "REGION",
    "MOTIF",
    "REPEAT",
    "ZN_FING",
    "COMPBIAS",
    "COILED",
    "TRANSMEM",
    "INTRAMEM",
]
WS_RE = re.compile(r"\s+")


def clean(desc: str) -> str:
    d = WS_RE.sub("_", desc.strip()).replace("|", "/").replace(",", ";")
    return d or "none"


def write_fasta(path: Path, rows) -> int:
    n = 0
    with open(path, "w") as fh:
        for name, seq in rows:
            fh.write(f">{name}\n{seq}\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument(
        "--accessions",
        type=Path,
        required=True,
        help="the whole-protein index's accession list",
    )
    ap.add_argument("--k-max", type=int, required=True)
    ap.add_argument("--min-cluster-aa", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flank = args.k_max - 1

    keep = set(Path(args.accessions).read_text().split())
    seqs = pl.read_parquet(
        args.sequences, columns=["accession", "is_mammal", "sequence"]
    ).filter(pl.col("accession").is_in(list(keep)))
    if seqs.height != len(keep):
        sys.exit(
            f"{len(keep) - seqs.height} reference accessions have no sequence: the "
            f"reference and the feature table come from different flat files"
        )
    if seqs["is_mammal"].any():
        sys.exit("a mammal is in the reference accession list")

    feats = (
        pl.read_parquet(args.features)
        .filter(pl.col("feature_type").is_in(FEATURE_TYPES))
        .join(seqs.select("accession", "sequence"), on="accession", how="inner")
    )
    feats = (
        feats.with_columns(
            pl.max_horizontal(pl.lit(1), pl.col("start") - flank).alias("cut_start"),
            pl.min_horizontal(
                pl.col("sequence").str.len_chars(), pl.col("end") + flank
            ).alias("cut_end"),
            pl.col("description")
            .map_elements(clean, return_dtype=pl.Utf8)
            .alias("desc_clean"),
        )
        .with_columns(
            pl.col("sequence")
            .str.slice(
                pl.col("cut_start") - 1, pl.col("cut_end") - pl.col("cut_start") + 1
            )
            .alias("cut_seq"),
            pl.concat_str(
                [
                    pl.col("accession"),
                    pl.col("feature_type"),
                    pl.col("desc_clean"),
                    pl.concat_str([pl.col("start"), pl.col("end")], separator="-"),
                ],
                separator="|",
            ).alias("name"),
            (pl.col("length") >= args.min_cluster_aa).alias("is_long"),
        )
        .with_columns(pl.col("cut_seq").str.len_chars().alias("cut_length"))
    )

    # The same feature listed twice on one entry (same type, note and coordinates) is one
    # region, not two.
    n_before = feats.height
    feats = feats.unique("name", keep="first", maintain_order=True).sort("name")
    n_dup_names = n_before - feats.height

    bad = feats.filter(
        pl.col("cut_length") != pl.col("cut_end") - pl.col("cut_start") + 1
    )
    if bad.height:
        sys.exit(f"{bad.height} cut-outs have the wrong length, e.g. {bad['name'][0]}")

    short = feats.filter(~pl.col("is_long"))
    short_rep = short.group_by("cut_seq", maintain_order=True).agg(
        pl.col("name").min().alias("short_rep")
    )
    feats = feats.join(short_rep, on="cut_seq", how="left").with_columns(
        pl.when(pl.col("is_long"))
        .then(None)
        .otherwise(pl.col("short_rep"))
        .alias("short_rep")
    )

    long_rows = feats.filter(pl.col("is_long")).select("name", "cut_seq").iter_rows()
    short_rows = (
        feats.filter(~pl.col("is_long") & (pl.col("name") == pl.col("short_rep")))
        .select("name", "cut_seq")
        .iter_rows()
    )
    n_long = write_fasta(args.out_dir / "regions_long.fasta", long_rows)
    n_short = write_fasta(args.out_dir / "regions_short.fasta", short_rows)

    (
        feats.select(
            "name",
            "accession",
            "feature_type",
            "description",
            "start",
            "end",
            pl.col("length").alias("feature_length"),
            "cut_start",
            "cut_end",
            "cut_length",
            "is_long",
            "short_rep",
            "experimental",
            pl.col("cut_seq").alias("sequence"),
        ).write_parquet(args.out_dir / "regions.parquet", compression="zstd")
    )

    print(
        f"[regions] k_max={args.k_max} flank={flank} reference_entries={seqs.height} "
        f"features={feats.height} (duplicate names dropped: {n_dup_names}) "
        f"long={n_long} short={short.height} short_distinct={n_short}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
