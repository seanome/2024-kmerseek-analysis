#!/usr/bin/env python3
"""Cut every Pfam domain out of the whole-protein index's entries: the Pfam index.

Added 2026-10-15 at Olga's request, as a third index next to the whole-protein and regions
indexes. It holds the same Swiss-Prot entries (reviewed, Mammalia removed: the accession
list build_clade_excluded_reference.py wrote), but only their Pfam domains, so the three
indexes differ in what part of each protein they keep, never in which proteins.

Input: Pfam-A.regions.tsv.gz (one row per Pfam domain on a pfamseq protein; pfamseq is the
UniProtKB reference proteomes, so Swiss-Prot entries outside them have no rows) and
Pfam-A.hmm.dat.gz (each family's name and type), from one Pfam release.

Rules:
- Rows for proteins outside the accession list are dropped: not Swiss-Prot, or a mammal.
- A row is kept only when its md5 matches our sequence: Pfam's coordinates refer to the
  pfamseq sequence, which may differ from the Swiss-Prot release in use. The count dropped
  for a changed sequence is printed.
- The domain is cut out from seq_start to seq_end (Pfam's envelope) with (k_max - 1)
  residues each side, clipped at the protein ends, as extract_regions.py cuts features.
- The entry's feature type is the Swiss-Prot type matching the family's Pfam type, so the
  scoring compares it with query features as it does for the regions index: Domain and
  Family -> DOMAIN (ZN_FING when the family name starts with "zf-"), Repeat -> REPEAT,
  Motif -> MOTIF, Coiled-coil -> COILED, Disordered -> REGION.
- Each entry is named accession|feature type|PFxxxxx/family name|start-end.

Outputs, in extract_regions.py's format, so the same clustering and finishing steps apply:
  regions_long.fasta, regions_short.fasta, regions.parquet, and pfam_filter.json (counts).
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import polars as pl

TYPE_MAP = {
    "Domain": "DOMAIN",
    "Family": "DOMAIN",
    "Repeat": "REPEAT",
    "Motif": "MOTIF",
    "Coiled-coil": "COILED",
    "Disordered": "REGION",
}
WS_RE = re.compile(r"\s+")


def pfam_families(hmm_dat: Path) -> dict[str, tuple[str, str]]:
    """{PFxxxxx: (family name, Swiss-Prot feature type)}."""
    out, acc, name, tp = {}, None, None, None
    with gzip.open(hmm_dat, "rt") as fh:
        for line in fh:
            if line.startswith("#=GF ID"):
                name = line.split(None, 2)[2].strip()
            elif line.startswith("#=GF AC"):
                acc = line.split(None, 2)[2].strip().split(".")[0]
            elif line.startswith("#=GF TP"):
                tp = line.split(None, 2)[2].strip()
            elif line.startswith("//"):
                if acc and name and tp:
                    ftype = TYPE_MAP[tp]
                    if ftype == "DOMAIN" and name.startswith("zf-"):
                        ftype = "ZN_FING"
                    out[acc] = (name, ftype)
                acc = name = tp = None
    return out


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
    ap.add_argument("--pfam-regions", type=Path, required=True)
    ap.add_argument("--pfam-hmm-dat", type=Path, required=True)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--accessions", type=Path, required=True)
    ap.add_argument("--k-max", type=int, required=True)
    ap.add_argument("--min-cluster-aa", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flank = args.k_max - 1

    keep = set(args.accessions.read_text().split())
    seqs = dict(
        pl.read_parquet(args.sequences, columns=["accession", "sequence"])
        .filter(pl.col("accession").is_in(list(keep)))
        .iter_rows()
    )
    md5 = {a: hashlib.md5(s.encode()).hexdigest() for a, s in seqs.items()}
    fams = pfam_families(args.pfam_hmm_dat)

    counts = {
        "rows": 0,
        "not_in_index": 0,
        "sequence_changed": 0,
        "unknown_family": 0,
        "bad_coordinates": 0,
        "kept": 0,
    }
    rows = []
    # Streamed line by line: the file is 5 GB compressed, and polars inflates a compressed
    # CSV fully in memory.
    with gzip.open(args.pfam_regions, "rt") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(head)}
        for line in fh:
            counts["rows"] += 1
            f = line.rstrip("\n").split("\t")
            acc = f[ix["pfamseq_acc"]]
            if acc not in keep:
                counts["not_in_index"] += 1
                continue
            if f[ix["md5"]] != md5[acc]:
                counts["sequence_changed"] += 1
                continue
            fam = f[ix["pfamA_acc"]].split(".")[0]
            if fam not in fams:
                counts["unknown_family"] += 1
                continue
            s, e = int(f[ix["seq_start"]]), int(f[ix["seq_end"]])
            seq = seqs[acc]
            if not 1 <= s <= e <= len(seq):
                counts["bad_coordinates"] += 1
                continue
            name, ftype = fams[fam]
            cs, ce = max(1, s - flank), min(len(seq), e + flank)
            desc = WS_RE.sub("_", f"{fam}/{name}").replace("|", "/").replace(",", ";")
            rows.append(
                (
                    f"{acc}|{ftype}|{desc}|{s}-{e}",
                    acc,
                    ftype,
                    f"{fam}/{name}",
                    s,
                    e,
                    e - s + 1,
                    cs,
                    ce,
                    ce - cs + 1,
                    seq[cs - 1 : ce],
                )
            )
            counts["kept"] += 1

    df = (
        pl.DataFrame(
            rows,
            orient="row",
            schema={
                "name": pl.Utf8,
                "accession": pl.Utf8,
                "feature_type": pl.Utf8,
                "description": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "feature_length": pl.Int64,
                "cut_start": pl.Int64,
                "cut_end": pl.Int64,
                "cut_length": pl.Int64,
                "sequence": pl.Utf8,
            },
        )
        .unique("name", keep="first", maintain_order=True)
        .sort("name")
    )
    df = df.with_columns(
        (pl.col("feature_length") >= args.min_cluster_aa).alias("is_long"),
        pl.lit(None, dtype=pl.Boolean).alias("experimental"),
    )
    short = df.filter(~pl.col("is_long"))
    rep = short.group_by("sequence", maintain_order=True).agg(
        pl.col("name").min().alias("short_rep")
    )
    df = df.join(rep, on="sequence", how="left").with_columns(
        pl.when(pl.col("is_long"))
        .then(None)
        .otherwise(pl.col("short_rep"))
        .alias("short_rep")
    )
    n_long = write_fasta(
        args.out_dir / "regions_long.fasta",
        df.filter(pl.col("is_long")).select("name", "sequence").iter_rows(),
    )
    n_short = write_fasta(
        args.out_dir / "regions_short.fasta",
        df.filter(~pl.col("is_long") & (pl.col("name") == pl.col("short_rep")))
        .select("name", "sequence")
        .iter_rows(),
    )
    df.select(
        "name",
        "accession",
        "feature_type",
        "description",
        "start",
        "end",
        "feature_length",
        "cut_start",
        "cut_end",
        "cut_length",
        "is_long",
        "short_rep",
        "experimental",
        "sequence",
    ).write_parquet(args.out_dir / "regions.parquet", compression="zstd")
    counts.update(
        {
            "entries": df.height,
            "long": n_long,
            "short_distinct": n_short,
            "proteins": df["accession"].n_unique(),
            "flank": flank,
            "types": dict(df["feature_type"].value_counts().iter_rows()),
        }
    )
    (args.out_dir / "pfam_filter.json").write_text(json.dumps(counts, indent=2))
    print(f"[pfam] {json.dumps(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
