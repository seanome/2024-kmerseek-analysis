#!/usr/bin/env python3
"""Carve the committed CI fixture out of the local mini data set.

Run by hand when the fixture needs regenerating, never by CI -- CI reads what this
commits. The point of committing it is that the test needs no QfO download, no AlphaFold
structures and no Pfam-A: 60-odd proteins of real sequence and real annotation, small
enough to live in git.

Proteins are chosen to cover the things the scoring path can get wrong, not at random:

  multi-instance proteins   several domains of DIFFERENT families in one protein, which is
                            what makes an instance-level stratum able to disagree with a
                            protein-level one. The recall-above-1.0 bug needed exactly this
                            shape to show up.
  varied domain length      instances spread across the feature_length_bin edges, so the
                            axis has more than one populated bin to compare.
  transferable families     every kept human family also occurs in the yeast target, or the
                            answer key would be unreachable and every metric would read 0.0
                            whether the code worked or not.
"""

import argparse
import shutil
from pathlib import Path

import polars as pl

MINI = Path("/Users/olga/data/qfo-pfam-region-benchmark-mini")


def read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    """accession -> (header line without '>', sequence)."""
    recs, name, hdr, buf = {}, None, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    recs[name] = (hdr, "".join(buf))
                hdr = line[1:].rstrip("\n")
                parts = hdr.split("|")
                name = parts[1] if len(parts) >= 2 else hdr.split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        recs[name] = (hdr, "".join(buf))
    return recs


def write_fasta(path: Path, recs: dict, keep: set) -> None:
    with open(path, "w") as f:
        for acc in sorted(keep):
            if acc not in recs:
                continue
            hdr, seq = recs[acc]
            f.write(f">{hdr}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mini", type=Path, default=MINI)
    p.add_argument("--outdir", type=Path, default=Path("tests/fixtures"))
    p.add_argument("--n-human", type=int, default=60)
    p.add_argument("--n-yeast", type=int, default=120)
    args = p.parse_args()

    ann = args.mini / "annotations"
    human = pl.read_parquet(ann / "human_pfam_domains.parquet").filter("has_position")
    yeast = pl.read_parquet(ann / "yeast_pfam_domains.parquet").filter("has_position")

    # Only families the target actually carries. An unreachable family makes every metric
    # 0.0 regardless of whether the code is right, which is the one thing a test must not
    # be unable to distinguish.
    shared = human.join(yeast.select("pfam_id").unique(), on="pfam_id", how="inner")

    per_protein = (
        shared.group_by("accession")
        .agg(pl.len().alias("n_dom"),
             pl.col("pfam_id").n_unique().alias("n_fam"),
             pl.col("domain_length").min().alias("min_len"),
             pl.col("domain_length").max().alias("max_len"))
    )
    # Multi-family proteins first: they are what separates an instance-level cut from a
    # protein-level one. Then fill up with the rest, widest length spread first.
    multi = per_protein.filter(pl.col("n_fam") > 1).sort("n_fam", descending=True)
    single = per_protein.filter(pl.col("n_fam") == 1).sort(
        (pl.col("max_len") - pl.col("min_len")), descending=True)
    keep_h = (multi["accession"].to_list() + single["accession"].to_list())[:args.n_human]
    keep_h = set(keep_h)

    h_ann = shared.filter(pl.col("accession").is_in(keep_h))
    fams = set(h_ann["pfam_id"].unique().to_list())

    # Yeast side: proteins carrying those families, so a transfer has somewhere to land.
    y_hits = yeast.filter(pl.col("pfam_id").is_in(fams))
    keep_y = set(y_hits["accession"].unique().to_list()[:args.n_yeast])
    y_ann = yeast.filter(pl.col("accession").is_in(keep_y))

    out = args.outdir
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "qfo").mkdir(parents=True, exist_ok=True)
    h_ann.write_parquet(out / "annotations/human_pfam_domains.parquet", compression="zstd")
    y_ann.write_parquet(out / "annotations/yeast_pfam_domains.parquet", compression="zstd")

    h_fa = read_fasta(args.mini / "qfo/Eukaryota/UP000005640_9606.fasta")
    y_fa = read_fasta(args.mini / "qfo/Eukaryota/UP000002311_559292.fasta")
    write_fasta(out / "qfo/human.fasta", h_fa, keep_h)
    write_fasta(out / "qfo/yeast.fasta", y_fa, keep_y)

    lens = h_ann["domain_length"]
    print(f"human proteins : {h_ann['accession'].n_unique()}")
    print(f"human instances: {h_ann.height} over {len(fams)} families")
    print(f"  multi-family : {multi.filter(pl.col('accession').is_in(keep_h)).height}")
    print(f"  domain length: {lens.min()} .. {lens.max()} (median {lens.median():.0f})")
    print(f"yeast proteins : {y_ann['accession'].n_unique()}, {y_ann.height} instances")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"fixture size   : {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
