"""Recompute distinct-word and protein-word counts from saved spectrum histograms."""

import argparse
import csv
import gzip
import re
from pathlib import Path


def audit(data_dir):
    rows = []
    for path in sorted(data_dir.glob("sprot.*.spectrum.csv.gz")):
        with gzip.open(path, "rt") as handle:
            header = dict(re.findall(r"(\w+)=([^\s]+)", handle.readline()))
            histogram = list(csv.DictReader(handle))
        unique = sum(int(row["n_kmers"]) for row in histogram)
        total = sum(int(row["occurrences"]) * int(row["n_kmers"]) for row in histogram)
        if unique != int(header["unique_kmers"]) or total != int(header["total_kmers"]):
            raise ValueError(f"Histogram/header count mismatch: {path}")
        if (
            not unique
            or abs(total / unique - float(header["mean_seqs_per_kmer"])) >= 0.0001
        ):
            raise ValueError(f"Histogram/header mean mismatch: {path}")
        rows.append(
            dict(
                file=path.name,
                moltype=header["moltype"],
                ksize=int(header["ksize"]),
                unique_kmers=unique,
                protein_word_incidences=total,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No spectrum files under {data_dir}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path.home() / "data/kmerseek-kmer-spectra"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("tables/024_spectrum_header_audit.csv")
    )
    args = parser.parse_args()
    rows = audit(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} spectrum histograms against their headers.")


if __name__ == "__main__":
    main()
