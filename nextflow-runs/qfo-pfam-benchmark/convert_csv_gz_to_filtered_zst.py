#!/usr/bin/env python3
"""
convert_csv_gz_to_filtered_zst.py

Convert existing kmerseek results from .csv.gz to .csv.zst, applying a
raw (uncorrected) poisson_pvalue filter to reduce file sizes.

For each human_vs_<species>.hp.k<N>.results.csv.gz:
  - Reads with gzip
  - Keeps rows where poisson_pvalue < PVALUE_THRESHOLD (default 0.05)
  - Writes .csv.zst with zstd -19 compression

Bonferroni correction (n_human * n_species protein pairs) is applied
in the analysis notebooks, not here.

Usage:
    python3 convert_csv_gz_to_filtered_zst.py [--dry-run] [--threshold 0.05]
    python3 convert_csv_gz_to_filtered_zst.py --delete-originals  # after verifying
"""

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path.home() / "data/qfo-pfam-benchmark/kmerseek-results"
PVALUE_COL  = 22   # 0-indexed; "poisson_pvalue" is the 23rd column
ZSTD_LEVEL  = 19   # high compression; use lower (e.g. 3) for speed


def convert_file(src: Path, threshold: float, dry_run: bool) -> dict:
    dst = src.with_suffix("").with_suffix(".csv.zst")  # drop .gz, replace .csv -> .csv.zst
    # Actually: src is like foo.results.csv.gz → dst is foo.results.csv.zst
    dst = Path(str(src).replace(".csv.gz", ".csv.zst"))

    src_size = src.stat().st_size

    if dry_run:
        print(f"[DRY RUN] {src.name} → {dst.name}")
        return {"src": src, "dst": dst, "skipped": True}

    print(f"Converting {src.name} ...", flush=True)

    n_in = n_out = 0
    zstd_proc = subprocess.Popen(
        ["zstd", f"-{ZSTD_LEVEL}", "-o", str(dst)],
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        with gzip.open(src, "rt") as fin:
            # Always write header
            header = fin.readline()
            zstd_proc.stdin.write(header)

            for line in fin:
                n_in += 1
                parts = line.split(",")
                try:
                    pval = float(parts[PVALUE_COL])
                except (ValueError, IndexError):
                    continue  # skip malformed rows
                if pval < threshold:
                    zstd_proc.stdin.write(line)
                    n_out += 1

        zstd_proc.stdin.close()
        zstd_proc.wait()
    except Exception as exc:
        zstd_proc.stdin.close()
        zstd_proc.wait()
        print(f"  ERROR: {exc}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return {"src": src, "dst": dst, "error": str(exc)}

    dst_size = dst.stat().st_size
    ratio = dst_size / src_size if src_size else float("nan")
    kept_pct = 100 * n_out / n_in if n_in else 0.0

    print(
        f"  {src_size/1e9:.2f} GB → {dst_size/1e9:.2f} GB  ({ratio:.2%})  "
        f"rows: {n_in:,} → {n_out:,} ({kept_pct:.1f}% kept)"
    )
    return {
        "src": src, "dst": dst,
        "src_size": src_size, "dst_size": dst_size,
        "n_in": n_in, "n_out": n_out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Raw poisson_pvalue cutoff (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--delete-originals", action="store_true",
                        help="Delete .csv.gz files after successful conversion")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"Directory with .csv.gz files (default: {RESULTS_DIR})")
    args = parser.parse_args()

    src_files = sorted(args.results_dir.glob("*.csv.gz"))
    if not src_files:
        print(f"No .csv.gz files found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(src_files)} files in {args.results_dir}")
    print(f"P-value threshold (raw, uncorrected): {args.threshold}")
    print(f"Zstd level: {ZSTD_LEVEL}")
    print()

    results = []
    for src in src_files:
        dst = Path(str(src).replace(".csv.gz", ".csv.zst"))
        if dst.exists() and not args.dry_run:
            print(f"Skipping {src.name} (destination already exists: {dst.name})")
            continue
        r = convert_file(src, args.threshold, args.dry_run)
        results.append(r)

    if args.dry_run:
        return

    # Summary
    succeeded = [r for r in results if "error" not in r and not r.get("skipped")]
    total_src = sum(r["src_size"] for r in succeeded)
    total_dst = sum(r["dst_size"] for r in succeeded)
    print()
    print(f"=== Summary ===")
    print(f"Files converted: {len(succeeded)}")
    if total_src:
        print(f"Total size: {total_src/1e9:.2f} GB → {total_dst/1e9:.2f} GB ({total_dst/total_src:.2%})")

    if args.delete_originals:
        for r in succeeded:
            print(f"Deleting {r['src'].name}")
            r["src"].unlink()
        print("Originals deleted.")
    else:
        print()
        print("Originals retained. Re-run with --delete-originals to remove them.")


if __name__ == "__main__":
    main()
