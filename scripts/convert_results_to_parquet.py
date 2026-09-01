#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
convert_results_to_parquet.py

Convert kmerseek human_vs_mouse.*.results.csv.zst files to .parquet in place.

Drops query_md5/target_md5 (the only columns confirmed unused downstream and
that don't meaningfully compress anyway) and keeps everything else, including
query_subseq/target_subseq (needed by notebooks 085/200).

Streams via polars scan_csv().sink_parquet() so multi-GB files never get
materialized in memory. Verifies row counts (parquet metadata vs a zstdcat
line count) before a file is considered converted; --delete-originals only
removes a source .csv.zst after its .parquet has been verified.

Resumable: files whose .parquet destination already exists are skipped, so a
killed/interrupted run can just be re-invoked.

Usage:
    python3 convert_results_to_parquet.py --results-dir <dir> --dry-run
    python3 convert_results_to_parquet.py --results-dir <dir>
    python3 convert_results_to_parquet.py --results-dir <dir> --delete-originals
"""

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

DROP_COLUMNS = ["query_md5", "target_md5"]


def csv_row_count(src: Path) -> int:
    """Streaming line count (decompress only, no CSV parsing) minus the header line."""
    p1 = subprocess.Popen(["zstdcat", str(src)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["wc", "-l"], stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
    p1.stdout.close()
    out, _ = p2.communicate()
    p1.wait()
    return max(int(out.strip()) - 1, 0)


def convert_file(src: Path, dst: Path, zstd_level: int) -> dict:
    src_size = src.stat().st_size

    if src_size == 0:
        print(f"  {src.name}: empty stub (0 bytes), skipping")
        return {"src": src, "dst": dst, "skipped": True}

    print(f"Converting {src.name} ({src_size / 1e9:.2f} GB) ...", flush=True)

    try:
        n_csv = csv_row_count(src)
        lf = pl.scan_csv(str(src), ignore_errors=True)
        cols = [c for c in lf.collect_schema().names() if c not in DROP_COLUMNS]
        lf.select(cols).sink_parquet(str(dst), compression="zstd", compression_level=zstd_level)

        n_parquet = pl.scan_parquet(str(dst)).select(pl.len()).collect().item()
        if n_parquet != n_csv:
            raise ValueError(f"row count mismatch: csv={n_csv:,} parquet={n_parquet:,}")
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return {"src": src, "dst": dst, "error": str(exc)}

    dst_size = dst.stat().st_size
    ratio = dst_size / src_size if src_size else float("nan")
    print(
        f"  {src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB ({ratio:.1%})  "
        f"rows: {n_csv:,} (verified match)"
    )
    return {"src": src, "dst": dst, "src_size": src_size, "dst_size": dst_size, "n_rows": n_csv}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="Directory with human_vs_mouse.*.results.csv.zst files")
    parser.add_argument("--zstd-level", type=int, default=9,
                         help="Parquet zstd compression level (default: 9)")
    parser.add_argument("--dry-run", action="store_true",
                         help="List what would be converted without writing anything")
    parser.add_argument("--delete-originals", action="store_true",
                         help="Delete .csv.zst files after their .parquet is verified")
    args = parser.parse_args()

    src_files = sorted(
        args.results_dir.glob("human_vs_mouse.*.results.csv.zst"),
        key=lambda p: p.stat().st_size,
    )
    if not src_files:
        print(f"No human_vs_mouse.*.results.csv.zst files found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(src_files)} files in {args.results_dir}")
    print(f"zstd level: {args.zstd_level}")
    print()

    results = []
    for src in src_files:
        dst = Path(str(src).replace(".results.csv.zst", ".results.parquet"))
        if dst.exists():
            print(f"Skipping {src.name} (destination already exists: {dst.name})")
            continue
        if args.dry_run:
            print(f"[DRY RUN] {src.name} -> {dst.name} ({src.stat().st_size / 1e9:.2f} GB)")
            continue
        results.append(convert_file(src, dst, args.zstd_level))

    if args.dry_run:
        return

    succeeded = [r for r in results if "error" not in r and not r.get("skipped")]
    failed = [r for r in results if "error" in r]
    total_src = sum(r["src_size"] for r in succeeded)
    total_dst = sum(r["dst_size"] for r in succeeded)
    print()
    print("=== Summary ===")
    print(f"Converted: {len(succeeded)}   Failed: {len(failed)}")
    if total_src:
        print(f"Total size: {total_src / 1e9:.2f} GB -> {total_dst / 1e9:.2f} GB ({total_dst / total_src:.1%})")
    if failed:
        print("Failed files (left untouched):")
        for r in failed:
            print(f"  {r['src'].name}: {r['error']}")

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
