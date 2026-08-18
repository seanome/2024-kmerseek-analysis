#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
convert_results_to_parquet_streaming.py

Convert kmerseek human_vs_mouse.*.results.csv.zst files to .parquet with BOUNDED memory.

Why this exists alongside convert_results_to_parquet.py
-------------------------------------------------------
That script's docstring claims `scan_csv().sink_parquet()` streams, but polars (1.34)
does NOT stream a *compressed* CSV: it inflates the whole file in memory before parsing.
Measured on human_vs_mouse.hp-kyte-doolittle.k28.results.csv.zst -- 1.9 GB compressed,
184.7 GB decompressed (97x) -- RSS hit 23 GB in 6 seconds with zero parquet bytes written,
heading for the full 184.7 GB against 137 GB of RAM. That is what OOM-kills the kernel.

Here `zstdcat` does the decompression as a pipe and pyarrow's incremental CSV reader
consumes it block by block, so peak memory is a few blocks regardless of file size.

Two safety properties the polars script lacks:

  * Atomic output. Parquet is written to `<dst>.partial` and renamed only after the row
    count is verified. An OOM-kill or Ctrl-C therefore leaves no `<dst>.parquet` behind.
    That matters because both scripts skip a combo when its .parquet exists, and
    `genome_wide_results_file` PREFERS .parquet -- so a truncated parquet would be
    silently treated as complete and quietly shorten every downstream analysis.
  * Shared schema. Types are taken from an already-converted parquet (--schema-from, or
    auto-detected) and applied to every file, so per-file type inference can't make one
    combo's column int64 and another's float64.

Usage:
    convert_results_to_parquet_streaming.py --results-dir <dir> --dry-run
    convert_results_to_parquet_streaming.py --results-dir <dir> --min-size-gb 5
    convert_results_to_parquet_streaming.py --results-dir <dir> --delete-originals
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DROP_COLUMNS = ["query_md5", "target_md5"]

# A kmerseek run over an empty index still writes a well-formed 13-byte empty zstd frame.
# Converting one produces a rowless parquet that would then shadow the real result for the
# same combo in another results dir (see EXTRA_DATA_DIRS in ortholog_analysis_utils.py).
EMPTY_STUB_MAX_BYTES = 1024

# Bytes of decompressed CSV pyarrow parses per block. Peak RSS is a small multiple of this.
BLOCK_SIZE = 64 << 20  # 64 MB

# A .partial untouched for longer than this is debris from a killed run; anything fresher
# is assumed to belong to a converter running right now (see `cleanup`).
PARTIAL_STALE_AFTER_S = 600


def zstdcat(src: Path) -> subprocess.Popen:
    return subprocess.Popen(["zstdcat", str(src)], stdout=subprocess.PIPE)


def csv_row_count(src: Path) -> int:
    """Streaming line count (decompress only, no CSV parsing) minus the header line.

    This costs a SECOND full decompression of the source, which is why it is opt-in
    (--verify-line-count) rather than the default. See `convert_file` for the single-pass
    checks that make it redundant in the normal case.
    """
    p1 = zstdcat(src)
    p2 = subprocess.Popen(["wc", "-l"], stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
    p1.stdout.close()
    out, _ = p2.communicate()
    p1.wait()
    return max(int(out.strip()) - 1, 0)


def reference_schema(results_dir: Path, explicit: Path | None) -> pa.Schema | None:
    """Schema from an already-converted, non-empty parquet, so all files share types."""
    if explicit:
        return pq.read_schema(str(explicit))
    for p in sorted(results_dir.glob("human_vs_mouse.*.results.parquet"),
                    key=lambda p: p.stat().st_size, reverse=True):
        try:
            if pq.ParquetFile(str(p)).metadata.num_rows > 0:
                print(f"Schema reference: {p.name}")
                return pq.read_schema(str(p))
        except Exception:
            continue
    print("No converted parquet to take a schema from; inferring per file.")
    return None


def convert_file(src: Path, dst: Path, schema: pa.Schema | None, zstd_level: int,
                 verify: bool) -> dict:
    src_size = src.stat().st_size
    if src_size <= EMPTY_STUB_MAX_BYTES:
        print(f"  {src.name}: {src_size}-byte empty stub, skipping (would shadow real data)")
        return {"src": src, "dst": dst, "skipped": True}

    print(f"Converting {src.name} ({src_size / 1e9:.2f} GB compressed) ...", flush=True)
    partial = dst.with_suffix(dst.suffix + ".partial")
    partial.unlink(missing_ok=True)

    convert_opts = pacsv.ConvertOptions(
        column_types={f.name: f.type for f in schema} if schema else None,
    )
    proc = zstdcat(src)
    writer = None
    n_rows = 0
    try:
        reader = pacsv.open_csv(
            proc.stdout,
            read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
            convert_options=convert_opts,
        )
        keep = [n for n in reader.schema.names if n not in DROP_COLUMNS]
        for batch in reader:
            table = pa.Table.from_batches([batch]).select(keep)
            if writer is None:
                writer = pq.ParquetWriter(str(partial), table.schema,
                                          compression="zstd", compression_level=zstd_level)
            writer.write_table(table)
            n_rows += table.num_rows
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)
        proc.kill()
        return {"src": src, "dst": dst, "error": str(exc)}
    finally:
        if writer is not None:
            writer.close()
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

    # Single-pass completeness check, costing no extra I/O: zstdcat exits non-zero on a
    # truncated or corrupt frame, and pyarrow raises (above) on a row it cannot parse. A
    # clean exit from both therefore means every byte of the source became a parquet row.
    # That is what makes the second decompression pass (--verify-line-count) redundant.
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        msg = f"zstdcat exited {proc.returncode} (truncated/corrupt source)"
        print(f"  ERROR: {msg}", file=sys.stderr)
        return {"src": src, "dst": dst, "error": msg}

    if verify:
        n_csv = csv_row_count(src)
        if n_csv != n_rows:
            partial.unlink(missing_ok=True)
            msg = f"row count mismatch: csv={n_csv:,} parquet={n_rows:,}"
            print(f"  ERROR: {msg}", file=sys.stderr)
            return {"src": src, "dst": dst, "error": msg}

    partial.rename(dst)  # atomic: dst appears only once complete and verified
    dst_size = dst.stat().st_size
    print(f"  {src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB "
          f"({dst_size / src_size:.1%})  rows: {n_rows:,}"
          f"{' (verified)' if verify else ' (unverified)'}", flush=True)
    return {"src": src, "dst": dst, "src_size": src_size, "dst_size": dst_size, "n_rows": n_rows}


def cleanup(results_dir: Path, dry_run: bool = False) -> None:
    """Drop files that are pure redundancy: a .csv.zst whose non-empty .parquet already
    exists, and stale .partial writes from a killed run. A source is only removed once its
    parquet is confirmed readable AND non-empty, so this can never be the only copy.

    Deliberately does NOT delete the 13-byte empty-index stubs: they live in a Nextflow
    storeDir, so removing one makes the pipeline recompute that combo on the next -resume.
    `genome_wide_results_file` already treats them as absent.
    """
    freed = n = 0
    tag = "[DRY RUN] would remove" if dry_run else "  removing"
    # A .partial belonging to a CONCURRENTLY RUNNING converter is a live write, not debris.
    # Deleting it would silently corrupt that run, so only touch ones untouched for a while.
    now = time.time()
    for stale in results_dir.glob("*.partial"):
        age = now - stale.stat().st_mtime
        if age < PARTIAL_STALE_AFTER_S:
            print(f"  skipping {stale.name}: modified {age:.0f}s ago, another converter "
                  f"is probably writing it")
            continue
        print(f"{tag} stale partial: {stale.name} (idle {age / 60:.0f} min)")
        if not dry_run:
            stale.unlink(missing_ok=True)

    for src in sorted(results_dir.glob("human_vs_mouse.*.results.csv.zst")):
        dst = Path(str(src).replace(".results.csv.zst", ".results.parquet"))
        if not dst.exists():
            continue
        try:
            if pq.ParquetFile(str(dst)).metadata.num_rows == 0:
                continue  # rowless parquet is not proof the source was converted
        except Exception:
            continue      # unreadable parquet -- keep the source, it's the only good copy
        freed += src.stat().st_size
        if not dry_run:
            src.unlink()
        n += 1
    verb = "would remove" if dry_run else "removed"
    print(f"Cleanup: {verb} {n} redundant .csv.zst, reclaiming {freed / 1e9:.1f} GB\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--schema-from", type=Path, default=None,
                    help="Parquet to copy column types from (default: largest converted one)")
    ap.add_argument("--min-size-gb", type=float, default=0.0,
                    help="Only convert sources at least this large (default: all)")
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--verify-line-count", action="store_true",
                    help="Also compare against a zstdcat line count. Costs a SECOND full "
                         "decompression; the single-pass checks make it redundant normally.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-originals", action="store_true",
                    help="Delete each .csv.zst after its .parquet is written and verified")
    ap.add_argument("--cleanup", action="store_true",
                    help="Before converting, delete any .csv.zst whose non-empty .parquet "
                         "already exists (reclaims space from earlier runs), and remove "
                         "stale .partial files. Never touches a source without a good parquet.")
    args = ap.parse_args()

    if args.cleanup:
        cleanup(args.results_dir, dry_run=args.dry_run)

    src_files = sorted(args.results_dir.glob("human_vs_mouse.*.results.csv.zst"),
                       key=lambda p: p.stat().st_size)
    src_files = [f for f in src_files if f.stat().st_size >= args.min_size_gb * 1e9]
    if not src_files:
        print(f"Nothing to convert in {args.results_dir} at >= {args.min_size_gb} GB")
        return

    print(f"Found {len(src_files)} candidate files in {args.results_dir}")
    if args.dry_run:
        for src in src_files:
            dst = Path(str(src).replace(".results.csv.zst", ".results.parquet"))
            size = src.stat().st_size
            if dst.exists():
                state = "SKIP (parquet exists)"
            elif size <= EMPTY_STUB_MAX_BYTES:
                state = "SKIP (empty stub)"
            else:
                state = "convert"
            print(f"  [{state}] {src.name} ({size / 1e9:.2f} GB)")
        return

    schema = reference_schema(args.results_dir, args.schema_from)
    results = []
    for src in src_files:
        dst = Path(str(src).replace(".results.csv.zst", ".results.parquet"))
        if dst.exists():
            print(f"Skipping {src.name} (already converted)")
            continue
        results.append(convert_file(src, dst, schema, args.zstd_level, args.verify_line_count))

    ok = [r for r in results if "error" not in r and not r.get("skipped")]
    failed = [r for r in results if "error" in r]
    print(f"\n=== Summary ===\nConverted: {len(ok)}   Failed: {len(failed)}")
    if ok:
        tot_s = sum(r["src_size"] for r in ok)
        tot_d = sum(r["dst_size"] for r in ok)
        print(f"Total: {tot_s / 1e9:.2f} GB -> {tot_d / 1e9:.2f} GB ({tot_d / tot_s:.1%})")
    for r in failed:
        print(f"  FAILED (original untouched): {r['src'].name}: {r['error']}")

    if args.delete_originals:
        for r in ok:
            r["src"].unlink()
        print(f"Deleted {len(ok)} verified originals.")


if __name__ == "__main__":
    main()
