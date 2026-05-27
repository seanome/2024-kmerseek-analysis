#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
download_alphafold.py

Download AlphaFold v4 CIF structures for proteins in a FASTA file.
Caches files: skips proteins already present in --cache dir.

Usage:
    download_alphafold.py --fasta <fasta> --outdir <dir> --cache <dir> [--max-workers N]

Rate-limited: respects EBI fair-use (~5 concurrent at most).
Proteins with no AlphaFold model are listed in missing_structures.txt.
"""

import argparse
import concurrent.futures
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif"


def read_accessions(fasta_path: str) -> list[str]:
    accs = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                header = line.strip()[1:]
                parts = header.split("|")
                acc = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                accs.append(acc)
    return accs


def download_one(acc: str, outdir: Path, cache: Path) -> tuple[str, bool]:
    """Download CIF for acc. Returns (acc, success)."""
    filename = f"AF-{acc}-F1-model_v4.cif"

    # Check cache first
    cache_file = cache / filename
    if cache_file.exists() and cache_file.stat().st_size > 0:
        dest = outdir / filename
        if not dest.exists():
            dest.symlink_to(cache_file.resolve())
        return acc, True

    url = AF_URL.format(acc=acc)
    dest = outdir / filename

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            # Also save to cache
            cache_file.write_bytes(data)
            return acc, True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return acc, False   # No AlphaFold model for this protein
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"WARNING: {acc} attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)

    return acc, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta",       required=True)
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--cache",       required=True)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cache  = Path(args.cache)
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    accs = read_accessions(args.fasta)
    print(f"Downloading AlphaFold structures for {len(accs)} proteins", file=sys.stderr)

    missing = []
    n_ok = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_one, acc, outdir, cache): acc for acc in accs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            acc, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                missing.append(acc)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(accs)} done, {n_ok} ok, {len(missing)} missing",
                      file=sys.stderr)
            time.sleep(0.1)   # polite rate-limiting between completions

    print(f"\nDone: {n_ok} structures downloaded, {len(missing)} not in AlphaFold",
          file=sys.stderr)

    missing_file = outdir / "missing_structures.txt"
    with open(missing_file, "w") as fh:
        for acc in missing:
            fh.write(acc + "\n")
    print(f"Missing proteins listed in {missing_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
