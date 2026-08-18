#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
download_af2_structures.py

Download AlphaFold v4 CIF structures for a list of UniProt accessions.

Reads a TSV produced by fetch_uniprot_ids.py (columns: domain_id, pdb_id,
chain_id, uniprot_acc) and downloads AF-{ACC}-F1-model_v4.cif from EBI.

Rows with uniprot_acc == NONE are skipped.
Structures already present in --cache are symlinked instead of re-downloaded.

Usage:
    download_af2_structures.py uniprot_map.tsv --outdir structures/ --cache /path/to/cache
"""

import argparse
import concurrent.futures
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


AF_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif"


def download_one(acc: str, outdir: Path, cache: Path) -> tuple[str, bool]:
    filename = f"AF-{acc}-F1-model_v4.cif"
    cache_file = cache / filename
    dest = outdir / filename

    if cache_file.exists() and cache_file.stat().st_size > 0:
        if not dest.exists():
            dest.symlink_to(cache_file.resolve())
        return acc, True

    url = AF_CIF_URL.format(acc=acc)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            cache_file.write_bytes(data)
            return acc, True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return acc, False
            time.sleep(2**attempt)
        except Exception as e:
            print(f"  WARNING: {acc} attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(2**attempt)
    return acc, False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uniprot_map", help="TSV with uniprot_acc column (header row expected)")
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--cache",       required=True)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cache  = Path(args.cache)
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Read unique non-NONE accessions
    accs = []
    seen = set()
    with open(args.uniprot_map) as fh:
        header = fh.readline().strip().split("\t")
        try:
            acc_col = header.index("uniprot_acc")
        except ValueError:
            acc_col = 3  # fallback: 4th column
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) <= acc_col:
                continue
            acc = parts[acc_col]
            if acc and acc != "NONE" and acc not in seen:
                accs.append(acc)
                seen.add(acc)

    print(f"Downloading AF2 structures for {len(accs)} unique UniProt accessions …",
          file=sys.stderr)

    missing = []
    n_ok = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(download_one, acc, outdir, cache): acc for acc in accs}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            acc, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                missing.append(acc)
            if (i + 1) % 50 == 0 or (i + 1) == len(accs):
                print(
                    f"  {i+1}/{len(accs)}  {n_ok} downloaded  {len(missing)} missing",
                    file=sys.stderr,
                )
            time.sleep(0.05)

    print(f"\nDone: {n_ok} structures, {len(missing)} not in AlphaFold DB.",
          file=sys.stderr)

    missing_file = outdir / "missing_structures.txt"
    missing_file.write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"Missing accessions listed in {missing_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
