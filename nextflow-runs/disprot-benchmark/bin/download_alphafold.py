#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
download_alphafold.py

Download AlphaFold CIF structures for proteins in a FASTA file.
Caches files: skips proteins already present in --cache dir, whatever model version the
cached file carries.

Usage:
    download_alphafold.py --fasta <fasta> --outdir <dir> --cache <dir> [--max-workers N]

The model version is RESOLVED, never assumed. Until 2026-09-20 this script asked for
AF-<acc>-F1-model_v4.cif by name, which the AlphaFold server had stopped serving (404 on
every accession) while the cache held 54_339 v6 files under a name this script never
looked for. Every protein of every species was listed as missing, every Foldseek search
ran on zero CIF files, and the DisProt report carried a Foldseek arm with no pair on any
proteome. Now the cache is checked for any AF-<acc>-F1-model_v*.cif, and a download asks
the AlphaFold API which file is current (https://alphafold.ebi.ac.uk/api/prediction/<acc>)
and fetches that.

Rate-limited: respects EBI fair-use (~5 concurrent at most).
Proteins with no AlphaFold model are listed in missing_structures.txt.
"""

import argparse
import concurrent.futures
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


# The API says which model file is current for an accession; the file URL is read off it.
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
CACHED = "AF-{acc}-F1-model_v*.cif"
VERSION = re.compile(r"-model_v(\d+)\.cif$")


def cached_model(cache: Path, acc: str) -> Path | None:
    """The cached CIF for acc at the highest model version present, or None."""
    found = [p for p in cache.glob(CACHED.format(acc=acc)) if p.stat().st_size > 0]
    if not found:
        return None
    return max(found, key=lambda p: int(VERSION.search(p.name).group(1)))


def current_cif_url(acc: str) -> str | None:
    """The current model's CIF URL from the AlphaFold API, or None when there is no
    model (404) for the accession."""
    with urllib.request.urlopen(AF_API.format(acc=acc), timeout=30) as resp:
        entries = json.loads(resp.read().decode())
    # One entry per fragment; F1 is the whole-chain model this pipeline uses.
    for e in entries if isinstance(entries, list) else [entries]:
        url = e.get("cifUrl")
        if url and "-F1-" in url:
            return url
    return None

# Circuit-breaker: set when DNS is confirmed dead so workers abort immediately.
_dns_dead = threading.Event()


def check_connectivity(host: str = "alphafold.ebi.ac.uk", timeout: int = 10) -> bool:
    """Return True if DNS resolves and TCP port 443 is reachable."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


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
    # Check cache first, at whatever model version it holds.
    cache_file = cached_model(cache, acc)
    if cache_file is not None:
        dest = outdir / cache_file.name
        if not dest.exists():
            dest.symlink_to(cache_file.resolve())
        return acc, True

    for attempt in range(3):
        if _dns_dead.is_set():
            return acc, False   # circuit-breaker: network is gone, abort immediately

        try:
            url = current_cif_url(acc)
            if url is None:
                return acc, False   # the API lists no whole-chain model
            filename = url.rsplit("/", 1)[-1]
            dest = outdir / filename
            cache_file = cache / filename
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            cache_file.write_bytes(data)
            return acc, True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return acc, False   # no AlphaFold model for this protein
            time.sleep(2 ** attempt)
        except OSError as e:
            # DNS resolution failure (errno 8 / EAI_NONAME) — network is dead.
            # Signal all other workers to stop retrying immediately.
            if "nodename nor servname" in str(e) or "Name or service not known" in str(e) or isinstance(e, socket.gaierror):
                print(f"ERROR: DNS resolution failed ({e}). Aborting all downloads.",
                      file=sys.stderr)
                _dns_dead.set()
                return acc, False
            print(f"WARNING: {acc} attempt {attempt+1} failed: {e}", file=sys.stderr)
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

    # Fast-fail if network is unreachable — avoids burning hours retrying.
    if not check_connectivity():
        print("ERROR: Cannot reach alphafold.ebi.ac.uk (DNS/network failure). "
              "Writing all proteins to missing_structures.txt and exiting.",
              file=sys.stderr)
        missing_file = outdir / "missing_structures.txt"
        with open(missing_file, "w") as fh:
            for acc in accs:
                fh.write(acc + "\n")
        sys.exit(1)

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
