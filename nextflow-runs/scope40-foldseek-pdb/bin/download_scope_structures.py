#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
Download PDB chain structures for SCOPe single-domain entries from RCSB.

For each domain ID (must end in _):
  pdb_code = domain_id[1:5]   (e.g. 1dlw)
  chain    = domain_id[5]     (SCOP lowercase → PDB: uppercase alpha, digit as-is, 'x' → ' ')
  Downloads {PDBID}.pdb from RCSB, extracts the chain, saves as {domain_id}.pdb.

Uses a two-level cache:
  --cache dir/raw/      raw full PDB files (one per PDB code, reused across chains)
  --outdir structures/  extracted chain PDB files (one per domain ID)

Usage:
  download_scope_structures.py domain_ids.txt --outdir structures/ --cache ~/data/scope/pdbstyle-2.08 --max-workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_DELAY = 5.0


def scop_chain_to_pdb(scop_chain: str) -> str:
    """Convert SCOP chain character to PDB chain ID."""
    if scop_chain == "x":
        return " "  # blank chain ID in PDB format
    if scop_chain.isalpha():
        return scop_chain.upper()
    return scop_chain  # digit or other


def download_pdb(pdb_id: str, raw_cache: Path, session: requests.Session) -> str | None:
    """Download a PDB file and return its content, using file-level cache."""
    cache_path = raw_cache / f"{pdb_id}.pdb"
    if cache_path.exists():
        return cache_path.read_text()

    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None  # genuinely missing
            if resp.status_code in RETRY_CODES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            resp.raise_for_status()
            content = resp.text
            cache_path.write_text(content)
            return content
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ERROR downloading {pdb_id}: {exc}", file=sys.stderr)
                return None
    return None


def extract_chain(pdb_content: str, chain_id: str) -> str:
    """Extract ATOM/HETATM records for a single chain from PDB-format text."""
    keep = []
    for line in pdb_content.splitlines(True):
        rec = line[:6].rstrip()
        if rec in ("ATOM", "HETATM", "TER"):
            if len(line) > 21 and line[21] == chain_id:
                keep.append(line)
        elif rec in ("HEADER", "COMPND", "SOURCE", "REMARK", "SEQRES", "CRYST1", "ORIGX1",
                     "ORIGX2", "ORIGX3", "SCALE1", "SCALE2", "SCALE3"):
            keep.append(line)
    if not any(l[:6].rstrip() in ("ATOM", "HETATM") for l in keep):
        return ""
    if not keep[-1].startswith("END"):
        keep.append("END\n")
    return "".join(keep)


def process_domain(domain_id: str, outdir: Path, raw_cache: Path,
                   session: requests.Session) -> tuple[str, str]:
    """Download and extract a single domain. Returns (domain_id, status)."""
    out_path = outdir / f"{domain_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return domain_id, "cached"

    pdb_code = domain_id[1:5]
    scop_chain = domain_id[5]
    pdb_chain = scop_chain_to_pdb(scop_chain)

    content = download_pdb(pdb_code, raw_cache, session)
    if content is None:
        return domain_id, "missing"

    extracted = extract_chain(content, pdb_chain)
    if not extracted:
        # Try original case before giving up (some PDB files have non-standard chain case)
        for alt_chain in (scop_chain, pdb_chain.lower()):
            extracted = extract_chain(content, alt_chain)
            if extracted:
                break
    if not extracted:
        return domain_id, f"no_chain_{pdb_chain}"

    out_path.write_text(extracted)
    return domain_id, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_ids", help="File with one SCOPe domain ID per line")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cache", required=True, help="Cache directory for raw PDB files")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    raw_cache = Path(args.cache) / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_cache.mkdir(parents=True, exist_ok=True)

    domain_ids = [line.strip() for line in Path(args.domain_ids).read_text().splitlines()
                  if line.strip() and line.strip().endswith("_")]
    print(f"Downloading structures for {len(domain_ids):,} single-domain entries", flush=True)

    stats = {"ok": 0, "cached": 0, "missing": 0, "no_chain": 0}

    with requests.Session() as session:
        session.headers["User-Agent"] = "kmerseek-analysis/1.0 (research; olga@seanome.org)"

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(process_domain, did, outdir, raw_cache, session): did
                       for did in domain_ids}
            done = 0
            for fut in as_completed(futures):
                did, status = fut.result()
                done += 1
                if status.startswith("no_chain"):
                    stats["no_chain"] += 1
                    print(f"  WARN no chain: {did} ({status})", file=sys.stderr)
                elif status == "missing":
                    stats["missing"] += 1
                    print(f"  WARN missing:  {did}", file=sys.stderr)
                elif status in stats:
                    stats[status] += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(domain_ids)} processed", flush=True)

    total_written = stats["ok"] + stats["cached"]
    print(f"\nDone: {stats['ok']} downloaded, {stats['cached']} cached, "
          f"{stats['missing']} missing PDB, {stats['no_chain']} no-chain-found")
    print(f"Structures available: {total_written:,}", flush=True)

    if total_written == 0:
        print("ERROR: No structures downloaded!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
