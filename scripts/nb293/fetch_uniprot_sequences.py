"""Fetch current UniProtKB sequences for a list of accessions, with a local cache.

Usage:
    python3 -I fetch_uniprot_sequences.py ACCESSIONS_TXT OUT_FASTA

Accessions already in OUT_FASTA are not fetched again. Queries
https://rest.uniprot.org/uniprotkb/accessions in batches of 100 (Swiss-Prot and TrEMBL).
Writes a log line with the number requested, found and missing, and the UniProt release
from the X-UniProt-Release response header.
"""

import sys
import urllib.request
from pathlib import Path

BATCH = 100
URL = "https://rest.uniprot.org/uniprotkb/accessions?accessions={}&format=fasta"


def cached_accessions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line[1:].split("|")[1]
        for line in path.read_text().splitlines()
        if line.startswith(">")
    }


def main() -> None:
    wanted = [a for a in Path(sys.argv[1]).read_text().split() if a]
    out = Path(sys.argv[2])
    have = cached_accessions(out)
    todo = [a for a in wanted if a not in have]
    release = "cached"
    for i in range(0, len(todo), BATCH):
        with urllib.request.urlopen(
            URL.format(",".join(todo[i : i + BATCH])), timeout=120
        ) as r:
            release = r.headers.get("X-UniProt-Release", "unknown")
            text = r.read().decode()
        if text and not text.startswith(">"):
            raise RuntimeError(f"UniProt did not return FASTA: {text[:200]}")
        with out.open("a") as fh:
            fh.write(text)
    found = cached_accessions(out)
    missing = sorted(set(wanted) - found)
    print(
        f"requested={len(wanted)} fetched_now={len(todo)} found={len(set(wanted) & found)} "
        f"missing={len(missing)} uniprot_release={release} missing_list={','.join(missing)}"
    )


if __name__ == "__main__":
    main()
