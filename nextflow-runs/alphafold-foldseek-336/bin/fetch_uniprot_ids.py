#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
fetch_uniprot_ids.py

Map SCOPe domain IDs to UniProt accessions via the RCSB REST API.

For each domain, extracts the PDB ID and chain from the SCOPe domain name
(format: d{pdb}{chain}{seg}, e.g. d1b9ha_ → PDB=1B9H, chain=A), then queries
https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb}/{chain}
to get the UniProt accession.

Input:
  TSV with at least a domain_id column. PDB ID is derived from the domain name.
  Optionally a second column with the PDB ID (--pdb-col 1).

Output:
  TSV: domain_id <TAB> pdb_id <TAB> chain_id <TAB> uniprot_acc
  Lines with no UniProt mapping are written with uniprot_acc=NONE.

Usage:
  fetch_uniprot_ids.py input.tsv output.tsv
  fetch_uniprot_ids.py input.tsv output.tsv --max-workers 5
"""

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


RCSB_INSTANCE_URL = (
    "https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb}/{chain}"
)


def parse_scop_domain(domain_id: str) -> tuple[str, str]:
    """d1b9ha_ → (PDB='1B9H', chain='A'); d1dlw__ → ('1DLW', '_')."""
    s = domain_id.lstrip("d")
    pdb = s[:4].upper()
    chain_char = s[4] if len(s) > 4 else "_"
    chain = chain_char.upper() if chain_char != "_" else "_"
    return pdb, chain


def fetch_uniprot(pdb: str, chain: str) -> str | None:
    """Return the first UniProt accession for this PDB chain, or None."""
    if chain == "_":
        # Try chain A as a fallback when no specific chain is encoded
        chain = "A"

    url = RCSB_INSTANCE_URL.format(pdb=pdb.lower(), chain=chain)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
            # rcsb_polymer_entity_instance → rcsb_uniprot_container_identifiers
            refs = (
                data.get("rcsb_polymer_entity_instance", {})
                .get("rcsb_uniprot_container_identifiers", {})
                .get("uniprot_ids", [])
            )
            if not refs:
                # Try via entity level: follow entity_id link
                entity_id = (
                    data.get("rcsb_polymer_entity_instance", {}).get("entity_id")
                    or data.get("entity_id")
                )
                if entity_id:
                    entity_url = (
                        f"https://data.rcsb.org/rest/v1/core/polymer_entity"
                        f"/{pdb.lower()}/{entity_id}"
                    )
                    with urllib.request.urlopen(entity_url, timeout=20) as er:
                        edata = json.loads(er.read())
                    refs = (
                        edata.get("rcsb_polymer_entity", {})
                        .get("rcsb_entity_host_organism", [{}])[0]
                        .get("ncbi_taxonomy_id", None)
                    )
                    # Extract from entity container identifiers
                    refs = (
                        edata.get("rcsb_polymer_entity_container_identifiers", {})
                        .get("uniprot_ids", [])
                        or []
                    )
            return refs[0] if refs else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2**attempt)
        except Exception as exc:
            print(f"  WARNING: {pdb}/{chain} attempt {attempt+1}: {exc}",
                  file=sys.stderr)
            time.sleep(2**attempt)
    return None


def process_domain(domain_id: str) -> tuple[str, str, str, str | None]:
    pdb, chain = parse_scop_domain(domain_id)
    uniprot = fetch_uniprot(pdb, chain)
    time.sleep(0.05)  # polite rate-limiting
    return domain_id, pdb, chain, uniprot


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_tsv",  help="TSV with domain_id in first column")
    parser.add_argument("output_tsv", help="Output TSV path")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent API requests (default: 8)")
    args = parser.parse_args()

    domain_ids = []
    with open(args.input_tsv) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            domain_ids.append(line.split("\t")[0])

    print(f"Fetching UniProt accessions for {len(domain_ids)} domains …",
          file=sys.stderr)

    results = []
    n_found = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(process_domain, d): d for d in domain_ids}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            domain_id, pdb, chain, uniprot = fut.result()
            results.append((domain_id, pdb, chain, uniprot or "NONE"))
            if uniprot:
                n_found += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(domain_ids):
                print(
                    f"  {i+1}/{len(domain_ids)} done  "
                    f"{n_found} with UniProt  "
                    f"{len(domain_ids) - n_found} missing",
                    file=sys.stderr,
                )

    # Sort by original order
    order = {d: i for i, d in enumerate(domain_ids)}
    results.sort(key=lambda r: order.get(r[0], 9999))

    with open(args.output_tsv, "w") as out:
        out.write("domain_id\tpdb_id\tchain_id\tuniprot_acc\n")
        for domain_id, pdb, chain, uniprot in results:
            out.write(f"{domain_id}\t{pdb}\t{chain}\t{uniprot}\n")

    print(
        f"\nDone: {n_found}/{len(domain_ids)} mapped to UniProt  "
        f"({len(domain_ids) - n_found} without AF2 model).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
