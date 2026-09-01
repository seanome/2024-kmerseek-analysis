#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
parse_disprot.py

Download and parse the DisProt current release JSON.
Extracts human proteins (taxon 9606) with their disordered regions.

Usage:
    parse_disprot.py [--local PATH] <output_tsv>

Options:
    --local PATH   Use a pre-downloaded JSON file instead of hitting the API

Output TSV columns:
    disprot_id, uniprot_acc, gene_name,
    n_disordered_regions, total_disordered_residues, protein_length,
    regions_json   (list of {start,end,length} dicts as JSON string)
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

DISPROT_API = "https://disprot.org/api/search?release=current&format=json&page_size=2000&page=1"
HUMAN_TAXON = "9606"


def fetch_disprot_json(url: str) -> dict:
    """Download DisProt JSON from API, handling pagination."""
    all_data = []
    page = 1
    while True:
        paged_url = url.replace("page=1", f"page={page}")
        print(f"Fetching DisProt page {page}: {paged_url}", file=sys.stderr)
        with urllib.request.urlopen(paged_url, timeout=60) as resp:
            payload = json.loads(resp.read().decode())

        entries = payload.get("data", payload.get("results", []))
        if not entries:
            break
        all_data.extend(entries)

        total = payload.get("count", payload.get("total", len(all_data)))
        print(f"  Retrieved {len(all_data)} / {total} entries", file=sys.stderr)
        if len(all_data) >= total:
            break
        page += 1
        time.sleep(0.5)   # be polite to the API

    return all_data


def parse_entry(entry: dict) -> dict | None:
    """Return parsed record for a human DisProt entry, or None to skip."""
    # ncbi_taxon_id is a direct integer field in the current API
    taxon = str(entry.get("ncbi_taxon_id", ""))
    if taxon != HUMAN_TAXON:
        return None

    disprot_id  = entry.get("disprot_id", "")
    uniprot_acc = entry.get("acc", entry.get("uniprot_acc", ""))

    # gene name lives in genes[0].name.value in the current API
    genes = entry.get("genes", [])
    if genes and isinstance(genes[0], dict):
        gene_name = genes[0].get("name", {}).get("value", "")
    else:
        gene_name = entry.get("gene_name", entry.get("gene", ""))

    protein_len = int(entry.get("length", 0))

    # Collect disordered regions — field name differs across release formats
    regions_raw = (
        entry.get("disprot_consensus", {}).get("regions", [])
        or entry.get("regions", [])
    )

    regions = []
    for r in regions_raw:
        start = int(r.get("start", 0))
        end   = int(r.get("end", 0))
        if start > 0 and end >= start:
            regions.append({"start": start, "end": end, "length": end - start + 1})

    # Merge overlapping regions
    regions.sort(key=lambda x: x["start"])
    merged = []
    for reg in regions:
        if merged and reg["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"]    = max(merged[-1]["end"], reg["end"])
            merged[-1]["length"] = merged[-1]["end"] - merged[-1]["start"] + 1
        else:
            merged.append(dict(reg))

    total_disordered = sum(r["length"] for r in merged)

    return {
        "disprot_id":               disprot_id,
        "uniprot_acc":              uniprot_acc,
        "gene_name":                gene_name,
        "n_disordered_regions":     len(merged),
        "total_disordered_residues": total_disordered,
        "protein_length":           protein_len,
        "regions_json":             json.dumps(merged),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", metavar="PATH", help="Pre-downloaded JSON file")
    parser.add_argument("output_tsv", help="Output TSV path")
    args = parser.parse_args()

    if args.local:
        print(f"Loading DisProt from local file: {args.local}", file=sys.stderr)
        with open(args.local) as fh:
            payload = json.load(fh)
        raw_entries = payload if isinstance(payload, list) else (
            payload.get("data", payload.get("results", []))
        )
    else:
        raw_entries = fetch_disprot_json(DISPROT_API)

    print(f"Total DisProt entries: {len(raw_entries)}", file=sys.stderr)

    cols = [
        "disprot_id", "uniprot_acc", "gene_name",
        "n_disordered_regions", "total_disordered_residues",
        "protein_length", "regions_json",
    ]

    records = []
    for entry in raw_entries:
        rec = parse_entry(entry)
        if rec is not None:
            records.append(rec)

    print(f"Human DisProt proteins: {len(records)}", file=sys.stderr)

    with open(args.output_tsv, "w") as out:
        out.write("\t".join(cols) + "\n")
        for rec in records:
            out.write("\t".join(str(rec[c]) for c in cols) + "\n")

    print(f"Wrote {args.output_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
