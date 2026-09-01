#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
parse_mobidb.py

Download and parse MobiDB disorder annotations for human proteins.

Streams the MobiDB bulk download for the human proteome (UP000005640).
Only proteins that have the requested disorder feature AND pass the
coverage filters are kept.

Coverage filters (configurable via args):
  --min-region  : minimum contiguous disordered region length (default 20)
  --min-frac    : minimum fraction of protein that must be disordered (default 0.20)

Output TSV columns (same schema as parse_disprot.py):
    disprot_id, uniprot_acc, gene_name,
    n_disordered_regions, total_disordered_residues, protein_length,
    regions_json

Usage:
    parse_mobidb.py [--local PATH] [--source curated|consensus]
                    [--min-region N] [--min-frac F]
                    <output_tsv>

Options:
    --local PATH       Pre-downloaded JSON-Lines file instead of API
    --source SOURCE    "curated"   – curated_disorder_literature only
                       "consensus" – mobidb_consensus_disorder_linear
                       [default: curated]
    --min-region N     Min region length in residues [default: 20]
    --min-frac F       Min disordered fraction of protein [default: 0.20]
"""

import argparse
import json
import sys
import urllib.request
from io import TextIOWrapper

MOBIDB_DOWNLOAD = "https://mobidb.org/api/download"
HUMAN_PROTEOME  = "UP000005640"

# MobiDB v5 field names (hyphenated, not underscored)
# "curated"   → merged curated disorder (DisProt + IDEAL + other curated sources)
# "consensus" → MobiDB-lite prediction (sequence-based, broadest coverage)
FEATURE_KEYS = {
    "curated":   "curated-disorder-merge",
    "consensus": "prediction-disorder-mobidb_lite",
}


def build_url(source: str) -> str:
    feature = FEATURE_KEYS[source]
    # MobiDB only includes a field when the protein has that annotation,
    # so requesting the projection just limits other fields returned.
    return (
        f"{MOBIDB_DOWNLOAD}"
        f"?proteome={HUMAN_PROTEOME}"
        f"&format=json"
        f"&projection={feature},acc,gene,length"
    )


def iter_entries_from_stream(stream) -> dict:
    """Yield parsed JSON objects from a JSON-Lines stream."""
    for raw_line in stream:
        line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode().strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj:   # skip empty {}
                yield obj
        except json.JSONDecodeError:
            pass


def iter_entries_from_file(path: str):
    with open(path) as fh:
        raw = fh.read().strip()
    if raw.startswith("["):
        for obj in json.loads(raw):
            if obj:
                yield obj
    else:
        for line in raw.splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj:
                        yield obj
                except json.JSONDecodeError:
                    pass


def extract_regions(feature_val) -> list[dict]:
    """
    Extract {start, end, length} dicts from a MobiDB v5 feature value.

    MobiDB v5 stores regions as:
        {"regions": [[start, end], ...], "content_fraction": …, ...}
    where each element is a 2-element list [start, end] (1-based, inclusive).
    """
    if not feature_val:
        return []

    if isinstance(feature_val, dict):
        raw = feature_val.get("regions", [])
        if not raw:
            return []
        result = []
        for r in raw:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                s, e = int(r[0]), int(r[1])
                result.append({"start": s, "end": e, "length": e - s + 1})
            elif isinstance(r, dict):
                s = int(r.get("start", 0))
                e = int(r.get("end", 0))
                if s and e:
                    result.append({"start": s, "end": e, "length": e - s + 1})
        return result

    return []


def merge_regions(regions: list[dict]) -> list[dict]:
    if not regions:
        return []
    regions = sorted(regions, key=lambda r: r["start"])
    merged = [dict(regions[0])]
    for reg in regions[1:]:
        if reg["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"]    = max(merged[-1]["end"], reg["end"])
            merged[-1]["length"] = merged[-1]["end"] - merged[-1]["start"] + 1
        else:
            merged.append(dict(reg))
    return merged


def parse_entry(entry: dict, feature_key: str,
                min_region: int, min_frac: float) -> dict | None:
    acc    = entry.get("acc", "")
    gene   = entry.get("gene", "")
    length = int(entry.get("length", 0))

    if not acc or length == 0:
        return None

    feature_val = entry.get(feature_key)
    if not feature_val:
        return None   # protein has no annotation for this feature

    regions = extract_regions(feature_val)
    regions = merge_regions(regions)
    regions = [r for r in regions if r["length"] >= min_region]
    if not regions:
        return None

    total_disordered = sum(r["length"] for r in regions)
    if total_disordered / length < min_frac:
        return None

    return {
        "disprot_id":                f"MOBIDB:{acc}",
        "uniprot_acc":               acc,
        "gene_name":                 gene,
        "n_disordered_regions":      len(regions),
        "total_disordered_residues": total_disordered,
        "protein_length":            length,
        "regions_json":              json.dumps(regions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local",      metavar="PATH",
                        help="Pre-downloaded JSON-Lines file")
    parser.add_argument("--source",     choices=["curated", "consensus"],
                        default="curated")
    parser.add_argument("--min-region", type=int, default=20,
                        help="Min contiguous disordered region length [default: 20]")
    parser.add_argument("--min-frac",   type=float, default=0.20,
                        help="Min disordered fraction of protein [default: 0.20]")
    parser.add_argument("output_tsv",   help="Output TSV path")
    args = parser.parse_args()

    feature_key = FEATURE_KEYS[args.source]
    cols = [
        "disprot_id", "uniprot_acc", "gene_name",
        "n_disordered_regions", "total_disordered_residues",
        "protein_length", "regions_json",
    ]

    n_total = 0
    n_no_feature = 0
    n_low_coverage = 0
    records = []

    if args.local:
        print(f"Loading MobiDB from local file: {args.local}", file=sys.stderr)
        entry_iter = iter_entries_from_file(args.local)
    else:
        url = build_url(args.source)
        print(f"Streaming MobiDB from: {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "kmerseek-disprot-benchmark/1.0"})
        response = urllib.request.urlopen(req, timeout=300)
        entry_iter = iter_entries_from_stream(TextIOWrapper(response, encoding="utf-8"))

    for entry in entry_iter:
        n_total += 1
        if n_total % 5000 == 0:
            print(f"  Processed {n_total} entries, kept {len(records)}...", file=sys.stderr)

        if not entry.get(feature_key):
            n_no_feature += 1
            continue

        rec = parse_entry(entry, feature_key, args.min_region, args.min_frac)
        if rec is None:
            n_low_coverage += 1
            continue
        records.append(rec)

    print(
        f"\nTotal entries: {n_total}  |  "
        f"No {args.source} feature: {n_no_feature}  |  "
        f"Below coverage threshold: {n_low_coverage}  |  "
        f"Kept: {len(records)}",
        file=sys.stderr,
    )

    with open(args.output_tsv, "w") as out:
        out.write("\t".join(cols) + "\n")
        for rec in records:
            out.write("\t".join(str(rec[c]) for c in cols) + "\n")

    print(f"Wrote {args.output_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
