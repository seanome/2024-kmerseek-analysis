#!/usr/bin/env python3
"""
Fetch Pfam domain annotations for QfO species from UniProt REST API.

For each species, queries UniProt for all proteins with Pfam annotations,
filters to the QfO canonical proteome accessions, and saves:
  - {species}_pfam_domains.parquet   — one row per (protein, Pfam domain)
  - {species}_architectures.parquet  — one row per protein with architecture string
  - {species}_pfam_summary.json      — stats

Domain positions (start/end) are fetched via the JSON feature API.
Architecture = Pfam IDs ordered by domain_start, joined by '-'.
If positions are unavailable, Pfam IDs are sorted alphabetically.

Usage:
    python build_pfam_architectures.py --species all
    python build_pfam_architectures.py --species human mouse fly
    python build_pfam_architectures.py --species human --force
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import polars as pl
import requests

# ---------------------------------------------------------------------------
# Species registry
# ---------------------------------------------------------------------------
SPECIES_METADATA = {
    "human":       {"taxon_id": "9606",   "mya": 0,    "name": "Homo sapiens",
                    "qfo_proteome": "UP000005640", "qfo_subdir": "Eukaryota"},
    "mouse":       {"taxon_id": "10090",  "mya": 100,  "name": "Mus musculus",
                    "qfo_proteome": "UP000000589", "qfo_subdir": "Eukaryota"},
    "chicken":     {"taxon_id": "9031",   "mya": 300,  "name": "Gallus gallus",
                    "qfo_proteome": "UP000000539", "qfo_subdir": "Eukaryota"},
    "zebrafish":   {"taxon_id": "7955",   "mya": 430,  "name": "Danio rerio",
                    "qfo_proteome": "UP000000437", "qfo_subdir": "Eukaryota"},
    "ciona":       {"taxon_id": "7719",   "mya": 550,  "name": "Ciona intestinalis",
                    "qfo_proteome": "UP000008144", "qfo_subdir": "Eukaryota"},
    "fly":         {"taxon_id": "7227",   "mya": 600,  "name": "Drosophila melanogaster",
                    "qfo_proteome": "UP000000803", "qfo_subdir": "Eukaryota"},
    "worm":        {"taxon_id": "6239",   "mya": 650,  "name": "Caenorhabditis elegans",
                    "qfo_proteome": "UP000001940", "qfo_subdir": "Eukaryota"},
    "yeast":       {"taxon_id": "559292", "mya": 900,  "name": "Saccharomyces cerevisiae S288C",
                    "qfo_proteome": "UP000002311", "qfo_subdir": "Eukaryota"},
    "arabidopsis": {"taxon_id": "3702",   "mya": 1500, "name": "Arabidopsis thaliana",
                    "qfo_proteome": "UP000006548", "qfo_subdir": "Eukaryota"},
    "ecoli":       {"taxon_id": "83333",  "mya": 2000, "name": "Escherichia coli K-12",
                    "qfo_proteome": "UP000000625", "qfo_subdir": "Bacteria"},
}

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500
REQUEST_DELAY = 0.5  # seconds between pages (be polite to UniProt)
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_accession(header: str) -> str | None:
    """Extract UniProt accession from a FASTA header line."""
    h = header.lstrip(">").strip()
    # Standard UniProt: sp|P12345|NAME or tr|A0B1C2|NAME
    m = re.match(r"^[st][rp]\|([A-Z0-9]+)\|", h)
    if m:
        return m.group(1)
    # Bare accession: P12345 or A0A000ABC1
    m = re.match(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b", h)
    if m:
        return m.group(1)
    return None


def get_qfo_accessions(species: str, qfo_dir: Path) -> set:
    meta = SPECIES_METADATA[species]
    fasta = qfo_dir / meta["qfo_subdir"] / f"{meta['qfo_proteome']}_{meta['taxon_id']}.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"QfO FASTA not found: {fasta}")
    accessions = set()
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                acc = extract_accession(line)
                if acc:
                    accessions.add(acc)
    print(f"  {species}: {len(accessions):,} canonical accessions in QfO FASTA")
    return accessions


def _get_with_retry(url: str, params: dict, max_retries: int = MAX_RETRIES) -> requests.Response:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# UniProt fetch (JSON for domain positions)
# ---------------------------------------------------------------------------

def fetch_pfam_domains_json(taxon_id: str) -> list[dict]:
    """
    Fetch Pfam domain annotations from UniProt REST API.

    Uses JSON format to capture domain start/end positions from the
    `features` array.  Falls back to xref_pfam cross-references when
    feature positions are not available.

    Returns list of dicts:
        accession, protein_length, pfam_id, domain_start, domain_end,
        domain_description, has_position
    """
    records = []
    params = {
        "query": f"(organism_id:{taxon_id}) AND (database:pfam)",
        "fields": "accession,length,xref_pfam,ft_domain",
        "format": "json",
        "size": str(PAGE_SIZE),
    }

    page = 0
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        elif "cursor" in params:
            del params["cursor"]

        resp = _get_with_retry(UNIPROT_API, params)
        data = resp.json()
        results = data.get("results", [])

        for entry in results:
            acc = entry.get("primaryAccession", "")
            length = (entry.get("sequence") or {}).get("length", 0)

            # --- Parse domain features (have start/end positions) ---
            features = entry.get("features", [])
            pfam_from_features: dict[str, dict] = {}  # pfam_id -> {start, end, desc}

            for feat in features:
                if feat.get("type") != "Domain":
                    continue
                desc = feat.get("description", "")
                loc = feat.get("location", {})
                start = (loc.get("start") or {}).get("value")
                end = (loc.get("end") or {}).get("value")
                # Look for Pfam ID in evidences
                for ev in feat.get("evidences", []):
                    src = ev.get("source", {})
                    if isinstance(src, dict) and src.get("name") == "Pfam":
                        pfam_id = src.get("id", "")
                        if pfam_id:
                            pfam_from_features[pfam_id] = {
                                "start": start,
                                "end": end,
                                "desc": desc,
                            }

            # --- Parse xref_pfam cross-references (no positions) ---
            xrefs = entry.get("uniProtKBCrossReferences", [])
            pfam_xref_ids = {x["id"] for x in xrefs if x.get("database") == "Pfam"}

            # Merge: prefer feature positions, fall back to xrefs
            all_pfam_ids = pfam_xref_ids | set(pfam_from_features.keys())

            for pfam_id in all_pfam_ids:
                if pfam_id in pfam_from_features:
                    info = pfam_from_features[pfam_id]
                    records.append({
                        "accession": acc,
                        "protein_length": length,
                        "pfam_id": pfam_id,
                        "domain_start": info["start"],
                        "domain_end": info["end"],
                        "domain_length": (info["end"] - info["start"] + 1)
                            if info["start"] is not None and info["end"] is not None else None,
                        "domain_description": info["desc"],
                        "has_position": True,
                    })
                else:
                    records.append({
                        "accession": acc,
                        "protein_length": length,
                        "pfam_id": pfam_id,
                        "domain_start": None,
                        "domain_end": None,
                        "domain_length": None,
                        "domain_description": "",
                        "has_position": False,
                    })

        page += 1
        total = data.get("totalResults", "?")
        print(f"  Page {page}: +{len(results)} proteins, cumulative domains: {len(records)} / ~{total} proteins")

        # Pagination via Link header
        link = resp.headers.get("Link", "")
        if 'rel="next"' in link:
            m = re.search(r'[?&]cursor=([^&>]+)', link)
            if m:
                cursor = m.group(1)
                time.sleep(REQUEST_DELAY)
                continue
        break

    return records


# ---------------------------------------------------------------------------
# Architecture builder
# ---------------------------------------------------------------------------

def build_architectures(domains_df: pl.DataFrame) -> pl.DataFrame:
    """
    Build per-protein architecture strings.

    Architecture = Pfam IDs ordered by domain_start (asc), ties broken
    alphabetically. Falls back to alphabetical sort when positions absent.
    """
    # Fill missing start with large number so they sort last
    df = domains_df.with_columns(
        pl.col("domain_start").fill_null(999999).alias("_sort_start")
    )

    # Sort within each accession by (sort_start, pfam_id)
    df = df.sort(["accession", "_sort_start", "pfam_id"])

    arch_df = (
        df.group_by("accession")
        .agg([
            pl.col("pfam_id").alias("pfam_ids"),
            pl.col("pfam_id").str.join("-").alias("architecture"),
            pl.col("protein_length").first().alias("protein_length"),
            pl.col("pfam_id").count().alias("n_domains"),
            pl.col("domain_length").drop_nulls().mean().alias("mean_domain_length"),
            pl.col("has_position").any().alias("any_position"),
        ])
    )
    return arch_df


# ---------------------------------------------------------------------------
# Per-species processing
# ---------------------------------------------------------------------------

def process_species(species: str, qfo_dir: Path, outdir: Path, force: bool = False):
    outdir.mkdir(parents=True, exist_ok=True)

    domains_path = outdir / f"{species}_pfam_domains.parquet"
    arch_path    = outdir / f"{species}_architectures.parquet"
    summary_path = outdir / f"{species}_pfam_summary.json"

    if domains_path.exists() and arch_path.exists() and not force:
        print(f"  {species}: already done — skipping (use --force to rerun)")
        return

    meta = SPECIES_METADATA[species]
    taxon_id = meta["taxon_id"]
    print(f"\n=== {species} ({meta['name']}, taxon {taxon_id}) ===")

    # QfO canonical accessions
    try:
        qfo_accessions = get_qfo_accessions(species, qfo_dir)
    except FileNotFoundError as e:
        print(f"  SKIPPING: {e}", file=sys.stderr)
        return

    # Fetch from UniProt
    print(f"  Fetching Pfam annotations from UniProt REST API...")
    records = fetch_pfam_domains_json(taxon_id)

    if not records:
        print(f"  WARNING: no Pfam records returned for {species}", file=sys.stderr)
        return

    df = pl.DataFrame(records).with_columns([
        pl.col("domain_start").cast(pl.Int32, strict=False),
        pl.col("domain_end").cast(pl.Int32, strict=False),
        pl.col("domain_length").cast(pl.Int32, strict=False),
        pl.col("protein_length").cast(pl.Int32, strict=False),
    ])

    # Filter to QfO canonical proteome
    before = df["accession"].n_unique()
    df = df.filter(pl.col("accession").is_in(qfo_accessions))
    after = df["accession"].n_unique()
    print(f"  After QfO filter: {after:,} proteins with Pfam ({before:,} before filter), {len(df):,} domain records")

    df.write_parquet(domains_path, compression="snappy")
    print(f"  Saved: {domains_path}")

    # Build architectures
    arch_df = build_architectures(df)
    arch_df.write_parquet(arch_path, compression="snappy")
    print(f"  Saved: {arch_path}")

    # Summary
    n_proteins  = df["accession"].n_unique()
    n_domains   = len(df)
    n_families  = df["pfam_id"].n_unique()
    n_multi     = arch_df.filter(pl.col("n_domains") >= 2).height
    pct_qfo     = round(100 * n_proteins / len(qfo_accessions), 1)

    top_pfam = (
        df.group_by("pfam_id")
        .agg(pl.col("accession").n_unique().alias("n_proteins"))
        .sort("n_proteins", descending=True)
        .head(10)
        .to_dicts()
    )

    summary = {
        "species": species,
        "taxon_id": taxon_id,
        "name": meta["name"],
        "mya": meta["mya"],
        "n_qfo_canonical": len(qfo_accessions),
        "n_proteins_with_pfam": n_proteins,
        "pct_qfo_with_pfam": pct_qfo,
        "n_total_domain_records": n_domains,
        "n_pfam_families": n_families,
        "n_multi_domain_proteins": n_multi,
        "top_pfam_families": top_pfam,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    print(f"  Summary: {n_proteins:,} proteins with Pfam ({pct_qfo}% of QfO), "
          f"{n_families:,} families, {n_multi:,} multi-domain")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--species", nargs="+", default=["all"],
        help="Species labels to process. 'all' = all species. Options: " + ", ".join(SPECIES_METADATA),
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("results/pfam_benchmark/annotations"),
        help="Output directory (default: results/pfam_benchmark/annotations)",
    )
    parser.add_argument(
        "--qfo-dir", type=Path,
        default=Path.home() / "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143",
        help="Root directory of QfO 2020 release",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even if output files exist")
    args = parser.parse_args()

    species_list = list(SPECIES_METADATA) if args.species == ["all"] else args.species
    unknown = [s for s in species_list if s not in SPECIES_METADATA]
    if unknown:
        print(f"Unknown species: {unknown}. Valid: {list(SPECIES_METADATA)}", file=sys.stderr)
        sys.exit(1)

    for sp in species_list:
        process_species(sp, args.qfo_dir, args.outdir, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
