#!/usr/bin/env python3
"""
Build Pfam domain annotations for QfO species from the Pfam-A bulk
regions file (run fetch_pfam_envelope_coords.py first).

For each species, filters the pre-fetched, pre-filtered Pfam-A regions
parquet to the QfO canonical proteome accessions, and saves:
  - {species}_pfam_domains.parquet   — one row per (protein, Pfam domain match)
  - {species}_architectures.parquet  — one row per protein with architecture string
  - {species}_pfam_summary.json      — stats

Domain positions (domain_start/domain_end) are Pfam's own envelope
coordinates (seq_start/seq_end in Pfam-A.regions.tsv.gz) — not UniProt's,
which does not expose them (verified: UniProt entry JSON has no
`Domain`-type features and `xref_pfam` cross-references carry no
start/end at all). ali_start/ali_end (the tighter HMM alignment
coordinates) are carried through too.

A protein with a repeat domain (e.g. WD40, ankyrin) now gets one row per
repeat instance, matching Pfam's own match count — architecture strings
will show real repeats ("PF00400-PF00400-PF00400") rather than a single
collapsed occurrence, which is what the old UniProt-xref-based approach
produced.

Architecture = Pfam IDs ordered by domain_start, joined by '-'.

Usage:
    python build_pfam_architectures.py --species all   # every registry species
    python build_pfam_architectures.py --species human mouse fly
    python build_pfam_architectures.py --species human --force
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Species registry
# ---------------------------------------------------------------------------
# Read from the benchmark's species registry rather than restated here. The registry holds
# all 78 species of QfO 2020_04 and is generated from the release itself; this dict used to
# carry ten, hand-typed, and was one of three copies of the same list that had drifted
# apart. See nextflow-runs/qfo-pfam-region-benchmark/bin/build_qfo_species_registry.py.
SPECIES_REGISTRY = (Path(__file__).resolve().parent.parent
                    / "nextflow-runs" / "qfo-pfam-region-benchmark"
                    / "assets" / "qfo_species.tsv")


def load_species_metadata(path: Path = SPECIES_REGISTRY) -> dict:
    """label -> {taxon_id, mya, name, qfo_proteome, qfo_subdir}, in registry order."""
    if not path.exists():
        raise SystemExit(
            f"species registry not found: {path}\n"
            f"Generate it with:\n"
            f"  nextflow-runs/qfo-pfam-region-benchmark/bin/build_qfo_species_registry.py "
            f"--release <QfO dir> --out <that path>"
        )
    meta = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            meta[r["label"]] = {
                "taxon_id": r["taxon"],
                # Empty for every species with no sourced divergence time. Carried through
                # to the summary JSON as null rather than as a number nobody measured.
                "mya": int(r["mya"]) if r.get("mya", "").strip() else None,
                "name": r["scientific_name"],
                "qfo_proteome": r["proteome"],
                "qfo_subdir": r["subdir"],
            }
    return meta


SPECIES_METADATA = load_species_metadata()

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


def get_qfo_lengths(species: str, qfo_dir: Path) -> dict:
    """Parse the QfO FASTA for {accession: sequence_length}."""
    meta = SPECIES_METADATA[species]
    fasta = qfo_dir / meta["qfo_subdir"] / f"{meta['qfo_proteome']}_{meta['taxon_id']}.fasta"
    lengths: dict[str, int] = {}
    acc = None
    seq_len = 0
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                if acc is not None:
                    lengths[acc] = seq_len
                acc = extract_accession(line)
                seq_len = 0
            else:
                seq_len += len(line.strip())
        if acc is not None:
            lengths[acc] = seq_len
    return lengths


# ---------------------------------------------------------------------------
# Bulk Pfam-A regions load (envelope + alignment positions)
# ---------------------------------------------------------------------------

def load_domains_from_bulk(
    species: str,
    qfo_accessions: set,
    lengths: dict,
    regions_df: pl.DataFrame,
) -> list[dict]:
    """
    Filter the global Pfam-A regions parquet (see fetch_pfam_envelope_coords.py)
    to this species' QfO canonical accessions.

    Returns list of dicts:
        accession, protein_length, pfam_id, domain_start, domain_end,
        domain_length, domain_ali_start, domain_ali_end, domain_description,
        has_position
    """
    sub = regions_df.filter(pl.col("accession").is_in(qfo_accessions))
    records = []
    for row in sub.iter_rows(named=True):
        acc = row["accession"]
        start, end = row["domain_start"], row["domain_end"]
        records.append({
            "accession": acc,
            "protein_length": lengths.get(acc),
            "pfam_id": row["pfam_id"],
            "domain_start": start,
            "domain_end": end,
            "domain_length": (end - start + 1) if start is not None and end is not None else None,
            "domain_ali_start": row["ali_start"],
            "domain_ali_end": row["ali_end"],
            "domain_description": "",
            "has_position": start is not None and end is not None,
        })
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

def process_species(species: str, qfo_dir: Path, outdir: Path, regions_df: pl.DataFrame, force: bool = False):
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

    # QfO canonical accessions + protein lengths
    try:
        qfo_accessions = get_qfo_accessions(species, qfo_dir)
        lengths = get_qfo_lengths(species, qfo_dir)
    except FileNotFoundError as e:
        print(f"  SKIPPING: {e}", file=sys.stderr)
        return

    print(f"  Filtering bulk Pfam-A regions to {species}'s QfO canonical accessions...")
    records = load_domains_from_bulk(species, qfo_accessions, lengths, regions_df)

    if not records:
        print(f"  WARNING: no Pfam records found for {species} in the bulk regions file", file=sys.stderr)
        return

    df = pl.DataFrame(records).with_columns([
        pl.col("domain_start").cast(pl.Int32, strict=False),
        pl.col("domain_end").cast(pl.Int32, strict=False),
        pl.col("domain_length").cast(pl.Int32, strict=False),
        pl.col("domain_ali_start").cast(pl.Int32, strict=False),
        pl.col("domain_ali_end").cast(pl.Int32, strict=False),
        pl.col("protein_length").cast(pl.Int32, strict=False),
    ])

    after = df["accession"].n_unique()
    print(f"  {after:,} proteins with Pfam matches, {len(df):,} domain-instance records (envelope positions from Pfam-A.regions.tsv.gz)")

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
    parser.add_argument(
        "--regions-parquet", type=Path,
        default=Path("results/pfam_benchmark/pfam_release_cache/pfam_regions_qfo.parquet"),
        help="Output of fetch_pfam_envelope_coords.py (run that first)",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess even if output files exist")
    args = parser.parse_args()

    species_list = list(SPECIES_METADATA) if args.species == ["all"] else args.species
    unknown = [s for s in species_list if s not in SPECIES_METADATA]
    if unknown:
        print(f"Unknown species: {unknown}. Valid: {list(SPECIES_METADATA)}", file=sys.stderr)
        sys.exit(1)

    if not args.regions_parquet.exists():
        print(f"Missing: {args.regions_parquet} — run fetch_pfam_envelope_coords.py first", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.regions_parquet}...")
    regions_df = pl.read_parquet(args.regions_parquet)
    print(f"  {len(regions_df):,} domain-instance rows, {regions_df['accession'].n_unique():,} accessions")

    for sp in species_list:
        process_species(sp, args.qfo_dir, args.outdir, regions_df, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
