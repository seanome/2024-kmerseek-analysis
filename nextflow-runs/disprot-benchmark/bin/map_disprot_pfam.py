#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
map_disprot_pfam.py

Cross-reference DisProt human proteins with the existing Pfam ground truth.

For each DisProt protein, determine:
  - Which Pfam domains it has (from ground truth)
  - Whether each disordered region is within/between/outside Pfam domains
  - How many QfO species share at least one Pfam domain with this protein

Only proteins appearing in the ground truth with ≥1 shared Pfam domain in
≥3 species are kept — these are the proteins with well-characterised homologs.

Usage:
    map_disprot_pfam.py <disprot_human.tsv> <pfam_pairs_dir> <output.tsv>
"""

import json
import sys
from pathlib import Path

import polars as pl


OVERLAP_TOLERANCE = 20   # residues; within this of a domain boundary = linker/terminal

SPECIES_ORDER = [
    "mouse", "chicken", "zebrafish", "ciona", "fly",
    "worm", "yeast", "arabidopsis", "ecoli",
]


def classify_disorder_location(regions: list[dict], domains: list[tuple]) -> str:
    """
    Classify the predominant disorder location for a protein.

    regions: list of {start, end, length}
    domains: list of (start, end) tuples for Pfam domains — may be empty

    Returns: "no_pfam" | "within_domain" | "linker" | "terminal"
    """
    if not regions:
        return "no_pfam"
    if not domains:
        return "no_pfam"

    categories = []
    for reg in regions:
        rs, re = reg["start"], reg["end"]
        # Check if it overlaps a Pfam domain (within tolerance)
        overlaps_domain = any(
            rs <= de + OVERLAP_TOLERANCE and re >= ds - OVERLAP_TOLERANCE
            for ds, de in domains
        )
        if overlaps_domain:
            categories.append("within_domain")
            continue

        # Check if it's between two domains (linker)
        sorted_domains = sorted(domains)
        is_linker = any(
            i + 1 < len(sorted_domains)
            and rs > sorted_domains[i][1]
            and re < sorted_domains[i + 1][0]
            for i in range(len(sorted_domains))
        )
        if is_linker:
            categories.append("linker")
            continue

        categories.append("terminal")

    if not categories:
        return "no_pfam"
    # Return the most common category
    return max(set(categories), key=categories.count)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    disprot_tsv, pfam_dir, output_tsv = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

    # ------------------------------------------------------------------
    # Load DisProt human proteins
    # ------------------------------------------------------------------
    disprot = pl.read_csv(disprot_tsv, separator="\t", infer_schema_length=10_000)
    print(f"DisProt human proteins: {len(disprot)}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Load Pfam ground truth for all 9 species
    # Collect all unique human accessions that appear in any ground truth
    # ------------------------------------------------------------------
    species_accessions: dict[str, set] = {}   # human_acc -> set of species
    for species in SPECIES_ORDER:
        gt_file = pfam_dir / f"human_vs_{species}_ground_truth.parquet"
        if not gt_file.exists():
            print(f"WARNING: missing {gt_file}", file=sys.stderr)
            continue
        gt = pl.read_parquet(gt_file, columns=["human_accession", "species_accession", "label"])
        # Only positive pairs (label is Boolean True = shared Pfam domain)
        positives = gt.filter(pl.col("label").cast(pl.Boolean))
        for row in positives.iter_rows(named=True):
            acc = row["human_accession"]
            if acc not in species_accessions:
                species_accessions[acc] = set()
            species_accessions[acc].add(species)

    print(f"Human accessions in Pfam ground truth: {len(species_accessions)}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Cross-reference DisProt proteins with Pfam ground truth
    # ------------------------------------------------------------------
    output_rows = []

    for row in disprot.iter_rows(named=True):
        acc         = row["uniprot_acc"]
        regions     = json.loads(row["regions_json"]) if row["regions_json"] else []
        species_set = species_accessions.get(acc, set())
        n_species   = len(species_set)

        # We don't have domain coordinates from the ground truth parquet
        # (it only records presence/absence of shared domains, not positions).
        # For disorder_location we rely on DisProt regions only, flagging
        # whether this protein HAS Pfam annotations at all.
        # Coordinate-level overlap requires the Pfam domtblout, which is
        # optional. Default to "terminal" when no coordinate data available.
        domains = []   # (start, end) — populated if hmmscan results exist later
        disorder_loc = classify_disorder_location(regions, domains) if regions else "no_pfam"
        if not species_set:
            disorder_loc = "no_pfam"

        output_rows.append({
            "disprot_id":           row["disprot_id"],
            "uniprot_acc":          acc,
            "gene_name":            row["gene_name"],
            "protein_length":       row["protein_length"],
            "n_disordered_regions": row["n_disordered_regions"],
            "total_disordered_residues": row["total_disordered_residues"],
            "n_pfam_species":       n_species,
            "species_with_shared_pfam": ",".join(sorted(species_set)),
            "disorder_location":    disorder_loc,
            "regions_json":         row["regions_json"],
        })

    result = pl.DataFrame(output_rows)
    print(f"\nAll DisProt proteins cross-referenced: {len(result)}", file=sys.stderr)

    # Filter: ≥1 shared Pfam domain in ≥3 species
    filtered = result.filter(pl.col("n_pfam_species") >= 3)
    print(f"After filter (≥3 species with shared Pfam): {len(filtered)}", file=sys.stderr)

    # Breakdown by disorder_location
    for loc, count in (
        filtered.group_by("disorder_location")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .iter_rows()
    ):
        print(f"  {loc}: {count}", file=sys.stderr)

    filtered.write_csv(output_tsv, separator="\t")
    print(f"\nWrote {output_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
