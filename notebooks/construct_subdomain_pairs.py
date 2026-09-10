#!/usr/bin/env python3
"""
Construct ground-truth benchmark pairs for the Pfam subdomain benchmark.

For each human × {species} pair:
  TRUE POSITIVE  — share ≥1 Pfam domain, different overall architecture,
                   AND at least one protein has ≥2 domains
                   (excludes single-domain full-length homologs which are
                    already covered by the Tier 1 benchmark)
  TRUE NEGATIVE  — share 0 Pfam domains

Sampling rules:
  - Cap ≤500 pairs per Pfam domain (prevents PF00069 kinase domination)
  - Target ≈20 K positives + ≈40 K negatives (or as many as available)
  - Random seed = 42 for reproducibility

Output per species pair:
  results/pfam_benchmark/pairs/human_vs_{species}_ground_truth.parquet
  results/pfam_benchmark/pairs/human_vs_{species}_summary.json

Usage:
    python construct_subdomain_pairs.py --species all
    python construct_subdomain_pairs.py --species mouse fly ecoli
    python construct_subdomain_pairs.py --species mouse --force
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
SPECIES_METADATA = {
    "human": {"taxon_id": "9606", "mya": 0},
    "mouse": {"taxon_id": "10090", "mya": 100},
    "chicken": {"taxon_id": "9031", "mya": 300},
    "zebrafish": {"taxon_id": "7955", "mya": 430},
    "ciona": {"taxon_id": "7719", "mya": 550},
    "fly": {"taxon_id": "7227", "mya": 600},
    "worm": {"taxon_id": "6239", "mya": 650},
    "yeast": {"taxon_id": "559292", "mya": 900},
    "arabidopsis": {"taxon_id": "3702", "mya": 1500},
    "ecoli": {"taxon_id": "83333", "mya": 2000},
}

TARGET_POSITIVES = 20_000
TARGET_NEGATIVES = 40_000
MAX_PAIRS_PER_DOMAIN = 500
RANDOM_SEED = 42
# ---------------------------------------------------------------------------


def load_annotations(
    annot_dir: Path, species: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load (domains_df, arch_df) for a species."""
    domains_path = annot_dir / f"{species}_pfam_domains.parquet"
    arch_path = annot_dir / f"{species}_architectures.parquet"
    for p in (domains_path, arch_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}  — run build_pfam_architectures.py first"
            )
    return pl.read_parquet(domains_path), pl.read_parquet(arch_path)


def build_lookup(domains_df: pl.DataFrame, arch_df: pl.DataFrame) -> dict:
    """Build in-memory lookup dicts for fast pair construction."""
    pfam_sets = {
        row["accession"]: set(row["pfam_ids"])
        for row in (
            domains_df.group_by("accession")
            .agg(pl.col("pfam_id").alias("pfam_ids"))
            .iter_rows(named=True)
        )
    }
    arch_map = {
        row["accession"]: row["architecture"] for row in arch_df.iter_rows(named=True)
    }
    n_dom = {
        row["accession"]: row["n_domains"] for row in arch_df.iter_rows(named=True)
    }
    plen = {
        row["accession"]: row["protein_length"] for row in arch_df.iter_rows(named=True)
    }
    return {"pfam_sets": pfam_sets, "arch_map": arch_map, "n_dom": n_dom, "plen": plen}


def sample_positives(
    human: dict,
    species: dict,
    rng: random.Random,
) -> list[tuple]:
    """
    Return list of (human_acc, species_acc, shared_pfam_csv, n_shared) tuples.

    TP criteria:
      1. Share ≥1 Pfam domain
      2. Different overall architecture
      3. At least one protein has ≥2 domains
    """
    h_pfam = human["pfam_sets"]
    s_pfam = species["pfam_sets"]
    h_arch = human["arch_map"]
    s_arch = species["arch_map"]
    h_ndom = human["n_dom"]
    s_ndom = species["n_dom"]

    # Index: pfam_id → species accessions carrying that domain
    pfam_to_species: dict[str, list] = defaultdict(list)
    for acc, pfam_set in s_pfam.items():
        for pfam_id in pfam_set:
            pfam_to_species[pfam_id].append(acc)

    # domain → candidate pairs
    domain_pairs: dict[str, set] = defaultdict(set)

    for h_acc, h_set in h_pfam.items():
        h_a = h_arch.get(h_acc, "")
        for pfam_id in h_set:
            for s_acc in pfam_to_species.get(pfam_id, []):
                s_a = s_arch.get(s_acc, "")
                # criteria 2 & 3
                if h_a == s_a:
                    continue
                if h_ndom.get(h_acc, 1) < 2 and s_ndom.get(s_acc, 1) < 2:
                    continue
                domain_pairs[pfam_id].add((h_acc, s_acc))

    # Cap per domain, then flatten
    positive_set: set = set()
    for pfam_id, pairs in domain_pairs.items():
        pairs_list = list(pairs)
        if len(pairs_list) > MAX_PAIRS_PER_DOMAIN:
            pairs_list = rng.sample(pairs_list, MAX_PAIRS_PER_DOMAIN)
        positive_set.update(pairs_list)

    positives = list(positive_set)
    if len(positives) > TARGET_POSITIVES:
        positives = rng.sample(positives, TARGET_POSITIVES)

    # Annotate with shared domain info
    result = []
    for h_acc, s_acc in positives:
        shared = h_pfam[h_acc] & s_pfam[s_acc]
        result.append((h_acc, s_acc, ";".join(sorted(shared)), len(shared)))

    return result


def sample_negatives(
    human: dict,
    species: dict,
    positive_pairs: set,
    rng: random.Random,
) -> list[tuple]:
    """
    Return list of (human_acc, species_acc) tuples with 0 shared Pfam domains.
    Avoids duplicating positive pairs.
    """
    h_pfam = human["pfam_sets"]
    s_pfam = species["pfam_sets"]
    s_accs = list(s_pfam.keys())

    negatives: list = []
    h_accs_shuffled = list(h_pfam.keys())
    rng.shuffle(h_accs_shuffled)

    for h_acc in h_accs_shuffled:
        if len(negatives) >= TARGET_NEGATIVES:
            break
        h_set = h_pfam[h_acc]
        # Sample a batch of species proteins and check disjointness
        candidates = rng.choices(s_accs, k=min(100, len(s_accs)))
        for s_acc in candidates:
            if len(negatives) >= TARGET_NEGATIVES:
                break
            if (h_acc, s_acc) in positive_pairs:
                continue
            if not (h_set & s_pfam[s_acc]):
                negatives.append((h_acc, s_acc))

    return negatives


def build_dataframe(
    positives: list[tuple],
    negatives: list[tuple],
    human: dict,
    species: dict,
) -> pl.DataFrame:
    h_arch = human["arch_map"]
    s_arch = species["arch_map"]
    h_ndom = human["n_dom"]
    s_ndom = species["n_dom"]
    h_plen = human["plen"]
    s_plen = species["plen"]

    rows = []
    for h_acc, s_acc, shared_csv, n_shared in positives:
        rows.append(
            {
                "human_accession": h_acc,
                "species_accession": s_acc,
                "label": True,
                "shared_pfam_ids": shared_csv,
                "n_shared_domains": n_shared,
                "human_architecture": h_arch.get(h_acc, ""),
                "species_architecture": s_arch.get(s_acc, ""),
                "human_n_domains": h_ndom.get(h_acc, 0),
                "species_n_domains": s_ndom.get(s_acc, 0),
                "human_protein_length": h_plen.get(h_acc),
                "species_protein_length": s_plen.get(s_acc),
            }
        )
    for h_acc, s_acc in negatives:
        rows.append(
            {
                "human_accession": h_acc,
                "species_accession": s_acc,
                "label": False,
                "shared_pfam_ids": "",
                "n_shared_domains": 0,
                "human_architecture": h_arch.get(h_acc, ""),
                "species_architecture": s_arch.get(s_acc, ""),
                "human_n_domains": h_ndom.get(h_acc, 0),
                "species_n_domains": s_ndom.get(s_acc, 0),
                "human_protein_length": h_plen.get(h_acc),
                "species_protein_length": s_plen.get(s_acc),
            }
        )

    schema = {
        "human_accession": pl.Utf8,
        "species_accession": pl.Utf8,
        "label": pl.Boolean,
        "shared_pfam_ids": pl.Utf8,
        "n_shared_domains": pl.Int32,
        "human_architecture": pl.Utf8,
        "species_architecture": pl.Utf8,
        "human_n_domains": pl.Int32,
        "species_n_domains": pl.Int32,
        "human_protein_length": pl.Int32,
        "species_protein_length": pl.Int32,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------


def process_pair(species: str, annot_dir: Path, pairs_dir: Path, force: bool = False):
    pairs_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = pairs_dir / f"human_vs_{species}_ground_truth.parquet"
    out_summary = pairs_dir / f"human_vs_{species}_summary.json"

    if out_parquet.exists() and not force:
        print(f"  human_vs_{species}: already done — skipping (use --force)")
        return

    print(f"\n=== human vs {species} ({SPECIES_METADATA[species]['mya']} MYA) ===")

    try:
        h_domains, h_arch = load_annotations(annot_dir, "human")
        s_domains, s_arch = load_annotations(annot_dir, species)
    except FileNotFoundError as e:
        print(f"  SKIPPING: {e}", file=sys.stderr)
        return

    human = build_lookup(h_domains, h_arch)
    species_ = build_lookup(s_domains, s_arch)

    rng = random.Random(RANDOM_SEED)

    print(
        f"  Sampling positives (target {TARGET_POSITIVES:,}, cap {MAX_PAIRS_PER_DOMAIN}/domain)..."
    )
    positives = sample_positives(human, species_, rng)
    print(f"  Positives: {len(positives):,}")

    positive_set = {(h, s) for h, s, *_ in positives}

    print(f"  Sampling negatives (target {TARGET_NEGATIVES:,})...")
    negatives = sample_negatives(human, species_, positive_set, rng)
    print(f"  Negatives: {len(negatives):,}")

    gt = build_dataframe(positives, negatives, human, species_)
    gt.write_parquet(out_parquet, compression="snappy")
    print(f"  Saved: {out_parquet}")

    mya = SPECIES_METADATA[species]["mya"]
    summary = {
        "species": species,
        "mya": mya,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_total": len(gt),
        "positive_rate": round(len(positives) / max(len(gt), 1), 4),
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_summary}")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["all"],
        help="Species to pair with human. 'all' = all non-human species.",
    )
    parser.add_argument(
        "--annot-dir",
        type=Path,
        default=Path("results/pfam_benchmark/annotations"),
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=Path("results/pfam_benchmark/pairs"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    non_human = [s for s in SPECIES_METADATA if s != "human"]
    species_list = (
        non_human
        if args.species == ["all"]
        else [s for s in args.species if s != "human"]
    )

    unknown = [s for s in species_list if s not in SPECIES_METADATA]
    if unknown:
        print(f"Unknown species: {unknown}", file=sys.stderr)
        sys.exit(1)

    for sp in species_list:
        process_pair(sp, args.annot_dir, args.pairs_dir, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
