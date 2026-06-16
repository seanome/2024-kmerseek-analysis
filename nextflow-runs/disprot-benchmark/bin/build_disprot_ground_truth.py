#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
build_disprot_ground_truth.py

Extract the DisProt subset of each species' Pfam ground truth and build a
filtered query FASTA from the human proteome.

Adds disorder metadata columns to each ground truth parquet:
  - human_mean_disorder    (placeholder 0.0 — filled in by evaluator after
                            metapredict runs; avoids running metapredict here)
  - human_disorder_location (from disprot_pfam_mapping.tsv)
  - n_disordered_regions
  - total_disordered_residues

Usage:
    build_disprot_ground_truth.py \\
        <disprot_pfam_mapping.tsv> \\
        <pfam_pairs_dir/> \\
        <human.fasta> \\
        <gt_outdir/> \\
        <query.fasta> \\
        <stats.txt>
"""

import sys
from pathlib import Path

import polars as pl


SPECIES_ORDER = [
    "mouse", "chicken", "zebrafish", "ciona", "fly",
    "worm", "yeast", "arabidopsis", "ecoli",
]


def read_fasta_index(fasta_path: str) -> dict[str, str]:
    """Read FASTA into {accession: full_header_line} dict (lazy, header only)."""
    index = {}
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                header = line.strip()[1:]
                # Extract UniProt accession: sp|P12345|GENE -> P12345
                parts = header.split("|")
                acc = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                index[acc] = header
    return index


def read_fasta_sequences(fasta_path: str, keep_accs: set[str]) -> dict[str, str]:
    """Read FASTA sequences for accessions in keep_accs."""
    seqs: dict[str, str] = {}
    current_acc = None
    current_seq = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if current_acc and current_acc in keep_accs:
                    seqs[current_acc] = "".join(current_seq)
                header = line.strip()[1:]
                parts = header.split("|")
                current_acc = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_acc and current_acc in keep_accs:
            seqs[current_acc] = "".join(current_seq)
    return seqs


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)

    mapping_tsv, pfam_dir, human_fasta, gt_outdir, query_fasta_out, stats_out = (
        sys.argv[1], Path(sys.argv[2]), sys.argv[3],
        Path(sys.argv[4]), sys.argv[5], sys.argv[6]
    )
    gt_outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load DisProt mapping
    # ------------------------------------------------------------------
    mapping = pl.read_csv(mapping_tsv, separator="\t", infer_schema_length=10_000)
    disprot_accs = set(mapping["uniprot_acc"].to_list())
    print(f"DisProt query proteins: {len(disprot_accs)}", file=sys.stderr)

    # Build lookup for metadata columns
    meta = mapping.select([
        "uniprot_acc", "disorder_location",
        "n_disordered_regions", "total_disordered_residues", "protein_length",
    ])

    # ------------------------------------------------------------------
    # Per-species ground truth
    # ------------------------------------------------------------------
    species_stats = []
    for species in SPECIES_ORDER:
        gt_file = pfam_dir / f"human_vs_{species}_ground_truth.parquet"
        if not gt_file.exists():
            print(f"WARNING: missing {gt_file}", file=sys.stderr)
            continue

        gt = pl.read_parquet(gt_file)

        # Filter to DisProt human proteins only
        disp_gt = gt.filter(pl.col("human_accession").is_in(disprot_accs))

        # Attach disorder metadata
        disp_gt = disp_gt.join(
            meta.rename({"uniprot_acc": "human_accession"}),
            on="human_accession",
            how="left",
        ).with_columns([
            pl.col("disorder_location").fill_null("unknown"),
            pl.col("n_disordered_regions").fill_null(0),
            pl.col("total_disordered_residues").fill_null(0),
            pl.col("protein_length").fill_null(0),
            # Placeholder: actual disorder scores added by evaluator
            pl.lit(0.0).alias("human_mean_disorder"),
            pl.lit(0.0).alias("human_max_region_disorder"),
        ])

        if "label" in disp_gt.columns:
            n_pos = int(disp_gt["label"].cast(pl.Boolean).sum())
        else:
            n_pos = 0
        n_neg = len(disp_gt) - n_pos
        species_stats.append((species, len(disp_gt), n_pos, n_neg))

        out_path = gt_outdir / f"human_vs_{species}_ground_truth.parquet"
        disp_gt.write_parquet(out_path)
        print(f"  {species}: {len(disp_gt)} pairs ({n_pos} TP, {n_neg} TN)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build query FASTA (DisProt proteins from human proteome)
    # ------------------------------------------------------------------
    seqs = read_fasta_sequences(human_fasta, disprot_accs)
    print(f"\nExtracted {len(seqs)} sequences from human FASTA "
          f"(of {len(disprot_accs)} requested)", file=sys.stderr)

    missing = disprot_accs - set(seqs.keys())
    if missing:
        print(f"WARNING: {len(missing)} DisProt proteins not found in human FASTA "
              f"(may be isoforms or obsolete entries)", file=sys.stderr)

    with open(query_fasta_out, "w") as fh:
        for acc, seq in seqs.items():
            fh.write(f">sp|{acc}|DISP_{acc}\n{seq}\n")
    print(f"Wrote query FASTA: {query_fasta_out} ({len(seqs)} sequences)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Write summary stats
    # ------------------------------------------------------------------
    with open(stats_out, "w") as fh:
        fh.write("=== DisProt Benchmark Summary ===\n\n")
        fh.write(f"Total DisProt query proteins: {len(disprot_accs)}\n")
        fh.write(f"Sequences found in human FASTA: {len(seqs)}\n")
        fh.write(f"Sequences missing: {len(missing)}\n\n")

        fh.write("Disorder location breakdown:\n")
        loc_counts = mapping["disorder_location"].value_counts().sort("count", descending=True)
        for row in loc_counts.iter_rows(named=True):
            fh.write(f"  {row['disorder_location']}: {row['count']}\n")

        fh.write("\nPer-species ground truth pairs:\n")
        fh.write(f"{'Species':<15} {'N_pairs':>8} {'N_TP':>8} {'N_TN':>8}\n")
        fh.write("-" * 42 + "\n")
        for species, n_pairs, n_pos, n_neg in species_stats:
            fh.write(f"{species:<15} {n_pairs:>8} {n_pos:>8} {n_neg:>8}\n")

    with open(stats_out) as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
