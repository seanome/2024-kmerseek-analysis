#!/usr/bin/env python3
"""
Download Pfam-A's bulk UniProt regions file and filter it to the QfO
canonical accessions of every species in the benchmark's species registry
(nextflow-runs/qfo-pfam-region-benchmark/assets/qfo_species.tsv -- all 78
of QfO 2020_04, not the original ten).

UniProt's own REST API does not expose Pfam domain positions: the
`ft_domain` feature type never appears in entry JSON (checked against
P04637/TP53, a protein with well-known Pfam domains — zero `Domain`-type
features returned), and `xref_pfam` cross-references carry no start/end
at all. This is why every row in results/pfam_benchmark/annotations/
*_pfam_domains.parquet currently has has_position=False.

Pfam-A.regions.tsv.gz is the authoritative bulk source instead: one row
per (protein, Pfam family, match instance), columns
    pfamseq_acc  seq_version  crc64  md5  pfamA_acc  seq_start  seq_end  ali_start  ali_end
where seq_start/seq_end are the envelope coordinates (the boundary Pfam
itself reports for the domain) and ali_start/ali_end are the tighter HMM
alignment coordinates.

Two steps:
  1. Download (resumable) Pfam-A.regions.tsv.gz from the EBI FTP (~4.7 GB
     compressed). This is a long, network-bound, stall-prone transfer —
     run it yourself and watch it; do not background it unattended.
  2. Single-pass awk filter to rows whose accession is in the union of
     our 10 species' QfO canonical proteomes (~175K accessions), then
     write a parquet.

Output:
    results/pfam_benchmark/pfam_release_cache/pfam_regions_qfo.parquet
    columns: accession, pfam_id, domain_start, domain_end, ali_start, ali_end

Usage:
    python fetch_pfam_envelope_coords.py
    python fetch_pfam_envelope_coords.py --force-download
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from build_pfam_architectures import SPECIES_METADATA, get_qfo_accessions

PFAM_REGIONS_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz"

# The release behind every *_pfam_domains.parquet in results/pfam_benchmark/annotations.
#
# current_release is a moving pointer, and the answer key must not move under half the
# species: the divergence axis compares tools ACROSS species, so a target annotated from a
# different Pfam release than its neighbours is a confound in exactly the axis this
# benchmark exists to measure. Checked 2026-09-03 -- current_release is 38.2, and its
# Pfam-A.regions.tsv.gz is byte-for-byte the size of the copy downloaded 2026-08-07 that
# the original ten species were built from, unchanged since 2026-01-22.
#
# The size check below is what turns a silent Pfam bump into a stop. If it fires, the file
# on the FTP is no longer the one the existing annotations came from: either pin the old
# release under previous_releases/ or rebuild ALL 78 species, not just the new ones.
PFAM_RELEASE          = "38.2"
PFAM_REGIONS_BYTES    = 5_053_457_363


def download_regions(cache_dir: Path, force: bool = False, max_attempts: int = 30) -> Path:
    dest = cache_dir / "Pfam-A.regions.tsv.gz"
    if force and dest.exists():
        dest.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Already complete: return without invoking curl at all. curl -C - against a finished
    # file does not reliably exit 0 (a zero-length range can come back as 416), so "the
    # file is already here" has to be decided before curl, not from its exit status.
    if dest.exists() and dest.stat().st_size == PFAM_REGIONS_BYTES:
        print(f"  have {dest} ({PFAM_REGIONS_BYTES:,} bytes, Pfam {PFAM_RELEASE})")
        return dest
    print(f"  {PFAM_REGIONS_URL}")
    print(f"  -> {dest}")
    print("  ~4.7 GB compressed. EBI FTP stalls intermittently on large transfers — each stall")
    print("  aborts that one curl attempt, but -C - resumes from the exact byte offset, so this")
    print(f"  loop just re-invokes curl (up to {max_attempts}x) until the file is fully down.")

    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            [
                "curl", "-L", "-C", "-",
                "--speed-limit", "10240", "--speed-time", "60",
                "-o", str(dest),
                PFAM_REGIONS_URL,
            ],
        )
        if result.returncode == 0:
            got = dest.stat().st_size
            if got != PFAM_REGIONS_BYTES:
                raise SystemExit(
                    f"\n{dest} is {got:,} bytes; Pfam {PFAM_RELEASE} is "
                    f"{PFAM_REGIONS_BYTES:,}.\n"
                    f"current_release has moved. The ten species already in "
                    f"results/pfam_benchmark/annotations were built from {PFAM_RELEASE}, "
                    f"and mixing releases across species is a confound in the divergence "
                    f"axis. Pin the old release or rebuild all 78."
                )
            return dest
        got = dest.stat().st_size / 1e9 if dest.exists() else 0.0
        print(f"  curl exit {result.returncode} on attempt {attempt}/{max_attempts} "
              f"({got:.2f} GB so far) — resuming...", file=sys.stderr)

    raise RuntimeError(f"Failed to download {PFAM_REGIONS_URL} after {max_attempts} attempts")


def write_accession_list(qfo_dir: Path, out_path: Path) -> set:
    accessions: set = set()
    for species in SPECIES_METADATA:
        accessions |= get_qfo_accessions(species, qfo_dir)
    out_path.write_text("\n".join(sorted(accessions)) + "\n")
    print(f"  {len(accessions):,} unique QfO canonical accessions across {len(SPECIES_METADATA)} species -> {out_path}")
    return accessions


def filter_regions(regions_gz: Path, accessions_txt: Path, out_tsv: Path) -> None:
    print(f"  Streaming {regions_gz.name} through awk, filtering to the accession list (single pass, no full decompress to disk)...")
    awk_prog = 'NR==FNR{acc[$1]; next} FNR==1{next} ($1 in acc)'
    cmd = (
        f"gzip -dc {shlex.quote(str(regions_gz))} | "
        f"awk -F'\\t' {shlex.quote(awk_prog)} {shlex.quote(str(accessions_txt))} - "
        f"> {shlex.quote(str(out_tsv))}"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"  Filtered rows written: {out_tsv}")


def build_parquet(filtered_tsv: Path, out_parquet: Path) -> pl.DataFrame:
    cols = ["accession", "seq_version", "crc64", "md5", "pfam_id",
            "seq_start", "seq_end", "ali_start", "ali_end"]
    df = pl.read_csv(filtered_tsv, separator="\t", has_header=False, new_columns=cols)
    df = df.select([
        "accession",
        "pfam_id",
        pl.col("seq_start").cast(pl.Int32).alias("domain_start"),
        pl.col("seq_end").cast(pl.Int32).alias("domain_end"),
        pl.col("ali_start").cast(pl.Int32),
        pl.col("ali_end").cast(pl.Int32),
    ])
    df.write_parquet(out_parquet, compression="snappy")
    print(f"  {len(df):,} domain-instance rows, {df['accession'].n_unique():,} distinct accessions -> {out_parquet}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--qfo-dir", type=Path,
        default=Path.home() / "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("results/pfam_benchmark/pfam_release_cache"),
    )
    parser.add_argument("--force-download", action="store_true", help="Refetch Pfam-A.regions.tsv.gz even if already cached")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    accessions_txt = args.cache_dir / "qfo_accessions.txt"
    filtered_tsv   = args.cache_dir / "pfam_regions_qfo.tsv"
    out_parquet    = args.cache_dir / "pfam_regions_qfo.parquet"

    print("=== Step 1/4: QfO accession list ===")
    write_accession_list(args.qfo_dir, accessions_txt)

    print("\n=== Step 2/4: download Pfam-A.regions.tsv.gz ===")
    regions_gz = download_regions(args.cache_dir, force=args.force_download)

    print("\n=== Step 3/4: filter to our accessions ===")
    filter_regions(regions_gz, accessions_txt, filtered_tsv)

    print("\n=== Step 4/4: write parquet ===")
    df = build_parquet(filtered_tsv, out_parquet)

    print("\n=== Coverage by species (QfO canonical accessions with >=1 Pfam-A envelope match) ===")
    for species in SPECIES_METADATA:
        sp_acc = get_qfo_accessions(species, args.qfo_dir)
        covered = df.filter(pl.col("accession").is_in(sp_acc))["accession"].n_unique()
        print(f"  {species:12s} {covered:6,} / {len(sp_acc):6,}")

    print(f"\nDone. Next:")
    print(f"  python build_pfam_architectures.py --species all --regions-parquet {out_parquet}")


if __name__ == "__main__":
    main()
