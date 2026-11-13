#!/usr/bin/env python3
"""Step 1 of notebook 341 for set A: every isoform pair that differs by one internal exon.

Reads the Patowary et al. 2024 isoforms (Dryad Data S1, Bulk.gtf, GRCh37) and per-isoform read
counts (Data S2, sheet B), assembles each isoform's mRNA from the GRCh37 genome, takes each
isoform's longest ORF, and writes:

  341_setA_exon_pairs.tsv.gz   one row per (including isoform, skipping isoform, exon)
  341_setA_exon_events.tsv     one row per exon event, with its read counts per sample, Ψ per
                               donor and region, and the proteins of one representative pair

The representative pair of an event is the one whose including and skipping isoforms have the
most reads together, among pairs whose ORFs are otherwise identical when there are any.

Usage:
  build_341_exon_table.py --gtf Bulk.gtf --counts DataS2.xlsx --genome GRCh37.primary_assembly.genome.fa.gz --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import condensate_exon_pairs as cp  # noqa: E402

PAIR_COLUMNS = [
    "gene_id",
    "gene_name",
    "chrom",
    "strand",
    "exon_start",
    "exon_end",
    "upstream_exon_end",
    "downstream_exon_start",
    "including_isoform",
    "skipping_isoform",
    "exon_nt",
    "frame_preserving",
    "exon_in_orf",
    "orf_otherwise_identical",
    "exon_first_residue",
    "exon_last_residue",
    "exon_residues",
    "protein_including",
    "protein_skipping",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_counts(xlsx: str) -> tuple[list[str], dict[str, list[int]]]:
    """Sheet B of Data S2: sample names and transcript_id -> read count per sample."""
    import openpyxl

    ws = openpyxl.load_workbook(xlsx, read_only=True)["B"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    samples = [h for h in header[3:]]
    counts = {r[2]: [int(x or 0) for x in r[3:]] for r in rows if r[2] is not None}
    return samples, counts


def psi_summary(inc: list[int], skip: list[int], samples: list[str]) -> dict:
    """Ψ pooled per donor and region, mean over donors per region, and CP minus GZ."""
    pooled = defaultdict(lambda: [0, 0])
    for s, i, k in zip(samples, inc, skip):
        donor, region = s.split("_")[0], s.split("_")[-1]
        pooled[(donor, region)][0] += i
        pooled[(donor, region)][1] += k
    out, per_region = {}, defaultdict(list)
    donors = sorted({d for d, _ in pooled})
    for (donor, region), (i, k) in sorted(pooled.items()):
        psi = i / (i + k) if i + k else None
        out[f"psi_{region}_{donor}"] = psi
        out[f"reads_{region}_{donor}"] = i + k
        if psi is not None:
            per_region[region].append(psi)
    for region in ("GZ", "CP"):
        vals = per_region.get(region, [])
        out[f"psi_{region}_mean"] = sum(vals) / len(vals) if vals else None
        out[f"n_donors_{region}"] = len(vals)
    paired = [
        out[f"psi_CP_{d}"] - out[f"psi_GZ_{d}"]
        for d in donors
        if out.get(f"psi_CP_{d}") is not None and out.get(f"psi_GZ_{d}") is not None
    ]
    out["delta_psi_cp_minus_gz"] = sum(paired) / len(paired) if paired else None
    out["n_donors_paired"] = len(paired)
    out["n_donors_cp_higher"] = sum(d > 0 for d in paired)
    out["n_donors_cp_lower"] = sum(d < 0 for d in paired)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--counts", required=True, help="DataS2.xlsx")
    ap.add_argument(
        "--genome", required=True, help="GRCh37 FASTA, chromosome names as in the GTF"
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("reading GTF")
    isoforms = cp.read_gtf_isoforms(args.gtf)
    log(f"{len(isoforms):_} isoforms")
    pairs = cp.find_exon_events(isoforms)
    log(f"{len(pairs):_} one-exon isoform pairs")

    by_chrom = defaultdict(list)
    for row in pairs:
        by_chrom[row["chrom"]].append(row)
    done = []
    for chrom, seq in cp.read_fasta_chromosomes(args.genome, set(by_chrom)):
        cache: dict = {}  # each isoform's mRNA and ORF, computed once per chromosome
        for row in by_chrom[chrom]:
            done.append(
                cp.pair_proteins(
                    row,
                    isoforms[row["including_isoform"]],
                    isoforms[row["skipping_isoform"]],
                    seq,
                    cache=cache,
                )
            )
        log(f"{chrom}: {len(by_chrom[chrom]):_} pairs translated")
    missing = set(by_chrom) - {r["chrom"] for r in done}
    if missing:
        sys.exit(f"chromosomes in the GTF but not in the genome: {sorted(missing)}")

    with gzip.open(out_dir / "341_setA_exon_pairs.tsv.gz", "wt", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=PAIR_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(done)
    log(f"wrote {len(done):_} pairs")

    log("reading counts")
    samples, counts = read_counts(args.counts)
    log(f"{len(samples)} samples, {len(counts):_} isoforms with counts")
    gene_isoforms = defaultdict(list)
    for iso in isoforms.values():
        gene_isoforms[iso.gene_id].append(iso)
    events = defaultdict(list)
    for row in done:
        events[cp.event_key(row)].append(row)
    rows = []
    for key, members in events.items():
        inc, skip = cp.event_read_counts(
            key, gene_isoforms[key[0]], counts, len(samples)
        )
        identical = [m for m in members if m["orf_otherwise_identical"]]
        pool = identical or members
        rep = max(
            pool,
            key=lambda m: sum(counts.get(m["including_isoform"], [0]))
            + sum(counts.get(m["skipping_isoform"], [0])),
        )
        rows.append(
            {
                **{c: rep[c] for c in PAIR_COLUMNS},
                "n_pairs": len(members),
                "n_pairs_orf_otherwise_identical": len(identical),
                "reads_including_total": sum(inc),
                "reads_skipping_total": sum(skip),
                **psi_summary(inc, skip, samples),
                **{f"inc_{s}": i for s, i in zip(samples, inc)},
                **{f"skip_{s}": k for s, k in zip(samples, skip)},
            }
        )
    fields = list(rows[0]) if rows else PAIR_COLUMNS
    with open(out_dir / "341_setA_exon_events.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_fp = sum(r["frame_preserving"] and r["orf_otherwise_identical"] for r in rows)
    log(
        f"wrote {len(rows):_} exon events; {n_fp:_} frame-preserving with ORFs otherwise identical"
    )


if __name__ == "__main__":
    main()
