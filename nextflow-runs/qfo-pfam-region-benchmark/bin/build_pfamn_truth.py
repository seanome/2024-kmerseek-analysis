#!/usr/bin/env python3
"""Pfam-N regions: the explicit "Pfam-A HMMs missed these" label set.

Why this one matters most for the hypothesis. The gray-zone convention stops the benchmark
counting a call in Pfam-silent territory as an error, but exclusion creates no credit --
those calls simply leave the denominator. Pfam-N converts a slice of that gray zone back
into scoreable TRUE positives: it is a deep-learning extension of Pfam that annotates
sequences and regions the profile HMMs did not reach, so a region kmerseek finds where
Pfam-A is silent can be checked against a label rather than left unadjudicated.

That also makes it the one truth set here that is NOT circular with the profile baselines.
phmmer/jackhmmer/hhblits are profile methods and Pfam-A is built from profile HMMs; Pfam-N
exists precisely where those HMMs failed.

Two facts that shape this script:

  Size.    Pfam-N.gz is ~17.4 GB of Stockholm alignments. It is streamed and filtered on
           the fly rather than downloaded -- only regions on accessions we actually query
           are kept, which is a few tens of thousands of rows out of millions.
  Vintage. It is published for Pfam35.0 only; releases 36 and 37 do not carry it (verified
           2026-08-20, both 404). So it is a frozen 2022 resource, and its Pfam accessions
           should be read against Pfam35 rather than assumed current.

Stockholm layout used here: `#=GF AC` gives the family, and each alignment row is
`<accession>/<start>-<end> <aligned sequence>`. Nothing else is needed, so the parser never
holds an alignment in memory.
"""

import argparse
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

GF_AC = re.compile(r"^#=GF\s+AC\s+(\S+)")
# P12345/23-100 or A0A0B4J2F0.1/5-77 -- the version suffix is dropped.
SEQ_LINE = re.compile(r"^([A-Z0-9]+)(?:\.\d+)?/(\d+)-(\d+)\s")

PFAMN_URL = ("https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-N.gz")


def stream(source: str):
    """Yield decoded lines from a local .gz or a URL, without storing the file."""
    if source.startswith(("http://", "https://")):
        # curl | gunzip rather than urllib so a stall is caught by the same speed floor
        # used everywhere else here, and memory stays flat regardless of file size.
        proc = subprocess.Popen(
            ["curl", "--fail", "--silent", "--show-error", "--location",
             "--speed-limit", "10240", "--speed-time", "120",
             "--retry", "5", "--retry-delay", "15", source],
            stdout=subprocess.PIPE,
        )
        with gzip.open(proc.stdout, "rt", errors="replace") as fh:
            yield from fh
        proc.wait()
        if proc.returncode not in (0, None):
            raise SystemExit(f"curl failed with {proc.returncode} on {source}")
    else:
        with gzip.open(source, "rt", errors="replace") as fh:
            yield from fh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=PFAMN_URL,
                   help="Pfam-N.gz URL or local path; streamed either way")
    p.add_argument("--annotations", required=True, type=Path,
                   help="Pfam annotation dir, used only for which accessions to keep")
    p.add_argument("--truth-out", required=True, type=Path)
    p.add_argument("--map-outdir", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    p.add_argument("--progress-every", type=int, default=20_000_000)
    args = p.parse_args()

    species_acc = {}
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        sp = path.name.replace("_pfam_domains.parquet", "")
        species_acc[sp] = set(pl.read_parquet(path)["accession"].unique().to_list())
    if "human" not in species_acc:
        raise SystemExit("no human_pfam_domains.parquet; cannot define the query side")
    wanted = set().union(*species_acc.values())

    rows = []
    family = None
    n_lines = 0
    for line in stream(args.source):
        n_lines += 1
        if n_lines % args.progress_every == 0:
            print(f"  {n_lines:,} lines, {len(rows):,} kept", file=sys.stderr, flush=True)
        if line.startswith("#=GF"):
            m = GF_AC.match(line)
            if m:
                family = m.group(1).split(".")[0]
            continue
        if line.startswith("#") or line.startswith("//") or not line.strip():
            continue
        m = SEQ_LINE.match(line)
        if not m or family is None:
            continue
        acc = m.group(1)
        if acc not in wanted:
            continue
        rows.append((acc, family, int(m.group(2)), int(m.group(3))))

    df = pl.DataFrame(
        rows, schema=["accession", "pfam_id", "domain_start", "domain_end"], orient="row"
    ).unique().filter(pl.col("domain_end") > pl.col("domain_start"))
    df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("protein_length"))

    human = df.filter(pl.col("accession").is_in(species_acc["human"]))
    human.write_parquet(args.truth_out, compression="zstd")

    args.map_outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": args.source,
        "note": "Pfam-N is published for Pfam35.0 only; releases 36 and 37 do not carry it",
        "lines_scanned": n_lines,
        "human": {"n_regions": human.height, "n_proteins": human["accession"].n_unique(),
                  "n_families": human["pfam_id"].n_unique()},
    }
    for sp, accs in species_acc.items():
        if sp == "human":
            continue
        sub = df.filter(pl.col("accession").is_in(accs))
        sub.write_parquet(args.map_outdir / f"{sp}_domain_map.parquet", compression="zstd")
        summary[sp] = {"n_regions": sub.height, "n_proteins": sub["accession"].n_unique()}

    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
