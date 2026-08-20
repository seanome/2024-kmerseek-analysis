#!/usr/bin/env python3
"""M-CSA catalytic residues as a truth set.

Function defined by MECHANISM, curated from the literature -- circular with neither the
profile HMMs the baselines use nor any fold classification. That independence is the point;
it is a different kind of evidence from Pfam or Swiss-Prot rather than more of the same.

Two limits, both real, both stated here so the numbers are not over-read:

  Coverage is small. 955 M-CSA proteins intersect this benchmark at 314 accessions, 95 of
  them human -- 0.5% of the human query set. That is a VIGNETTE, not a population
  statistic, and it should be read the way the MHC block is. Do not let an n=95 stratum
  carry a headline claim.

  Catalytic residues are points, not domains. Each is one residue, so a boundary IoU
  against it is meaningless. They are widened to a small window and flagged is_point, and
  the useful question is recall -- does the tool put a region on the catalytic machinery --
  not how precisely it drew an edge.

Numbering comes from the API's `residue_sequences`, which is UNIPROT-numbered. The
curated_data.csv flat file gives PDB numbering with a separate chain column, which needs
SIFTS to map and is silently wrong if used directly.

The higher-value use of M-CSA is not this truth set at all: Folddisco published an M-CSA
benchmark during review (713 queries, sensitivity-to-first-FP) with per-query results
deposited, so running that query set makes kmerseek's numbers directly comparable to a
published NBT table. That is a separate exercise from the QfO domain sweep.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import polars as pl

API = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json&page_size=100"


def fetch_entries(url: str, max_pages: int = 50) -> list[dict]:
    entries, page = [], 0
    while url and page < max_pages:
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.load(r)
        entries.extend(d.get("results", []))
        url = d.get("next")
        page += 1
        # The API is a courtesy; do not hammer it.
        time.sleep(0.3)
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--truth-out", required=True, type=Path)
    p.add_argument("--map-outdir", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    p.add_argument("--window", type=int, default=5,
                   help="residues either side of a catalytic residue, so a point becomes a "
                        "scoreable interval; a 1-residue truth cannot be scored at all")
    args = p.parse_args()

    species_acc = {}
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        sp = path.name.replace("_pfam_domains.parquet", "")
        species_acc[sp] = set(pl.read_parquet(path)["accession"].unique().to_list())
    if "human" not in species_acc:
        raise SystemExit("no human_pfam_domains.parquet; cannot define the query side")
    wanted = set().union(*species_acc.values())

    entries = fetch_entries(API)
    print(f"fetched {len(entries)} M-CSA entries")

    rows = []
    for e in entries:
        mcsa = f"MCSA{e.get('mcsa_id')}"
        for res in e.get("residues", []):
            for rs in res.get("residue_sequences", []):
                acc, resid = rs.get("uniprot_id"), rs.get("resid")
                if not acc or acc not in wanted or not resid:
                    continue
                start = max(1, int(resid) - args.window)
                end = int(resid) + args.window
                # Labelled by M-CSA entry, so "same family" means the same catalytic
                # mechanism -- the transfer question becomes "does this target carry the
                # same machinery", which is what M-CSA actually asserts.
                rows.append((acc, mcsa, start, end, int(resid)))

    df = pl.DataFrame(
        rows,
        schema=["accession", "pfam_id", "domain_start", "domain_end", "catalytic_resid"],
        orient="row",
    ).unique()
    df = df.with_columns(
        pl.lit(None, dtype=pl.Int32).alias("protein_length"),
        pl.lit(True).alias("is_point"),
    )

    human = df.filter(pl.col("accession").is_in(species_acc["human"]))
    human.write_parquet(args.truth_out, compression="zstd")

    args.map_outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_entries": len(entries),
        "window": args.window,
        "note": "catalytic residues are POINT features widened by --window; small coverage, "
                "read as a vignette not a population statistic",
        "human": {"n_residues": human.height, "n_proteins": human["accession"].n_unique()},
    }
    for sp, accs in species_acc.items():
        if sp == "human":
            continue
        sub = df.filter(pl.col("accession").is_in(accs))
        sub.write_parquet(args.map_outdir / f"{sp}_domain_map.parquet", compression="zstd")
        summary[sp] = {"n_residues": sub.height, "n_proteins": sub["accession"].n_unique()}

    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
