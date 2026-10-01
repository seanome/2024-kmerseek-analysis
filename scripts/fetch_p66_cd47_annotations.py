#!/usr/bin/env python3
"""Fetch the P66 and CD47 sequences and every UniProt feature, in both numberings.

Two numbering systems are in play for each protein and mixing them is the main risk
in notebook 245, so every row carries both:

  P66  (UniProt H7C7N8, Borreliella burgdorferi): 618-residue precursor,
       SIGNAL 1..21, CHAIN 22..618, so the mature protein is 597 aa.
       mature = uniprot - 21.
  CD47 (UniProt Q08722, human): 323-residue precursor, SIGNAL 1..18,
       CHAIN 19..323, mature 305 aa. mature = uniprot - 18.

The published residues are in mature numbering in both papers, so they are written
here with the offset applied, and each one is checked against the fetched sequence:
the script fails if the residue at a named position is not the residue the name says.

  P66:  the loop required for integrin binding, mature 181-187, is UniProt 202-208,
        with the two aspartates D184 and D186 at UniProt 205 and 207.
  CD47: the residues that contact SIRP-alpha, mature Tyr37, Asp46, Glu97, Thr99,
        Glu100, Thr102, Arg103, Glu104, Glu106, are UniProt 55, 64, 115, 117, 118,
        120, 121, 122, 124.

Writes tables/245_p66_cd47_annotations.csv.

Usage:
  fetch_p66_cd47_annotations.py [--out tables/245_p66_cd47_annotations.csv]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tables" / "245_p66_cd47_annotations.csv"

API = "https://rest.uniprot.org/uniprotkb/{acc}.json"

# protein -> (accession, precursor length, signal peptide end = mature offset)
PROTEINS = {
    "P66": ("H7C7N8", 618, 21),
    "CD47": ("Q08722", 323, 18),
}

# The residues from the literature, in the mature numbering the papers use.
# (protein, name, first mature position, last mature position, expected residues, source)
PUBLISHED = [
    ("P66", "loop required for integrin binding, proposed to bind SIRP-alpha",
     181, 187, None,
     "Coburn et al., integrin-binding loop of P66 (mature 181-187)"),
    ("P66", "Asp184", 184, 184, "D",
     "Coburn et al., integrin-binding loop of P66 (mature 181-187)"),
    ("P66", "Asp186", 186, 186, "D",
     "Coburn et al., integrin-binding loop of P66 (mature 181-187)"),
]
CD47_CONTACTS = [("Tyr37", 37, "Y"), ("Asp46", 46, "D"), ("Glu97", 97, "E"),
                 ("Thr99", 99, "T"), ("Glu100", 100, "E"), ("Thr102", 102, "T"),
                 ("Arg103", 103, "R"), ("Glu104", 104, "E"), ("Glu106", 106, "E")]
CD47_SOURCE = "Hatherley et al., CD47 residues contacting SIRP-alpha (mature numbering)"
for _name, _pos, _res in CD47_CONTACTS:
    PUBLISHED.append(("CD47", _name, _pos, _pos, _res, CD47_SOURCE))


def fetch(acc: str) -> dict:
    with urllib.request.urlopen(API.format(acc=acc), timeout=60) as fh:
        return json.loads(fh.read().decode())


def evidence_codes(feature: dict) -> str:
    """The ECO codes on a feature, so a predicted feature is distinguishable from a
    measured one. Empty when UniProt gives none."""
    return ";".join(e.get("evidenceCode", "") for e in feature.get("evidences", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows, seqs = [], {}
    for protein, (acc, expect_len, offset) in PROTEINS.items():
        entry = fetch(acc)
        seq = entry["sequence"]["value"]
        if len(seq) != expect_len:
            raise SystemExit(
                f"{protein} ({acc}): UniProt returned {len(seq)} residues, expected "
                f"{expect_len}. The numbering in notebook 245 is built on {expect_len}; "
                "check the entry before going further."
            )
        seqs[protein] = seq
        rows.append(dict(
            protein=protein, accession=acc, feature_type="SEQUENCE", name="full sequence",
            start_uniprot=1, end_uniprot=len(seq),
            start_mature=1 - offset, end_mature=len(seq) - offset,
            evidence="", source="UniProt", sequence=seq,
        ))
        for ft in entry.get("features", []):
            loc = ft["location"]
            if loc["start"]["value"] is None or loc["end"]["value"] is None:
                continue
            start, end = int(loc["start"]["value"]), int(loc["end"]["value"])
            rows.append(dict(
                protein=protein, accession=acc,
                feature_type=ft["type"], name=ft.get("description", ""),
                start_uniprot=start, end_uniprot=end,
                start_mature=start - offset, end_mature=end - offset,
                evidence=evidence_codes(ft), source="UniProt",
                sequence=seq[start - 1:end],
            ))

    # The published residues, with the offset applied and each one checked.
    for protein, name, start_m, end_m, expect, source in PUBLISHED:
        acc, _, offset = PROTEINS[protein]
        start_u, end_u = start_m + offset, end_m + offset
        got = seqs[protein][start_u - 1:end_u]
        if expect is not None and got != expect:
            raise SystemExit(
                f"{protein} {name}: mature {start_m} is UniProt {start_u}, which holds "
                f"{got!r}, not {expect!r}. Either the offset or the cited position is "
                "wrong; nothing downstream is trustworthy until this agrees."
            )
        rows.append(dict(
            protein=protein, accession=acc, feature_type="PUBLISHED", name=name,
            start_uniprot=start_u, end_uniprot=end_u,
            start_mature=start_m, end_mature=end_m,
            evidence="", source=source, sequence=got,
        ))

    # The P66 loop must hold an aspartate at both of the positions the paper names.
    loop = seqs["P66"][202 - 1:208]
    if loop[205 - 202] != "D" or loop[207 - 202] != "D":
        raise SystemExit(f"P66 UniProt 202-208 is {loop!r}; expected D at 205 and 207.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(args.out)
    print(f"wrote {args.out}: {df.height} rows")
    print(f"P66 loop, UniProt 202-208 = mature 181-187: {loop}")
    print("CD47 contact residues, UniProt: " + ", ".join(
        f"{p}{seqs['CD47'][p - 1]}" for p in (55, 64, 115, 117, 118, 120, 121, 122, 124)))


if __name__ == "__main__":
    main()
