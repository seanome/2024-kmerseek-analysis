#!/usr/bin/env python3
"""Fetch the P66 and CD47 sequences, every UniProt feature, and the published residues.

Every position is written in both numberings, because mixing them is the main way to
get a wrong answer in notebook 245:

  P66  (UniProt H7C7N8, Borreliella burgdorferi): 618-residue precursor,
       SIGNAL 1..21, CHAIN 22..618, mature 597 aa, mature = UniProt - 21.
  CD47 (UniProt Q08722, human): 323-residue precursor, SIGNAL 1..18,
       CHAIN 19..323, mature 305 aa, mature = UniProt - 18.

Where the published positions come from, and which numbering each paper uses:

P66, the loop required for integrin binding. Ristow et al. (2015) deleted amino acids
202-208 and mutated the two aspartates at 205 and 207, and both changes cut binding to
integrin alpha-v beta-3 by two to three orders of magnitude. Defoe and Coburn (2001)
found that a synthetic peptide of amino acids 203-209, ENDKDTP, competed with whole
bacteria for binding to integrin alpha-IIb beta-3 while its scrambled version DNEKPDT did
not, and that no other peptide across 142-384 competed. Both
papers number the precursor, the same numbering as the UniProt entry: UniProt H7C7N8
has aspartate at 205 and at 207. No offset is applied to these positions.

  Ristow, L. C., M. Bonde, Y.-P. Lin, H. Sato, M. Curtis, E. Wesley, B. L. Hahn, et al.
  "Integrin binding by Borrelia burgdorferi P66 facilitates dissemination but is not
  required for infectivity." *Cellular Microbiology* 17, no. 7 (2015): 1021-36.
  https://doi.org/10.1111/cmi.12418
  Defoe, G., and J. Coburn. "Delineation of Borrelia burgdorferi p66 sequences required
  for integrin alpha(IIb)beta(3) recognition." *Infection and Immunity* 69, no. 5
  (2001): 3455-59. https://doi.org/10.1128/IAI.69.5.3455-3459.2001

CD47, the residues that contact SIRP-alpha. Rather than copy a residue list out of the
paper, this script reads the structure the paper deposited, PDB 2JJS, the 1.85 A complex
of human CD47 with human SIRP-alpha, and records every CD47 residue with an atom within
4 A of SIRP-alpha. The structure numbers the mature chain; its UniProt mapping (PDBe
SIFTS) gives author residue 1 = UniProt 19, which is the 18-residue offset above.

  Hatherley, D., S. C. Graham, J. Turner, K. Harlos, D. I. Stuart, and A. N. Barclay.
  "Paired receptor specificity explained by structures of signal regulatory proteins
  alone and complexed with CD47." *Molecular Cell* 31, no. 2 (2008): 266-77.
  https://doi.org/10.1016/j.molcel.2008.05.026

Both sequence lengths and the residue at every published position are checked against
the fetched sequences. The script fails if any of them disagrees.

Writes tables/245_p66_cd47_annotations.csv.

Usage:
  fetch_p66_cd47_annotations.py [--out tables/245_p66_cd47_annotations.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tables" / "245_p66_cd47_annotations.csv"

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{acc}.json"
PDB_CIF = "https://files.rcsb.org/download/{pdb}.cif"
SIFTS_API = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"

# protein -> (accession, precursor length, signal peptide end = mature offset)
PROTEINS = {
    "P66": ("H7C7N8", 618, 21),
    "CD47": ("Q08722", 323, 18),
}

RISTOW = ("Ristow et al. 2015, Cell Microbiol 17:1021-36, "
          "https://doi.org/10.1111/cmi.12418")
DEFOE = ("Defoe and Coburn 2001, Infect Immun 69:3455-59, "
         "https://doi.org/10.1128/IAI.69.5.3455-3459.2001")
HATHERLEY = ("Hatherley et al. 2008, Mol Cell 31:266-77, PDB 2JJS, "
             "https://doi.org/10.1016/j.molcel.2008.05.026")

# P66, in the precursor numbering both papers use, which is UniProt numbering.
# (name, first, last, expected residues or None, source)
P66_PUBLISHED = [
    ("loop required for integrin binding, proposed to bind SIRP-alpha; "
     "deleting it cuts integrin alpha-v beta-3 binding ~1370-fold", 202, 208, "QENDKDT",
     RISTOW),
    ("Asp205; mutating it and Asp207 to alanine cuts integrin binding ~225-fold",
     205, 205, "D", RISTOW),
    ("Asp207; mutating it and Asp205 to alanine cuts integrin binding ~225-fold",
     207, 207, "D", RISTOW),
    # Defoe and Coburn Table 2: peptide 203-9 is ENDKDTP, and its scrambled version
    # DNEKPDT does not compete, so the order of the residues is what matters.
    ("synthetic peptide that competes with whole bacteria for integrin "
     "alpha-IIb beta-3 binding; its scrambled version does not", 203, 209, "ENDKDTP",
     DEFOE),
]

CD47_PDB = "2JJS"
# CD47 chain -> the SIRP-alpha chain it is in contact with, in 2JJS.
CD47_PAIRS = {"C": "A", "D": "B"}
CONTACT_CUTOFF_A = 4.0

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Pyroglutamate: the mature N-terminal glutamine, cyclised.
    "PCA": "Q",
}


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as fh:
        return json.loads(fh.read().decode())


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as fh:
        return fh.read().decode()


def evidence_codes(feature: dict) -> str:
    """The ECO codes on a feature, so a predicted feature is distinguishable from a
    measured one. Empty when UniProt gives none."""
    return ";".join(e.get("evidenceCode", "") for e in feature.get("evidences", []))


def sifts_offset(pdb: str, chain: str, accession: str) -> int:
    """UniProt number minus author residue number for this chain, from PDBe SIFTS.
    Checked to be one constant offset over the whole chain."""
    mappings = fetch_json(SIFTS_API.format(pdb=pdb.lower()))[pdb.lower()]["UniProt"]
    if accession not in mappings:
        raise SystemExit(f"{pdb}: no {accession} mapping")
    for m in mappings[accession]["mappings"]:
        if m["chain_id"] != chain:
            continue
        start_author = m["start"]["author_residue_number"]
        if start_author is None:
            start_author = m["start"]["residue_number"]
        offset = int(m["unp_start"]) - int(start_author)
        span = int(m["unp_end"]) - int(m["unp_start"])
        end_author = m["end"]["author_residue_number"]
        if end_author is not None and int(end_author) - int(start_author) != span:
            raise SystemExit(f"{pdb} chain {chain}: the UniProt mapping is not one "
                             "constant offset; the contact numbering needs checking")
        return offset
    raise SystemExit(f"{pdb}: no mapping for chain {chain}")


def cif_atoms(cif: str) -> list[dict]:
    """Every atom of the first model, from the mmCIF atom_site loop."""
    out = []
    for line in cif.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        f = line.split()
        if f[-1] != "1":  # first model only
            continue
        if f[5] not in THREE_TO_ONE:  # waters, sugars, ions: not part of either chain
            continue
        out.append({"comp": f[5], "auth_seq": f[16], "auth_asym": f[18],
                    "xyz": (float(f[10]), float(f[11]), float(f[12]))})
    return out


def cd47_contacts(offset: int) -> dict[int, dict]:
    """CD47 residues with an atom within the cutoff of SIRP-alpha in PDB 2JJS, keyed by
    UniProt position. Both copies of the complex in the crystal are measured, and the
    row records whether they agree, so a contact seen in only one copy is visible."""
    atoms = cif_atoms(fetch_text(PDB_CIF.format(pdb=CD47_PDB)))
    by_chain: dict[str, list[dict]] = {}
    for a in atoms:
        by_chain.setdefault(a["auth_asym"], []).append(a)

    per_copy: dict[str, dict[int, float]] = {}
    for cd47_chain, sirp_chain in CD47_PAIRS.items():
        near: dict[int, float] = {}
        for a in by_chain[cd47_chain]:
            for b in by_chain[sirp_chain]:
                d2 = sum((x - y) ** 2 for x, y in zip(a["xyz"], b["xyz"]))
                if d2 <= CONTACT_CUTOFF_A ** 2:
                    pos = int(a["auth_seq"]) + offset
                    near[pos] = min(near.get(pos, math.inf), math.sqrt(d2))
        per_copy[cd47_chain] = near

    residue_name = {int(a["auth_seq"]) + offset: a["comp"]
                    for a in by_chain["C"]}
    out = {}
    for pos in sorted(set().union(*(d.keys() for d in per_copy.values()))):
        dists = {c: d[pos] for c, d in per_copy.items() if pos in d}
        out[pos] = {
            "residue": THREE_TO_ONE.get(residue_name.get(pos, ""), "?"),
            "min_distance_angstrom": round(min(dists.values()), 2),
            "n_copies": len(dists),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows, seqs = [], {}
    for protein, (acc, expect_len, offset) in PROTEINS.items():
        entry = fetch_json(UNIPROT_API.format(acc=acc))
        seq = entry["sequence"]["value"]
        if len(seq) != expect_len:
            raise SystemExit(
                f"{protein} ({acc}): UniProt returned {len(seq)} residues, expected "
                f"{expect_len}. The numbering in notebook 245 is built on {expect_len}; "
                "check the entry before going further.")
        seqs[protein] = seq
        rows.append(dict(
            protein=protein, accession=acc, feature_type="SEQUENCE", name="full sequence",
            start_uniprot=1, end_uniprot=len(seq),
            start_mature=1 - offset, end_mature=len(seq) - offset,
            evidence="", source="UniProt", sequence=seq, distance_angstrom=None,
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
                sequence=seq[start - 1:end], distance_angstrom=None,
            ))

    # P66: the published positions are already in the precursor numbering.
    p66_acc, _, p66_offset = PROTEINS["P66"]
    for name, start_u, end_u, expect, source in P66_PUBLISHED:
        got = seqs["P66"][start_u - 1:end_u]
        if expect is not None and got != expect:
            raise SystemExit(
                f"P66 {name}: UniProt {start_u} holds {got!r}, not {expect!r}. Either "
                "the cited position or the entry is wrong; nothing downstream is "
                "trustworthy until this agrees.")
        rows.append(dict(
            protein="P66", accession=p66_acc, feature_type="PUBLISHED", name=name,
            start_uniprot=start_u, end_uniprot=end_u,
            start_mature=start_u - p66_offset, end_mature=end_u - p66_offset,
            evidence="", source=source, sequence=got, distance_angstrom=None,
        ))

    # CD47: every contact with SIRP-alpha, measured in the deposited structure.
    cd47_acc, _, cd47_offset = PROTEINS["CD47"]
    pdb_offset = sifts_offset(CD47_PDB, "C", cd47_acc)
    if pdb_offset != cd47_offset:
        raise SystemExit(
            f"PDB {CD47_PDB} maps author residue 1 to UniProt {pdb_offset + 1}, but the "
            f"signal peptide gives an offset of {cd47_offset}. The contact positions "
            "would be shifted; check both before going further.")
    contacts = cd47_contacts(pdb_offset)
    if not contacts:
        raise SystemExit(f"no CD47 contacts found in {CD47_PDB}; the structure or the "
                         "parsing is wrong")
    for pos, c in contacts.items():
        got = seqs["CD47"][pos - 1]
        if c["residue"] != got:
            raise SystemExit(
                f"CD47 {pos}: the structure has {c['residue']} there, the UniProt "
                f"sequence has {got}. The two numberings do not line up.")
        rows.append(dict(
            protein="CD47", accession=cd47_acc, feature_type="CONTACT",
            name=f"{got}{pos - cd47_offset} contacts SIRP-alpha "
                 f"({c['min_distance_angstrom']} A, in {c['n_copies']} of "
                 f"{len(CD47_PAIRS)} copies of the complex)",
            start_uniprot=pos, end_uniprot=pos,
            start_mature=pos - cd47_offset, end_mature=pos - cd47_offset,
            evidence="", source=HATHERLEY, sequence=got,
            distance_angstrom=c["min_distance_angstrom"],
        ))

    # The loop must hold an aspartate at both positions the paper names.
    loop = seqs["P66"][202 - 1:208]
    if loop[205 - 202] != "D" or loop[207 - 202] != "D":
        raise SystemExit(f"P66 UniProt 202-208 is {loop!r}; expected D at 205 and 207.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(args.out)
    print(f"wrote {args.out}: {df.height} rows")
    print(f"P66 loop, UniProt 202-208 = mature 181-187: {loop}")
    print(f"CD47 residues contacting SIRP-alpha within {CONTACT_CUTOFF_A} A in "
          f"{CD47_PDB} ({len(contacts)}):")
    print("  " + ", ".join(
        f"{c['residue']}{p} (mature {p - cd47_offset}, {c['min_distance_angstrom']} A"
        + ("" if c["n_copies"] == len(CD47_PAIRS) else ", 1 copy only") + ")"
        for p, c in contacts.items()))


if __name__ == "__main__":
    main()
