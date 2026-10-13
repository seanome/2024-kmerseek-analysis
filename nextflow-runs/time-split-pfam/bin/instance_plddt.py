#!/usr/bin/env python3
"""Mean AlphaFold pLDDT of each instance's query residues inside its t1 domain (notebook 280).

    instance_plddt.py --structures <dir> --inputs <dir> [<dir> ...] --out <tsv>

<dir> under --structures holds AF-<accession>-F1.cif files, as
nextflow-runs/qfo-pfam-region-benchmark/bin/fetch_alphafold_structures.sh writes them.
Each --inputs dir is a make_search_inputs.py queries output (queries.fasta, instances.tsv).

pLDDT is read with read_plddt() from qfo-pfam-region-benchmark's build_query_covariates.py
(one value per CA atom, the same parse as notebook 202). A model counts only when its
residues are the query's sequence: AlphaFold DB models the current UniProt sequence, which
can differ from the Pfam 33.1 one. Bands: < 50, 50-70, >= 70, and "no model" (no file, or
a model of a different sequence).

Writes one row per instance: query, pfam_acc, t1_start, t1_end, model (yes / none /
other sequence), mean_plddt_region, plddt_band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "qfo-pfam-region-benchmark/bin"))
from build_query_covariates import CA_ATOM_NAME_FIELD, read_plddt  # noqa: E402
from make_search_inputs import read_fasta  # noqa: E402

COMP_ID_FIELD = 5  # label_comp_id in AlphaFold DB mmCIF atom_site rows
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BANDS = ("< 50", "50-70", ">= 70", "no model")


def model_sequence(path: Path) -> str:
    """The model's residues, one letter per CA atom, in file order (unknown residues X)."""
    with open(path) as fh:
        return "".join(
            THREE_TO_ONE.get(p[COMP_ID_FIELD], "X")
            for line in fh
            if line.startswith("ATOM")
            and len(p := line.split()) > COMP_ID_FIELD
            and p[CA_ATOM_NAME_FIELD] == "CA"
        )


def band(v: float | None) -> str:
    if v is None:
        return "no model"
    return "< 50" if v < 50 else ("50-70" if v < 70 else ">= 70")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    seqs, inst = {}, []
    for d in a.inputs:
        seqs.update(read_fasta(d / "queries.fasta"))
        inst.append(
            pl.read_csv(d / "instances.tsv", separator="\t").select(
                "query", "pfam_acc", "t1_start", "t1_end"
            )
        )
    inst = pl.concat(inst).unique().sort("query", "t1_start", "pfam_acc")

    tracks, status = {}, {}
    for acc in inst["query"].unique().sort().to_list():
        path = a.structures / f"AF-{acc}-F1.cif"
        if not path.exists():
            status[acc] = "none"
            continue
        if model_sequence(path) != seqs[acc].upper():
            status[acc] = "other sequence"
            continue
        tracks[acc] = read_plddt(path)
        status[acc] = "yes"

    rows = []
    for q, fam, s, e in inst.iter_rows():
        t = tracks.get(q)
        v = sum(t[s - 1 : e]) / (e - s + 1) if t else None  # 1-based, end included
        rows.append((q, fam, s, e, status[q], v, band(v)))
    out = pl.DataFrame(
        rows,
        orient="row",
        schema=["query", "pfam_acc", "t1_start", "t1_end", "model", "mean_plddt_region", "plddt_band"],
    )
    out.write_csv(a.out, separator="\t")
    print(out.group_by("model").len().sort("model"))
    print(out.group_by("plddt_band").len().sort("plddt_band"))


if __name__ == "__main__":
    main()
