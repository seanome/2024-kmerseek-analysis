#!/usr/bin/env python3
"""Does each ELM ortholog have an AlphaFold model that matches its QfO sequence? (notebook 253)

Foldseek and Reseek search AlphaFold models, so an ortholog with no model in the run's
structure folder cannot be hit by them, and one whose model is not its QfO sequence is hit in
a different numbering from the projected motif. reduce_elm_cover.py checks this for the human
query only; this checks the target side.

For every ortholog in assets/elm_cover_projections.tsv, per species:
  model        "same"     model length = QfO length and >= 95% identical residues
               "differs"  a model exists but fails that test
               "no model" no AF-<accession>-F1*.cif in structures/<species>
  qfo_length, model_length, identity (share of identical residues over the QfO length)

Writes one TSV. Run on Sherlock inside a job; it reads one CIF per ortholog (~3_000 files):
  srun -A ayeletv -p normal -t 30 -c 2 --mem 8G apptainer exec --bind /scratch <image> \
      python3 scripts/elm_ortholog_model_status.py --cover <data/elm-cover> \
      --projections <assets/elm_cover_projections.tsv> --out elm_ortholog_model_status.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# QfO 2020_04 reference proteome per species, as named under qfo/Eukaryota.
PROTEOMES = {
    "chicken": "UP000000539_9031",
    "mouse": "UP000000589_10090",
    "zebrafish": "UP000000437_7955",
    "ciona": "UP000008144_7719",
    "fly": "UP000000803_7227",
    "worm": "UP000001940_6239",
    "yeast": "UP000002311_559292",
    "arabidopsis": "UP000006548_3702",
}
# Same rule reduce_elm_cover.py applies to the human query models.
MIN_MODEL_IDENTITY = 0.95
THREE_TO_ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G",
                    HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S",
                    THR="T", TRP="W", TYR="Y", VAL="V", SEC="U", PYL="O")


def read_fasta(path: Path) -> dict[str, str]:
    """Accession -> sequence, from a UniProt FASTA (sp|ACC|NAME ...)."""
    seqs: dict[str, list[str]] = {}
    acc = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                head = line[1:].split()[0]
                acc = head.split("|")[1] if "|" in head else head
                seqs[acc] = []
            elif acc is not None:
                seqs[acc].append(line.strip())
    return {a: "".join(v) for a, v in seqs.items()}


def model_sequence(cif: Path) -> str:
    """The residues an AlphaFold model's ATOM records carry, in label_seq_id order."""
    res: dict[int, str] = {}
    with open(cif) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                f = line.split()
                res[int(f[8])] = THREE_TO_ONE.get(f[5], "X")
    return "".join(res[i] for i in sorted(res))


def species_status(cover: Path, species: str, orthologs: list[str]) -> pl.DataFrame:
    """One row per ortholog accession of one species."""
    qfo = read_fasta(cover / "qfo" / "Eukaryota" / f"{PROTEOMES[species]}.fasta")
    rows = []
    for acc in sorted(orthologs):
        seq = qfo.get(acc)
        cifs = sorted((cover / "structures" / species).glob(f"AF-{acc}-F1*.cif"))
        if not cifs:
            rows.append((species, acc, "no model", len(seq) if seq else None, None, None))
            continue
        model = model_sequence(cifs[0])
        identity = sum(a == b for a, b in zip(model, seq)) / len(seq) if seq else None
        same = seq is not None and len(model) == len(seq) and identity >= MIN_MODEL_IDENTITY
        rows.append((species, acc, "same" if same else "differs", len(seq) if seq else None,
                     len(model), identity))
    return pl.DataFrame(rows, orient="row", schema={
        "species": pl.String, "ortholog": pl.String, "model": pl.String,
        "qfo_length": pl.Int64, "model_length": pl.Int64, "identity": pl.Float64})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cover", type=Path, required=True, help="the region benchmark's data/elm-cover")
    ap.add_argument("--projections", type=Path, required=True, help="assets/elm_cover_projections.tsv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    proj = pl.read_csv(args.projections, separator="\t")
    frames = [species_status(args.cover, sp, proj.filter(pl.col("species") == sp)["ortholog"].unique().to_list())
              for sp in PROTEOMES if sp in set(proj["species"])]
    out = pl.concat(frames)
    out.write_csv(args.out, separator="\t")
    print(out.group_by("species", "model").len().sort("species", "model"))


if __name__ == "__main__":
    main()
