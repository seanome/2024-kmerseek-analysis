#!/usr/bin/env python3
"""Write one protein per gene for the four species OrthoFinder groups in notebook 282.

OrthoFinder counts proteins, not genes. A gene with three isoforms in the input becomes
three "copies" in its family, so every proteome here is cut to one protein per gene
before the run. Each output header is a short gene ID with no spaces or "|", because
OrthoFinder keeps the header up to the first space as the gene name.

Species and what "one per gene" means for each:

- Botryllus schlosseri (`botryllus`): the proteome the dark-set run used,
  `Eukaryota/BOTSCH2026_30301.fasta` (45,339 records). 2,019 gene names there carry 2-6
  records each, renamed `<name>.1` .. `<name>.n` by
  `nextflow-runs/qfo-pfam-region-benchmark/bin/prepare_botryllus_fasta.py` (isoforms).
  The longest record per name is kept. Swap in Ayelet Voskoboynik's newer genome with
  `--botryllus`.
- Human (`human`): the 19,732 GENCODE v49 canonical proteins of notebooks 241-256, already
  one per gene. ID = Ensembl gene ID without its version, so it joins to OHNOLOGS v2.
- Amphioxus (`amphioxus`): Branchiostoma lanceolatum, the reference protein per coding
  locus from Marletaz et al. 2018 (`Bla_annot_v4_best_Aac.fa.gz`, 35,956 records from
  amphiencode.github.io/Data). ID = the BL gene ID, so it joins to their Supplementary
  Datasets 7 and 9.
- Ciona (`ciona`): the Quest for Orthologs 2020_04 reference proteome UP000008144
  (16,678 records, one per gene). ID = the UniProt accession.

DIAMOND stops on any character that is not a letter. The amphioxus proteins end in "."
(a stop codon, 29,538 of 35,956), so a trailing "." or "*" is removed and any other
non-letter becomes X, the unknown-residue code. Both are counted in `inputs.json`.

Writes `<out>/{botryllus,human,amphioxus,ciona}.fa`, `<out>/gene_ids.tsv` (species, gene
ID, original header, length) and `<out>/inputs.json` (path, sha256, records in and out).

Usage:
    python3 scripts/prepare_282_proteomes.py --out ~/data/282-ohnolog-families/proteomes
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

QFO = (
    Path.home()
    / "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
)
DEFAULTS = {
    "botryllus": QFO / "Eukaryota/BOTSCH2026_30301.fasta",
    "human": Path.home()
    / "data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    "amphioxus": Path.home() / "data/marletaz2018/Bla_annot_v4_best_Aac.fa.gz",
    "ciona": QFO / "Eukaryota/UP000008144_7719.fasta",
}


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """(header without '>', sequence) in file order; reads .gz transparently."""
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def clean_sequence(seq: str) -> tuple[str, int, int]:
    """Drop one trailing stop ('.' or '*'), turn any other non-letter into X.

    Returns the sequence, 1 if a stop was dropped (else 0), and the number of X written.
    """
    stop = 1 if seq[-1:] in (".", "*") else 0
    if stop:
        seq = seq[:-1]
    cleaned = "".join(c if c.isalpha() else "X" for c in seq)
    return cleaned, stop, sum(a != b for a, b in zip(seq, cleaned))


def gene_id(species: str, header: str) -> str:
    """The one gene ID each species' header carries (see the module docstring)."""
    first = header.split()[0]
    if species == "botryllus":
        # FUN000023_UBE2D2B, or FUN000023_UBE2D2B.2 for the second isoform of that name.
        return re.sub(r"\.\d+$", "", first)
    if species == "human":
        # ENSP...|ENST...|ENSG00000186092.7|... -> ENSG00000186092
        return first.split("|")[2].split(".")[0]
    if species == "amphioxus":
        # BL61971_cuf5 -> BL61971
        return first.split("_")[0]
    if species == "ciona":
        # tr|A0A140TAT7|A0A140TAT7_CIOIN -> A0A140TAT7
        return first.split("|")[1]
    raise ValueError(species)


def longest_per_gene(
    species: str, records: list[tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """gene ID -> (header, sequence) of its longest protein; ties go to the first in the file."""
    best: dict[str, tuple[str, str]] = {}
    for header, seq in records:
        gid = gene_id(species, header)
        if gid not in best or len(seq) > len(best[gid][1]):
            best[gid] = (header, seq)
    return best


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    for species, default in DEFAULTS.items():
        parser.add_argument(f"--{species}", type=Path, default=default)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    with open(args.out / "gene_ids.tsv", "w") as ids:
        ids.write("species\tgene_id\toriginal_header\tlength_aa\n")
        for species in DEFAULTS:
            path = getattr(args, species)
            raw = read_fasta(path)
            records, n_stops, n_x = [], 0, 0
            for header, seq in raw:
                seq, stop, x = clean_sequence(seq)
                records.append((header, seq))
                n_stops, n_x = n_stops + stop, n_x + x
            best = longest_per_gene(species, records)
            with open(args.out / f"{species}.fa", "w") as fa:
                for gid, (header, seq) in best.items():
                    fa.write(f">{gid}\n")
                    for i in range(0, len(seq), 60):
                        fa.write(seq[i : i + 60] + "\n")
                    ids.write(f"{species}\t{gid}\t{header}\t{len(seq)}\n")
            inputs[species] = {
                "path": str(path),
                "sha256": sha256(path),
                "records_in": len(records),
                "genes_out": len(best),
                "trailing_stops_removed": n_stops,
                "other_non_letters_to_x": n_x,
            }
            print(
                f"{species}: {len(records):_} records -> {len(best):_} genes; "
                f"{n_stops:_} trailing stops removed, {n_x:_} other non-letters -> X"
            )
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
