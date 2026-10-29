"""Build the nb 357 condensate-region table from PhaSePro and DisProt.

Rule (notebooks/357_preregistration.md, "The new index"): every PhaSePro region and every
DisProt region annotated with a phase-separation term, cut out of its protein sequence with
10-residue flanks, human and mouse entries removed.

DisProt's only phase-separation term in the current release is GO:0140693, "molecular
condensate scaffold activity" (a protein that drives formation of a condensate by phase
separation). Its other term matched by the word "phase", GO:0072766 ("centromere clustering
at the mitotic interphase nuclear envelope"), is not about phase separation and is left out.

Sequences come from the database records themselves (PhaSePro `sequence`, DisProt
`sequence`), which both copy from UniProt. Coordinates are 1-based and inclusive in both.

Usage:
    python scripts/357_condensate_regions.py PHASEPRO_JSON DISPROT_JSON OUT_TSV
"""

import hashlib
import json
import sys
from pathlib import Path

import polars as pl

#: Residues kept on each side of a region (spec: "10-residue flanks").
FLANK = 10
#: Species removed: set A is human, and mouse is the nearest model (nb 244 filter).
EXCLUDED_TAXA = {9606: "Homo sapiens", 10090: "Mus musculus"}
#: DisProt term for phase separation.
DISPROT_LLPS_TERM = "GO:0140693"


def md5(path: Path) -> str:
    """md5 of a file, hex."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def cut(sequence: str, start: int, end: int) -> tuple[str, int, int]:
    """Region start..end (1-based, inclusive) plus FLANK residues each side.

    Returns the cut sequence and its 1-based first and last residue in the protein.
    """
    first = max(1, start - FLANK)
    last = min(len(sequence), end + FLANK)
    return sequence[first - 1 : last], first, last


def phasepro_rows(path: Path) -> list[dict]:
    """One row per PhaSePro region. PhaSePro gives organism names, not taxon ids."""
    rows = []
    for acc, entry in json.loads(path.read_text()).items():
        for part in entry["boundaries"].split(";"):
            start, end = (int(x) for x in part.strip().split("-"))
            rows.append(
                dict(
                    source="PhaSePro",
                    region_id=f"phasepro_{entry['id']}_{start}_{end}",
                    accession=acc,
                    protein_name=entry["common_name"],
                    organism=entry["organism"],
                    taxon_id=None,
                    term=entry["segment"],
                    region_start=start,
                    region_end=end,
                    sequence=entry["sequence"],
                    release_or_date=entry["date"],
                )
            )
    return rows


def disprot_rows(path: Path) -> list[dict]:
    """One row per DisProt region annotated with DISPROT_LLPS_TERM."""
    rows = []
    for entry in json.loads(path.read_text())["data"]:
        for region in entry["regions"]:
            if region["term_id"] != DISPROT_LLPS_TERM:
                continue
            rows.append(
                dict(
                    source="DisProt",
                    region_id=region["region_id"],
                    accession=entry["acc"],
                    protein_name=entry["name"],
                    organism=entry["organism"],
                    taxon_id=entry["ncbi_taxon_id"],
                    term=f"{region['term_id']} {region['term_name']}",
                    region_start=region["start"],
                    region_end=region["end"],
                    sequence=entry["sequence"],
                    release_or_date=region["released"],
                )
            )
    return rows


def newest_disprot_release(path: Path) -> str:
    """Newest `released` tag over every DisProt region, a stand-in for the release name."""
    return max(
        region["released"]
        for entry in json.loads(path.read_text())["data"]
        for region in entry["regions"]
    )


def main(phasepro: Path, disprot: Path, out: Path) -> None:
    rows = phasepro_rows(phasepro) + disprot_rows(disprot)
    # A region that ends past the stored sequence means the database's coordinates refer
    # to another version of the protein, so none of that protein's coordinates are trusted.
    bad_coordinates = {
        (r["source"], r["accession"])
        for r in rows
        if not 1 <= r["region_start"] <= r["region_end"] <= len(r["sequence"])
    }
    excluded_names = set(EXCLUDED_TAXA.values())
    for row in rows:
        row["coordinates_past_sequence_end"] = (
            row["source"],
            row["accession"],
        ) in bad_coordinates
        seq = row.pop("sequence")
        protein_length = len(seq)
        excluded_species = (
            row["taxon_id"] in EXCLUDED_TAXA or row["organism"] in excluded_names
        )
        if row["coordinates_past_sequence_end"]:
            row["protein_length"] = protein_length
            row["region_sequence"] = row["indexed_sequence"] = None
            row["indexed_first_residue"] = row["indexed_last_residue"] = None
            row["excluded_species"] = excluded_species
            continue
        row["protein_length"] = protein_length
        row["region_sequence"] = seq[row["region_start"] - 1 : row["region_end"]]
        (
            row["indexed_sequence"],
            row["indexed_first_residue"],
            row["indexed_last_residue"],
        ) = cut(seq, row["region_start"], row["region_end"])
        row["excluded_species"] = excluded_species
    table = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        (pl.col("region_end") - pl.col("region_start") + 1).alias("region_length_aa"),
        (~pl.col("excluded_species") & ~pl.col("coordinates_past_sequence_end")).alias(
            "in_index"
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out, separator="\t")
    provenance = out.with_suffix(".provenance.tsv")
    provenance.write_text(
        "\n".join(
            [
                f"PhaSePro file\t{phasepro.name}\tmd5 {md5(phasepro)}",
                "PhaSePro URL\thttps://phasepro.elte.hu/download_full.json",
                "PhaSePro release\tno numbered release; newest entry date "
                + max(r["release_or_date"] for r in rows if r["source"] == "PhaSePro"),
                f"DisProt file\t{disprot.name}\tmd5 {md5(disprot)}",
                "DisProt URL\thttps://disprot.org/api/search?release=current"
                "&show_ambiguous=true&show_obsolete=false&format=json&page_size=5000",
                "DisProt release\tcurrent on 2026-10-20; newest region release tag "
                + newest_disprot_release(disprot),
                f"DisProt term\t{DISPROT_LLPS_TERM}",
                f"flank\t{FLANK}",
                f"region table md5\t{md5(out)}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
