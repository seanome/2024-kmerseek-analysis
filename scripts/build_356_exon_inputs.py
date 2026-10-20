#!/usr/bin/env python3
"""Inputs for notebook 356: 1:1 ortholog pairs and the target proteins' exon table.

Pairs. For each of the 998 midi-plus human queries (notebooks/356_data/
midi_plus_query_accessions.txt), its 1:1 ortholog in chicken and zebrafish: a human protein
and a target protein in the same OMA group, each the group's only member from its species
among the proteins in that species' QfO FASTA. This is the rule of
scripts/elm_stage0_flanks.py (oma_pairs) and nextflow-runs/qfo-dnds-omega/bin/
build_ortholog_pairs.py, applied to the midi-plus queries instead of the ELM ones.

Exons. UniProt's genomic coordinates file shipped with the QfO 2020_04 release
(<proteome>_<taxon>_gcoord.xml) maps each protein onto the coding exons of one or more
Ensembl transcripts. Per target protein, the first transcript (by transcript ID) whose
exons reach the protein's last residue is kept. The reading frame step is done here from
each exon's coding length: with c0 coding nucleotides before the exon and n in it, the exon
codes residues c0 // 3 to (c0 + n - 1) // 3, 0-based and inclusive, so a residue whose codon
is split between two exons belongs to both. That interval is checked against UniProt's own
protein positions for the exon (column agrees_uniprot).

Every interval is written 0-based and end-exclusive: [prot_start, prot_end).

Writes, into --out-dir (default notebooks/356_data):
  orthologs_1to1.tsv   query_acc, species, target_acc
  exons.parquet        one row per (species, target protein, exon)
  exon_inputs.json     counts behind each step

Usage:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/build_356_exon_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
QFO = Path.home() / (
    "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota"
)
HUMAN = "UP000005640_9606"
#: species -> QfO proteome file stem. The target FASTAs' md5 matches the midi-plus run's
#: copy on Sherlock (data/midi-plus/qfo/Eukaryota), checked 2026-10-20.
SPECIES = {"chicken": "UP000000539_9031", "zebrafish": "UP000000437_7955"}
NS = "{http://www.ebi.ac.uk/proteins/api/doc/xsd/feature}"


def read_fasta(path: Path) -> dict[str, str]:
    """Accession -> sequence from a UniProt FASTA (>sp|ACC|NAME ...)."""
    seqs: dict[str, list[str]] = {}
    acc = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split("|")[1]
                seqs[acc] = []
            else:
                seqs[acc].append(line.strip())
    return {a: "".join(s) for a, s in seqs.items()}


def oma_groups(path: Path, keep: set[str]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            acc, db, value = line.rstrip("\n").split("\t")
            if db == "OMA" and acc in keep:
                groups[value].add(acc)
    return groups


def oma_pairs(human_keep: set[str], species: str) -> dict[str, str]:
    """Human accession -> target accession, 1:1 in OMA (elm_stage0_flanks.oma_pairs)."""
    stem = SPECIES[species]
    t_keep = set(read_fasta(QFO / f"{stem}.fasta"))
    hg = oma_groups(QFO / f"{HUMAN}.idmapping", human_keep)
    tg = oma_groups(QFO / f"{stem}.idmapping", t_keep)
    out = {}
    for grp in set(hg) & set(tg):
        if len(hg[grp]) == 1 and len(tg[grp]) == 1:
            out[next(iter(hg[grp]))] = next(iter(tg[grp]))
    return out


def _pos(loc: ET.Element, which: str) -> int | None:
    """Position of a begin/end child, or of a single-residue <position> child."""
    el = loc.find(f"{NS}{which}")
    if el is None:
        el = loc.find(f"{NS}position")
    if el is None or el.get("position") is None:
        return None
    return int(el.get("position"))


def parse_gcoord(path: Path, keep: set[str]) -> dict[str, list[dict]]:
    """Accession -> list of transcripts, each {transcript, gene, exons: [...]}."""
    out: dict[str, list[dict]] = defaultdict(list)
    for _, entry in ET.iterparse(path, events=("end",)):
        if entry.tag != "gnEntry":
            continue
        acc = entry.findtext("accession")
        if acc in keep:
            for coord in entry.findall("gnCoordinate"):
                exons = []
                for ex in coord.iter("exon"):
                    p = ex.find("proteinLocation")
                    g = ex.find("genomeLocation")
                    exons.append(
                        {
                            "exon_id": ex.get("id"),
                            "uniprot_begin": _pos(p, "begin") if p is not None else None,
                            "uniprot_end": _pos(p, "end") if p is not None else None,
                            "g_begin": _pos(g, "begin") if g is not None else None,
                            "g_end": _pos(g, "end") if g is not None else None,
                        }
                    )
                out[acc].append(
                    {
                        "transcript": coord.get("ensembl_transcript_id"),
                        "gene": coord.get("ensembl_gene_id"),
                        "exons": exons,
                    }
                )
        entry.clear()
    return out


def exon_rows(species: str, acc: str, seq_len: int, transcripts: list[dict]) -> tuple[
    list[dict], str
]:
    """Rows for the kept transcript of one protein, and why one was or was not kept."""
    for tx in sorted(transcripts, key=lambda t: t["transcript"] or ""):
        ex = tx["exons"]
        if not ex or any(e["g_begin"] is None or e["g_end"] is None for e in ex):
            continue
        if any(e["uniprot_begin"] is None for e in ex):
            continue
        ends = [e["uniprot_end"] or e["uniprot_begin"] for e in ex]
        if max(ends) != seq_len or min(e["uniprot_begin"] for e in ex) != 1:
            continue
        rows, c0 = [], 0
        for rank, e in enumerate(ex, start=1):
            n_nt = abs(e["g_end"] - e["g_begin"]) + 1
            first = c0 // 3
            last = min((c0 + n_nt - 1) // 3, seq_len - 1)
            u_begin = e["uniprot_begin"] - 1
            u_end = e["uniprot_end"] or e["uniprot_begin"]
            rows.append(
                {
                    "species": species,
                    "target_acc": acc,
                    "transcript": tx["transcript"],
                    "gene": tx["gene"],
                    "exon_id": e["exon_id"],
                    "exon_rank": rank,
                    "n_exons": len(ex),
                    "internal": 1 < rank < len(ex),
                    "start_phase": c0 % 3,
                    "n_nt": n_nt,
                    "prot_start": first,
                    "prot_end": last + 1,
                    "exon_len": last + 1 - first,
                    "uniprot_start": u_begin,
                    "uniprot_end": u_end,
                    "agrees_uniprot": (first == u_begin) and (last + 1 == u_end),
                    "target_len": seq_len,
                }
            )
            c0 += n_nt
        return rows, "kept"
    return [], "no transcript reaches both protein ends"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "notebooks" / "356_data")
    args = ap.parse_args()

    queries = set(
        (args.out_dir / "midi_plus_query_accessions.txt").read_text().split()
    )
    summary: dict = {
        "n_queries": len(queries),
        "qfo_release": str(QFO),
        "gcoord_md5": {},
        "species": {},
    }
    pair_rows, all_exons = [], []
    for species, stem in SPECIES.items():
        pairs = oma_pairs(queries, species)
        targets = set(pairs.values())
        seqs = read_fasta(QFO / f"{stem}.fasta")
        gpath = QFO / f"{stem}_gcoord.xml"
        summary["gcoord_md5"][species] = hashlib.md5(gpath.read_bytes()).hexdigest()
        gcoord = parse_gcoord(gpath, targets)
        why: dict[str, int] = defaultdict(int)
        for t in sorted(targets):
            if t not in gcoord:
                why["not in gcoord.xml"] += 1
                continue
            rows, reason = exon_rows(species, t, len(seqs[t]), gcoord[t])
            why[reason] += 1
            all_exons.extend(rows)
        pair_rows.extend(
            {"query_acc": q, "species": species, "target_acc": t}
            for q, t in sorted(pairs.items())
        )
        summary["species"][species] = {
            "n_pairs_1to1": len(pairs),
            "targets_by_outcome": dict(why),
        }

    pl.DataFrame(pair_rows).write_csv(args.out_dir / "orthologs_1to1.tsv", separator="\t")
    exons = pl.DataFrame(all_exons)
    # Two transcripts are never both kept, but one transcript can list the same protein
    # interval twice (a duplicated exon record); keep it once.
    n_before = exons.height
    exons = exons.unique(
        subset=["species", "target_acc", "prot_start", "prot_end"],
        keep="first",
        maintain_order=True,
    )
    exons.write_parquet(args.out_dir / "exons.parquet")
    for species in SPECIES:
        e = exons.filter(pl.col("species") == species)
        summary["species"][species].update(
            n_targets_with_exons=e["target_acc"].n_unique(),
            n_exons=e.height,
            n_internal_le60=e.filter(pl.col("internal") & (pl.col("exon_len") <= 60)).height,
            n_agrees_uniprot=int(e["agrees_uniprot"].sum()),
            n_single_exon_proteins=e.filter(pl.col("n_exons") == 1)["target_acc"].n_unique(),
        )
    summary["n_duplicate_intervals_dropped"] = n_before - exons.height
    (args.out_dir / "exon_inputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
