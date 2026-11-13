#!/usr/bin/env python3
"""Notebook 285, section 9: the reverse direction (BCL2 -> Ced9) from the dark-set run's landmark pairs.

The kmerseek 0.4 invertebrate dark-set midi run (nextflow-runs/invertebrate-dark-set, branch
olgabot/dark-set-kmerseek-0.4, finished 2026-10-12) searched each species' proteome as
queries against a Pfam-annotated reference with that species' clade removed, at 230
alphabet-ksize-scaled-extension settings, reporting regions with E <= 10
(--evalue_report 10; regions with no E-value fit are kept as E = inf). It follows three
pairs by name (params.landmarks):

  human     BCL2 (P10415)  -> Ced9 (P41958)   reverse of notebook 241's direction
  worm      Ced9 (P41958)  -> BCL2 (P10415)   same direction as notebook 241
  borrelia  P66  (H7C7N8)  -> CD47 (Q08722)   same direction as notebook 241

CD47 -> P66 was never searched: P66 is not in the reference (checked in
minus_Mammalia_reference_pfam.parquet).

Inputs: <species>_landmarks.parquet copied from
$SCRATCH/invertebrate-dark-set/results-0.4/<species>/ (2026-10-12) to
~/data/invertebrate-dark-set/results-0.4/landmarks/; UniProt sequences and features
(~/data/uniprot/285/, fetched 2026-10-13).

Writes notebooks/285_dark_set_landmarks.csv (every setting, three pairs) and
notebooks/285_dark_set_landmark_calls.csv (the calls, with residues and UniProt features).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import requests

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
LANDMARKS = Path("/Users/olga/data/invertebrate-dark-set/results-0.4/landmarks")
REFERENCE = Path("/Users/olga/data/invertebrate-dark-set/reference-pfam")
UNIPROT = Path("/Users/olga/data/uniprot/285")
NAMES = {"P10415": "BCL2", "P41958": "Ced9", "H7C7N8": "P66", "Q08722": "CD47"}
SPECIES = ["human", "worm", "borrelia"]
# Is the target-side partner alone in its Pfam family within its own proteome? (UniProt search)
FAMILY_CHECKS = [("Ced9", "PF00452", 6239, "Caenorhabditis elegans"),
                 ("P66", "PF11263", 224326, "Borreliella burgdorferi B31")]


def sequence(acc: str) -> str:
    return json.loads((UNIPROT / f"{acc}.json").read_text())["sequence"]["value"]


def features(acc: str) -> list[tuple[str, int, int]]:
    p = UNIPROT / f"{acc}_features.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return [(f["type"] + (f" {f['description']}" if f.get("description") else ""),
             f["location"]["start"]["value"], f["location"]["end"]["value"])
            for f in d.get("features", []) if f["type"] != "Topological domain"]


def overlapping(acc: str, start1: int, end1: int) -> str:
    """UniProt features (1-based, inclusive) that overlap a 1-based inclusive stretch."""
    return "; ".join(f"{t} {a}-{b}" for t, a, b in features(acc) if a <= end1 and b >= start1)


def main() -> None:
    d = pl.concat([pl.read_parquet(LANDMARKS / f"{s}_landmarks.parquet") for s in SPECIES])
    d = d.with_columns(query_name=pl.col("query").replace_strict(NAMES),
                       target_name=pl.col("target").replace_strict(NAMES))
    d.drop("arm").write_csv(NB / "285_dark_set_landmarks.csv")
    ref = pl.scan_parquet(REFERENCE / "minus_Mammalia_reference_pfam.parquet")
    print("P66 (H7C7N8) in the human run's reference:",
          ref.filter(pl.col("accession") == "H7C7N8").select(pl.len()).collect().item() > 0)
    print(d.group_by("species", "query_name", "target_name").agg(
        settings=pl.len(), with_a_call=pl.col("found").sum(),
        with_finite_e=pl.col("region_evalue").is_finite().sum()).sort("species"))
    n_ref = ref.select(pl.col("accession").n_unique()).collect().item()
    print(f"human run's reference: {n_ref:_} distinct proteins, "
          f"{n_ref / 19_732:.1f} times notebook 241's 19_732")
    fam_rows = []
    for name, pfam, taxon, organism in FAMILY_CHECKS:
        cache = UNIPROT / f"{pfam}_{taxon}.tsv"
        if not cache.exists():
            r = requests.get("https://rest.uniprot.org/uniprotkb/search", timeout=60, params={
                "query": f"xref:pfam-{pfam} AND organism_id:{taxon}",
                "fields": "accession,gene_primary,length", "format": "tsv"})
            r.raise_for_status()
            if not r.text.startswith("Entry"):
                raise SystemExit(f"UniProt did not return TSV: {r.text[:200]!r}")
            cache.write_text(r.text)
        acc = [line.split("\t")[0] for line in cache.read_text().splitlines()[1:] if line]
        fam_rows.append({"protein": name, "pfam": pfam, "organism": organism,
                         "n_uniprot_proteins_with_pfam": len(acc), "accessions": " ".join(acc)})
    fam = pl.DataFrame(fam_rows)
    fam.write_csv(NB / "285_dark_set_partner_family_check.csv")
    print(fam)
    rows = []
    for r in d.filter(pl.col("found")).iter_rows(named=True):
        qs, qe, ts = int(r["region_start"]), int(r["region_end"]), int(r["target_start"])
        q, t = sequence(r["query"])[qs:qe], sequence(r["target"])[ts:ts + qe - qs]
        rows.append({"query": r["query_name"], "target": r["target_name"],
                     "alphabet": r["alphabet"], "k": r["ksize"], "scaled": r["scaled"],
                     "extension": r["extension"], "region_evalue": r["region_evalue"],
                     "query_start0": qs, "query_end0": qe, "target_start0": ts,
                     "target_end0": ts + qe - qs, "query_residues": q, "target_residues": t,
                     "n_identical": sum(a == b for a, b in zip(q, t)),
                     "query_features": overlapping(r["query"], qs + 1, qe),
                     "target_features": overlapping(r["target"], ts + 1, ts + qe - qs)})
    calls = pl.DataFrame(rows)
    calls.write_csv(NB / "285_dark_set_landmark_calls.csv")
    print(calls.drop("query_residues", "target_residues"))


if __name__ == "__main__":
    main()
