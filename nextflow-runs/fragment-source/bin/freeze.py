#!/usr/bin/env python3
"""Freeze the notebook 354 inputs: database, 500 peptide calls, the split, the source-absent
set and the shuffled controls. Every random draw has a recorded seed.

Usage: freeze.py --raw $OUT/raw --out $OUT/frozen

Inputs (downloaded by sbatch/download.sbatch, listed with sha256 in raw/manifest.tsv):
  PEAKS/PEAKS_qExHF01_02580_de novo peptides.csv  PEAKS de novo calls, HeLa run 1 of PXD005280
  PXD005280_HeLa_1_qExHF01_02580.msf  Proteome Discoverer 1.4 search of the same run
  UP000005640_9606.fasta.gz           UniProt human reference proteome, release 2026_03
  crap.fasta                          cRAP contaminants (The GPM)

The peptide call is the PEAKS de novo sequence of one spectrum, as exported, with its
per-residue local confidence. Nothing is removed for low confidence or for disagreeing with
the database search; the PEAKS export itself holds only calls with ALC >= 50%.

The accepted source is the database search of the same spectrum: Mascot, MS Amanda and
SEQUEST rank-1 matches at Percolator confidence "High" (PD 1.4 ConfidenceLevel 3, q < 0.01).
A spectrum counts when every engine with such a match gives the same peptide (I read as L).
Its accepted sources are all proteins PD maps that peptide to, kept if the frozen database
has them. PEAKS and PD number the spectra differently (raw scan number against the order of
an MGF), so a call is joined to its PD spectrum on the precursor (`match_spectrum`).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import lookup

SEED_DRAW = 354  # which 500 spectra
SEED_SPLIT = 3540  # which source-protein groups go to development
SEED_ABSENT = 3541  # which test cases lose their source
SEED_SHUFFLE = 3542  # shuffled controls
N_CALLS = 500
ABSENT_FRACTION = 0.2
N_SHUFFLE_DEV = 10  # composition shuffles per development peptide, to set each method's threshold
N_SHUFFLE_TEST = 5  # composition and dipeptide shuffles per test peptide, the controls
PEAKS_FILE = "PEAKS/PEAKS_qExHF01_02580_de novo peptides.csv"
MSF_FILE = "PXD005280_HeLa_1_qExHF01_02580.msf"
ENGINES = {2: "Mascot", 4: "MS Amanda", 5: "SEQUEST"}
PROTON = 1.007276  # PD stores MH+; PEAKS the neutral mass
MATCH_PPM = 10.0  # the search's precursor tolerance
MATCH_RT_MIN = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_database(raw: Path) -> tuple[dict[str, str], list[dict]]:
    """Human reference proteome by accession, plus cRAP entries that are not already in it."""
    human = {}
    for name, seq in lookup.read_fasta(raw / "UP000005640_9606.fasta.gz").items():
        human[name.split("|")[1]] = seq
    seqs = set(human.values())
    db = dict(human)
    crap_log = []
    for name, seq in lookup.read_fasta(raw / "crap.fasta").items():
        entry = name.replace("sp|", "").strip("|")
        kept = seq not in seqs
        crap_log.append({"crap_entry": entry, "length": len(seq), "kept": kept})
        if kept:
            db[f"cRAP_{entry}"] = seq
            seqs.add(seq)
    return db, crap_log


def accepted_sources(msf: Path) -> list[dict]:
    """PD spectra whose engines agree: precursor neutral mass, charge, retention time,
    accepted peptide and proteins."""
    con = sqlite3.connect(msf)
    rows = con.execute(
        """select p.ProcessingNodeNumber, p.SpectrumID, p.Sequence, p.PeptideID
           from Peptides p
           where p.SearchEngineRank = 1 and p.ConfidenceLevel = 3""").fetchall()
    prot_acc = {pid: desc.split("|")[1] for pid, desc in
                con.execute("select ProteinID, Description from ProteinAnnotations")}
    pep_prots = defaultdict(set)
    for node, pep_id, prot_id in con.execute("select ProcessingNodeNumber, PeptideID, ProteinID from PeptidesProteins"):
        pep_prots[(node, pep_id)].add(prot_acc[prot_id])
    header = {sid: (rt, z, mh - PROTON) for sid, rt, z, mh in
              con.execute("select SpectrumID, RetentionTime, Charge, Mass from SpectrumHeaders")}
    by_spec = defaultdict(list)
    for node, sid, seq, pep_id in rows:
        by_spec[sid].append((node, seq, pep_id))
    out = []
    for sid, hits in by_spec.items():
        if len({seq.replace("I", "L") for _, seq, _ in hits}) != 1:
            continue
        accs = set()
        for node, _, pep_id in hits:
            accs |= pep_prots[(node, pep_id)]
        rt, z, mass = header[sid]
        out.append({"pd_spectrum": sid, "rt_min": rt, "z": z, "mass": mass, "peptide": hits[0][1],
                    "accessions": accs, "engines": sorted(ENGINES[n] for n, _, _ in hits)})
    return out


def match_spectrum(row: dict, by_mass: list, masses: list[float]) -> tuple[dict | None, str]:
    """The PD spectrum a PEAKS call was read from.

    PD read an MGF numbered 1, 2, 3, ... while PEAKS read the raw file's scan numbers, so
    the two are joined on the precursor: same charge, neutral mass within MATCH_PPM, and
    retention time within MATCH_RT_MIN (PEAKS rounds it to 0.01 min). If several PD
    spectra qualify the closest in retention time is taken; a tie leaves the call unmatched.
    """
    m, rt, z = float(row["Mass"]), float(row["RT"]), int(row["z"])
    lo = bisect.bisect_left(masses, m * (1 - MATCH_PPM * 1e-6))
    hi = bisect.bisect_right(masses, m * (1 + MATCH_PPM * 1e-6))
    cands = [s for _, s in by_mass[lo:hi] if s["z"] == z and abs(s["rt_min"] - rt) <= MATCH_RT_MIN]
    if not cands:
        return None, "no PD spectrum with an agreed match"
    cands.sort(key=lambda s: abs(s["rt_min"] - rt))
    if len(cands) > 1 and abs(cands[0]["rt_min"] - rt) == abs(cands[1]["rt_min"] - rt):
        return None, "tie"
    return cands[0], "ok"


def split_by_protein(calls: list[dict]) -> None:
    """Put each group of calls that share an accepted protein wholly in dev or test."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in calls:
        accs = sorted(c["accepted_in_db"])
        for a in accs[1:]:
            parent[find(a)] = find(accs[0])
    groups = defaultdict(list)
    for c in calls:
        groups[find(sorted(c["accepted_in_db"])[0])].append(c)
    keys = sorted(groups)
    random.Random(SEED_SPLIT).shuffle(keys)
    n_dev = 0
    for key in keys:
        side = "dev" if n_dev < len(calls) / 2 else "test"
        for c in groups[key]:
            c["split"] = side
        n_dev += len(groups[key]) if side == "dev" else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    db, crap_log = load_database(args.raw)
    sources = accepted_sources(args.raw / MSF_FILE)

    by_mass = sorted(((s["mass"], s) for s in sources), key=lambda x: x[0])
    masses = [m for m, _ in by_mass]
    match_log = defaultdict(int)
    calls = []
    with open(args.raw / PEAKS_FILE) as fh:
        for row in csv.DictReader(fh):
            scan = int(row["Scan"])
            src, status = match_spectrum(row, by_mass, masses)
            match_log[status] += 1
            if src is None:
                continue
            in_db = {a for a in src["accessions"] if a in db}
            if not in_db:
                match_log["accepted source not in the frozen database"] += 1
                continue
            bare = re.sub(r"\([^)]*\)", "", row["Peptide"])
            calls.append({
                "call_id": f"scan{scan}",
                "scan": scan,
                "peptide": bare,
                "peaks_peptide": row["Peptide"],
                "length": len(bare),
                "alc_percent": float(row["ALC (%)"]),
                "local_confidence_percent": row["local confidence (%)"],
                "peaks_ptm": row["PTM"],
                "peaks_tag": row["tag (>=0%)"],
                "mz": row["m/z"], "z": row["z"], "rt_min": row["RT"], "ppm": row["ppm"],
                "pd_spectrum": src["pd_spectrum"],
                "pd_rt_min": round(src["rt_min"], 4),
                "precursor_ppm_peaks_vs_pd": round((float(row["Mass"]) - src["mass"]) / src["mass"] * 1e6, 2),
                "db_search_peptide": src["peptide"],
                "db_search_engines": ";".join(src["engines"]),
                "accepted_accessions_pd": ";".join(sorted(src["accessions"])),
                "accepted_in_db": in_db,
                "denovo_equals_db_search": lookup.normalise(bare) == lookup.normalise(src["peptide"]),
            })
    n_eligible = len(calls)
    calls.sort(key=lambda c: c["scan"])
    eligible = list(calls)
    calls = random.Random(SEED_DRAW).sample(calls, N_CALLS)
    calls.sort(key=lambda c: c["scan"])

    split_by_protein(calls)
    test = [c for c in calls if c["split"] == "test"]
    n_absent = round(ABSENT_FRACTION * len(test))
    absent_ids = set(c["call_id"] for c in random.Random(SEED_ABSENT).sample(sorted(test, key=lambda c: c["scan"]), n_absent))
    removed = set()
    for c in calls:
        c["source_absent"] = c["call_id"] in absent_ids
        if c["source_absent"]:
            removed |= c["accepted_in_db"]
    db_absent = {a: s for a, s in db.items() if a not in removed}

    # Files
    lookup.write_fasta(db, out / "db_full.fasta")
    lookup.write_fasta(db_absent, out / "db_source_absent.fasta")
    fields = ["call_id", "scan", "split", "source_absent", "peptide", "length", "alc_percent",
              "local_confidence_percent", "peaks_peptide", "peaks_ptm", "peaks_tag", "mz", "z",
              "rt_min", "ppm", "pd_spectrum", "pd_rt_min", "precursor_ppm_peaks_vs_pd", "db_search_peptide", "db_search_engines", "accepted_in_db",
              "accepted_accessions_pd", "denovo_equals_db_search"]
    with open(out / "calls.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for c in calls:
            w.writerow({**c, "accepted_in_db": ";".join(sorted(c["accepted_in_db"]))})
    sets = {
        "dev": [c for c in calls if c["split"] == "dev"],
        "test_present": [c for c in test if not c["source_absent"]],
        "test_absent": [c for c in test if c["source_absent"]],
    }
    for name, rows in sets.items():
        lookup.write_fasta({c["call_id"]: c["peptide"] for c in rows}, out / f"queries_{name}.fasta")
    rng = random.Random(SEED_SHUFFLE)
    shuf = {}
    for c in sets["dev"]:
        for i in range(N_SHUFFLE_DEV):
            shuf[f"{c['call_id']}_comp{i}"] = lookup.composition_shuffle(c["peptide"], rng)
    lookup.write_fasta(shuf, out / "queries_dev_shuffled.fasta")
    ctrl = {}
    for c in test:
        for i in range(N_SHUFFLE_TEST):
            ctrl[f"{c['call_id']}_comp{i}"] = lookup.composition_shuffle(c["peptide"], rng)
        for i in range(N_SHUFFLE_TEST):
            ctrl[f"{c['call_id']}_dipep{i}"] = lookup.dipeptide_shuffle(c["peptide"], rng)
    lookup.write_fasta(ctrl, out / "queries_test_controls.fasta")
    with open(out / "crap_entries.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["crap_entry", "length", "kept"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(crap_log)

    summary = {
        "seeds": {"draw": SEED_DRAW, "split": SEED_SPLIT, "absent": SEED_ABSENT, "shuffle": SEED_SHUFFLE},
        "n_peaks_calls_with_accepted_source_in_db": n_eligible,
        "n_spectra_with_agreed_high_confidence_psm": len(sources),
        "peaks_to_pd_matching": dict(match_log),
        "eligible_denovo_equals_db_search": sum(c["denovo_equals_db_search"] for c in eligible),
        "n_calls": len(calls),
        "n_dev": len(sets["dev"]), "n_test": len(test),
        "n_test_present": len(sets["test_present"]), "n_test_absent": len(sets["test_absent"]),
        "n_proteins_removed_for_absent": len(removed),
        "db_full_proteins": len(db), "db_source_absent_proteins": len(db_absent),
        "crap_entries": len(crap_log), "crap_kept": sum(r["kept"] for r in crap_log),
        "dev_test_protein_overlap": len(set().union(*[c["accepted_in_db"] for c in sets["dev"]])
                                        & set().union(*[c["accepted_in_db"] for c in test])),
    }
    (out / "freeze_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out / "manifest.tsv", "w") as fh:
        fh.write("file\tbytes\tsha256\n")
        for p in sorted(out.iterdir()):
            if p.name != "manifest.tsv" and p.is_file():
                fh.write(f"{p.name}\t{p.stat().st_size}\t{sha256(p)}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
