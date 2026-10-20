#!/usr/bin/env python3
"""Freeze the notebook 353 inputs and build the evidence collection from them.

Copies the raw files into data/kill_tests/short_stretch/raw/, writes MANIFEST.tsv (file,
source URL, download date, SHA-256), and builds one evidence record per ELM true-positive
instance and per measured yeast transcription-factor tile, as notebooks/353_prereg.md
specifies.

Inputs (read-only):
  ELM instance and class exports and the UniProt sequences of every ELM protein, downloaded
  2026-09-24 for notebook 250 (scripts/fetch_elm.py and scripts/prep_elm_inputs.py on
  branch olgabot/elm-motif-transfer), in /Users/olga/data/elm-motif-transfer/.
  Sanborn et al. 2021 eLife 10:e68068, Figure 1 source data 1, elife-68068-fig1-data1-v3.xlsx.

Outputs in data/kill_tests/short_stretch/:
  raw/                       the four raw files (the three ELM files gzipped)
  MANIFEST.tsv               one row per raw file
  records.tsv                one row per evidence record (columns described in COLUMNS)
  records.fasta              the record sequences, full 20-letter alphabet, id = record_id
  freeze_counts.tsv          how many rows each step kept and dropped, with the reason

Coordinates are 0-based, end-exclusive, in the source protein (kmerseek's convention).

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/353_freeze_short_stretch_data.py \
        --elm-dir /Users/olga/data/elm-motif-transfer --sanborn-xlsx <path to the xlsx>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import shutil
import sys
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "kill_tests" / "short_stretch"

RECORD_LENGTH = 53  # the Sanborn tile length; ELM windows are made the same length
SANBORN_DOI = "10.7554/eLife.68068"
SANBORN_URL = (
    "https://cdn.elifesciences.org/articles/68068/elife-68068-fig1-data1-v3.xlsx"
)
SANBORN_SHEET = "TF tiles, see Fig 1"
SANBORN_SEQ_SHEET = "TF sequences, see Fig 1"
TILE_PREFIX = (
    "A_tiles_"  # row names of the TF tiles; controls and random sequences differ
)

COLUMNS = [
    "record_id",  # E:<ELM instance> or T:<protein ID>:<position>
    "source",  # elm or yeast_tile
    "accession",  # source protein accession as given (ELM Primary_Acc may be an isoform)
    "base_accession",  # accession without an isoform suffix; the split is on this
    "protein_name",
    "organism",
    "record_start",  # record window in the source protein
    "record_end",
    "feature_start",  # the measured feature: the ELM motif, or the tile itself
    "feature_end",
    "sequence",  # record residues, full alphabet
    "feature_sequence",
    "measurement",  # one line an analyst reads on the evidence card
    "elm_class",
    "elm_instance",
    "methods",
    "activation",  # Sanborn activation, empty for ELM
    "z_score",
    "num_cells",
    "reference",  # PMIDs for ELM, the DOI for tiles
    "split",  # dev or held_out
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(base_accession: str) -> str:
    """Development half when the SHA-256 of the accession, read as an integer, is even."""
    return (
        "dev"
        if int(hashlib.sha256(base_accession.encode()).hexdigest(), 16) % 2 == 0
        else "held_out"
    )


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    acc = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            acc = line[1:].split()[0].split("|")[1]
            seqs[acc] = []
        elif acc is not None:
            seqs[acc].append(line.strip())
    return {a: "".join(s) for a, s in seqs.items()}


def header_value(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"#{key}:"):
            return line.split(":", 1)[1].strip()
    return ""


def elm_records(elm_dir: Path, counts: list[tuple[str, int, str]]) -> list[dict]:
    text = (elm_dir / "elm_instances.tsv").read_text()
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body), delimiter="\t"))
    counts.append(("elm: instances in the export", len(rows), ""))
    tp = [r for r in rows if r["InstanceLogic"] == "true positive"]
    counts.append(("elm: true positive", len(tp), "other InstanceLogic values dropped"))
    seqs = read_fasta(elm_dir / "elm_proteins.fasta")
    out, no_seq, out_of_range = [], 0, 0
    for r in tp:
        acc = r["Primary_Acc"]
        seq = seqs.get(acc)
        if seq is None:
            no_seq += 1
            continue
        m_start, m_end = int(r["Start"]) - 1, int(r["End"])
        if m_start < 0 or m_end > len(seq) or m_end <= m_start:
            out_of_range += 1
            continue
        motif_len = m_end - m_start
        if len(seq) <= RECORD_LENGTH:
            w_start, w_end = 0, len(seq)
        elif motif_len >= RECORD_LENGTH:
            w_start, w_end = m_start, m_end
        else:
            centre = (m_start + m_end) / 2
            w_start = int(round(centre - RECORD_LENGTH / 2))
            w_start = min(max(w_start, 0), len(seq) - RECORD_LENGTH)
            w_end = w_start + RECORD_LENGTH
        base = acc.split("-")[0]
        out.append(
            {
                "record_id": f"E:{r['Accession']}",
                "source": "elm",
                "accession": acc,
                "base_accession": base,
                "protein_name": r["ProteinName"],
                "organism": r["Organism"],
                "record_start": w_start,
                "record_end": w_end,
                "feature_start": m_start,
                "feature_end": m_end,
                "sequence": seq[w_start:w_end],
                "feature_sequence": seq[m_start:m_end],
                "measurement": f"ELM {r['ELMIdentifier']} motif {seq[m_start:m_end]} at {m_start + 1}-{m_end}; evidence: {r['Methods']}",
                "elm_class": r["ELMIdentifier"],
                "elm_instance": r["Accession"],
                "methods": r["Methods"],
                "activation": "",
                "z_score": "",
                "num_cells": "",
                "reference": "PMID " + ", ".join(r["References"].split()),
                "split": split_of(base),
            }
        )
    counts.append(("elm: no UniProt sequence", no_seq, "dropped"))
    counts.append(("elm: coordinates outside the sequence", out_of_range, "dropped"))
    counts.append(("elm: records", len(out), ""))
    return out


def tile_records(xlsx: Path, counts: list[tuple[str, int, str]]) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    tf_seq = {}
    for i, row in enumerate(wb[SANBORN_SEQ_SHEET].iter_rows(values_only=True)):
        if i and row[1]:
            tf_seq[row[1]] = row[2]
    rows = list(wb[SANBORN_SHEET].iter_rows(values_only=True))
    header, rows = rows[0], rows[1:]
    col = {name: j for j, name in enumerate(header) if name}
    tiles = [r for r in rows if r[0] and r[0].startswith(TILE_PREFIX)]
    counts.append(
        (
            "tiles: rows named A_tiles_ in the sheet",
            len(tiles),
            "controls and random sequences are other rows",
        )
    )
    out, unmeasured, mismatch = [], 0, 0
    for r in tiles:
        seq, name, acc, pos = (
            r[col["protein sequence"]],
            r[col["protein"]],
            r[col["protein ID"]],
            int(r[col["position"]]),
        )
        act = r[col["activation"]]
        if act is None:
            unmeasured += 1
            continue
        if tf_seq.get(acc, "")[pos : pos + len(seq)] != seq:
            mismatch += 1
        z, cells = r[col["Z score"]], r[col["num cells"]]
        out.append(
            {
                "record_id": f"T:{acc}:{pos}",
                "source": "yeast_tile",
                "accession": acc,
                "base_accession": acc,
                "protein_name": name,
                "organism": "Saccharomyces cerevisiae",
                "record_start": pos,
                "record_end": pos + len(seq),
                "feature_start": pos,
                "feature_end": pos + len(seq),
                "sequence": seq,
                "feature_sequence": seq,
                "measurement": f"{name} tile {pos + 1}-{pos + len(seq)}: activation {act:.2f} (Z score {z:.2f}, {cells:.0f} cells), yeast activation-domain screen",
                "elm_class": "",
                "elm_instance": "",
                "methods": "GAL4-DBD fusion activation screen, sorted by GFP reporter (Sanborn et al. 2021)",
                "activation": f"{act:.6g}",
                "z_score": f"{z:.6g}",
                "num_cells": f"{cells:.6g}",
                "reference": f"doi:{SANBORN_DOI}",
                "split": split_of(acc),
            }
        )
    counts.append(
        (
            "tiles: activation empty (too few cells)",
            unmeasured,
            "dropped: no measurement",
        )
    )
    counts.append(
        (
            "tiles: sequence differs from the TF sequence sheet at its position",
            mismatch,
            "kept; checked only",
        )
    )
    counts.append(("tiles: records", len(out), ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elm-dir", type=Path, required=True)
    ap.add_argument("--sanborn-xlsx", type=Path, required=True)
    args = ap.parse_args()

    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    elm_text = (args.elm_dir / "elm_instances.tsv").read_text()
    cls_text = (args.elm_dir / "elm_classes.tsv").read_text()
    manifest = [
        (
            "raw/elm_instances.tsv.gz",
            "http://elm.eu.org/instances.tsv?q=*",
            header_value(elm_text, "ELM_Instance_Download_Date"),
            f"ELM instance export format {header_value(elm_text, 'ELM_Instance_Download_Version')}",
        ),
        (
            "raw/elm_classes.tsv.gz",
            "http://elm.eu.org/elms/elms_index.tsv",
            header_value(cls_text, "ELM_Classes_Download_Date"),
            f"ELM class export format {header_value(cls_text, 'ELM_Classes_Download_Version')}",
        ),
        (
            "raw/elm_proteins.fasta.gz",
            "https://rest.uniprot.org/uniprotkb/accessions?accessions=...&format=fasta",
            "2026-09-24",
            "UniProt sequences of every ELM true-positive protein (scripts/prep_elm_inputs.py, branch olgabot/elm-motif-transfer)",
        ),
        (
            "raw/elife-68068-fig1-data1-v3.xlsx",
            SANBORN_URL,
            "2026-10-20",
            f"Sanborn et al. 2021 eLife 10:e68068, doi:{SANBORN_DOI}, Figure 1 source data 1",
        ),
    ]
    # Gzipped byte for byte: git's core.autocrlf=input would rewrite the ELM exports' CRLF
    # line ends in a plain .tsv, and the checked-out file would no longer match its SHA-256.
    for name in ("elm_instances.tsv", "elm_classes.tsv", "elm_proteins.fasta"):
        with open(args.elm_dir / name, "rb") as f, gzip.GzipFile(
            raw / f"{name}.gz", "wb", mtime=0
        ) as g:
            shutil.copyfileobj(f, g)
    shutil.copyfile(args.sanborn_xlsx, raw / "elife-68068-fig1-data1-v3.xlsx")

    counts: list[tuple[str, int, str]] = []
    records = elm_records(args.elm_dir, counts) + tile_records(
        raw / "elife-68068-fig1-data1-v3.xlsx", counts
    )
    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids)), "record ids are not unique"
    for split in ("dev", "held_out"):
        for src in ("elm", "yeast_tile"):
            n = sum(r["split"] == split and r["source"] == src for r in records)
            p = len(
                {
                    r["base_accession"]
                    for r in records
                    if r["split"] == split and r["source"] == src
                }
            )
            counts.append((f"{split}: {src} records ({p} proteins)", n, ""))

    with open(OUT / "records.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    with open(OUT / "records.fasta", "w") as f:
        for r in records:
            f.write(f">{r['record_id']}\n{r['sequence']}\n")
    with open(OUT / "freeze_counts.tsv", "w") as f:
        f.write("step\tn\tnote\n")
        for step, n, note in counts:
            f.write(f"{step}\t{n}\t{note}\n")
            print(f"{n:>7_}  {step}  {note}")
    manifest += [
        (
            "records.tsv",
            "scripts/353_freeze_short_stretch_data.py",
            "2026-10-20",
            "evidence collection",
        ),
        (
            "records.fasta",
            "scripts/353_freeze_short_stretch_data.py",
            "2026-10-20",
            "record sequences",
        ),
    ]
    with open(OUT / "MANIFEST.tsv", "w") as f:
        f.write(
            "file\tsource\tdownload_or_build_date\tdescription\tsha256\tsha256_uncompressed\n"
        )
        for path, src, date, desc in manifest:
            plain = ""
            if path.endswith(".gz"):
                with gzip.open(OUT / path, "rb") as g:
                    plain = hashlib.sha256(g.read()).hexdigest()
            f.write(f"{path}\t{src}\t{date}\t{desc}\t{sha256(OUT / path)}\t{plain}\n")
    print((OUT / "MANIFEST.tsv").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
