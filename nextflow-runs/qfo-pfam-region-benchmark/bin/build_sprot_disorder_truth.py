#!/usr/bin/env python3
"""Truth for the disorder and low-complexity baselines: Swiss-Prot REGION "Disordered"
and COMPBIAS features of the query proteins.

The main Swiss-Prot truth set (build_swissprot_truth.py) keeps REGION whatever its note
says and has no COMPBIAS. A disorder predictor can only be asked about the REGION
features whose note is "Disordered", and SEG about compositional bias, so this table
holds just those two, labelled:

    REGION_DISORDERED   FT REGION with /note="Disordered"
    COMPBIAS            FT COMPBIAS, any note

Each label is written to its own table and becomes its own truth set,
`swissprot_disordered` and `swissprot_compbias`, scored at the stratum "all". Two tables
rather than one with a feature-type stratum, because evaluate_domain_calls.py only cuts a
feature-type stratum for the labels build_swissprot_truth.py knows, and editing either
script would change the task hash of every scoring task already cached for the main runs.
No row already scored against the main Swiss-Prot truth set changes.

Caution for the reader of any score against it: in UniProtKB most "Disordered" REGION
and COMPBIAS features carry the evidence code ECO:0000256 with source MobiDB-lite, a
consensus of disorder predictors that includes IUPred. A disorder predictor scored here is
partly being compared with a cousin of itself. The `evidence` column keeps that source so
the share can be counted.

Same schema as the other truth tables (accession, pfam_id, domain_start, domain_end,
protein_length, is_point); `pfam_id` holds the label.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import polars as pl

LABELS = ("REGION_DISORDERED", "COMPBIAS")
FT_RE = re.compile(r"^FT   (\w+)\s+([<>?]?\d+)(?:\.\.([<>?]?\d+))?\s*$")
QUAL_RE = re.compile(r'^FT\s{19}/(\w+)="?([^"]*)"?\s*$')


def accessions_of_fasta(path: Path) -> set[str]:
    out = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                first = line[1:].split()[0]
                parts = first.split("|")
                out.add(parts[1] if len(parts) >= 2 and parts[1] else first)
    return out


def parse(dat: Path, wanted: set[str]) -> list[dict]:
    """REGION and COMPBIAS features with their /note and /evidence qualifiers.

    Fuzzy endpoints (<, >, ?) are dropped, as in build_swissprot_truth.py.
    """
    rows: list[dict] = []
    acc, length, current = None, None, None

    def flush():
        nonlocal current
        if current is not None:
            if current["type"] == "COMPBIAS":
                current["pfam_id"] = "COMPBIAS"
            elif current["note"] == "Disordered":
                current["pfam_id"] = "REGION_DISORDERED"
            if "pfam_id" in current:
                rows.append(current)
        current = None

    opener = gzip.open if str(dat).endswith(".gz") else open
    with opener(dat, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("ID "):
                flush()
                acc, length = None, None
                parts = line.split()
                if len(parts) >= 4 and parts[-1] == "AA.":
                    length = int(parts[-2])
            elif line.startswith("AC ") and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif line.startswith("FT "):
                if acc not in wanted:
                    continue
                m = FT_RE.match(line.rstrip("\n"))
                if m:
                    flush()
                    ftype, a, b = m.groups()
                    if ftype not in ("REGION", "COMPBIAS"):
                        continue
                    if any(c in a for c in "<>?") or (b and any(c in b for c in "<>?")):
                        continue
                    current = {"accession": acc, "type": ftype,
                               "domain_start": int(a), "domain_end": int(b or a),
                               "protein_length": length, "note": None,
                               "evidence": None}
                    continue
                q = QUAL_RE.match(line.rstrip("\n"))
                if q and current is not None:
                    key, val = q.groups()
                    if key in ("note", "evidence") and current[key] is None:
                        current[key] = val
            elif line.startswith("//"):
                flush()
                acc, length = None, None
    flush()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sprot-dat", required=True, type=Path)
    p.add_argument("--query-fasta", required=True, type=Path)
    p.add_argument("--disordered-out", required=True, type=Path,
                   help="REGION_DISORDERED features (truth set swissprot_disordered)")
    p.add_argument("--compbias-out", required=True, type=Path,
                   help="COMPBIAS features (truth set swissprot_compbias)")
    p.add_argument("--summary-out", required=True, type=Path)
    a = p.parse_args()

    wanted = accessions_of_fasta(a.query_fasta)
    rows = parse(a.sprot_dat, wanted)
    schema = {"accession": pl.String, "pfam_id": pl.String, "domain_start": pl.Int64,
              "domain_end": pl.Int64, "protein_length": pl.Int64, "is_point": pl.Boolean,
              "evidence": pl.String}
    t = (pl.DataFrame(rows, infer_schema_length=None)
         .with_columns(pl.lit(False).alias("is_point"))
         .select(list(schema)).cast(schema)
         .unique().sort("accession", "pfam_id", "domain_start", "domain_end")
         if rows else pl.DataFrame(schema=schema))
    t.filter(pl.col("pfam_id") == "REGION_DISORDERED").write_parquet(a.disordered_out)
    t.filter(pl.col("pfam_id") == "COMPBIAS").write_parquet(a.compbias_out)
    summary = {
        "n_query_proteins": len(wanted),
        "by_label": {
            lab: {
                "n_features": int(t.filter(pl.col("pfam_id") == lab).height),
                "n_proteins": int(t.filter(pl.col("pfam_id") == lab)["accession"].n_unique()),
                "n_mobidb_lite": int(t.filter((pl.col("pfam_id") == lab)
                                              & pl.col("evidence").fill_null("")
                                              .str.contains("MobiDB-lite")).height),
            }
            for lab in LABELS
        },
    }
    a.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
