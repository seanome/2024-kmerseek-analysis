"""Read the few Swiss-Prot fields the stage diagnostic needs from the flat .dat file.

One pass over uniprot_sprot.<release>.dat.gz, written once to a parquet cache next to it.
Kept per entry: accession, entry name, sequence and its version, length, organism,
taxonomy, keywords, the family from the SIMILARITY comment, whether the FUNCTION comment
cites experimental evidence (ECO:0000269), and the PEPTIDE and CHAIN features (the mature
toxin is one of these, not the precursor with its signal and propeptide).
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import polars as pl

FAMILY_RE = re.compile(r"Belongs to the (.+?)(?:\. |\.$| \{)")
SEQ_VERSION_RE = re.compile(r"sequence version (\d+)")
FEATURE_RE = re.compile(r"^FT   (PEPTIDE|CHAIN)\s+<?(\d+)\.\.>?(\d+)")


def _finish(e: dict) -> dict:
    cc = e.pop("_cc")
    blocks = re.split(r"-!- ", cc)
    fam, func_exp, func_any = None, False, False
    for b in blocks:
        if b.startswith("SIMILARITY:") and fam is None:
            m = FAMILY_RE.search(b.replace("\n", " "))
            fam = m.group(1) if m else None
        if b.startswith("FUNCTION:"):
            func_any = True
            func_exp = func_exp or "ECO:0000269" in b
    e["family"] = fam
    e["function_annotated"] = func_any
    e["function_experimental"] = func_exp
    e["keywords"] = "; ".join(k for k in e["keywords"] if k)
    e["sequence"] = "".join(e["sequence"])
    e["length"] = len(e["sequence"])
    e["features"] = "; ".join(e["features"])
    return e


def parse_dat(path: Path):
    """Yield one dict per entry. Plain line parser: Biopython's is ~5x slower here."""
    opener = gzip.open if str(path).endswith(".gz") else open
    e = None
    in_seq = False
    with opener(path, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                e = dict(
                    entry_name=line.split()[1],
                    accession=None,
                    seq_version=None,
                    organism="",
                    taxonomy="",
                    keywords=[],
                    sequence=[],
                    features=[],
                    _cc="",
                    protein_existence=None,
                )
                in_seq = False
            elif tag == "AC" and e["accession"] is None:
                e["accession"] = line[5:].split(";")[0].strip()
            elif tag == "DT" and "sequence version" in line:
                e["seq_version"] = int(SEQ_VERSION_RE.search(line).group(1))
            elif tag == "OS":
                e["organism"] += line[5:].strip() + " "
            elif tag == "OC":
                e["taxonomy"] += line[5:].strip() + " "
            elif tag == "PE":
                e["protein_existence"] = int(line[5])
            elif tag == "KW":
                for k in line[5:].strip().rstrip(".").split(";"):
                    e["keywords"].append(re.sub(r"\s*\{.*?\}", "", k).strip())
            elif tag == "CC":
                e["_cc"] += line[9:] if line.startswith("CC       ") else line[5:]
            elif tag == "FT":
                m = FEATURE_RE.match(line)
                if m:
                    e["features"].append(f"{m.group(1)}:{m.group(2)}-{m.group(3)}")
            elif tag == "SQ":
                in_seq = True
            elif tag == "//":
                e["organism"] = e["organism"].strip()
                e["taxonomy"] = e["taxonomy"].strip()
                yield _finish(e)
                e, in_seq = None, False
            elif in_seq:
                e["sequence"].append(line.replace(" ", "").strip())


def load_slim(dat_path: Path) -> pl.DataFrame:
    """Return the cached slim table, building it on first use."""
    cache = dat_path.with_name(dat_path.name.replace(".dat.gz", "") + ".stage_diagnostic.parquet")
    if cache.exists() and cache.stat().st_mtime > dat_path.stat().st_mtime:
        return pl.read_parquet(cache)
    df = pl.DataFrame(list(parse_dat(dat_path)))
    df.write_parquet(cache)
    return df
