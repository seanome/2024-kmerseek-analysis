"""Sort each probe set into the best sequence source its array annotation names, for full-length proteins.

Usage: probe_set_sources.py <GPL table> <out.tsv>
The rules and their order are in probe_protein_plan.md. Reads GPL570-style tables (U133 Plus 2: GB_ACC, Sequence Type,
RefSeq Transcript ID) and GPL17586 (HTA 2.0: mrna_assignment). Writes one row per probe set: its source, the
accessions to fetch for that source, and how many there are. Nothing is fetched here.
"""

import re
import sys

ORDER = ["refseq_mrna", "ensembl_coding", "refseq_model", "genbank_mrna", "est", "other_only", "none"]


def u133(cells, column):
    refseq = [a.strip() for a in cells[column["RefSeq Transcript ID"]].split("///") if a.strip()]
    mrna = [a for a in refseq if a.startswith("NM_")]
    if mrna:
        return "refseq_mrna", mrna
    model = [a for a in refseq if a.startswith("XM_")]
    if model:
        return "refseq_model", model
    design = cells[column["GB_ACC"]].strip()
    if not design:
        return ("other_only", refseq) if refseq else ("none", [])
    if cells[column["Sequence Type"]].startswith("Consensus"):
        return "est", [design]
    return "genbank_mrna", [design]


def hta(cells, column):
    entries = [e.split(" // ") for e in cells[column["mrna_assignment"]].split(" /// ") if e.strip() not in ("", "---")]
    by_kind = {kind: [] for kind in ORDER}
    for e in entries:
        accession, source = e[0].strip(), e[1].strip() if len(e) > 1 else ""
        if accession.startswith("NM_"):
            by_kind["refseq_mrna"].append(accession)
        elif accession.startswith("ENST") and len(e) > 2 and "transcript_biotype:protein_coding" in e[2]:
            by_kind["ensembl_coding"].append(accession)
        elif accession.startswith("XM_"):
            by_kind["refseq_model"].append(accession)
        elif source == "GenBank":
            by_kind["genbank_mrna"].append(accession)
        else:
            by_kind["other_only"].append(accession)
    for kind in ORDER:
        if by_kind[kind]:
            return kind, sorted(set(by_kind[kind]))
    return "none", []


table, out = sys.argv[1:]
with open(table, errors="replace") as f, open(out, "w") as w:
    w.write("probe_set\tsource\taccessions\tcount\n")
    column = None
    for line in f:
        if line.startswith("ID\t"):
            column = {name: i for i, name in enumerate(line.rstrip("\n").split("\t"))}
            pick = hta if "mrna_assignment" in column else u133
            continue
        if column is None or line.startswith(("!", "#")) or not line.strip():
            continue
        cells = line.rstrip("\n").split("\t")
        cells += [""] * (len(column) - len(cells))
        source, accessions = pick(cells, column)
        w.write(f"{cells[0]}\t{source}\t{','.join(accessions)}\t{len(accessions)}\n")
