"""Turn NovoBoard's human data (PRIDE PXD055277) into experiment 358's inputs.

Standard library only, Python 3.6 or later, so it runs with Sherlock's /usr/bin/python3.
The rules are in ../FROZEN_PLAN.md, section "Amendment, 2026-10-20".

    convert_novoboard.py inspect ZIP                       # what is in the zip, as JSON
    convert_novoboard.py collection SPROT_TAR CRAP OUTDIR  # human Swiss-Prot 2024_01 + cRAP
    convert_novoboard.py convert ZIP OUTDIR                # calls.tsv, accepted.tsv, inputs.json
"""

import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REAL_MGF = "Human_Benchmark_DDA_1hr.mgf"
CASANOVO = "eval_human/Casanovo/Sample 1.denovo.csv"
DB_PSMS = "eval_human/db.psms.csv"
N_CALLS = 500
MIN_LEN, MAX_LEN = 7, 30
SALT = "358-novoboard:"
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
MASS = r"[+-]\d+(?:\.\d+)?"
NTERM_RE = re.compile(r"^(?:%s)+" % MASS)
RESIDUE_RE = re.compile(r"([A-Z])((?:%s)*)" % MASS)

csv.field_size_limit(1 << 30)


def read_member(z, name):
    """Rows of one CSV member of the zip, as dicts."""
    with z.open(name) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))


def scan_number(text):
    """'14293' or 'F1:14293' -> 14293."""
    m = re.search(r"(\d+)\s*$", str(text))
    if not m:
        raise ValueError("no scan number in %r" % text)
    return int(m.group(1))


def casanovo_tokens(sequence):
    """Split a Casanovo sequence into its scored tokens.

    '+43.006LLKPM+15.995K' -> [None, 'L', 'L', 'K', 'P', 'M', 'K']. None is an N-terminal
    mass tag, which Casanovo scores as one token. Raises on anything else."""
    tokens, pos = [], 0
    m = NTERM_RE.match(sequence)
    if m:
        tokens.append(None)
        pos = m.end()
    while pos < len(sequence):
        m = RESIDUE_RE.match(sequence, pos)
        if not m:
            raise ValueError("cannot read %r at position %d" % (sequence, pos))
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def bare_peptide(text):
    """'AC(+57.02)M(Oxidation)K' or 'K.PEPTIDE.R' -> letters of the peptide only."""
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    if re.match(r"^[A-Z-]\.[A-Z]+\.[A-Z-]$", text):
        text = text.split(".")[1]
    return re.sub(r"[^A-Z]", "", text)


def unit_hash(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16) / 16 ** 15


# --------------------------------------------------------------------------- inspect


def inspect(zip_path):
    """Per member: size, rows, source files, distinct scans. Plus how Casanovo's real-spectra
    scans overlap the database search, and the spectrum count of the real MGF."""
    z = zipfile.ZipFile(str(zip_path))
    out = {"zip": str(zip_path), "members": []}
    scans = {}
    for info in z.infolist():
        entry = {"name": info.filename, "bytes": info.file_size}
        if info.filename.endswith(".csv"):
            rows = read_member(z, info.filename)
            sources = Counter(r.get("Source File", "") for r in rows)
            entry["rows"] = len(rows)
            entry["columns"] = list(rows[0].keys()) if rows else []
            entry["source_files"] = dict(sources)
            real = [r for r in rows if r.get("Source File") == REAL_MGF]
            entry["distinct_scans_real_mgf"] = len({scan_number(r["Scan"]) for r in real})
            scans[info.filename] = {scan_number(r["Scan"]) for r in real}
        elif info.filename.endswith(".mgf") and "/decoy" not in info.filename:
            with z.open(info.filename) as raw:
                entry["spectra"] = sum(1 for line in raw if line.startswith(b"BEGIN IONS"))
        out["members"].append(entry)
    if CASANOVO in scans and DB_PSMS in scans:
        c, d = scans[CASANOVO], scans[DB_PSMS]
        out["casanovo_scans"] = len(c)
        out["db_psm_scans"] = len(d)
        out["casanovo_scans_with_db_psm"] = len(c & d)
    return out


# --------------------------------------------------------------------------- collection


def read_fasta_text(handle):
    name, parts = None, []
    for line in handle:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(parts)
            name, parts = line, []
        elif line:
            parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


def accession(header):
    word = header[1:].split()[0]
    fields = word.split("|")
    return fields[1] if len(fields) >= 3 else word


def collection(sprot_tar, crap_fasta, outdir):
    """Human canonical Swiss-Prot entries from the release tarball, and cRAP without the
    accessions already in the human set."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    human = {}
    with tarfile.open(str(sprot_tar), "r|gz") as tar:
        for member in tar:
            if Path(member.name).name != "uniprot_sprot.fasta.gz":
                continue
            with gzip.open(tar.extractfile(member), "rt") as fa:
                for header, seq in read_fasta_text(fa):
                    if " OX=9606 " in header + " ":
                        human[accession(header)] = (header, seq)
            break
    if not human:
        raise SystemExit("no uniprot_sprot.fasta.gz with OX=9606 entries in %s" % sprot_tar)
    with open(str(outdir / "proteins_human.fasta"), "w") as f:
        for header, seq in human.values():
            f.write("%s\n%s\n" % (header, seq))
    kept, dropped = [], []
    with open(str(crap_fasta)) as fa:
        for header, seq in read_fasta_text(fa):
            (dropped if accession(header) in human else kept).append((header, seq))
    with open(str(outdir / "proteins_contaminants.fasta"), "w") as f:
        for header, seq in kept:
            f.write("%s\n%s\n" % (header, seq))
    report = {"human_entries": len(human), "crap_entries_kept": len(kept),
              "crap_entries_dropped_already_human": [accession(h) for h, _ in dropped]}
    (outdir / "collection_report.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- convert


def convert(zip_path, outdir):
    """calls.tsv, accepted.tsv, inputs.json and conversion_report.json in outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(str(zip_path))
    report = Counter()

    # accepted peptides: best -10LgP per scan of the real spectra
    by_scan = defaultdict(list)
    for r in read_member(z, DB_PSMS):
        report["db_rows"] += 1
        if r["Source File"] != REAL_MGF:
            report["db_rows_other_source_file"] += 1
            continue
        by_scan[scan_number(r["Scan"])].append((float(r["-10LgP"]), bare_peptide(r["Peptide"])))
    accepted = {}
    for scan, psms in by_scan.items():
        psms.sort(reverse=True)
        if len(psms) > 1:
            report["db_scans_with_several_psms"] += 1
            if psms[0][0] == psms[1][0] and psms[0][1] != psms[1][1]:
                report["db_scans_dropped_tied_psms"] += 1
                continue
        accepted[scan] = psms[0]

    # calls: Casanovo on the real spectra
    # a spectrum with more than one Casanovo row has no single call: count over all real rows
    rows_per_scan = Counter()
    for r in read_member(z, CASANOVO):
        if r["Source File"] == REAL_MGF:
            rows_per_scan[scan_number(r["Scan"])] += 1
    duplicated = {scan for scan, n in rows_per_scan.items() if n > 1}
    pool = {}
    for r in read_member(z, CASANOVO):
        report["casanovo_rows"] += 1
        if r["Source File"] != REAL_MGF:
            report["casanovo_rows_other_source_file"] += 1
            continue
        scan = scan_number(r["Scan"])
        try:
            tokens = casanovo_tokens(r["sequence"])
        except ValueError:
            report["casanovo_unreadable_sequence"] += 1
            continue
        scores = [s for s in r["opt_ms_run[1]_aa_scores"].split(",") if s.strip()]
        if len(scores) != len(tokens):
            report["casanovo_score_count_mismatch"] += 1
            continue
        seq = "".join(t for t in tokens if t)
        res_scores = [s for t, s in zip(tokens, scores) if t]
        if scan in duplicated:
            report["casanovo_duplicate_scan_rows"] += 1
            continue
        if set(seq) - CANONICAL:
            report["casanovo_noncanonical_letter"] += 1
            continue
        if not MIN_LEN <= len(seq) <= MAX_LEN:
            report["casanovo_length_outside_7_30"] += 1
            continue
        if scan not in accepted:
            report["casanovo_no_db_psm"] += 1
            continue
        pool[scan] = (seq, r["Casanovo score"], res_scores)
    report["pool"] = len(pool)
    chosen = sorted(pool, key=lambda s: unit_hash(SALT + str(s)))[:N_CALLS]
    report["chosen"] = len(chosen)

    with open(str(outdir / "calls.tsv"), "w") as f:
        f.write("call_id\trun_id\tscan_id\tcalled_sequence\tcall_score\tresidue_scores\talternatives\n")
        for scan in sorted(chosen):
            seq, score, rs = pool[scan]
            f.write("scan%d\t%s\t%d\t%s\t%s\t%s\t\n" % (
                scan, REAL_MGF.rsplit(".", 1)[0], scan, seq, score, ",".join(rs)))
    with open(str(outdir / "accepted.tsv"), "w") as f:
        f.write("call_id\taccepted_peptide\tevidence_tool\tevidence_qvalue\tevidence_score\n")
        for scan in sorted(chosen):
            lgp, pep = accepted[scan]
            f.write("scan%d\t%s\tPEAKS DB, deposited db.psms.csv (1%% peptide FDR)\t0.01\t%s\n"
                    % (scan, pep, lgp))
    exact = sum(pool[s][0].replace("I", "L") == accepted[s][1].replace("I", "L") for s in chosen)
    report["chosen_call_equals_accepted_IL"] = exact
    (outdir / "inputs.json").write_text(json.dumps({
        "data_origin": "experimental",
        "source": "PRIDE PXD055277 eval_human.zip; Casanovo 4.1.0 calls; deposited db.psms.csv",
        "proteins": [{"path": "proteins_human.fasta", "role": "human"},
                     {"path": "proteins_contaminants.fasta", "role": "contaminant"}],
        "calls": "calls.tsv", "accepted": "accepted.tsv"}, indent=2))
    (outdir / "conversion_report.json").write_text(json.dumps(dict(report), indent=2))
    return dict(report)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd = argv[1]
    if cmd == "inspect" and len(argv) == 3:
        print(json.dumps(inspect(argv[2]), indent=2))
    elif cmd == "collection" and len(argv) == 5:
        print(json.dumps(collection(argv[2], argv[3], argv[4]), indent=2))
    elif cmd == "convert" and len(argv) == 4:
        print(json.dumps(convert(argv[2], argv[3]), indent=2))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
