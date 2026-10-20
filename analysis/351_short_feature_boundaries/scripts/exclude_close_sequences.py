"""Remove the query proteins and their close relatives from Swiss-Prot, and check it worked.

A close relative is a sequence with MMseqs2 fident >= 0.90 (fraction of identical
aligned residues) over an alignment covering at least half of either sequence. The
query proteins themselves are removed by accession and by identical sequence.

Swiss-Prot headers are first rewritten to the bare accession (>P12345), because MMseqs2
reads sp|P12345|NAME as P12345 while HMMER and BLAST keep the whole first word, and
kmerseek writes the whole header line, commas included, into its CSV.

Usage:
  python -I exclude_close_sequences.py prep <uniprot_sprot.fasta> <swissprot.acc.fasta>
  python -I exclude_close_sequences.py remove <swissprot.fasta> <queries.fasta> <hits.m8> <out.fasta> <removed.tsv>
  python -I exclude_close_sequences.py check <hits_after.m8> <check.txt>

hits.m8 columns (MMseqs2 --format-output):
  query,target,fident,alnlen,qstart,qend,tstart,tend,qlen,tlen,evalue
"""

import sys

MIN_IDENTITY = 0.90
MIN_COVER = 0.5
M8_COLS = "query,target,fident,alnlen,qstart,qend,tstart,tend,qlen,tlen,evalue".split(
    ","
)


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name:
            seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def accession(name):
    """sp|P12345|NAME_HUMAN -> P12345; a bare accession stays as it is."""
    parts = name.split("|")
    return parts[1] if len(parts) >= 3 else name


def read_m8(path):
    for line in open(path):
        row = dict(zip(M8_COLS, line.rstrip("\n").split("\t")))
        for c in ("fident",):
            row[c] = float(row[c])
        for c in ("alnlen", "qstart", "qend", "tstart", "tend", "qlen", "tlen"):
            row[c] = int(row[c])
        yield row


def is_close(row):
    qcov = (row["qend"] - row["qstart"] + 1) / row["qlen"]
    tcov = (row["tend"] - row["tstart"] + 1) / row["tlen"]
    return row["fident"] >= MIN_IDENTITY and max(qcov, tcov) >= MIN_COVER


def prep(in_path, out_path):
    seqs = read_fasta(in_path)
    with open(out_path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{accession(name)}\n{seq}\n")
    print(f"{len(seqs)} Swiss-Prot sequences written with bare accessions")


def remove(sp_path, q_path, m8_path, out_path, removed_path):
    sp = read_fasta(sp_path)
    queries = read_fasta(q_path)
    # An isoform query (P12345-2) also removes its canonical entry (P12345).
    q_acc = {accession(q).split("-")[0] for q in queries}
    q_seq = set(queries.values())
    reason = {}
    for name, seq in sp.items():
        if accession(name).split("-")[0] in q_acc:
            reason[name] = "query accession"
        elif seq in q_seq:
            reason[name] = "identical to a query"
    for row in read_m8(m8_path):
        if is_close(row) and row["target"] not in reason:
            reason[row["target"]] = f"fident {row['fident']:.3f} to {row['query']}"
    with open(out_path, "w") as fh:
        for name, seq in sp.items():
            if name not in reason:
                fh.write(f">{name}\n{seq}\n")
    with open(removed_path, "w") as fh:
        fh.write("target\treason\n")
        for name in sorted(reason):
            fh.write(f"{name}\t{reason[name]}\n")
    print(f"Swiss-Prot {len(sp)}; removed {len(reason)}; kept {len(sp) - len(reason)}")


def check(m8_path, out_path):
    """Print the closest remaining sequence; fail if any is at or above the cutoff."""
    best = None
    n = 0
    for row in read_m8(m8_path):
        qcov = (row["qend"] - row["qstart"] + 1) / row["qlen"]
        tcov = (row["tend"] - row["tstart"] + 1) / row["tlen"]
        if max(qcov, tcov) < MIN_COVER:
            continue
        n += 1
        if best is None or row["fident"] > best["fident"]:
            best = row
    lines = [f"alignments covering >= {MIN_COVER:.0%} of either sequence: {n}"]
    if best is None:
        lines.append("highest fident: none (no such alignment)")
        ok = True
    else:
        lines.append(
            f"highest fident: {best['fident']:.3f} ({best['query']} vs {best['target']}, "
            f"{best['alnlen']} aligned residues)"
        )
        ok = best["fident"] < MIN_IDENTITY
    lines.append(
        f"exclusion check: {'PASSED' if ok else 'FAILED'} (cutoff {MIN_IDENTITY})"
    )
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(*sys.argv[2:4])
    elif sys.argv[1] == "remove":
        remove(*sys.argv[2:7])
    elif sys.argv[1] == "check":
        check(*sys.argv[2:4])
    else:
        sys.exit(f"unknown step {sys.argv[1]}")
