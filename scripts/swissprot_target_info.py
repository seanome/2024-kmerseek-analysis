#!/usr/bin/env python3
"""Species, length and curated domain positions of chosen Swiss-Prot entries.

    python3 swissprot_target_info.py <uniprot_sprot.dat.gz> <accessions.txt> <out_prefix>

Writes <out_prefix>_info.tsv (accession, entry_name, organism, length, description: the first
RecName or SubName Full=), <out_prefix>_seqs.fasta
(each entry's sequence under its requested accession) and <out_prefix>_domains.tsv (accession,
start, end, name): one row per FT DOMAIN feature, 1-based inclusive, as UniProt curators placed it (positions with < or > are kept without the
sign; features with an unknown end are skipped). The organism is the OS line without its
trailing period.

Added 2026-10-13 for notebook 281's page, so each target shows its species and domains.
Python 3.6, standard library only.
"""
import gzip
import re
import sys

POS = re.compile(r"^FT   DOMAIN          [<>]?(\d+)\.\.[<>]?(\d+)\s*$")
NOTE = re.compile(r'^FT                   /note="([^"]*)"?')


def main(dat, acc_path, prefix):
    wanted = set(l.strip() for l in open(acc_path) if l.strip())
    info, doms, seqs = open(prefix + "_info.tsv", "w"), open(prefix + "_domains.tsv", "w"), open(prefix + "_seqs.fasta", "w")
    info.write("accession\tentry_name\torganism\tlength\tdescription\n")
    doms.write("accession\tstart\tend\tname\n")
    accs, name, os_, length, pending, entry_doms, seq, in_seq, de = [], "", "", 0, None, [], [], False, ""
    with gzip.open(dat, "rt") as fh:
        for line in fh:
            if in_seq and not line.startswith("//"):
                seq.append(line.replace(" ", "").strip())
                continue
            if line.startswith("SQ   "):
                in_seq = True
            elif line.startswith("ID   "):
                f = line.split()
                name, length = f[1], int(f[3])
            elif line.startswith("AC   "):
                accs += [a.strip() for a in line[5:].split(";") if a.strip()]
            elif line.startswith("DE   ") and not de and "Full=" in line:
                de = line.split("Full=", 1)[1].split("{")[0].strip().rstrip(";")
            elif line.startswith("OS   "):
                os_ += (" " if os_ else "") + line[5:].strip()
            elif line.startswith("FT   DOMAIN"):
                m = POS.match(line.rstrip("\n"))
                pending = [int(m.group(1)), int(m.group(2)), ""] if m else None
                if pending:
                    entry_doms.append(pending)
            elif pending is not None and line.startswith("FT                   /note="):
                m = NOTE.match(line.rstrip("\n"))
                pending[2] = m.group(1) if m else ""
                pending = None
            elif line.startswith("FT   "):
                pending = None
            elif line.startswith("//"):
                hit = [a for a in accs if a in wanted]
                for a in hit:
                    info.write("%s\t%s\t%s\t%d\t%s\n" % (a, name, os_.rstrip("."), length, de))
                    for s, e, n in entry_doms:
                        doms.write("%s\t%d\t%d\t%s\n" % (a, s, e, n))
                    seqs.write(">%s\n%s\n" % (a, "".join(seq)))
                accs, name, os_, length, pending, entry_doms, seq, in_seq, de = [], "", "", 0, None, [], [], False, ""
    info.close()
    doms.close()
    seqs.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
