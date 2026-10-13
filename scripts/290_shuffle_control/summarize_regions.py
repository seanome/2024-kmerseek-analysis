#!/usr/bin/env python3
"""Read one kmerseek region CSV on stdin and keep what notebook 290 needs from it.

    <regions CSV> | summarize_regions.py <out_prefix> <source_label>

The same code reads the run's real region files and this experiment's searches, so real
and shuffled reach are counted the same way. Writes, each first as .tmp and renamed only
when the whole input was read:

  <out_prefix>.min_evalue.csv.gz   one row per query with any region: regions, regions with a
                                   finite E-value, and the smallest finite region_evalue
                                   ("inf" when every region says inf). Same rule as
                                   tools/dark-regions min-evalue (PR 57).
  <out_prefix>.regions_e1.csv.gz   every region with a finite region_evalue below 1, with the
                                   columns tools/dark-regions keeps, for the low-complexity and
                                   helix filter and for the residues.
  <out_prefix>.check.tsv           rows read, queries, and an order-free hash of the rows
                                   (sum of each line's md5, mod 2^128), so two files with the
                                   same rows in a different order compare equal.

region_evalue can be the string "inf"; Python's float() reads it as infinity, never as 0.
Only query_name can be quoted (kmerseek quotes an id holding a comma), and region_evalue is
the second-to-last column (region_n_chained is last); the header is checked for both. A
search refused for want of a Karlin-Altschul fit writes a three-column header and no rows:
that file gives empty outputs and status "nofit".
"""
import csv
import gzip
import hashlib
import io
import os
import sys

csv.field_size_limit(10 ** 9)
KEEP = ["query_name", "target_name", "ksize", "scaled", "moltype", "region_start", "region_end",
        "region_subseq", "target_start", "target_end", "target_subseq", "moltype_seq",
        "region_length", "region_n_mismatches", "region_ka_bits", "region_evalue"]
MOD = 1 << 128


def main(prefix, label):
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="")
    header_line = stdin.readline()
    header = next(csv.reader([header_line]))
    status = "ok"
    if header == ["query_name", "target_name", "region_evalue"]:
        status = "nofit"
    elif header[0] != "query_name" or header[-2:] != ["region_evalue", "region_n_chained"]:
        sys.exit("unexpected header: first {} last {}".format(header[0], header[-2:]))
    idx = {k: header.index(k) for k in KEEP} if status == "ok" else {}
    stats = {}
    n = 0
    multiset = 0
    tmp = {s: prefix + s + ".tmp" for s in (".min_evalue.csv.gz", ".regions_e1.csv.gz", ".check.tsv")}
    with gzip.open(tmp[".regions_e1.csv.gz"], "wt", newline="") as e1:
        w = csv.writer(e1, lineterminator="\n")
        w.writerow(["source_file"] + KEEP)
        for line in stdin:
            n += 1
            multiset = (multiset + int(hashlib.md5(line.encode()).hexdigest(), 16)) % MOD
            q = next(csv.reader([line]))[0] if line.startswith('"') else line.split(",", 1)[0]
            ev = float(line.rstrip("\r\n").rsplit(",", 2)[1])
            s = stats.get(q)
            if s is None:
                s = stats[q] = [0, 0, float("inf")]
            s[0] += 1
            if ev != float("inf"):
                s[1] += 1
                if ev < s[2]:
                    s[2] = ev
                if ev < 1:
                    row = next(csv.reader([line]))
                    if float(row[idx["region_evalue"]]) != ev:
                        sys.exit("region_evalue column read two ways disagrees on a row of " + q)
                    w.writerow([label] + [row[idx[k]] for k in KEEP])
    if status == "nofit" and n:
        sys.exit("a three-column header with {} rows".format(n))
    with gzip.open(tmp[".min_evalue.csv.gz"], "wt", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["source_file", "query_name", "regions", "regions_finite", "min_region_evalue"])
        for q in sorted(stats):
            r, f, m = stats[q]
            w.writerow([label, q, r, f, repr(m)])
    with open(tmp[".check.tsv"], "w") as fh:
        fh.write("source_file\tstatus\trows\tqueries\theader_md5\trow_hash\n")
        fh.write("{}\t{}\t{}\t{}\t{}\t{:032x}\n".format(label, status, n, len(stats),
                                                     hashlib.md5(header_line.encode()).hexdigest(), multiset))
    for s, t in tmp.items():
        os.replace(t, prefix + s)
    print("{}: {} rows, {} queries, status {}".format(label, n, len(stats), status))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
