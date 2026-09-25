#!/usr/bin/env python3
"""Turn phmmer's saved alignment (-A) and domain table into one row per aligned domain.

phmmer builds its model from the query, so model match state i is query residue i. In the
Stockholm file, `#=GC RF` marks the match columns with `x`; in a hit row an uppercase letter
in a match column is a target residue aligned to that query residue, a lowercase letter is
a target residue inserted between two query residues, and `-` in a match column is a query
residue with no target residue opposite. Row names are `<target>/<ali from>-<ali to>`, the
same coordinates as the domain table's `ali from`/`ali to`, which is how each row gets its
E-value (the domain's independent E-value, i-Evalue).

Only rows whose target is a family protein other than the query are written.

    parse_phmmer.py query.fasta family.fasta hits.sto hits.domtbl out.tsv
"""

import re
import sys

import polars as pl

query_fa, family_fa, sto, domtbl, out = sys.argv[1:6]


def fasta(path):
    recs, h = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            h = line[1:].split()[0]
            recs[h] = ""
        elif h:
            recs[h] += line.strip()
    return recs


acc = lambda name: name.split("|")[1]
((query_name, query_seq),) = fasta(query_fa).items()
family = {acc(n) for n in fasta(family_fa)}

# Stockholm: concatenate each row's blocks
rows, rf = {}, ""
for line in open(sto):
    if not line.strip() or line.startswith(
        ("# STOCKHOLM", "//", "#=GF", "#=GS", "#=GR")
    ):
        continue
    if line.startswith("#=GC RF"):
        rf += line.split()[-1]
        continue
    if line.startswith("#"):
        continue
    name, s = line.split()
    rows[name] = rows.get(name, "") + s

# domain table: target, query, ali from, ali to -> i-Evalue, score
evalue = {}
for line in open(domtbl):
    if line.startswith("#"):
        continue
    f = line.split()
    # columns: 0 target, 3 query, 12 i-Evalue, 13 score, 17 ali from, 18 ali to
    evalue[(f[0], int(f[17]), int(f[18]))] = (float(f[12]), float(f[13]))

out_rows = []
for name, s in rows.items():
    target, span = name.rsplit("/", 1)
    t_from, t_to = map(int, span.split("-"))
    if acc(target) not in family or acc(target) == acc(query_name):
        continue
    q, t = 0, t_from - 1
    pairs, qaln, taln = [], [], []
    for col, ch in enumerate(s):
        match = rf[col] == "x"
        if match:
            q += 1
        if ch.isalpha():
            t += 1
        if match and ch.isupper():
            pairs.append((q, t))
            qaln.append(query_seq[q - 1])
            taln.append(ch)
        elif match and ch in "-.":
            qaln.append(query_seq[q - 1])
            taln.append("-")
        elif not match and ch.isalpha():
            qaln.append("-")
            taln.append(ch.upper())
    assert t == t_to, (name, t, t_to)
    # trim the unaligned model positions at either end
    first = next(i for i, (a, b) in enumerate(zip(qaln, taln)) if a != "-" and b != "-")
    last = max(i for i, (a, b) in enumerate(zip(qaln, taln)) if a != "-" and b != "-")
    e, score = evalue[(target, t_from, t_to)]
    out_rows.append(
        dict(
            tool="phmmer",
            query=acc(query_name),
            target=acc(target),
            evalue=e,
            bitscore=score,
            qstart=pairs[0][0],
            qend=pairs[-1][0],
            tstart=pairs[0][1],
            tend=pairs[-1][1],
            qaln="".join(qaln[first : last + 1]),
            taln="".join(taln[first : last + 1]),
        )
    )

schema = dict(
    tool=pl.String,
    query=pl.String,
    target=pl.String,
    evalue=pl.Float64,
    bitscore=pl.Float64,
    qstart=pl.Int64,
    qend=pl.Int64,
    tstart=pl.Int64,
    tend=pl.Int64,
    qaln=pl.String,
    taln=pl.String,
)
pl.DataFrame(out_rows, schema=schema).write_csv(out, separator="\t")
print(f"{acc(query_name)}: {len(out_rows)} family domains")
