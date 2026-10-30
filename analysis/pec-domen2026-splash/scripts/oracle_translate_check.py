#!/usr/bin/env python3
"""Check `pec-tools translate` against Biopython on random and real DNA.

Usage: oracle_translate_check.py <pec-tools binary> <out_dir> [real.fasta]

Biopython's Seq.translate (table 1) gives the six frames; every maximal run of
residues without '*' or 'X' of length >= 20 is a peptide. The two sets of
(strand, start, end, peptide) must be identical. Writes oracle_check.json.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

from Bio.Seq import Seq

MIN_AA = 20


def oracle(seq):
    n = len(seq)
    out = set()
    for strand, s in (("+", seq), ("-", str(Seq(seq).reverse_complement()))):
        for off in range(3):
            sub = s[off:]
            sub = sub[: len(sub) // 3 * 3]
            # Biopython turns codons with N into X; other IUPAC codes may translate, so mask them first
            codons = [sub[i:i + 3] for i in range(0, len(sub), 3)]
            aa = "".join("X" if any(c not in "ACGT" for c in cod) else str(Seq(cod).translate()) for cod in codons)
            start = 0
            for i in range(len(aa) + 1):
                if i == len(aa) or aa[i] in "*X":
                    if i - start >= MIN_AA:
                        a, b = off + 3 * start, off + 3 * i
                        ns, ne = (a, b) if strand == "+" else (n - b, n - a)
                        out.add((strand, ns, ne, aa[start:i]))
                    start = i + 1
    return out


def main():
    binary, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(361)
    seqs = {}
    for i in range(3000):
        L = rng.randint(60, 600)
        # few stops in some, many N in others
        alphabet = "ACGT" if i % 5 else "ACGTN"
        seqs[f"r{i}"] = "".join(rng.choice(alphabet) for _ in range(L))
    if len(sys.argv) > 3:
        name = None
        for line in open(sys.argv[3]):
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = ""
            else:
                seqs[name] += line.strip().upper()
    fa = out_dir / "oracle_input.fa"
    fa.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    subprocess.run([binary, "translate", str(fa), str(out_dir / "oracle"), str(MIN_AA)], check=True)
    got = {}
    with open(out_dir / "oracle.peptides.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(header, line.rstrip("\n").split("\t")))
            got.setdefault(r["seq_id"], set()).add((r["strand"], int(r["nt_start"]), int(r["nt_end"]), r["peptide"]))
    n_pep, mism = 0, []
    for k, s in seqs.items():
        want = oracle(s)
        n_pep += len(want)
        if want != got.get(k, set()):
            mism.append(k)
    res = dict(sequences=len(seqs), oracle_peptides=n_pep,
               pec_tools_peptides=sum(len(v) for v in got.values()), mismatched_sequences=len(mism),
               first_mismatches=mism[:10], min_aa=MIN_AA)
    (out_dir / "oracle_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if mism:
        sys.exit(1)


if __name__ == "__main__":
    main()
