#!/usr/bin/env python3
"""Decoy queries for Foldseek and ProstT5: shuffled copies whose hits are chance hits.

    structure_decoys.py sequence --in proteome.fasta --max-len 6000 --out-real real.fasta \
        --out-decoy decoy.fasta --skipped skipped.tsv [--seed 0]
    structure_decoys.py columns --aa aa.fasta --ss ss.fasta \
        --out-aa decoy_aa.fasta --out-ss decoy_ss.fasta [--seed 0]

sequence  For ProstT5. Proteins longer than --max-len are left out of both files and
          listed in skipped.tsv (ProstT5 runs each sequence whole and the QfO region
          benchmark capped it at 6_000 residues); they are "not run" for ProstT5. Each
          remaining protein's residues are reordered with the dipeptide shuffle in
          nextflow-runs/shared/decoys.py (same first and last residue, same count of every
          adjacent residue pair). ProstT5 then predicts 3Di letters for the shuffled
          sequence, as it does for the real one.
columns   For Foldseek. A shuffled protein has no structure, so a structure cannot be
          shuffled by shuffling its sequence. Foldseek scores an alignment from two
          letters per residue, the amino acid and the 3Di letter that describes the
          residue's structural neighbourhood. Here each residue's (amino acid, 3Di) pair is
          one symbol, and the dipeptide shuffle reorders those symbols, so the decoy keeps
          the protein's amino-acid and 3Di composition and the count of every adjacent pair
          of pairs. The decoy has no C-alpha coordinates; Foldseek's default alignment
          (3Di and amino acids) and its E-value do not use them.

Each protein gets its own random generator, seeded from --seed and the protein's name, so a
decoy does not depend on which other proteins are in the file.

nextflow-runs/shared/decoys.py must be importable: the Nextflow process stages it as
decoys.py and puts the task folder on PYTHONPATH.
"""
import argparse
import random
import sys
import zlib

from decoys import dipeptide_shuffle


def read_fasta(path):
    recs, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name, buf = line[1:].rstrip("\n"), []
            else:
                buf.append(line.strip())
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def rng_for(seed, name):
    return random.Random((int(seed) << 32) ^ zlib.crc32(name.split()[0].encode()))


def write_fasta(path, recs):
    with open(path, "w") as fh:
        for name, seq in recs:
            fh.write(">%s\n%s\n" % (name, seq))


def shuffle_columns(aa, ss, rng):
    """Dipeptide-shuffle the residues of one protein as (amino acid, 3Di) pairs."""
    if len(aa) != len(ss):
        raise ValueError("amino-acid and 3Di strings differ in length: %d, %d" % (len(aa), len(ss)))
    code, back = {}, []
    sym = []
    for pair in zip(aa, ss):
        if pair not in code:
            # Private-use code points: one character per distinct pair, so the shared
            # string shuffle can be used unchanged.
            code[pair] = chr(0xE000 + len(back))
            back.append(pair)
        sym.append(code[pair])
    out = dipeptide_shuffle("".join(sym), rng)
    pairs = [back[ord(c) - 0xE000] for c in out]
    return "".join(p[0] for p in pairs), "".join(p[1] for p in pairs)


def cmd_sequence(args):
    recs = read_fasta(args.inp)
    keep = [(n, s) for n, s in recs if len(s) <= args.max_len]
    with open(args.skipped, "w") as fh:
        fh.write("accession\tlength\n")
        for n, s in recs:
            if len(s) > args.max_len:
                fh.write("%s\t%d\n" % (n.split()[0], len(s)))
    write_fasta(args.out_real, keep)
    write_fasta(args.out_decoy, [(n, dipeptide_shuffle(s, rng_for(args.seed, n))) for n, s in keep])
    print("%d sequences, %d longer than %d left out; %d decoys"
          % (len(recs), len(recs) - len(keep), args.max_len, len(keep)))


def cmd_columns(args):
    aa, ss = read_fasta(args.aa), read_fasta(args.ss)
    if [n.split()[0] for n, _ in aa] != [n.split()[0] for n, _ in ss]:
        sys.exit("%s and %s do not list the same proteins in the same order" % (args.aa, args.ss))
    out_aa, out_ss = [], []
    for (name, a), (_n, s) in zip(aa, ss):
        da, ds = shuffle_columns(a, s, rng_for(args.seed, name))
        out_aa.append((name, da))
        out_ss.append((name, ds))
    write_fasta(args.out_aa, out_aa)
    write_fasta(args.out_ss, out_ss)
    print("%d decoy structures" % len(aa))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("sequence")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--max-len", type=int, required=True)
    s.add_argument("--out-real", required=True)
    s.add_argument("--out-decoy", required=True)
    s.add_argument("--skipped", required=True)
    s.add_argument("--seed", default=0, type=int)
    c = sub.add_parser("columns")
    c.add_argument("--aa", required=True)
    c.add_argument("--ss", required=True)
    c.add_argument("--out-aa", required=True)
    c.add_argument("--out-ss", required=True)
    c.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    if args.cmd == "sequence":
        cmd_sequence(args)
    elif args.cmd == "columns":
        cmd_columns(args)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
