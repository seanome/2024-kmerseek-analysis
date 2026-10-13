#!/usr/bin/env python3
"""Shuffled copies of the dark-set run's query chunks, for notebook 290's chance control.

    shuffle_chunks.py <chunks.tsv> <out_dir>

<chunks.tsv> has one row per real query chunk the run searched (species, chunk, path, md5;
written by tasks.py). For each chunk and each kind of shuffle this writes
<out_dir>/<species>/<chunk>.<kind>.fasta: the real chunk with every protein replaced by its
shuffled copy, in the same order, named <accession><suffix>. A shuffled chunk keeps the
real chunk's membership because kmerseek builds k-mer frequencies over the whole query
file and --max-query-pvalue uses query-level statistics.

Two kinds:
  dipeptide  every adjacent residue pair keeps its count, and the first and last residues
             stay (Altschul and Erickson 1985). shuffle_dipeptide() is imported unchanged
             from nextflow-runs/scope40-lowcomplexity-null/bin/shuffle_fasta.py, the code
             that made data/279_bhf_dipeptide_shuffles.fasta.
  window10   residues shuffled within non-overlapping 10-residue windows from the first
             residue (the last window may be shorter). window_shuffle() is copied from
             nextflow-runs/regions-index-pilot/bin/make_window_decoys.py (PR 114). A
             hydrophobic stretch or a low-complexity run stays where it was.

Each protein's random generator is seeded by sha256("290:<kind>:<accession>"), so a copy
does not depend on the chunk or the file order. The script checks every copy (dipeptide
counts, or the residues of every window) and refuses to overwrite a file whose content
differs. It writes <out_dir>/shuffles.tsv: md5, proteins, and proteins left unchanged
(shorter than 3 residues, or a shuffle that happened to give the same sequence).
Standard library only; Python 3.8 or later (run it with the kmerseek image's python3:
shuffle_fasta.py uses annotations Sherlock's system Python 3.6 cannot read).
"""
import csv
import hashlib
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nextflow-runs", "scope40-lowcomplexity-null", "bin"))
from shuffle_fasta import dipeptide_counts, shuffle_dipeptide  # noqa: E402

SEED = 290
WINDOW = 10
KINDS = {"dipeptide": "_SHUFDP", "window10": "_SHUFW10"}


def window_shuffle(seq, rng, window):
    """Copied from nextflow-runs/regions-index-pilot/bin/make_window_decoys.py (PR 114)."""
    out = []
    for i in range(0, len(seq), window):
        w = list(seq[i : i + window])
        rng.shuffle(w)
        out.extend(w)
    return "".join(out)


def read_fasta(path):
    """(name, sequence) in file order; name is the header's first word."""
    records, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line.strip():
                buf.append(line.strip())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def rng_for(kind, name):
    h = hashlib.sha256("{}:{}:{}".format(SEED, kind, name).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def shuffle_one(kind, name, seq):
    rng = rng_for(kind, name)
    if kind == "dipeptide":
        out = shuffle_dipeptide(seq, rng)
        if dipeptide_counts(out) != dipeptide_counts(seq) or out[:1] != seq[:1] or out[-1:] != seq[-1:]:
            raise ValueError("dipeptide counts or end residues changed in " + name)
    elif kind == "window10":
        out = window_shuffle(seq, rng, WINDOW)
        for i in range(0, len(seq), WINDOW):
            if Counter(seq[i : i + WINDOW]) != Counter(out[i : i + WINDOW]):
                raise ValueError("window composition changed in {} at {}".format(name, i))
    else:
        raise ValueError("unknown kind " + kind)
    if Counter(out) != Counter(seq):
        raise ValueError("composition changed in " + name)
    return out


def md5_of(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main(chunks_tsv, out_dir):
    rows = list(csv.DictReader(open(chunks_tsv), delimiter="\t"))
    report = []
    for r in rows:
        if md5_of(r["path"]) != r["md5"]:
            sys.exit("{} changed: md5 is not {}".format(r["path"], r["md5"]))
        records = read_fasta(r["path"])
        for kind, suffix in KINDS.items():
            text = []
            unchanged = 0
            for name, seq in records:
                out = shuffle_one(kind, name, seq)
                unchanged += out == seq
                text.append(">{}{}\n{}\n".format(name, suffix, out))
            text = "".join(text)
            md5 = hashlib.md5(text.encode()).hexdigest()
            dest = os.path.join(out_dir, r["species"], "{}.{}.fasta".format(r["chunk"], kind))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and md5_of(dest) != md5:
                sys.exit("{} exists with different content; not overwriting".format(dest))
            with open(dest, "w") as fh:
                fh.write(text)
            report.append([r["species"], r["chunk"], kind, dest, md5, len(records), unchanged])
            print("{} {} {}: {} proteins, {} unchanged, md5 {}".format(
                r["species"], r["chunk"], kind, len(records), unchanged, md5))
    with open(os.path.join(out_dir, "shuffles.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "chunk", "kind", "path", "md5", "proteins", "unchanged"])
        w.writerows(report)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
