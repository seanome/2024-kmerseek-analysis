"""Build 30-letter HP k-mers from each probe set's proteins, and the values behind them for each series.

Usage: hp_kmer_table.py <probe_set_proteins.tsv.gz> <array name> <GPL table> <out folder> <proteins.fasta>[,...]
       [<series matrix>...]
The rules are in hp_table_plan.md. Writes, for each alphabet, <array>.<alphabet>.kmers.npz and .summary.tsv, and for
each series matrix <series>.probe_set_values.tsv.gz and <series>.<alphabet>.shared_kmers.tsv.gz.
"""

import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

K = 30
ALPHABETS = {"hp_lehninger2": "AFGILMPVWY", "hp_thomas_dill2": "ACFILMVWY"}
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
NUCLEOTIDE_SOURCES = {"est", "genomic_clone"}


def hp_codes(protein, hydrophobic):
    """Each window's k-mer packed as K bits, hydrophobic = 1 and the first residue highest; windows with another
    letter are left out."""
    if len(protein) < K:
        return np.zeros(0, dtype=np.uint64)
    bits = np.array([residue in hydrophobic for residue in protein], dtype=np.uint64)
    good = np.array([residue in AMINO_ACIDS for residue in protein], dtype=np.int64)
    weights = np.uint64(1) << np.arange(K - 1, -1, -1, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(bits, K)
    codes = (windows * weights).sum(axis=1, dtype=np.uint64)
    complete = np.convolve(good, np.ones(K, dtype=np.int64), mode="valid") == K
    return codes[complete]


def as_letters(code):
    return "".join("h" if (int(code) >> (K - 1 - i)) & 1 else "p" for i in range(K))


def read_fasta(paths):
    sequences, name, parts = {}, None, []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    if name:
                        sequences[name] = "".join(parts)
                    name, parts = line[1:].split()[0], []
                else:
                    parts.append(line.strip())
    if name:
        sequences[name] = "".join(parts)
    return sequences


def probeset_ids(gpl):
    """HTA 2.0 series matrices name probe sets by probeset_id; map those to the platform's ID. Empty for U133."""
    with open(gpl, errors="replace") as f:
        for line in f:
            if line.startswith("ID\t"):
                header = line.rstrip("\n").split("\t")
                if "probeset_id" not in header:
                    return {}
                column = header.index("probeset_id")
                return {
                    cells[column]: cells[0] for cells in (l.rstrip("\n").split("\t") for l in f) if len(cells) > column
                }
    return {}


def read_series(path, rename):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = iter(f)
        for line in lines:
            if line.startswith("!series_matrix_table_begin"):
                break
        samples = [s.strip('"') for s in next(lines).rstrip("\n").split("\t")[1:]]
        values = {}
        for line in lines:
            if line.startswith("!series_matrix_table_end"):
                break
            name, *cells = line.rstrip("\n").split("\t")
            name = name.strip('"')
            values[rename.get(name, name)] = np.array([float(c) if c not in ("", "null") else np.nan for c in cells])
    return samples, values


pairs_file, array, gpl, out, fasta, *series = sys.argv[1:]
out = Path(out)
stem = array.replace(" ", "_")  # file names without spaces
proteins_of = defaultdict(set)
with gzip.open(pairs_file, "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["array"] == array and r["source"] not in NUCLEOTIDE_SOURCES:
            proteins_of[r["probe_set"]].add(r["sequence"])
sequences = read_fasta(fasta.split(","))
probe_sets = sorted(proteins_of)
rename = probeset_ids(gpl)
series_values = {Path(p).name.split("_")[0]: read_series(p, rename) for p in series}
for gse, (samples, values) in series_values.items():
    with gzip.open(out / f"{gse}.probe_set_values.tsv.gz", "wt") as f:
        f.write("probe_set\t" + "\t".join(samples) + "\n")
        for name in probe_sets:
            if name in values:
                f.write(name + "\t" + "\t".join(f"{v:.4f}" for v in values[name]) + "\n")

for alphabet, hydrophobic in ALPHABETS.items():
    hydrophobic = set(hydrophobic)
    codes, owners = [], []
    missing = 0
    for index, name in enumerate(probe_sets):
        found = [sequences[p] for p in proteins_of[name] if p in sequences]
        missing += len(proteins_of[name]) - len(found)
        own = np.unique(np.concatenate([hp_codes(p, hydrophobic) for p in found])) if found else np.zeros(0, np.uint64)
        codes.append(own)
        owners.append(np.full(len(own), index, dtype=np.int32))
    codes, owners = np.concatenate(codes), np.concatenate(owners)
    order = np.lexsort((owners, codes))
    codes, owners = codes[order], owners[order]
    distinct, starts, sharing = np.unique(codes, return_index=True, return_counts=True)
    np.savez_compressed(
        out / f"{stem}.{alphabet}.kmers.npz",
        kmer=distinct,
        start=starts,
        probe_sets=sharing,
        owner=owners,
        names=np.array(probe_sets),
    )
    spread = Counter(sharing.tolist())
    with open(out / f"{stem}.{alphabet}.summary.tsv", "w") as f:
        f.write(
            "array\talphabet\tk\tprobe_sets_with_a_protein\tprotein_names_not_in_fasta\tprobe_sets_with_a_kmer\t"
            "distinct_kmers\tkmers_in_one_probe_set\tkmers_in_2_or_more\tkmers_in_10_or_more\tmost_probe_sets\n"
        )
        f.write(
            f"{array}\t{alphabet}\t{K}\t{len(probe_sets)}\t{missing}\t{len(np.unique(owners))}\t{len(distinct)}\t"
            f"{spread[1]}\t{int((sharing >= 2).sum())}\t{int((sharing >= 10).sum())}\t{int(sharing.max())}\n"
        )
    with open(out / f"{stem}.{alphabet}.sharing.tsv", "w") as f:
        f.write("probe_sets_sharing_a_kmer\tkmers\n")
        f.writelines(f"{n}\t{c}\n" for n, c in sorted(spread.items()))
    shared = np.flatnonzero(sharing >= 2)
    for gse, (samples, values) in series_values.items():
        table = np.array([values.get(n, np.full(len(samples), np.nan)) for n in probe_sets])
        with gzip.open(out / f"{gse}.{alphabet}.shared_kmers.tsv.gz", "wt") as f:
            f.write("kmer\tprobe_sets\t" + "\t".join(samples) + "\n")
            for i in shared:
                rows = owners[starts[i] : starts[i] + sharing[i]]
                mean = np.nanmean(table[rows], axis=0)
                f.write(f"{as_letters(distinct[i])}\t{sharing[i]}\t" + "\t".join(f"{v:.4f}" for v in mean) + "\n")
