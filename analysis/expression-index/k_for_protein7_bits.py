"""Choose, for each kmerseek alphabet, the k whose k-mers carry as many bits as a protein 7-mer.

Bits per letter are the Shannon entropy of the alphabet's letters, with each letter's
frequency the summed Swiss-Prot frequency of the amino acids it groups. Bits per k-mer
are k times that. Protein k = 7 is the reference; each alphabet gets the k whose bits are
nearest to it (ties go to the larger k).

The amino acid groups are read from kmerseek's own source (src/rust/alphabets.rs) and,
for dayhoff6, from sourmash's encodings.rs, so this script holds no grouping of its own.

    python k_for_protein7_bits.py KMERSEEK_DIR SWISSPROT_FASTA_GZ OUT_TSV
"""

import gzip
import math
import re
import sys
from collections import Counter
from pathlib import Path

AA20 = "ACDEFGHIKLMNPQRSTVWY"
REF_K = 7


def swissprot_frequencies(path):
    counts = Counter()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                counts.update(line.strip())
    total = sum(counts[a] for a in AA20)
    return {a: counts[a] / total for a in AA20}, total


def alphabet_groups(kmerseek_dir):
    """moltype -> list of amino acid groups, parsed from kmerseek's source."""
    src = (kmerseek_dir / "src/rust/alphabets.rs").read_text()
    # enum variant -> moltype name
    moltype = dict(re.findall(r"Self::(\w+) => \"(\w+)\"", src.split("fn to_moltype")[1].split("fn ")[0]))
    # enum variant -> static table name, from partition()
    part = src.split("fn partition")[1].split("\n    pub fn ")[0]
    static_of = {v: s for v, s in re.findall(r"Self::(\w+) => &(\w+),", part)}
    groups = {}
    for name, h, p in re.findall(r"static (\w+): LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| build_hp\(b\"(\w+)\", b\"(\w+)\"\)\)", src):
        groups[name] = [h, p]
    for name, h, p, c in re.findall(r"static (\w+): LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| build_hpc\(b\"(\w+)\", b\"(\w+)\", b\"(\w+)\"\)\)", src):
        groups[name] = [h, p, c]
    clusters = {n: re.findall(r"\"(\w+)\"", body) for n, body in re.findall(r"const (\w+)_CLUSTERS: &\[&str\] =\s*&\[([^\]]*)\]", src)}
    for name, cl in re.findall(r"static (\w+): LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| build_reduced\((\w+)_CLUSTERS\)\)", src):
        groups[name] = clusters[cl]
    out = {}
    for variant, mt in moltype.items():
        if variant in static_of:
            out[mt] = groups[static_of[variant]]
    out["protein20"] = list(AA20)
    # sourmash's Dayhoff and HP tables
    enc = (kmerseek_dir / "REFERENCE_sourmash/src/core/src/encodings.rs").read_text()
    for table, mt in (("DAYHOFFTABLE", "dayhoff6"), ("HPTABLE", "hp_lehninger2")):
        body = enc.split(f"static {table}")[1].split("collect()")[0]
        by_letter = {}
        for aa, letter in re.findall(r"\(b'(\w)', b'(\w)'\)", body):
            if aa in AA20:
                by_letter.setdefault(letter, []).append(aa)
        g = ["".join(v) for v in by_letter.values()]
        if mt in out and sorted(map(sorted, out[mt])) != sorted(map(sorted, g)):
            raise SystemExit(f"{mt}: kmerseek and sourmash groups differ")
        out[mt] = g
    for mt, g in out.items():
        if sorted("".join(g)) != sorted(AA20):
            raise SystemExit(f"{mt}: groups do not cover the 20 amino acids once: {g}")
    return out


def main(kmerseek_dir, fasta, out_tsv):
    freq, n_residues = swissprot_frequencies(fasta)
    groups = alphabet_groups(Path(kmerseek_dir))
    h_protein = -sum(f * math.log2(f) for f in freq.values())
    target = REF_K * h_protein
    rows = []
    for mt, g in groups.items():
        p = [sum(freq[a] for a in grp) for grp in g]
        h = -sum(x * math.log2(x) for x in p if x > 0)
        k = max(range(1, 200), key=lambda k: (-abs(k * h - target), k))
        rows.append((mt, len(g), h, k, k * h, " ".join(g)))
    rows.sort(key=lambda r: (r[1], r[0]))
    with open(out_tsv, "w") as fh:
        fh.write(f"# Swiss-Prot residues counted: {n_residues}; protein k={REF_K} bits: {target:.3f}\n")
        fh.write("alphabet\tletters\tbits_per_letter\tk\tbits_per_kmer\tgroups\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t{r[3]}\t{r[4]:.3f}\t{r[5]}\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
