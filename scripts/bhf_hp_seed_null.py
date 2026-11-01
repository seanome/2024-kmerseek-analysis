#!/usr/bin/env python3
"""Does BHF share more exact H/P k-mers with Swiss-Prot than its composition alone gives?

    python3 bhf_hp_seed_null.py <bhf.fasta> <reference.fasta> <out_prefix> [--kmin 11] [--kmax 16] [--shuffles 100]

For k = kmin..kmax, encodes BHF and every reference sequence in hp_lehninger2 (sourmash
0.20.0 aa_to_hp, the encoder kmerseek uses for that alphabet: A F G I L M P V W Y are h;
N C S T D E R H K Q are p; any other letter breaks a k-mer), and counts, for BHF and for
`--shuffles` copies of BHF with its residues shuffled (same composition, seed 0):

  proteins_ge1/2/3/5   reference proteins sharing at least 1, 2, 3 or 5 distinct k-mers
  max_shared           the most distinct k-mers BHF (or the copy) shares with one protein
  best_target          that protein

Shared means the same H/P k-mer anywhere in both, not on one diagonal, so it counts seeds,
not regions, and gives no E-value. The empirical p for BHF is (1 + shuffles scoring at
least BHF's value) / (1 + shuffles).

Writes <out_prefix>.counts.tsv (one row per k and query) and <out_prefix>.bhf_top.tsv (the
20 proteins sharing the most k-mers with BHF at each k). Added 2026-10-12 for notebook 279
(PR 120): it decides whether indexes below hp_lehninger2's k_min of 17 are worth building
for BHF. Run with the kmerseek image's python (numpy).
"""

import argparse

import numpy as np

H = set("AFGILMPVWY")
P = set("NCSTDERHKQ")


def read_fasta(path):
    names, seqs, buf, name = [], [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append("".join(buf))
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        names.append(name)
        seqs.append("".join(buf))
    return names, seqs


BIT = np.zeros(256, dtype=np.uint8)
VALID = np.zeros(256, dtype=bool)
for a in H:
    BIT[ord(a)], VALID[ord(a)] = 1, True
for a in P:
    VALID[ord(a)] = True


def encode(seq):
    """bit (1 = h, 0 = p) and validity per residue, through 256-entry lookup tables."""
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return BIT[raw], VALID[raw]


def kmer_codes(bits, valid, k):
    """Integer code of every k-mer start and whether the whole window is valid."""
    n = len(bits)
    if n < k:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    codes = np.zeros(n - k + 1, dtype=np.int64)
    for j in range(k):
        codes = (codes << 1) | bits[j:n - k + 1 + j]
    bad = np.concatenate([[0], np.cumsum(~valid)])
    ok = (bad[k:] - bad[:-k]) == 0
    return codes, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bhf")
    ap.add_argument("reference")
    ap.add_argument("out_prefix")
    ap.add_argument("--kmin", type=int, default=11)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--shuffles", type=int, default=100)
    args = ap.parse_args()

    _, (bhf,) = read_fasta(args.bhf)
    rnames, rseqs = read_fasta(args.reference)
    # One string with a separator that breaks k-mers, and the protein index of each residue
    sep = "*"
    joined = sep.join(rseqs)
    lengths = np.array([len(s) for s in rseqs], dtype=np.int64)
    pid = np.repeat(np.arange(len(rseqs), dtype=np.int64), lengths + 1)[:len(joined)]
    rbits, rvalid = encode(joined)
    rng = np.random.default_rng(0)
    queries = [("BHF", bhf)] + [(f"shuffle_{i:03d}", "".join(rng.permutation(list(bhf)))) for i in range(args.shuffles)]
    print(f"reference: {len(rseqs):_} proteins, {len(joined):_} residues; BHF {len(bhf)} aa", flush=True)

    rows, top = [], []
    for k in range(args.kmin, args.kmax + 1):
        codes, ok = kmer_codes(rbits, rvalid, k)
        keys = np.unique(pid[:len(codes)][ok] * (1 << k) + codes[ok])
        pairs_pid, pairs_code = keys >> k, keys & ((1 << k) - 1)
        for qname, qseq in queries:
            qb, qv = encode(qseq)
            qc, qok = kmer_codes(qb, qv, k)
            mask = np.zeros(1 << k, dtype=bool)
            mask[qc[qok]] = True
            counts = np.bincount(pairs_pid[mask[pairs_code]], minlength=len(rseqs))
            best = int(np.argmax(counts))
            rows.append((k, qname, int(mask.sum()), int((counts >= 1).sum()), int((counts >= 2).sum()),
                         int((counts >= 3).sum()), int((counts >= 5).sum()), int(counts[best]), rnames[best]))
            if qname == "BHF":
                for i in np.argsort(-counts, kind="stable")[:20]:
                    top.append((k, rnames[i], int(counts[i]), len(rseqs[i])))
        b = rows[-(len(queries))]
        print(f"k={k}: BHF {b[2]} distinct k-mers, {b[3]:_} proteins share >=1, max {b[7]} with {b[8]}", flush=True)

    with open(args.out_prefix + ".counts.tsv", "w") as fh:
        fh.write("k\tquery\tdistinct_kmers\tproteins_ge1\tproteins_ge2\tproteins_ge3\tproteins_ge5\tmax_shared\tbest_target\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    with open(args.out_prefix + ".bhf_top.tsv", "w") as fh:
        fh.write("k\ttarget\tshared_kmers\ttarget_length\n")
        for r in top:
            fh.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()
