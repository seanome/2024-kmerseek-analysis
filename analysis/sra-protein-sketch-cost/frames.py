"""Write the four translation choices for one FASTQ and count what each keeps.

Choices, per read:
  a  all six frames, stops kept as '*' (what `sourmash sketch translate` hashes)
  b  every stop-free stretch of 30 or more amino acids, in any frame
  c  the single longest stop-free stretch across the six frames
  d  the stop-free stretch (10+ aa) with the most protein 10-mers found in Swiss-Prot;
     reads with no Swiss-Prot 10-mer in any frame are dropped (an orpheum-like choice)

Each record of the input (a read, or a contig) counts as one "read" below.

Usage: python frames.py reads.fastq.gz|contigs.fa[.gz|.zst] uniprot_sprot.fasta.gz n_reads out_dir
Writes out_dir/{a,b,c,d}.faa and out_dir/frames_summary.json.
"""

import gzip
import json
import sys
import time

import numpy as np

K = 10
MIN_LEN_B = 30
AA = "ACDEFGHIKLMNPQRSTVWY"
CODE = np.full(256, 255, dtype=np.uint8)
for i, a in enumerate(AA):
    CODE[ord(a)] = i

BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: AMINO[16 * i + 4 * j + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}
COMP = str.maketrans("ACGTN", "TGCAN")


def kmer_codes(seq_bytes):
    """Base-20 codes of every K-mer of standard amino acids in a byte string."""
    c = CODE[np.frombuffer(seq_bytes, dtype=np.uint8)]
    n = len(c) - K + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    code = np.zeros(n, dtype=np.uint64)
    bad = np.zeros(n, dtype=bool)
    for j in range(K):
        w = c[j:j + n]
        bad |= w == 255
        code = code * np.uint64(20) + np.where(w == 255, 0, w).astype(np.uint64)
    return code[~bad]


def swissprot_kmers(path):
    seqs, cur = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    # '#' separates proteins so no K-mer spans two of them.
    return np.unique(kmer_codes("#".join(seqs).encode())), len(seqs)


def lines_of(path):
    """Lines of a plain, gzip or zstd file. For zstd, fail if zstdcat exits non-zero."""
    if path.endswith(".zst"):
        import io
        import subprocess

        proc = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE)
        read_all = False
        try:
            yield from io.TextIOWrapper(proc.stdout)
            read_all = True
        finally:
            proc.stdout.close()
            if not read_all:  # the caller stopped early
                proc.terminate()
            code = proc.wait()
        if code != 0:
            raise RuntimeError(f"zstdcat failed on {path} (exit {code}): truncated stream?")
    else:
        with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
            yield from f


def records(path):
    """Sequences from FASTQ or (multi-line) FASTA, plain, gzip or zstd."""
    f = lines_of(path)
    first = next(f, "")
    if first.startswith("@"):
        for i, line in enumerate(f, start=1):
            if i % 4 == 1:
                yield line.strip()
        return
    cur = []
    for line in f:
        if line.startswith(">"):
            if cur:
                yield "".join(cur).upper()
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        yield "".join(cur).upper()


def translate(dna):
    return "".join(CODON.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def six_frames(read):
    rc = read.translate(COMP)[::-1]
    return [translate(s[o:]) for s in (read, rc) for o in range(3)]


def main():
    fq, sprot, n_reads, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    t0 = time.time()
    ref, n_prot = swissprot_kmers(sprot)
    t_ref = time.time() - t0

    outs = {x: open(f"{out}/{x}.faa", "w") for x in "abcd"}
    res = {x: 0 for x in "abcd"}
    seqs = {x: 0 for x in "abcd"}
    reads = bases = 0
    t_trans = t_look = 0.0
    for read in records(fq):
        if reads >= n_reads:
            break
        reads += 1
        bases += len(read)
        t1 = time.time()
        frames = six_frames(read)
        pieces = [p for fr in frames for p in fr.split("*")]
        t_trans += time.time() - t1
        for fi, fr in enumerate(frames):
            outs["a"].write(f">r{reads}_f{fi}\n{fr}\n")
            res["a"] += len(fr)
            seqs["a"] += 1
        for pi, p in enumerate(pieces):
            if len(p) >= MIN_LEN_B:
                outs["b"].write(f">r{reads}_p{pi}\n{p}\n")
                res["b"] += len(p)
                seqs["b"] += 1
        longest = max(pieces, key=len)
        if longest:
            outs["c"].write(f">r{reads}\n{longest}\n")
            res["c"] += len(longest)
            seqs["c"] += 1
        t1 = time.time()
        best, best_hits = None, 0
        for p in pieces:
            if len(p) < K:
                continue
            codes = kmer_codes(p.encode())
            pos = np.searchsorted(ref, codes)
            pos[pos == len(ref)] = 0
            hits = int((ref[pos] == codes).sum())
            if hits > best_hits:
                best, best_hits = p, hits
        t_look += time.time() - t1
        if best is not None:
            outs["d"].write(f">r{reads}\n{best}\n")
            res["d"] += len(best)
            seqs["d"] += 1
    for h in outs.values():
        h.close()
    summary = {
        "reads": reads, "bases": bases, "swissprot_proteins": n_prot,
        "swissprot_unique_10mers": int(len(ref)), "swissprot_build_s": round(t_ref, 1),
        "python_translate_s": round(t_trans, 1), "python_lookup_s": round(t_look, 1),
        "residues": res, "sequences": seqs,
        "residues_per_base": {x: res[x] / bases for x in "abcd"},
        "reads_kept_d": seqs["d"] / reads,
    }
    json.dump(summary, open(f"{out}/frames_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
