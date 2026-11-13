#!/usr/bin/env python3
"""Changed k-mers to protein queries, for every labelling at once.

K-mer path steps 3 and 4 of notebooks/360_data/preregistered.md:

  3. k-mers with padj < 0.05 and the same sign of log2 fold change are joined where they
     overlap by k - 1 nt into maximal unbranched runs. Only k-mers that were each called
     changed are joined; nothing else is assembled.
  4. each run is translated in 6 frames and cut at stop codons; pieces of at least 5 amino
     acids are the queries.

    de_kmer_contigs.py out_prefix <labelling>.deseq2.tsv.gz ...

Writes
  <out_prefix>.queries.fasta    one record per distinct peptide, named p0000001 ...
  <out_prefix>.peptide_map.tsv  peptide_id, labelling, run_id, sign, strand, frame, run_length_nt
  <out_prefix>.runs.tsv         labelling, run_id, sign, length_nt, n_kmers, sequence
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

PADJ = 0.05
MIN_PEPTIDE = 5
COMP = str.maketrans("ACGTN", "TGCAN")
CODON = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        ((x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
}


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def unitigs(kmers: set[str]) -> list[tuple[str, int]]:
    """Maximal unbranched runs over canonical k-mers, each run reported on one strand.

    Both strands of every k-mer are graph nodes; a run grows from x to y = x[1:] + b only
    when y is x's only successor and x is y's only predecessor. Marking both strands of a
    used k-mer stops the same run coming out again read backwards. k is odd (31), so no
    k-mer is its own reverse complement.
    """
    nodes = kmers | {rc(x) for x in kmers}

    def succ(x: str) -> list[str]:
        return [y for y in (x[1:] + b for b in "ACGT") if y in nodes]

    def pred(x: str) -> list[str]:
        return [y for y in (b + x[:-1] for b in "ACGT") if y in nodes]

    def starts_run(x: str) -> bool:
        p = pred(x)
        return not (len(p) == 1 and len(succ(p[0])) == 1)

    used: set[str] = set()
    runs: list[tuple[str, int]] = []

    def walk(x: str) -> None:
        path = [x]
        used.update((x, rc(x)))
        cur = x
        while True:
            s = succ(cur)
            if len(s) != 1:
                break
            nxt = s[0]
            if nxt in used or len(pred(nxt)) != 1:
                break
            path.append(nxt)
            used.update((nxt, rc(nxt)))
            cur = nxt
        seq = path[0] + "".join(p[-1] for p in path[1:])
        runs.append((seq, len(path)))

    for x in sorted(nodes):
        if x not in used and starts_run(x):
            walk(x)
    for x in sorted(nodes):  # what is left lies on cycles
        if x not in used:
            walk(x)
    return runs


def translate(nt: str) -> str:
    return "".join(CODON.get(nt[i : i + 3], "X") for i in range(0, len(nt) - 2, 3))


def main(prefix: str, files: list[str]) -> None:
    pep_id: dict[str, str] = {}
    map_rows, run_rows = [], []
    for f in sorted(files):
        lab = Path(f).name.split(".")[-4]  # <prefix>.<labelling>.deseq2.tsv.gz
        d = pl.read_csv(f, separator="\t", null_values=["NA"], schema_overrides={"padj": pl.Float64, "log2FoldChange": pl.Float64})
        sig = d.filter(pl.col("padj") < PADJ)
        for sign, sub in (("up", sig.filter(pl.col("log2FoldChange") > 0)), ("down", sig.filter(pl.col("log2FoldChange") < 0))):
            for i, (seq, n) in enumerate(unitigs(set(sub["feature"].to_list()))):
                run_id = f"{lab}.{sign}.{i:06d}"
                run_rows.append((lab, run_id, sign, len(seq), n, seq))
                for strand, s in (("+", seq), ("-", rc(seq))):
                    for frame in range(3):
                        for piece in translate(s[frame:]).split("*"):
                            if len(piece) < MIN_PEPTIDE:
                                continue
                            pid = pep_id.setdefault(piece, f"p{len(pep_id) + 1:07d}")
                            map_rows.append((pid, lab, run_id, sign, strand, frame, len(seq)))
        print(f"{lab}: {sig.height} changed k-mers", file=sys.stderr)
    with open(f"{prefix}.queries.fasta", "w") as fh:
        for piece, pid in pep_id.items():
            fh.write(f">{pid}\n{piece}\n")
    pl.DataFrame(map_rows, schema=["peptide_id", "labelling", "run_id", "sign", "strand", "frame", "run_length_nt"], orient="row").write_csv(
        f"{prefix}.peptide_map.tsv", separator="\t"
    )
    pl.DataFrame(run_rows, schema=["labelling", "run_id", "sign", "length_nt", "n_kmers", "sequence"], orient="row").write_csv(
        f"{prefix}.runs.tsv", separator="\t"
    )
    print(f"{len(pep_id)} distinct peptides from {len(run_rows)} runs", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
