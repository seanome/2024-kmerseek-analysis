#!/usr/bin/env python3
"""Local alignment of query-target protein pairs that sourmash proposed (no database search).

Usage: pair_align.py <pairs.tsv> <queries.faa> <targets.faa[.gz]> <out.parquet> [n_procs]

pairs.tsv needs columns `query` and `target` (FASTA IDs, first word of the header). Each pair is
aligned with parasail Smith-Waterman (BLOSUM62, gap open 11, extend 1). Bit score and E-value use
BLAST's published BLOSUM62 11/1 gapped constants (lambda 0.267, K 0.041), the query length and
the total residues of the target FASTA: E = m * N * 2^-bits. The rule from the analysis plan:
a pair is supported when E <= 1e-5 and the alignment covers >= 50 % of the query.
"""
import gzip
import math
import sys
from multiprocessing import get_context

import parasail
import polars as pl

LAMBDA, K = 0.267, 0.041
OPEN, EXTEND = 11, 1


def read_fasta(path, keep=None):
    seqs, name, buf = {}, None, []
    op = gzip.open if str(path).endswith(".gz") else open
    total = 0
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    s = "".join(buf)
                    total += len(s)
                    if keep is None or name in keep:
                        seqs[name] = s
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        s = "".join(buf)
        total += len(s)
        if keep is None or name in keep:
            seqs[name] = s
    return seqs, total


_Q = _T = None
_N = 0


def _init(q, t, n):
    global _Q, _T, _N
    _Q, _T, _N = q, t, n


MAX_TRACE_CELLS = 2e8  # longer pairs get a score-only alignment (traceback memory)


def align(pair):
    qn, tn = pair
    q, t = _Q[qn], _T[tn]
    row = dict(query=qn, target=tn, query_len=len(q), target_len=len(t))
    if len(q) * len(t) > MAX_TRACE_CELLS:
        r = parasail.sw_striped_sat(q, t, OPEN, EXTEND, parasail.blosum62)
        bits = (LAMBDA * r.score - math.log(K)) / math.log(2)
        row.update(raw_score=r.score, bit_score=round(bits, 2), evalue=len(q) * _N * 2.0 ** (-bits),
                   traceback=False)
        row["supported"] = None  # coverage unknown without a traceback
        return row
    r = parasail.sw_trace_striped_sat(q, t, OPEN, EXTEND, parasail.blosum62)
    tb = r.traceback
    ident = sum(a == b and a != "-" for a, b in zip(tb.query, tb.ref))
    aln_len = len(tb.query)
    q_aln = len(tb.query.replace("-", ""))
    t_aln = len(tb.ref.replace("-", ""))
    bits = (LAMBDA * r.score - math.log(K)) / math.log(2)
    evalue = len(q) * _N * 2.0 ** (-bits)
    # parasail end positions are 0-based and inclusive; report 1-based inclusive coordinates
    row.update(raw_score=r.score, bit_score=round(bits, 2), evalue=evalue, traceback=True,
               aln_len=aln_len, identities=ident,
               pct_identity=round(100 * ident / aln_len, 2) if aln_len else 0.0,
               query_coverage=round(q_aln / len(q), 4), target_coverage=round(t_aln / len(t), 4),
               query_start=r.end_query - q_aln + 2, query_end=r.end_query + 1,
               target_start=r.end_ref - t_aln + 2, target_end=r.end_ref + 1,
               aligned_query=tb.query, aligned_target=tb.ref,
               supported=bool(evalue <= 1e-5 and q_aln / len(q) >= 0.5))
    return row


def main():
    pairs_p, qfa, tfa, out = sys.argv[1:5]
    n_procs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    pairs = pl.read_csv(pairs_p, separator="\t", infer_schema_length=0).select("query", "target").unique()
    if pairs.height == 0:
        pl.DataFrame(schema={"query": pl.Utf8, "target": pl.Utf8}).write_parquet(out)
        print("pairs 0")
        return
    q, _ = read_fasta(qfa, set(pairs["query"]))
    t, n_total = read_fasta(tfa, set(pairs["target"]))
    missing = set(pairs["query"]) - set(q) | set(pairs["target"]) - set(t)
    if missing:
        raise SystemExit(f"{len(missing)} IDs not found in FASTA, e.g. {sorted(missing)[:5]}")
    todo = list(zip(pairs["query"], pairs["target"]))
    with get_context("spawn").Pool(n_procs, initializer=_init, initargs=(q, t, n_total)) as pool:
        rows = pool.map(align, todo, chunksize=256)
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(n_total).alias("db_residues"))
    df.write_parquet(out)
    print(f"pairs {df.height}, supported {df['supported'].sum()}, db residues {n_total}")


if __name__ == "__main__":
    main()
