"""Does a sequence aligner find each nb 232 pair? phmmer pairwise (-Z 20000, a proteome-sized
database) and MMseqs2 -s 7.5 all-vs-all over the pair set's own sequences.

Writes one row per pair with the best E-value from each tool (null = no hit reported).

Usage:
    python scripts/aligner_pair_check.py --dir /Users/olga/data/pfam/233_aligner_check --procs 14
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import polars as pl

PHMMER = "/Users/olga/anaconda3/envs/hmmer/bin/phmmer"
PIXI = "/Users/olga/.pixi/bin/pixi"
REPO = Path(__file__).resolve().parents[1]
Z = 20_000

_SEQS: dict[str, str] = {}


def init(fa: str):
    name = None
    with open(fa) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
                _SEQS[name] = ""
            else:
                _SEQS[name] += line.strip()


def phmmer_pair(job):
    q, t = job
    with tempfile.TemporaryDirectory() as d:
        qf, tf, out = Path(d) / "q.fa", Path(d) / "t.fa", Path(d) / "tbl"
        qf.write_text(f">{q}\n{_SEQS[q]}\n")
        tf.write_text(f">{t}\n{_SEQS[t]}\n")
        subprocess.run([PHMMER, "--noali", "-Z", str(Z), "--domZ", str(Z), "--tblout", str(out), str(qf), str(tf)],
                       capture_output=True, check=False)
        ev = None
        for line in out.read_text().splitlines():
            if line.startswith("#"):
                continue
            ev = float(line.split()[4])
            break
    return {"query": q, "target": t, "phmmer_evalue": ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()
    d = Path(args.dir)
    pairs = pl.read_csv(d / "pairs.tsv", separator="\t")
    fa = str(d / "segments.fa")

    # MMseqs2 all-vs-all, then look the pairs up.
    mm_out = d / "mmseqs_hits.tsv"
    if not mm_out.exists():
        subprocess.run([PIXI, "run", "-e", "mmseqs2", "mmseqs", "easy-search", fa, fa, str(mm_out), str(d / "mmtmp"),
                        "-s", "7.5", "-e", "10", "--max-seqs", "4000", "--threads", str(args.procs),
                        "--format-output", "query,target,evalue,bits,pident"], cwd=REPO, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mm = (pl.read_csv(mm_out, separator="\t", has_header=False, new_columns=["query", "target", "mmseqs_evalue", "mmseqs_bits", "mmseqs_pident"])
          .group_by("query", "target").agg(pl.col("mmseqs_evalue").min(), pl.col("mmseqs_bits").max(), pl.col("mmseqs_pident").max()))
    out = pairs.join(mm, on=["query", "target"], how="left")
    print("mmseqs done:", out["mmseqs_evalue"].is_not_null().sum(), "of", out.height, "pairs have a hit at E<=10", flush=True)

    jobs = list(zip(pairs["query"], pairs["target"]))
    rows = []
    with Pool(args.procs, initializer=init, initargs=(fa,)) as pool:
        for i, r in enumerate(pool.imap_unordered(phmmer_pair, jobs, chunksize=50)):
            rows.append(r)
            if i % 10_000 == 0:
                print(f"  phmmer {i}", flush=True)
    ph = pl.DataFrame(rows)
    out = out.join(ph, on=["query", "target"], how="left")
    out.write_parquet(d / "233_aligner_pair_evalues.parquet")
    print(out.select(pl.col("phmmer_evalue").is_not_null().mean(), (pl.col("phmmer_evalue") <= 0.01).mean(), (pl.col("mmseqs_evalue") <= 0.01).mean()))


if __name__ == "__main__":
    main()
