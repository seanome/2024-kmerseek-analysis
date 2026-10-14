"""Align exactly the listed protein pairs with MMseqs2, one and two alignments per pair.

Usage:
    python3 -I run_mmseqs_pairs.py PAIRS_PARQUET FASTA OUT_PARQUET WORK_DIR
        [--shuffles 10] [--seed 293] [--mmseqs PATH] [--threads 8]

Each pair (and each of its shuffled copies, made exactly as in run_offset_test.py) gets
its own query and target entry, and a prefilter result listing only that pair is written
with `mmseqs tsv2db`, so `mmseqs align` aligns only the listed pairs; no prefilter step can
drop a pair. Alignment settings are MMseqs2 defaults (E-value 1e-3) apart from:
    C1: -a  (one alignment per pair, the best)
    C2: -a --alt-ali 2  (up to three alignments per pair: the best plus two alternatives;
        the two best are used)
Output: one row per alignment, with method (C1 or C2), copy, alignment rank within the
pair, and 1-based inclusive coordinates as MMseqs2 reports them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import kmer_offset_utils as ku  # noqa: E402

COLUMNS = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,qaln,taln"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pairs")
    p.add_argument("fasta")
    p.add_argument("out")
    p.add_argument("work")
    p.add_argument("--shuffles", type=int, default=10)
    p.add_argument("--seed", type=int, default=293)
    p.add_argument(
        "--mmseqs", default="/Users/olga/anaconda3/envs/orthofinder/bin/mmseqs"
    )
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(args.pairs).select("pair_set", "accession_a", "accession_b")
    seqs = ku.read_fasta(Path(args.fasta))

    meta = []
    with open(work / "q.fasta", "w") as fq, open(work / "t.fasta", "w") as ft:
        for idx, pair in enumerate(pairs.iter_rows(named=True)):
            seq_a, seq_b = seqs[pair["accession_a"]], seqs[pair["accession_b"]]
            copies = [(-1, seq_b)] + [
                (c, ku.shuffled(seq_b, args.seed * 1_000_003 + idx * 101 + c))
                for c in range(args.shuffles)
            ]
            for copy, b in copies:
                pid = f"p{idx}_c{copy}"
                fq.write(f">{pid}\n{seq_a}\n")
                ft.write(f">{pid}\n{b}\n")
                meta.append({"pair_id": pid, **pair, "copy": copy})
    mm = args.mmseqs
    run([mm, "createdb", str(work / "q.fasta"), str(work / "qdb")])
    run([mm, "createdb", str(work / "t.fasta"), str(work / "tdb")])
    # Same header in q and t, so the internal keys line up: key i in qdb pairs with key i in tdb.
    qkeys = dict(
        line.split("\t")[1::-1]
        for line in (work / "qdb.lookup").read_text().splitlines()
    )
    tkeys = dict(
        line.split("\t")[1::-1]
        for line in (work / "tdb.lookup").read_text().splitlines()
    )
    with open(work / "pairs.tsv", "w") as fh:
        for m in meta:
            fh.write(f"{qkeys[m['pair_id']]}\t{tkeys[m['pair_id']]}\t0\n")
    run(
        [
            mm,
            "tsv2db",
            str(work / "pairs.tsv"),
            str(work / "pref"),
            "--output-dbtype",
            "7",
        ]
    )
    frames = []
    for method, extra in [("C1", []), ("C2", ["--alt-ali", "2"])]:
        aln = work / f"aln_{method}"
        for suffix in ["", ".index", ".dbtype"]:
            Path(f"{aln}{suffix}").unlink(missing_ok=True)
        run(
            [
                mm,
                "align",
                str(work / "qdb"),
                str(work / "tdb"),
                str(work / "pref"),
                str(aln),
                "-a",
                "--threads",
                str(args.threads),
                *extra,
            ]
        )
        out_tsv = work / f"aln_{method}.tsv"
        run(
            [
                mm,
                "convertalis",
                str(work / "qdb"),
                str(work / "tdb"),
                str(aln),
                str(out_tsv),
                "--format-output",
                COLUMNS,
            ]
        )
        df = pl.read_csv(
            out_tsv, separator="\t", has_header=False, new_columns=COLUMNS.split(",")
        )
        df = df.filter(pl.col("query") == pl.col("target")).with_columns(
            pl.lit(method).alias("method"),
            pl.int_range(pl.len()).over("query").alias("alignment_rank"),
        )
        frames.append(df)
    res = pl.concat(frames).rename({"query": "pair_id"}).drop("target")
    res = pl.DataFrame(meta).join(res, on="pair_id", how="left")
    res.write_parquet(args.out)
    version = subprocess.run(
        [mm, "version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"mmseqs={version} pairs={pairs.height} rows={res.height} -> {args.out}")


if __name__ == "__main__":
    main()
