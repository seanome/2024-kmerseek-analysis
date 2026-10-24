#!/usr/bin/env python3
"""Notebook 346: score every query against every protein of its database, and decoys.

For each database in tables/346_databases.tsv, every protein is scored three times against
each query that searches that database: as it is, after a uniform shuffle of its residues
(notebook 232's null), and after a shuffle that keeps every dipeptide count (the shuffle in
nextflow-runs/scope40-lowcomplexity-null/bin/shuffle_fasta.py, which keeps runs of H and P
residues). The four scores of linked_pairs_utils.diagonal_scan (B2, the chosen linked pair,
its randstrobe, B3), their key counts and the shared 6- and 8-letter word counts are
written per (query, protein, kind) to ~/data/linked-pairs/346_scan_<database>.parquet.

Swiss-Prot is searched by the Pfam pairs' queries (domain sequences from Pfam seed
alignments, whose proteins are mostly not in Swiss-Prot). Each pair's partner domain is
added to the database as one more protein named "partner|<pair>|<name>"; the notebook
removes, for each query, the other pairs' partner domains and any Swiss-Prot entry of the
query's or partner's own protein.

Every shuffle has a fixed seed per protein (SEED_UNIFORM or SEED_DIPEPTIDE, and the
protein's index in the file), so a rerun gives the same decoys.

Usage (project interpreter, repository root):

    python scripts/346_database_scan.py --workers 12            # all databases
    python scripts/346_database_scan.py --sample 300 --workers 12  # time a sample, write nothing
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "nextflow-runs/scope40-lowcomplexity-null/bin"))
import linked_pairs_utils as lp  # noqa: E402
import shuffle_fasta  # noqa: E402

OUT = Path(os.environ.get("NB346_DIR", "/Users/olga/data/linked-pairs"))
SEED_UNIFORM = 343_001  # fixed when this notebook was numbered 343; it fixes the decoys
SEED_DIPEPTIDE = 343_002
KINDS = ["real", "uniform", "dipeptide"]
COLS = ["B2", "B2_count", "linked", "linked_count", "randstrobe", "randstrobe_count",
        "B3", "B3_count", "words6", "words8"]  # fmt: skip
SLUG = {
    "human GENCODE v49 canonical": "human_gencode_v49",
    "C. elegans UniProt UP000001940": "celegans_uniprot_UP000001940",
    "B. burgdorferi B31 UniProt UP000001807": "borrelia_uniprot_UP000001807",
    "Swiss-Prot": "swissprot",
}


def read_fasta(path) -> tuple[list[str], list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    names, seqs, buf = [], [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if names:
                    seqs.append("".join(buf))
                names.append(line[1:].strip())
                buf = []
            else:
                buf.append(line.strip())
    if names:
        seqs.append("".join(buf))
    return names, seqs


def decoys(seq: str, index: int) -> tuple[str, str, bool]:
    """The uniform and the dipeptide-preserving shuffle of one protein, and whether the
    dipeptide shuffle came back unchanged: either shuffle_fasta gave up after 1,000 tries of
    its Euler-path sampling, or no other order with the same dipeptide counts exists (short
    proteins). The output column is named shuffle_failed; such decoys are left out."""
    rng = np.random.default_rng([SEED_UNIFORM, index])
    arr = np.frombuffer(seq.encode(), np.uint8)
    uniform = arr[rng.permutation(arr.size)].tobytes().decode()
    di = shuffle_fasta.shuffle_dipeptide(
        seq, random.Random(f"{SEED_DIPEPTIDE}:{index}")
    )
    return uniform, di, (di == seq and len(seq) >= 3)


_QUERIES: list[tuple[np.ndarray, np.ndarray]] = []


def _init(query_seqs: list[str]) -> None:
    global _QUERIES
    _QUERIES = []
    for s in query_seqs:
        q = lp.hp_classes(s)
        _QUERIES.append((q, lp.randstrobe_partner(q, lp.DB_ANCHOR, lp.DB_WINDOW)))


def _work(task):
    """Score one chunk of proteins: returns (rows array, failed-shuffle flags)."""
    items = task
    out = []
    for index, seq in items:
        uniform, di, failed = decoys(seq, index)
        for k, s in enumerate([seq, uniform, di]):
            t = lp.hp_classes(s)
            tp = lp.randstrobe_partner(t, lp.DB_ANCHOR, lp.DB_WINDOW)
            for qi, (q, qp) in enumerate(_QUERIES):
                r = lp.diagonal_scan(q, t, qp, tp)
                out.append(
                    [qi, index, k, len(s), int(failed and k == 2)]
                    + [r[c] for c in COLS]
                )
    return np.array(out, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="time this many proteins per database, write nothing",
    )
    ap.add_argument(
        "--only",
        default="",
        help="run only this database (its name in 346_databases.tsv)",
    )
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument(
        "--db-file", default="", help="read the --only database from this file instead"
    )
    ap.add_argument(
        "--part",
        type=int,
        default=-1,
        help="score only proteins with index %% nparts == part",
    )
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument(
        "--merge",
        action="store_true",
        help="join the --only database's part files into one",
    )
    args = ap.parse_args()
    if args.merge:
        slug = SLUG[args.only]
        parts = sorted(OUT.glob(f"346_scan_{slug}.part*.parquet"))
        df = pl.concat([pl.read_parquet(f) for f in parts]).sort(
            "query", "target_index", "kind"
        )
        df.write_parquet(OUT / f"346_scan_{slug}.parquet")
        print(
            f"merged {len(parts)} parts, {df.height:_} rows -> {OUT / f'346_scan_{slug}.parquet'}"
        )
        return

    known = pl.read_csv(ROOT / "tables/346_known_pairs.tsv", separator="\t")
    dbs = pl.read_csv(ROOT / "tables/346_databases.tsv", separator="\t")
    seq_names, seq_list = read_fasta(ROOT / "tables/346_sequences.fasta")
    seqs = dict(zip(seq_names, seq_list))
    OUT.mkdir(parents=True, exist_ok=True)

    for db in dbs.iter_rows(named=True):
        if args.only and db["database"] != args.only:
            continue
        searches = known.filter(pl.col("query_db") == db["database"])
        queries = searches["query"].to_list()
        names, dseqs = read_fasta(args.db_file or db["file"])
        if len(names) != db["n_proteins"]:
            raise ValueError(
                f"{len(names)} proteins in the file, {db['n_proteins']} in 346_databases.tsv"
            )
        if db["database"] == "Swiss-Prot":
            for r in searches.iter_rows(named=True):
                names.append(f"partner|{r['pair']}|{r['partner']}")
                dseqs.append(seqs[r["partner"]])
        n = len(names)
        idx = np.arange(n)
        if args.part >= 0:
            idx = idx[idx % args.nparts == args.part]
        if args.sample:
            idx = np.random.default_rng(0).choice(n, min(args.sample, n), replace=False)
        tasks = [
            [(int(i), dseqs[i]) for i in idx[c : c + args.chunk]]
            for c in range(0, idx.size, args.chunk)
        ]
        n_res = sum(len(dseqs[i]) for i in idx)
        print(
            f"{db['database']}: {idx.size:_} proteins ({n_res:_} residues) x 3 kinds x "
            f"{len(queries)} queries ({', '.join(queries)}), {len(tasks)} chunks",
            flush=True,
        )
        t0 = time.time()
        parts = []
        with Pool(
            args.workers, initializer=_init, initargs=([seqs[q] for q in queries],)
        ) as pool:
            done_res = 0
            for k, arr in enumerate(pool.imap_unordered(_work, tasks)):
                parts.append(arr)
                done_res += int(arr[(arr[:, 0] == 0) & (arr[:, 2] == 0), 3].sum())
                if k % 50 == 0 or k == len(tasks) - 1:
                    el = time.time() - t0
                    eta = el / max(done_res, 1) * (n_res - done_res)
                    print(
                        f"  {k + 1}/{len(tasks)} chunks, {el / 60:.1f} min, about {eta / 60:.0f} min left",
                        flush=True,
                    )
        el = time.time() - t0
        print(
            f"  {db['database']}: {el / 60:.1f} min wall, "
            f"{el * args.workers / max(n_res, 1) * 1e6:.1f} us of worker time per protein residue",
            flush=True,
        )
        if args.sample:
            full = (
                el
                * dbs.filter(pl.col("database") == db["database"])["n_residues"][0]
                / n_res
            )
            print(
                f"  projected for the whole database: {full / 60:.0f} min", flush=True
            )
            continue
        a = np.concatenate(parts)
        df = pl.DataFrame(
            a,
            schema=["query_i", "target_index", "kind_i", "length", "shuffle_failed"]
            + COLS,
            orient="row",
        )
        df = (
            df.with_columns(
                pl.col("query_i")
                .replace_strict(dict(enumerate(queries)), return_dtype=pl.String)
                .alias("query"),
                pl.col("kind_i")
                .replace_strict(dict(enumerate(KINDS)), return_dtype=pl.String)
                .alias("kind"),
                pl.col("target_index")
                .replace_strict(dict(enumerate(names)), return_dtype=pl.String)
                .alias("target"),
            )
            .drop("query_i", "kind_i")
            .sort("query", "target_index", "kind")
        )
        part = f".part{args.part:02d}" if args.part >= 0 else ""
        path = OUT / f"346_scan_{SLUG[db['database']]}{part}.parquet"
        df.write_parquet(path)
        print(f"  wrote {path} ({df.height:_} rows)", flush=True)


if __name__ == "__main__":
    main()
