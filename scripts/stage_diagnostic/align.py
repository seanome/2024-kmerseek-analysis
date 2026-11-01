"""Stages A and B: seedless local alignment of every query against every reference.

Writes tables/350_stage_diagnostic/v1/<dataset>/align.parquet, one row per query, reference
and scorer ("full" = A, an alphabet name = B with that alphabet), with the score, and the
alignment coordinates for the pairs the evaluation reports (positives, and each query's
top 20 under that scorer). Per query and scorer: wall time and the worker's peak memory go
to align_cost.tsv.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from panels import read_fasta
from scoring import Scorer

TOP_N_COORDS = 20
_STATE = {}


def _init(refs: dict, scorers: list[str]):
    _STATE["refs"] = refs
    _STATE["scorers"] = {k: Scorer(k) for k in scorers}
    _STATE["prepped"] = {
        k: {rid: s.prep(seq) for rid, seq in refs.items()} for k, s in _STATE["scorers"].items()
    }


def _one(args):
    qid, qseq, positives = args
    rows, cost = [], []
    for kind, sc in _STATE["scorers"].items():
        t0 = time.perf_counter()
        q = sc.prep(qseq)
        refs = _STATE["prepped"][kind]
        scores = {rid: sc.score(q, t) for rid, t in refs.items()}
        top = set(sorted(scores, key=lambda r: (-scores[r], r))[:TOP_N_COORDS])
        for rid, s in scores.items():
            co = sc.coords(q, refs[rid]) if (rid in top or rid in positives) and s > 0 else None
            rows.append(
                dict(
                    query=qid, reference=rid, scorer=kind, score=s,
                    q_start=co[0] if co else None, q_end=co[1] if co else None,
                    t_start=co[2] if co else None, t_end=co[3] if co else None,
                )
            )  # fmt: skip
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1 << 20) if sys.platform == "darwin" else peak / 1024
        cost.append(
            dict(query=qid, scorer=kind, seconds=time.perf_counter() - t0, worker_peak_mb=peak_mb,
                 n_references=len(refs))
        )  # fmt: skip
    return rows, cost


def run(data_dir: Path, out_dir: Path, scorers: list[str], positives: dict, workers: int):
    queries = read_fasta(data_dir / "queries.fasta")
    refs = read_fasta(data_dir / "references.fasta")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(q, s, positives.get(q, set())) for q, s in queries.items()]
    rows, cost = [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(refs, scorers)) as ex:
        for r, c in ex.map(_one, jobs, chunksize=1):
            rows += r
            cost += c
    wall = time.perf_counter() - t0
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out_dir / "align.parquet")
    pl.DataFrame(cost).with_columns(
        pl.lit(wall).alias("total_wall_seconds"), pl.lit(workers).alias("workers")
    ).write_csv(out_dir / "align_cost.tsv", separator="\t")
    return wall


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) - 2))
