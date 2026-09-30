#!/usr/bin/env python3
"""Shuffled-BHF comparison for notebook 256: does BHF find anything in the human
proteome that a scrambled BHF does not?

BHF is searched against the human canonical proteome together with 300 shuffled copies
of itself, on all 152 alphabet-ksize pairs of the notebook 241 sweep (same
indexes, same search flags as 241 and 242). Each copy keeps BHF's length, its amino-acid
counts and its counts of every adjacent pair of residues (a dipeptide shuffle,
Altschul-Erickson 1985), so it has BHF's composition and its short repeats, and no
real homology.

For each query, alphabet-ksize pair and ranking metric the run keeps: the best value over the human
proteins it hit, the protein that value belongs to and how many proteins tie with it,
the number of proteins hit, the ten best proteins, and the rank of each human protein
named as a BHF hit in the 2024 Evolgenome talk. A human protein's value is that of its
best region, as in 241. A query that hit no human protein on an alphabet-ksize pair has no row for that
alphabet-ksize pair: count it as "found nothing", not as missing data.

Queries are searched 25 at a time on alphabet-ksize pairs at or below 20.5 bits of seed information and
all at once on the others (chunk_tags in alphabet_ensemble_utils.py, shared with 242).
Each chunk's CSV is reduced and deleted, so the run needs little disk and resumes chunk
by chunk.

Output: /Users/olga/data/botryllus/alphabet-ranking-three-cases/null_bhf_dipeptide/
  queries.fa           BHF first, then shuf000..shuf299
  arms.json            the alphabet-ksize pairs searched
  top/<tag>.parquet    one row per (query, alphabet-ksize pair, metric)

Usage: 256_bhf_dipeptide_shuffle_null.py [--workers 3] [--n 300] [--seed 0]
                                         [--alphabets a,b] [--out DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from alphabet_ensemble_utils import chunk_tags  # noqa: E402

_spec = importlib.util.spec_from_file_location("null242", HERE / "242_null_queries.py")
null242 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(null242)

KMERSEEK = null242.KMERSEEK
D = null242.D
OUT = D / "null_bhf_dipeptide"
CASE = "BHF"

# Human proteins named as BHF hits in the 2024 Evolgenome talk (k = 24, orpheum +
# sourmash, hp alphabet). TSHZ1-3 are paralogs; SFI1 and CETN2 were one hit.
TALK_GENES = ["ZNF292", "RSF1", "TSHZ1", "TSHZ2", "TSHZ3", "RNMT", "SFI1", "CETN2",
              "TRAPPC10", "NDNF"]

# The ranking metrics of 241_alphabet_ranking_collect.py: (column, lower is better).
METRICS = {
    "E-value": ("region_evalue", True),
    "bit score": ("region_ka_bits", False),
    "mean IDF": ("region_mean_idf", False),
    "tf-idf": ("region_tfidf", False),
    "enrichment": ("region_enrichment", False),
    "Poisson score": ("region_poisson_score", False),
    "Poisson p-value": ("region_tail_probability", True),
    "shared k-mers": ("region_n_shared_kmers", False),
    "containment": ("containment", False),
    "protein enrichment": ("query_enrichment", False),
    "protein Poisson p-value": ("query_poisson_pvalue", True),
}


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985, Mol Biol Evol 2:526).

    The pairs are edges of a graph on residues; `seq` is one walk that uses every edge
    once. Pick, for every residue except the last one, which of its outgoing edges is
    used last, such that those edges lead to the last residue from everywhere (else
    the walk strands). Shuffle the other edges, put the chosen one at the end, walk."""
    if len(seq) < 3:
        return seq
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out[a].append(b)
    last = seq[-1]
    while True:
        final = {v: rng.choice(ends) for v, ends in out.items() if v != last}
        ok = True
        for v in final:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = final[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, ends in out.items():
        rest = list(ends)
        if v in final:
            rest.remove(final[v])
        rng.shuffle(rest)
        order[v] = rest + ([final[v]] if v in final else [])
    walk, u = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        u = order[u].pop(0)
        walk.append(u)
    return "".join(walk)


def pair_counts(s: str) -> Counter:
    return Counter(zip(s, s[1:]))


def make_queries(n: int, seed: int) -> list[tuple[str, str]]:
    bhf = null242.read_fasta(D / "queries.fa")[CASE]
    rng = random.Random(seed)
    qs = [(CASE, bhf)]
    for i in range(n):
        s = dipeptide_shuffle(bhf, rng)
        assert pair_counts(s) == pair_counts(bhf) and s[0] == bhf[0] and s[-1] == bhf[-1]
        qs.append((f"shuf{i:03d}", s))
    return qs


def reduce_chunk(csv: Path, arm: dict) -> pl.DataFrame:
    """One row per (query, metric) for the queries in this chunk that hit anything."""
    cols = ["query_name", "target_name", "region_ka_lambda"] + [c for c, _ in METRICS.values()]
    if csv.stat().st_size == 0:
        return pl.DataFrame()
    df = (pl.read_csv(csv, columns=cols, infer_schema_length=0)
          .with_columns([pl.col(c).cast(pl.Float64, strict=False)
                         for c in cols if c not in ("query_name", "target_name")])
          .with_columns(pl.col("query_name").str.strip_chars(),
                        pl.col("target_name").str.split("|").list.get(6).alias("gene")))
    # No lambda means no score scale: the E-value is inf and the bit score 0 for every
    # region of the alphabet-ksize pair. Ranking on either would tie the whole proteome (see 241).
    df = df.with_columns(pl.when(pl.col("region_ka_lambda") > 0).then(pl.col("region_ka_bits"))
                         .alias("region_ka_bits"))
    rows = []
    for label, (col, lower) in METRICS.items():
        best = (pl.col(col).min() if lower else pl.col(col).max()).alias("v")
        per = (df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
                 .group_by("query_name", "gene").agg(best)
                 .with_columns(pl.col("v").rank("min", descending=not lower)
                               .over("query_name").alias("rank")))
        if per.height == 0:
            continue
        # Order within a query is fixed by value then gene name, so the ten best are the
        # same on every run (ties would otherwise fall in polars' arbitrary row order).
        per = per.sort("query_name", "v", "gene", descending=[False, not lower, False])
        top = per.group_by("query_name", maintain_order=True).agg(
            pl.col("v").first().alias("top_value"),
            pl.col("gene").first().alias("top_gene"),
            (pl.col("rank") == 1).sum().alias("n_tied_top"),
            pl.len().alias("n_hit"),
            pl.col("gene").head(10).alias("top10"),
        )
        talk = (per.filter(pl.col("gene").is_in(TALK_GENES))
                   .group_by("query_name").agg(pl.col("rank").min().alias("talk_best_rank")))
        rows.append(top.join(talk, on="query_name", how="left")
                       .with_columns(pl.lit(label).alias("metric")))
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows).with_columns(
        pl.lit(arm["alphabet"]).alias("alphabet"), pl.lit(arm["k"]).alias("k"),
        pl.lit(arm["bits"]).alias("bits"), pl.lit(arm["nofit"]).alias("nofit"))


def one_chunk(tag: str, chunk: list[tuple[str, str]], arm: dict) -> str:
    done = OUT / "top" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    fa, csv = OUT / "tmp" / f"{tag}.fa", OUT / "tmp" / f"{tag}.csv"
    fa.write_text("".join(f">{h}\n{s}\n" for h, s in chunk))
    idx = D / "idx" / f"human.{arm['alphabet']}.k{arm['k']}.rocksdb"
    pen = (["--extend-mismatch-penalty", "0"] if arm["nofit"] else
           ["--extend-mismatch-penalty", str(arm["penalty"]), "--extend-xdrop", str(arm["xdrop"])])
    cmd = [str(KMERSEEK), "search", "-q", str(fa), "-t", str(idx), "-k", str(arm["k"]),
           "-a", arm["alphabet"], "--threshold", "0", "--min-shared-kmers", "1",
           "--max-query-pvalue", "1", "--min-region-score", "0", *pen, "-o", str(csv)]
    with open(OUT / "logs" / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not csv.exists():
        return f"{tag} FAILED rc={rc} (log: {OUT / 'logs' / (tag + '.log')})"
    reduce_chunk(csv, arm).write_parquet(done)
    csv.unlink(missing_ok=True)
    fa.unlink(missing_ok=True)
    return f"{tag} ok"


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (a smoke test writes elsewhere)")
    ap.add_argument("--alphabets", default="", help="comma-separated subset, for a smoke test")
    args = ap.parse_args()
    OUT = args.out
    for d in ("top", "tmp", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    # Cached chunks hold the queries they were searched with; a different --n or --seed
    # would silently mix two query sets, so refuse it.
    manifest = OUT / "run.json"
    want = {"n": args.n, "seed": args.seed}
    if manifest.exists() and json.loads(manifest.read_text()) != want:
        sys.exit(f"{OUT} holds a run with {manifest.read_text().strip()}; use another --out")
    if not manifest.exists() and any((OUT / "top").glob("*.parquet")):
        sys.exit(f"{OUT}/top holds chunks from an unknown query set; use another --out")
    manifest.write_text(json.dumps(want))
    qs = make_queries(args.n, args.seed)
    (OUT / "queries.fa").write_text("".join(f">{h}\n{s}\n" for h, s in qs))
    A = [a for a in null242.arms("all")
         if not args.alphabets or a["alphabet"] in args.alphabets.split(",")]
    (OUT / "arms.json").write_text(json.dumps(A, indent=1))
    jobs = []
    for arm in A:
        tags = chunk_tags(CASE, arm["alphabet"], arm["k"], arm["bits"], len(qs))
        size = 25 if len(tags) > 1 else len(qs)
        jobs += [(tag, qs[i * size:(i + 1) * size], arm) for i, tag in enumerate(tags)]
    print(f"{len(qs)} queries, {len(jobs)} chunks over {len(A)} alphabet-ksize pairs", file=sys.stderr, flush=True)
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda j: one_chunk(*j), jobs):
            failed += "FAILED" in msg
            print(msg, file=sys.stderr, flush=True)
    print(f"done: {len(jobs) - failed} of {len(jobs)} chunks ok", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
