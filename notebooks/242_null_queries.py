#!/usr/bin/env python3
"""Query-side null for the alphabet ensemble (notebook 242).

Does Ced9 rank BCL2 higher than a random human protein of Ced9's length does, and
does P66 rank CD47 higher than a random protein of P66's length? And does combining
alphabets widen that gap?

For each case, 300 random human proteins within +/- 25% of the query's length are
searched, together with the true query, against the same 19 indexes: one arm per
alphabet, its lowest-bit arm (about 16 bits; 18.7 to 21.8 for hsdm17, uniprot18,
protein20 and wass14, whose ladders start there). That is where the partners are in
view most often: BCL2 for Ced9 in 13 of these 19 arms, CD47 for P66 in 16. The rule
does not look at where the partner ranks. Search flags are those of 241_alphabet_ranking_driver.py. A
query's own protein is removed from its hit list before ranking. Both partners, BCL2
and CD47, are ranked for every query, so the run also says how often each partner
ranks high for any query at all.

Queries are searched 25 at a time; each chunk's CSV is reduced to the partners'
ranks and deleted, so the run needs little disk and resumes chunk by chunk.

Output: /Users/olga/data/botryllus/alphabet-ranking-three-cases/null/ranks/*.parquet
with one row per (case, query, alphabet, k, metric, partner): rank among the proteins
the query hit (None when the partner was not hit), and n_hit.

About 1.6 billion regions in all (gbmr7 k=8 is 442 M of them), about 3 hours with
3 workers on the Mac.

Usage: 242_null_queries.py [--workers 3] [--n 300] [--chunk 25]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

KMERSEEK = Path("/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek")
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
D = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
OUT = D / "null"
PARTNERS = ["BCL2", "CD47"]
# Multi-domain Bcl-2 family members share BCL2's fold; as a random query they would be
# true homologs of the partner, not a null. CD47 has no paralog in this proteome.
EXCLUDE = {"BCL2", "BCL2L1", "BCL2L2", "BCL2L10", "BCL2A1", "MCL1", "BAX", "BAK1", "BOK", "CD47"}
CASES = {"Ced9": 280, "P66": 597}
METRICS = {
    "E-value": ("region_evalue", True),
    "mean IDF": ("region_mean_idf", False),
    "tf-idf": ("region_tfidf", False),
    "enrichment": ("region_enrichment", False),
    "Poisson p-value": ("region_tail_probability", True),
}


def set_out(p: Path) -> None:
    global OUT
    OUT = p


def read_fasta(p: Path) -> dict[str, str]:
    out, name, buf = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    out[name] = "".join(buf)
    return out


def arms() -> list[dict]:
    """One arm per alphabet, its lowest-bit k, with the penalty the sweep used and
    whether its index has a Karlin-Altschul fit."""
    plan = pl.DataFrame(json.loads((D / "plan.json").read_text()))
    pick = plan.sort("bits").group_by("alphabet").first().sort("bits")
    out = []
    for a, k, b, c, x in pick.select("alphabet", "ksize", "bits", "penalty", "xdrop").iter_rows():
        out.append(dict(alphabet=a, k=k, bits=b, penalty=c, xdrop=x,
                        nofit=(D / "search" / f"{a}.k{k}.nofit").exists()))
    return out


def query_sets(n: int, seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    """case -> [(header, sequence)], the true query first, then n random human proteins
    within +/- 25% of its length. Headers of human proteins are their full GENCODE
    header, so a self-hit can be recognised."""
    human = read_fasta(HUMAN)
    true = read_fasta(D / "queries.fa")
    rng = random.Random(seed)
    sets = {}
    for case, L in CASES.items():
        pool = sorted(h for h, s in human.items()
                      if 0.75 * L <= len(s) <= 1.25 * L and h.split("|")[6] not in EXCLUDE)
        pick = rng.sample(pool, n)
        sets[case] = [(case, true[case])] + [(h, human[h]) for h in pick]
    return sets


def reduce_chunk(csv: Path, case: str, arm: dict) -> pl.DataFrame:
    cols = ["query_name", "target_name"] + [c for c, _ in METRICS.values()]
    if csv.stat().st_size == 0:
        return pl.DataFrame()
    df = pl.read_csv(csv, columns=cols, infer_schema_length=0).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c, _ in METRICS.values()]
    ).with_columns(pl.col("query_name").str.strip_chars(),
                   pl.col("target_name").str.split("|").list.get(6).alias("gene"))
    df = df.filter(pl.col("query_name") != pl.col("target_name"))  # the self-hit
    rows = []
    for label, (col, lower) in METRICS.items():
        best = (pl.col(col).min() if lower else pl.col(col).max()).alias("v")
        per = (df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
                 .group_by("query_name", "gene").agg(best)
                 .with_columns(pl.col("v").rank("average", descending=not lower).over("query_name").alias("rank"),
                               pl.len().over("query_name").alias("n_hit")))
        hit = per.filter(pl.col("gene").is_in(PARTNERS)).select("query_name", "gene", "rank", "n_hit")
        nh = per.group_by("query_name").agg(pl.len().alias("n_hit"))
        grid = nh.join(pl.DataFrame({"gene": PARTNERS}), how="cross")
        rows.append(grid.join(hit.drop("n_hit"), on=["query_name", "gene"], how="left")
                        .with_columns(pl.lit(label).alias("metric")))
    return pl.concat(rows).with_columns(pl.lit(case).alias("case"), pl.lit(arm["alphabet"]).alias("alphabet"),
                                        pl.lit(arm["k"]).alias("k"), pl.lit(arm["bits"]).alias("bits"))


def one_chunk(case: str, i: int, chunk: list[tuple[str, str]], arm: dict) -> str:
    tag = f"{case}.{arm['alphabet']}.k{arm['k']}.c{i:02d}"
    done = OUT / "ranks" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    fa = OUT / "tmp" / f"{tag}.fa"
    csv = OUT / "tmp" / f"{tag}.csv"
    fa.write_text("".join(f">{h}\n{s}\n" for h, s in chunk))
    idx = D / "idx" / f"human.{arm['alphabet']}.k{arm['k']}.rocksdb"
    pen = ["--extend-mismatch-penalty", "0"] if arm["nofit"] else \
          ["--extend-mismatch-penalty", str(arm["penalty"]), "--extend-xdrop", str(arm["xdrop"])]
    cmd = [str(KMERSEEK), "search", "-q", str(fa), "-t", str(idx), "-k", str(arm["k"]), "-a", arm["alphabet"],
           "--threshold", "0", "--min-shared-kmers", "1", "--max-query-pvalue", "1", "--min-region-score", "0",
           *pen, "-o", str(csv)]
    with open(OUT / "logs" / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return f"{tag} FAILED rc={rc}"
    reduce_chunk(csv, case, arm).write_parquet(done)
    csv.unlink(missing_ok=True)
    fa.unlink(missing_ok=True)
    return f"{tag} ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (a smoke test writes elsewhere)")
    ap.add_argument("--alphabets", default="", help="comma-separated subset, for a smoke test")
    args = ap.parse_args()
    set_out(args.out)
    for d in ("ranks", "tmp", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    sets = query_sets(args.n)
    (OUT / "query_sets.json").write_text(json.dumps({c: [h for h, _ in v] for c, v in sets.items()}, indent=1))
    A = [a for a in arms() if not args.alphabets or a["alphabet"] in args.alphabets.split(",")]
    (OUT / "arms.json").write_text(json.dumps(A, indent=1))
    jobs = []
    for arm in A:
        for case, qs in sets.items():
            for i in range(0, len(qs), args.chunk):
                jobs.append((case, i // args.chunk, qs[i:i + args.chunk], arm))
    print(f"{len(jobs)} chunks over {len(A)} arms", file=sys.stderr, flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda j: one_chunk(*j), jobs):
            print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
