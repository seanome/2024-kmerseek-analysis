#!/usr/bin/env python3
"""Notebook 284, shuffled-query control: how much do the 19 alphabets agree by chance?

Ced9, P66 and BHF, each with 20 dipeptide-shuffled copies, are searched against the
GENCODE v49 canonical human proteome at notebook 241's 152 alphabet-ksize settings, with
notebook 256's search command (same binary, indexes and flags). A copy keeps its query's
length, first and last residue and the count of every adjacent residue pair
(Altschul & Erickson 1985), so it has the query's composition and short repeats and no
real homology. Every region is kept.

Then, for each query and copy, the agreement counts of alphabet_agreement_utils (the same
code that made the real numbers in scripts/284_*.py), at each E-value cut:
  - for Ced9 and its copies, the best spot on BCL2; for P66 and its copies, on CD47;
  - the proteome's highest count, and how many proteins reach each count.
If the real query's agreement on its partner, or its proteome-wide agreement, sits inside
the spread of its copies, agreement is what chance gives.

Steps (each resumes):
  search      one region table per setting: <OUT>/regions/<alphabet>.k<k>.<chunk>.parquet
  summarise   <OUT>/284_shuffle_agreement_partner.csv and _proteome.csv (small; copy back)

Paths come from the environment:
  NB284_DIR       output folder (default $SCRATCH/284-shuffle-agreement)
  NB241_KMERSEEK  kmerseek 982a055 binary (notebook 241's build)
  NB241_IDX       folder of the 152 rebuilt human indexes, human.<alphabet>.k<k>.rocksdb
                  (notebook 283's rebuild: $SCRATCH/283-bhf-low-complexity/indexes/idx)
  NB256_ARMS      notebook 256's arms.json (alphabet, k, bits, penalty, xdrop, nofit)
  NB241_QUERIES   notebook 241's queries.fa (Ced9, P66, BHF)

Usage: 284_shuffle_agreement.py search|summarise [--workers 3] [--n 20] [--seed 284]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alphabet_agreement_utils as au  # noqa: E402

ENV = os.environ
SCRATCH = Path(ENV.get("SCRATCH", "/tmp"))
OUT = Path(ENV.get("NB284_DIR", SCRATCH / "284-shuffle-agreement"))
KMERSEEK = Path(
    ENV.get(
        "NB241_KMERSEEK",
        SCRATCH / "241-alphabet-ranking-kmerseek/target/release/kmerseek",
    )
)
IDX = Path(ENV.get("NB241_IDX", SCRATCH / "283-bhf-low-complexity/indexes/idx"))
ARMS = Path(
    ENV.get("NB256_ARMS", SCRATCH / "241-alphabet-ranking/null_bhf_dipeptide/arms.json")
)
QUERIES = Path(ENV.get("NB241_QUERIES", SCRATCH / "241-alphabet-ranking/queries.fa"))
CASES = {"Ced9": "BCL2", "P66": "CD47", "BHF": None}
HEAVY_BITS = 20.5  # settings at or below this many seed bits are searched 25 queries at a time (as in 256)
KEEP = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "region_evalue",
]
# Cuts summarised; the full list in alphabet_agreement_utils costs more time than it adds.
SUMMARY_CUTS = [None, 1e4, 1e2, 10]


def read_fasta(p: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = ""
        else:
            out[name] += line.strip()
    return out


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """A random sequence with the same first residue, last residue and count of every
    adjacent residue pair as `seq` (Altschul & Erickson 1985). Notebook 256's function.
    """
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


def make_queries(n: int, seed: int) -> list[tuple[str, str]]:
    real = read_fasta(QUERIES)
    rng = random.Random(seed)
    qs = []
    for case in CASES:
        qs.append((case, real[case]))
        for i in range(n):
            s = dipeptide_shuffle(real[case], rng)
            assert Counter(zip(s, s[1:])) == Counter(zip(real[case], real[case][1:]))
            qs.append((f"{case}_shuf{i:02d}", s))
    return qs


def one_chunk(tag: str, chunk: list[tuple[str, str]], arm: dict) -> str:
    done = OUT / "regions" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    fa, csv = OUT / "tmp" / f"{tag}.fa", OUT / "tmp" / f"{tag}.csv"
    fa.write_text("".join(f">{h}\n{s}\n" for h, s in chunk))
    idx = IDX / f"human.{arm['alphabet']}.k{arm['k']}.rocksdb"
    pen = (
        ["--extend-mismatch-penalty", "0"]
        if arm["nofit"]
        else [
            "--extend-mismatch-penalty",
            str(arm["penalty"]),
            "--extend-xdrop",
            str(arm["xdrop"]),
        ]
    )
    cmd = [
        str(KMERSEEK),
        "search",
        "-q",
        str(fa),
        "-t",
        str(idx),
        "-k",
        str(arm["k"]),
        "-a",
        arm["alphabet"],
        "--threshold",
        "0",
        "--min-shared-kmers",
        "1",
        "--max-query-pvalue",
        "1",
        "--min-region-score",
        "0",
        *pen,
        "-o",
        str(csv),
    ]
    with open(OUT / "logs" / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not csv.exists():
        return f"{tag} FAILED rc={rc} (log: {OUT / 'logs' / (tag + '.log')})"
    if csv.stat().st_size == 0:
        df = pl.DataFrame(schema={c: pl.String for c in KEEP})
    else:
        df = (
            pl.scan_csv(csv, infer_schema_length=0)
            .select(KEEP)
            .filter(
                pl.col("region_start").is_not_null() & (pl.col("region_start") != "")
            )
            .with_columns(
                pl.col("query_name").str.strip_chars(),
                *[pl.col(c).cast(pl.Float64, strict=False) for c in KEEP[2:]],
            )
            .with_columns(
                gene=pl.col("target_name").str.split("|").list.get(6),
                alphabet=pl.lit(arm["alphabet"]),
                ksize_arm=pl.lit(arm["k"]),
            )
            .collect()
        )
    df.write_parquet(done.with_suffix(".tmp"))
    done.with_suffix(".tmp").rename(done)
    csv.unlink(missing_ok=True)
    fa.unlink(missing_ok=True)
    return f"{tag} ok"


def step_search(workers: int, n: int, seed: int) -> None:
    for d in ("regions", "tmp", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    want = {"n": n, "seed": seed}
    run = OUT / "run.json"
    if run.exists() and json.loads(run.read_text()) != want:
        sys.exit(f"{OUT} holds a run with {run.read_text()}; use another NB284_DIR")
    run.write_text(json.dumps(want))
    qs = make_queries(n, seed)
    (OUT / "queries.fa").write_text("".join(f">{h}\n{s}\n" for h, s in qs))
    arms = json.loads(ARMS.read_text())
    missing = [
        a
        for a in arms
        if not (IDX / f"human.{a['alphabet']}.k{a['k']}.rocksdb").exists()
    ]
    if missing:
        sys.exit(
            f"{len(missing)} of {len(arms)} indexes missing in {IDX} (first: {missing[0]}); "
            "run notebook 283's `indexes` step first"
        )
    jobs = []
    for a in arms:
        base = f"{a['alphabet']}.k{a['k']}"
        if a["bits"] <= HEAVY_BITS:
            jobs += [
                (f"{base}.c{i:02d}", qs[i * 25 : (i + 1) * 25], a)
                for i in range(-(-len(qs) // 25))
            ]
        else:
            jobs.append((f"{base}.all", qs, a))
    print(
        f"{len(qs)} queries, {len(jobs)} chunks over {len(arms)} settings",
        file=sys.stderr,
        flush=True,
    )
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(one_chunk, *zip(*[(t, c, a) for t, c, a in jobs])):
            print(msg, file=sys.stderr, flush=True)
            if "FAILED" in msg:
                failed.append(msg)
    if failed:
        sys.exit(f"{len(failed)} chunks failed; rerun to retry them")


def step_summarise() -> None:
    files = sorted((OUT / "regions").glob("*.parquet"))
    names = [h for h in read_fasta(OUT / "queries.fa")]
    partner_rows, proteome_rows = [], []
    for q in names:
        case = q.split("_shuf")[0]
        d = au.prepare(pl.scan_parquet(files).filter(pl.col("query_name") == q))
        for cut in SUMMARY_CUTS:
            s = au.spots(d, cut)
            label = "all regions" if cut is None else f"E < {cut:g}"
            for c in au.COUNTS:
                b = au.best_per_target(s, c)
                hist = b.group_by(c).len()
                for v, k in hist.iter_rows():
                    proteome_rows.append(
                        dict(
                            query=q,
                            case=case,
                            real=q == case,
                            e_cut=label,
                            count=c,
                            agreement=v,
                            n_proteins=k,
                        )
                    )
                if CASES[case]:
                    p = b.filter(pl.col("gene") == CASES[case])
                    v = int(p[c].max()) if p.height else 0
                    partner_rows.append(
                        dict(
                            query=q,
                            case=case,
                            real=q == case,
                            partner=CASES[case],
                            e_cut=label,
                            count=c,
                            partner_value=v,
                            proteins_with_more=int(b.filter(pl.col(c) > v).height),
                            proteome_max=int(b[c].max()) if b.height else 0,
                        )
                    )
        print(f"{q} summarised", file=sys.stderr, flush=True)
    pl.DataFrame(partner_rows).write_csv(OUT / "284_shuffle_agreement_partner.csv")
    pl.DataFrame(proteome_rows).write_csv(OUT / "284_shuffle_agreement_proteome.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["search", "summarise"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=284)
    args = ap.parse_args()
    if args.step == "search":
        step_search(args.workers, args.n, args.seed)
    else:
        step_summarise()


if __name__ == "__main__":
    main()
