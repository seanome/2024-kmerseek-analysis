#!/usr/bin/env python3
"""score_hitlist.py against Foldseek's own bench.noselfhit.awk on made-up hit lists.

Builds a small made-up SCOPe set and random hit lists (distinct scores, so file order and
score order agree, plus self-hits and targets outside the list), runs both, and compares
every query's family, superfamily and fold sensitivity. The awk is fetched once from
github.com/steineggerlab/foldseek-analysis into tests/ (or pass --awk).

Run: python tests/test_score_hitlist.py [--awk path/to/bench.noselfhit.awk]
"""

import argparse
import random
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
AWK_URL = ("https://raw.githubusercontent.com/steineggerlab/foldseek-analysis/main/"
           "scopbenchmark/scripts/bench.noselfhit.awk")


def made_up_set(rng: random.Random) -> list[tuple[str, str]]:
    dom = []
    for c in "ab":
        for fo in range(1, 4):
            for sf in range(1, 4):
                for fa in range(1, 4):
                    for i in range(rng.randint(0, 3)):
                        dom.append((f"d{c}{fo}{sf}{fa}{i}_", f"{c}.{fo}.{sf}.{fa}"))
    return dom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--awk", type=Path, default=HERE / "bench.noselfhit.awk")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    if not a.awk.exists():
        urllib.request.urlretrieve(AWK_URL, a.awk)
    bad = 0
    for seed in range(a.seeds):
        rng = random.Random(seed)
        dom = made_up_set(rng)
        ids = [d for d, _ in dom]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "lookup.tsv").write_text("".join(f"{d}\t{s}\n" for d, s in dom))
            rows = []
            for q in ids:
                targets = rng.sample(ids, rng.randint(0, min(len(ids), 25))) + (["dUNKNOWN"] if rng.random() < 0.1 else [])
                rng.shuffle(targets)
                for k, t in enumerate(targets):
                    rows.append((q, t, 1000.0 - k + rng.random() * 0.1))
            # The awk reads hits in file order: best score first within each query.
            (tmp / "hits.tsv").write_text("".join(f"{q}\t{t}\t{s:.4f}\n" for q, t, s in rows))
            awk = subprocess.run(["awk", "-f", str(a.awk), str(tmp / "lookup.tsv"), str(tmp / "hits.tsv")],
                                 capture_output=True, text=True, check=True).stdout
            want = pl.read_csv(awk.encode(), separator="\t")
            (tmp / "dom.tsv").write_text("domain_id\tscop_id\n" + "".join(f"{d}\t{s}\n" for d, s in dom))
            subprocess.run([sys.executable, str(HERE.parent / "bin" / "score_hitlist.py"), "--domains",
                            str(tmp / "dom.tsv"), "--hits", str(tmp / "hits.tsv"), "--columns", "query,target,bits",
                            "--score", "bits", "--higher-first", "--tool", "t", "--setting", "s",
                            "--out-prefix", str(tmp / "out")], check=True, capture_output=True)
            got = pl.read_parquet(tmp / "out.per_query.parquet").pivot(
                on="level", index="query", values="sensitivity")
            j = want.join(got, left_on="NAME", right_on="query", how="full")
            if j.height != want.height or j.height != got.height:
                print(f"seed {seed}: {want.height} awk queries, {got.height} ours")
                bad += 1
                continue
            for awk_col, ours in (("FAM", "family"), ("SFAM", "superfamily"), ("FOLD", "fold")):
                d = (j[awk_col] - j[ours]).abs().max()
                if d > 1e-5:
                    print(f"seed {seed} {ours}: max difference {d}")
                    bad += 1
    print("score_hitlist.py matches bench.noselfhit.awk on", a.seeds - bad, "of", a.seeds, "made-up sets")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
