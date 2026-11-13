#!/usr/bin/env python3
"""Does the size-scaled whole-protein p-value (seanome/kmerseek PR 136) rank BCL2 and CD47
better than the other measures?

For each setting, the index is built once with the PR 136 build (search-time change only, so
the index is the same) and searched with both kmerseek main and PR 136. Then BCL2's rank for
Ced9 and CD47's rank for P66 are read off under every measure, with ties, by the same code
the notebook's collector uses (rank = 1 + the number of proteins strictly better).

Settings: with --all, every alphabet and k-size pair in plan.json (152). Without it, a
quick ten: the eight where a partner had its best untied rank under some measure in
ranks.csv, plus hp_lehninger2 k=17 (the BCL2/Ced9 core) and dayhoff6 k=15 (MCL1).

Usage (paths from the environment, as in the driver; 241_length_scaled_check.sbatch sets
them on Sherlock):
  NB241_MAIN=<kmerseek main> NB241_KMERSEEK=<PR 136 build> NB241_CHECK_DIR=<out> \
      241_length_scaled_check.py [--all] [--workers 2]            # index and search
      241_length_scaled_check.py [--all] --collect                # the table (polars)
Writes $NB241_CHECK_DIR/length_scaled_ranks.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drv = load("nb241_driver", "241_alphabet_ranking_driver.py")
OUT = Path(os.environ["NB241_CHECK_DIR"])
MAIN = Path(os.environ["NB241_MAIN"])
QUERIES = HERE / "241_queries.fa"
SETTINGS = [
    ("polarity4", 9),
    ("gbmr4", 16),
    ("wass14", 6),
    ("wwmj5", 9),
    ("hp_lehninger_hpc3", 14),
    ("hp_lehninger_hpc3", 18),
    ("hp_lehninger_hpc3", 21),
    ("sdm12", 6),
    ("hp_lehninger2", 17),
    ("dayhoff6", 15),
]


def read_plan() -> dict:
    """The first run's settings (k, penalty, give-up margin per pair), committed so the check
    uses exactly the settings ranks.csv came from. Regenerating it reads the kappa table,
    which has changed since: funcgroups8 now gets penalty 0.29 instead of the fallback 2.
    """
    plan = json.loads((HERE / "241_plan.json").read_text())
    return {(a["alphabet"], a["ksize"]): a for a in plan}


def one_setting(a: str, k: int, arm: dict) -> dict:
    new, old = OUT / "pr136", OUT / "main"
    c, x = arm["penalty"], arm["xdrop"]
    rec = drv.one_arm(a, k, c, x, False, queries=QUERIES, out=new, pairs={})
    tag = f"{a}.k{k}"
    # Same search as one_arm, run with kmerseek main on the same index.
    ext = ["--extend-mismatch-penalty", "0"]
    if not (new / "search" / f"{tag}.nofit").exists():
        ext = ["--extend-mismatch-penalty", c, "--extend-xdrop", x]
    csv = old / "search" / f"{tag}.csv"
    if not csv.exists() and rec.get("index_rc") == 0:
        cmd = [
            str(MAIN),
            "search",
            "-q",
            str(QUERIES),
            "-t",
            str(new / "idx" / f"human.{tag}.rocksdb"),
            "-k",
            str(k),
            "-a",
            a,
            "--threshold",
            "0",
            "--min-shared-kmers",
            "1",
            "--max-query-pvalue",
            "1",
            "--min-region-score",
            "0",
            *ext,
            "-o",
            str(csv),
        ]
        with open(new / "logs" / f"{tag}.main_search.log", "w") as fh:
            rec["main_search_rc"] = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT
            ).returncode
    return rec


def run(settings: list[tuple[str, int]], workers: int) -> None:
    plan = read_plan()
    for d in ("idx", "search", "ka_survival", "pair", "logs"):
        (OUT / "pr136" / d).mkdir(parents=True, exist_ok=True)
    (OUT / "main" / "search").mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_setting, a, k, plan[(a, k)]) for a, k in settings]
        for fut in cf.as_completed(futs):
            print(json.dumps(fut.result()), file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every pair in plan.json")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--collect",
        action="store_true",
        help="build the table (needs polars) instead of running",
    )
    args = ap.parse_args()
    settings = sorted(read_plan()) if args.all else SETTINGS
    collect(settings) if args.collect else run(settings, args.workers)


def collect(settings: list[tuple[str, int]]) -> None:
    import polars as pl

    col = load("nb241_collect", "241_alphabet_ranking_collect.py")
    plan = read_plan()
    rows = []
    for build, d in (("main", OUT / "main"), ("PR 136", OUT / "pr136")):
        for a, k in settings:
            df = col.read_search(f"{a}.k{k}", out=d)
            if df is None:
                continue
            for r in col.rank_rows(
                df, a, k, plan[(a, k)]["bits"], queries=["Ced9", "P66"]
            ):
                rows.append({"build": build, **r})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(
        OUT / "length_scaled_ranks.csv"
    )
    print(f"{len(rows)} rows -> {OUT / 'length_scaled_ranks.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
