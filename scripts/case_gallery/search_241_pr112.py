"""Rerun notebook 241's 152 searches with kmerseek PR #112 (run-evalue-v2, 9878ff8).

Same indexes, queries and flags as notebooks/241_alphabet_ranking_driver.py; arms with no
Karlin-Altschul fit are searched with exact regions (--extend-mismatch-penalty 0), as there.
Writes to sources.D241 / search; notebook 241's own search/ (in sources.D241_ORIGINAL) is not touched.
A search whose CSV already exists is skipped.
"""
import concurrent.futures as cf
import json
import subprocess
import time
from pathlib import Path

from sources import D241, D241_ORIGINAL as D

KMERSEEK = "/Users/olga/code/kmerseek-run-evalue-v2/target/release/kmerseek"
OUT = D241 / "search"
OUT.mkdir(exist_ok=True)

last = {}
for line in open(D / "runs.jsonl"):
    r = json.loads(line)
    last[r["tag"]] = r


def one(r):
    tag, a, k, c, x = r["tag"], r["alphabet"], str(r["ksize"]), r["penalty"], r["xdrop"]
    csv = OUT / f"{tag}.csv"
    if csv.exists():
        return tag, 0, 0.0
    cmd = [KMERSEEK, "search", "-q", str(D / "queries.fa"), "-t", str(D / "idx" / f"human.{a}.k{k}.rocksdb"),
           "-k", k, "-a", a, "--threshold", "0", "--min-shared-kmers", "1",
           "--max-query-pvalue", "1", "--min-region-score", "0"]
    if (D / "search" / f"{tag}.nofit").exists():
        cmd += ["--extend-mismatch-penalty", "0"]
    else:
        cmd += ["--extend-mismatch-penalty", c, "--extend-xdrop", x]
    cmd += ["-o", str(csv) + ".part"]
    t = time.time()
    with open(OUT / f"{tag}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        Path(str(csv) + ".part").rename(csv)
    return tag, rc, time.time() - t


with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for tag, rc, dt in ex.map(one, last.values()):
        print(tag, rc, round(dt, 1), flush=True)
