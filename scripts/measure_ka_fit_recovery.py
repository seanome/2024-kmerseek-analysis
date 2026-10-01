"""How many kmerseek alphabet-k combinations get a Karlin-Altschul fit on the deep-twilight
database, with the x = 0 bin fix and --ka-reference-shuffles (kmerseek PR #89 on #88).

For each row of nextflow-runs/deep-twilight-controls/assets/arms.tsv, builds the index with
the row's penalty and give-up margin and 1, 4 and then 32 reference shuffles, stopping at
the first that fits, and records what the index step printed. The index itself is deleted.

    python scripts/measure_ka_fit_recovery.py KMERSEEK_BINARY DATABASE_FASTA OUT_TSV [--workers 4]

DATABASE_FASTA is the pipeline's results/database/database.fasta. Output columns: alphabet,
ksize, shuffles, fitted, n_queries, n_regions, n_reference_queries, n_reference_regions,
seconds.
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
ARMS = REPO / "nextflow-runs" / "deep-twilight-controls" / "assets" / "arms.tsv"
SHUFFLES = (1, 4, 32)
REFUSED = re.compile(
    r"(\d+) queries gave (\d+) regions and (\d+) shuffled queries gave (\d+) chance regions"
)
FITTED = re.compile(r"fitted now: (\d+) database queries, (\d+) regions")


def one_arm(binary, database, arm):
    rows = []
    for n in SHUFFLES:
        tmp = tempfile.mkdtemp(prefix="kafit_")
        t0 = time.time()
        proc = subprocess.run(
            [
                binary, "index", "--input", database, "--output", f"{tmp}/idx",
                "--ksize", str(arm["ksize"]), "--alphabet", arm["alphabet"],
                "--extend-mismatch-penalty", str(arm["penalty"]),
                "--extend-xdrop", str(arm["xdrop"]),
                "--ka-reference-shuffles", str(n),
            ],
            capture_output=True, text=True,
            env={**os.environ, "RAYON_NUM_THREADS": "3"},
        )
        seconds = time.time() - t0
        shutil.rmtree(tmp, ignore_errors=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{arm} shuffles {n}: {proc.stderr[-2000:]}")
        fit, refused = FITTED.search(proc.stderr), REFUSED.search(proc.stderr)
        row = dict(alphabet=arm["alphabet"], ksize=arm["ksize"], shuffles=n,
                   fitted=fit is not None, seconds=round(seconds, 1))
        if fit:
            row.update(n_queries=int(fit[1]), n_regions=int(fit[2]),
                       n_reference_queries=None, n_reference_regions=None)
        elif refused:
            row.update(n_queries=int(refused[1]), n_regions=int(refused[2]),
                       n_reference_queries=int(refused[3]), n_reference_regions=int(refused[4]))
        else:
            raise RuntimeError(f"{arm} shuffles {n}: no fit line in\n{proc.stderr[-2000:]}")
        rows.append(row)
        print(f"{arm['alphabet']} k{arm['ksize']} shuffles {n}: "
              f"{'fitted' if row['fitted'] else 'refused'} ({seconds:.0f} s)", flush=True)
        if row["fitted"]:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("database")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    arms = pl.read_csv(ARMS, separator="\t").to_dicts()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda a: one_arm(args.binary, args.database, a), arms))
    out = pl.DataFrame([r for rows in results for r in rows]).sort("alphabet", "ksize", "shuffles")
    out.write_csv(args.out, separator="\t")
    first = out.group_by("alphabet", "ksize").agg(
        pl.col("shuffles").filter(pl.col("fitted")).min().alias("shuffles_to_fit"))
    print(first.group_by("shuffles_to_fit").len().sort("shuffles_to_fit"))


if __name__ == "__main__":
    main()
