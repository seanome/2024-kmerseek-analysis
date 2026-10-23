#!/usr/bin/env python3
"""Searches for notebook 254: the 87 notebook-244 cases, exact and extended, on the new build.

For every (species, arm) the 87 cases need, this builds two indexes of the whole target
proteome and runs one search on each, all in the image notebook 252 searched with
(deep-twilight-controls/nextflow.config, params.kmerseek_search_image):

  exact     `kmerseek index` with the midi-plus flags (alphabet, k, --remove-low-complexity),
            then `kmerseek search` with the midi-plus search flags.
  extended  the same index command plus --extend-mismatch-penalty, --extend-xdrop and
            --ka-reference-shuffles, as deep-twilight-controls/main.nf builds its index
            (the Karlin-Altschul fit is built into the index); the same search command plus
            --extend-mismatch-penalty and --extend-xdrop.

The search flags are the midi-plus defaults from qfo-pfam-region-benchmark/main.nf
(--threshold 0, --min-shared-kmers 2, --max-query-pvalue 0.05, --min-region-score 1.3) for
both conditions, so extension is the only difference between them. The queries are the
human proteins of the 87 cases only. No statistic kmerseek reports depends on the other
queries in the file: the Bonferroni count downstream is region_search_space times
db_n_targets, both set by the target proteome.

Each search's CSV is written to parquet the way kmerseekSearch does it (same dropped
columns), under OUT_DIR/<condition>/human_vs_<species>.<alphabet>.k<k>.lctrue.regions.parquet,
so reduce_swissprot_instance_landing.py reads it unchanged.

    python scripts/run_254_extension_searches.py [--workers 4]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import hero_example_utils as he  # noqa: E402

IMAGE = (
    "docker.io/olgabot/kmerseek@sha256:"
    "f08eeda6b7dc719c7e03fc69ac9779591498639bf75c00a0e91dc90cbcb52dd5"
)
OUT_DIR = Path("/Users/olga/data/hero-244-extension")
CASES = REPO / "tables" / "244_hero_candidates.csv"
ARMS = REPO / "tables" / "254_extension_arms.tsv"
#: kmerseekSearch's search filters in qfo-pfam-region-benchmark/main.nf (midi-plus).
SEARCH_FLAGS = [
    "--threshold",
    "0.0",
    "--min-shared-kmers",
    "2",
    "--max-query-pvalue",
    "0.05",
    "--min-region-score",
    "1.3",
]
#: Columns kmerseekSearch drops before writing parquet.
DROP = ["query_md5", "target_md5", "region_subseq", "target_subseq", "moltype_seq"]
N_EXPECTED = 87


def load_cases() -> pl.DataFrame:
    cases = pl.read_csv(CASES).filter(pl.col("admitted_under") == "all criteria")
    if cases.height != N_EXPECTED:
        raise SystemExit(f"expected {N_EXPECTED} cases, found {cases.height}")
    pattern = r"^kmerseek\.(.+)_k(\d+)_lc(True|False)$"
    arm = pl.col("kmerseek_chosen_arm")
    return cases.with_columns(
        alphabet=arm.str.extract(pattern, 1),
        ksize=arm.str.extract(pattern, 2).cast(pl.Int64),
        lc=arm.str.extract(pattern, 3),
    )


def write_queries(accessions: set[str], out: Path) -> None:
    """The human query records, headers kept as the QfO file has them."""
    keep = False
    n = 0
    with open(he.proteome_fasta("human")) as fh, open(out, "w") as w:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split()[0].split("|")[1]
                keep = acc in accessions
                n += keep
            if keep:
                w.write(line)
    if n != len(accessions):
        raise SystemExit(f"found {n} of {len(accessions)} query proteins")


def docker(args: list[str], mounts: list[Path], log: Path) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-e",
        "RAYON_NUM_THREADS=4",
    ]
    for m in mounts:
        cmd += ["-v", f"{m}:{m}"]
    cmd += [IMAGE, "kmerseek", *args]
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(args) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"exit {rc}: {' '.join(args)} (see {log})")


def search_to_parquet(csv: Path, out: Path) -> None:
    if csv.stat().st_size == 0:
        out.touch()
        return
    lf = pl.scan_csv(csv, ignore_errors=True)
    cols = [c for c in lf.collect_schema().names() if c not in DROP]
    lf.select(cols).sink_parquet(out, compression="zstd", compression_level=9)
    csv.unlink()


def one(species: str, arm: dict, queries: Path) -> str:
    a, k = arm["alphabet"], arm["ksize"]
    stem = f"human_vs_{species}.{a}.k{k}.lctrue"
    target = he.proteome_fasta(species)
    work = OUT_DIR / "indexes"
    work.mkdir(parents=True, exist_ok=True)
    mounts = [OUT_DIR, target.parent, queries.parent]
    t0 = time.time()
    for cond, extra_index, extra_search in (
        ("exact", [], []),
        (
            "extended",
            [
                "--extend-mismatch-penalty",
                str(arm["penalty"]),
                "--extend-xdrop",
                str(arm["xdrop"]),
                "--ka-reference-shuffles",
                str(arm["ka_reference_shuffles"]),
            ],
            [
                "--extend-mismatch-penalty",
                str(arm["penalty"]),
                "--extend-xdrop",
                str(arm["xdrop"]),
            ],
        ),
    ):
        out_dir = OUT_DIR / cond
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{stem}.regions.parquet"
        if out.exists():
            continue
        log = out_dir / f"{stem}.log"
        idx = work / f"{species}.{a}.k{k}.lctrue.{cond}.rocksdb"
        if idx.exists():
            shutil.rmtree(idx)
        docker(
            [
                "index",
                "--alphabet",
                a,
                "--ksize",
                str(k),
                "--input",
                str(target),
                "--output",
                str(idx),
                "--remove-low-complexity",
                *extra_index,
            ],
            mounts,
            log,
        )
        csv = out_dir / f"{stem}.regions.csv"
        docker(
            [
                "search",
                "--alphabet",
                a,
                "--ksize",
                str(k),
                "--query",
                str(queries),
                "--target",
                str(idx),
                "--remove-low-complexity",
                *SEARCH_FLAGS,
                *extra_search,
                "--output",
                str(csv),
            ],
            mounts,
            log,
        )
        search_to_parquet(csv, out)
        shutil.rmtree(idx)
    return f"{species} {a} k{k}: {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="run one species, for testing")
    args = ap.parse_args()

    cases = load_cases()
    if (cases["lc"] != "True").any():
        raise SystemExit(
            "a case has the low-complexity mask off; this runner assumes it on"
        )
    arms = pl.read_csv(ARMS, separator="\t")
    need = cases.select("species", "alphabet", "ksize").unique()
    missing = need.join(arms, on=["alphabet", "ksize"], how="anti")
    if missing.height:
        raise SystemExit(
            f"no extension settings for:\n{missing.select('alphabet', 'ksize').unique()}"
        )
    jobs = (
        need.join(arms, on=["alphabet", "ksize"]).sort("species", "alphabet").to_dicts()
    )
    if args.only:
        jobs = [j for j in jobs if j["species"] == args.only]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = OUT_DIR / "queries_254.fasta"
    write_queries(set(cases["query"]), queries)
    print(
        f"{cases['query'].n_unique()} query proteins; {len(jobs)} (species, arm) pairs; "
        f"two indexes and two searches each",
        flush=True,
    )
    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(lambda j: one(j["species"], j, queries), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
