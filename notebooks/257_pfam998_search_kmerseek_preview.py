#!/usr/bin/env python3
"""Notebook 243's all-against-all search of the 998 Pfam-annotated human proteins, run
again with a newer kmerseek, so notebook 255's ranking metrics can be scored on it.

Why: in the notebook 243 run (kmerseek-ka-lambda-region, 982a055) 2_005 of the 5_649
labelled hp_lehninger2 k=24 matches have no E-value. That build solves lambda from each
region's own two stretches and gives no E-value when those stretches agree by chance more
often than C/(1+C). The preview build (kmerseek 8978e78, the tip of the local branch
olgabot/extend-10-default-extension-v2, not yet pushed or reviewed) differs in three ways:

  - lambda comes from the two whole proteins' class frequencies, not the region's;
  - every row gets `region_evalue`: the extension E-value (`region_ka_evalue`) when the
    pair has a positive lambda, otherwise the exact-run E-value (`region_run_evalue`),
    with `region_evalue_source` saying which;
  - a refused Karlin-Altschul fit no longer refuses the search: regions are still
    extended, and `region_evalue` is the run E-value.

What is the same as 243: the 998 proteins (PR #44's truth set, QfO 2020_04 sequences),
the 152 alphabet-ksize pairs of notebook 241's plan.json with their penalty and give-up
margin, --remove-low-complexity, a fit on 200 queries (seed 1) when the index is built,
and the search flags (every region kept). What differs: every alphabet-ksize pair is
extended (243 kept regions exact where notebook 241's human search had no fit), and there
is no calibrate retry (this build has no calibrate subcommand, and does not need one).

Each alphabet-ksize pair's CSV is reduced to KEEP and written as
<out>/regions/<alphabet>.k<k>.parquet; the CSV and index are then deleted. Resumes pair by
pair. The kmerseek binary and its commit are written to <out>/run.json and checked on a
resume, so two builds cannot mix in one folder.

Usage: 257_pfam998_search_kmerseek_preview.py [--workers 3] [--alphabets a,b] [--out DIR]
                                              [--kmerseek PATH]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search243", HERE / "243_pfam998_search.py")
s243 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s243)

KMERSEEK = Path("/Users/olga/code/kmerseek-preview-8978e78/target/release/kmerseek")
OUT = Path("/Users/olga/data/alphabet-logreg-pfam998-kmerseek-8978e78")
# 243's columns, minus region_ka_lambda (this build does not write it; whether a pair had a
# positive lambda is region_evalue_source), plus the new E-value columns and the four
# metrics 243 did not keep (so notebook 255 can score them per alphabet-ksize pair).
KEEP = [c for c in s243.KEEP if c != "region_ka_lambda"] + [
    "region_ka_evalue", "region_run_evalue", "region_poisson_evalue", "region_run_length", "region_pr_same",
    "region_evalue_source", "region_poisson_score", "containment", "query_enrichment", "query_poisson_pvalue"]
TEXT = {"query_name", "target_name", "region_evalue_source"}


def build_commit(kmerseek: Path) -> str:
    """The git commit of the checkout the binary was built in (target/release/kmerseek)."""
    root = kmerseek.resolve().parents[2]
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def one_pair(pair: dict, fa: Path, kmerseek: Path) -> str:
    try:
        return _one_pair(pair, fa, kmerseek)
    except Exception as e:  # noqa: BLE001  one pair's failure must not stop the others
        return f"{pair['alphabet']}.k{pair['ksize']} FAILED {type(e).__name__}: {e}"[:300]


def _one_pair(pair: dict, fa: Path, kmerseek: Path) -> str:
    a, k, c, x = pair["alphabet"], pair["ksize"], str(pair["penalty"]), str(pair["xdrop"])
    tag = f"{a}.k{k}"
    done = OUT / "regions" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    idx, csv = OUT / "tmp" / f"{tag}.rocksdb", OUT / "tmp" / f"{tag}.csv"
    surv, logs = OUT / "ka_survival" / f"{tag}.csv", OUT / "logs"
    shutil.rmtree(idx, ignore_errors=True)
    rc = s243.run([str(kmerseek), "index", "-i", str(fa), "-o", str(idx), "-k", str(k), "-s", "1", "-a", a,
                   "--remove-low-complexity", "--extend-mismatch-penalty", c, "--extend-xdrop", x,
                   "--ka-queries", "200", "--ka-seed", "1", "--ka-survival-out", str(surv)],
                  logs / f"{tag}.index.log")
    if rc != 0:
        return f"{tag} INDEX FAILED rc={rc} (log: {logs / (tag + '.index.log')})"
    # This build's survival CSV has no `fitted` column; the index log says whether a fit
    # was stored for this penalty and give-up margin.
    fit = "Stored in the index for" in (logs / f"{tag}.index.log").read_text()
    rc = s243.run([str(kmerseek), "search", "-q", str(fa), "-t", str(idx), "-k", str(k), "-a", a,
                   "--threshold", "0", "--min-shared-kmers", "1", "--max-query-pvalue", "1",
                   "--min-region-score", "0", "--extend-mismatch-penalty", c, "--extend-xdrop", x,
                   "-o", str(csv)], logs / f"{tag}.search.log")
    if rc != 0 or not csv.exists():
        return f"{tag} SEARCH FAILED rc={rc} (log: {logs / (tag + '.search.log')})"
    if csv.stat().st_size == 0:
        df = pl.DataFrame(schema={c_: pl.Utf8 for c_ in KEEP})
    else:
        df = pl.read_csv(csv, columns=KEEP, infer_schema_length=0)
    df = (df.with_columns([pl.col(c_).cast(pl.Float64, strict=False) for c_ in KEEP if c_ not in TEXT])
            .with_columns(pl.col("query_name").str.strip_chars(), pl.col("target_name").str.strip_chars())
            .filter(pl.col("query_name") != pl.col("target_name"))
            .with_columns(pl.lit(a).alias("alphabet"), pl.lit(k).alias("ksize"),
                          pl.lit(pair["bits"]).alias("bits"), pl.lit(True).alias("extended"),
                          pl.lit(fit is True).alias("fitted_here")))
    df.write_parquet(done)
    csv.unlink(missing_ok=True)
    shutil.rmtree(idx, ignore_errors=True)
    return f"{tag} ok {df.height} regions fit_here={fit}"


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--alphabets", default="")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--kmerseek", type=Path, default=KMERSEEK)
    args = ap.parse_args()
    OUT = args.out
    for d in ("regions", "tmp", "logs", "ka_survival"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    want = {"kmerseek": str(args.kmerseek), "commit": build_commit(args.kmerseek)}
    manifest = OUT / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != want:
        sys.exit(f"{OUT} holds a run with {manifest.read_text().strip()}; use another --out")
    if not manifest.exists() and any((OUT / "regions").glob("*.parquet")):
        sys.exit(f"{OUT}/regions holds results from an unknown kmerseek; use another --out")
    manifest.write_text(json.dumps(want))
    fa = OUT / "human_pfam_truth.fa"
    if not fa.exists():
        print(f"{s243.write_fasta(fa)} proteins written to {fa}", file=sys.stderr)
    plan = json.loads((s243.SWEEP / "plan.json").read_text())
    pairs = sorted((p for p in plan if not args.alphabets or p["alphabet"] in args.alphabets.split(",")),
                   key=lambda p: p["bits"])  # heaviest first
    print(f"{len(pairs)} alphabet-ksize pairs, kmerseek {want['commit'][:7]}", file=sys.stderr, flush=True)
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda p: one_pair(p, fa, args.kmerseek), pairs):
            failed += "FAILED" in msg
            print(msg, file=sys.stderr, flush=True)
    print(f"done: {len(pairs) - failed} of {len(pairs)} alphabet-ksize pairs ok", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
