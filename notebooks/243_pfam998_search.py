#!/usr/bin/env python3
"""All-against-all search of the 998 Pfam-annotated human proteins with every arm of the
notebook 241 sweep (notebook 243's training data).

The protein set and its Pfam truth are PR #44's (build_labeled_benchmark.py): the 998
human proteins with a Pfam domain in the midi-plus truth set, sequences from the QfO
2020_04 human proteome. Each arm (one alphabet at one k, 152 in all, the plan.json of
notebook 241) gets its own index on these 998 proteins with the same penalty, X-drop
and Karlin-Altschul fit rule as 241_alphabet_ranking_driver.py: a fit on 200 sequences,
retried with 1_000 x 8 shuffles when refused unless both curves are already dense, and
exact regions when still refused. The search keeps every region.

Every arm is searched the way the same arm was searched against the human proteome in
notebook 241, so a feature means the same thing in training and when the model is
applied: an arm that got exact regions there (a .nofit marker) gets exact regions here,
with no fit; an arm that was extended there is extended here, and if this database's fit
is refused it borrows the human fit's K (--ka-k). Without this, 18 of the first 117 arms
differed, and an extended region's mean IDF (summed IDF over the region / shared k-mers)
reached 100 times what the model had seen in training.

Each arm's CSV is reduced to the columns notebook 243 uses and written as
<out>/regions/<alphabet>.k<k>.parquet; the CSV and the index are then deleted, so the
run needs little disk and resumes arm by arm.

The heaviest arm, gbmr7 k=8, gives 59 M region pairs in about 250 s; all 152 arms take
roughly 30 to 60 minutes with 3 workers.

Usage: 243_pfam998_search.py [--workers 3] [--alphabets a,b] [--min-bits B] [--out DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

KMERSEEK = Path("/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek")
TRUTH = Path("/Users/olga/data/qfo-pfam-region-midi-plus/truth/human_domain_truth.parquet")
PROTEOME = Path("/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
                "/Eukaryota/UP000005640_9606.fasta")
SWEEP = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
OUT = Path("/Users/olga/data/alphabet-logreg-pfam998")
KEEP = ["query_name", "target_name", "region_start", "region_end", "target_start", "target_end",
        "region_length", "region_n_shared_kmers", "region_n_mismatches", "region_mean_idf",
        "region_tfidf", "region_enrichment", "region_tail_probability", "region_evalue",
        "region_ka_bits", "region_ka_lambda", "db_n_targets"]
RETRY_QUERIES, RETRY_SHUFFLES, DENSE = 1000, 8, 100_000
_COUNTS = re.compile(r"(\d+) queries gave (\d+) regions and (\d+) shuffled queries gave (\d+) chance regions")


def write_fasta(out: Path) -> int:
    """The truth-set proteins, by UniProt accession, from the QfO human proteome."""
    want = set(pl.read_parquet(TRUTH)["accession"].to_list())
    kept, name, seq = 0, None, []
    with out.open("w") as fh:
        def flush():
            nonlocal kept
            if name and name in want:
                fh.write(f">{name}\n{''.join(seq)}\n")
                kept += 1
        for line in PROTEOME.open():
            if line.startswith(">"):
                flush()
                parts = line[1:].strip().split("|")
                name = parts[1] if len(parts) > 2 else line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
        flush()
    return kept


def fitted(survival: Path) -> bool | None:
    if not survival.exists():
        return None
    with open(survival) as fh:
        header = fh.readline().rstrip("\n").split(",")
        row = fh.readline().rstrip("\n").split(",")
    if "fitted" not in header or len(row) < len(header):
        return None
    return row[header.index("fitted")].strip().lower() == "true"


def run(cmd: list[str], log: Path) -> int:
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def human_setting(tag: str) -> tuple[bool, float | None]:
    """(extended in the human search, the human fit's K) for one arm."""
    if (SWEEP / "search" / f"{tag}.nofit").exists():
        return False, None
    for f in (SWEEP / "ka_survival" / f"{tag}.retry.csv", SWEEP / "ka_survival" / f"{tag}.csv"):
        if fitted(f):
            with open(f) as fh:
                header = fh.readline().rstrip("\n").split(",")
                row = fh.readline().rstrip("\n").split(",")
            return True, float(row[header.index("k")])
    return True, None


def one_arm(arm: dict, fa: Path) -> str:
    a, k, c, x = arm["alphabet"], arm["ksize"], str(arm["penalty"]), str(arm["xdrop"])
    tag = f"{a}.k{k}"
    extend, human_k = human_setting(tag)
    done = OUT / "regions" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    idx = OUT / "tmp" / f"{tag}.rocksdb"
    csv = OUT / "tmp" / f"{tag}.csv"
    surv = OUT / "ka_survival" / f"{tag}.csv"
    logs = OUT / "logs"
    shutil.rmtree(idx, ignore_errors=True)
    fit_args = (["--extend-mismatch-penalty", c, "--extend-xdrop", x, "--ka-queries", "200", "--ka-seed", "1",
                 "--ka-survival-out", str(surv)] if extend else ["--ka-queries", "0"])
    rc = run([str(KMERSEEK), "index", "-i", str(fa), "-o", str(idx), "-k", str(k), "-s", "1", "-a", a,
              "--remove-low-complexity", *fit_args], logs / f"{tag}.index.log")
    if rc != 0:
        return f"{tag} INDEX FAILED rc={rc}"
    fit = fitted(surv) if extend else None
    if extend and fit is False:
        m = _COUNTS.search((logs / f"{tag}.index.log").read_text())
        dense = m and min(int(m.group(2)), int(m.group(4))) >= DENSE
        if not dense:
            retry = OUT / "ka_survival" / f"{tag}.retry.csv"
            run([str(KMERSEEK), "calibrate", "-t", str(idx), "--extend-mismatch-penalty", c, "--extend-xdrop", x,
                 "--ka-queries", str(RETRY_QUERIES), "--ka-reference-shuffles", str(RETRY_SHUFFLES),
                 "--ka-survival-out", str(retry)], logs / f"{tag}.calibrate.log")
            fit = fitted(retry)
    if not extend:
        pen = ["--extend-mismatch-penalty", "0"]
    elif fit is True:
        pen = ["--extend-mismatch-penalty", c, "--extend-xdrop", x]
    elif human_k is not None:
        pen = ["--extend-mismatch-penalty", c, "--extend-xdrop", x, "--ka-k", str(human_k)]
    else:
        return f"{tag} NO FIT: extended in the human search but no fit here and no human K"
    rc = run([str(KMERSEEK), "search", "-q", str(fa), "-t", str(idx), "-k", str(k), "-a", a,
              "--threshold", "0", "--min-shared-kmers", "1", "--max-query-pvalue", "1", "--min-region-score", "0",
              *pen, "-o", str(csv)], logs / f"{tag}.search.log")
    if rc != 0:
        return f"{tag} SEARCH FAILED rc={rc}"
    if csv.stat().st_size == 0:
        df = pl.DataFrame(schema={c_: pl.Utf8 for c_ in KEEP})
    else:
        df = pl.read_csv(csv, columns=KEEP, infer_schema_length=0)
    num = [c_ for c_ in KEEP if c_ not in ("query_name", "target_name")]
    df = (df.with_columns([pl.col(c_).cast(pl.Float64, strict=False) for c_ in num])
            .with_columns(pl.col("query_name").str.strip_chars(), pl.col("target_name").str.strip_chars())
            .filter(pl.col("query_name") != pl.col("target_name"))
            .with_columns(pl.lit(a).alias("alphabet"), pl.lit(k).alias("ksize"),
                          pl.lit(arm["bits"]).alias("bits"), pl.lit(extend).alias("extended"),
                          pl.lit(fit is True).alias("fitted_here")))
    df.write_parquet(done)
    csv.unlink(missing_ok=True)
    shutil.rmtree(idx, ignore_errors=True)
    return f"{tag} ok {df.height} regions extended={extend} fit_here={fit}"


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--alphabets", default="")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-bits", type=float, default=0.0, help="only arms at or above this seed information")
    args = ap.parse_args()
    OUT = args.out
    for d in ("regions", "tmp", "logs", "ka_survival"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    fa = OUT / "human_pfam_truth.fa"
    if not fa.exists():
        print(f"{write_fasta(fa)} proteins written to {fa}", file=sys.stderr)
    plan = json.loads((SWEEP / "plan.json").read_text())
    arms = [p for p in plan if (not args.alphabets or p["alphabet"] in args.alphabets.split(","))
            and p["bits"] >= args.min_bits]
    # heaviest (lowest bits) first, so the long arms do not trail at the end
    arms.sort(key=lambda p: p["bits"])
    print(f"{len(arms)} arms", file=sys.stderr, flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda p: one_arm(p, fa), arms):
            print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
