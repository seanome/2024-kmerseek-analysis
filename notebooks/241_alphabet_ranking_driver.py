#!/usr/bin/env python3
"""Search three query proteins against the human proteome with every kmerseek alphabet.

The three cases:
  Ced9 (C. elegans)      -> the right answer is BCL2, a known homolog. The gold standard.
  P66  (B. burgdorferi)  -> the proposed answer is CD47. The reach.
  BHF  (B. schlosseri)   -> no known answer. The application.

One background for all three: GENCODE v49 canonical human proteins (19_732 proteins,
11.35 M residues). BCL2 and CD47 are both in it, so every metric kmerseek writes
(E-value, mean IDF, tf-idf, enrichment, Poisson score) is computed on the same database
for every case and every alphabet.

For each alphabet the k ladder is chosen so a seed carries a fixed number of bits,
k = round(bits / H), where H is the Shannon entropy of the alphabet's class shares
measured on this proteome (not log2 of the class count: gbmr7 has 7 classes but only
2.0 bits per position because one class holds most residues). Every alphabet is
therefore compared at the same seed information, 16 to 44 bits.

Each index is built with region extension at the alphabet's own optimal mismatch
penalty from its measured copy rate kappa (assets/kappa_by_alphabet.tsv in the dark-set
pipeline; notebook 230) and a Karlin-Altschul fit on 200 database sequences. Where the
fit is refused the search still runs and writes E = inf; the ka_survival CSV records
the refusal.

Usage:
  241_alphabet_ranking_driver.py [--alphabets a,b,c] [--workers 2] [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import subprocess
import sys
import time
from pathlib import Path

KMERSEEK = Path("/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek")
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
OUT = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
QUERIES = OUT / "queries.fa"
KAPPA_TSV = Path(
    "/Users/olga/code/2024-kmerseek-analysis/.claude/worktrees/dark-set-v4/"
    "nextflow-runs/invertebrate-dark-set/assets/kappa_by_alphabet.tsv"
)

# The full header of each known partner in the human FASTA (kmerseek pair matches the
# whole header or its first token; these headers have no spaces so the token is the
# whole header).
BCL2 = "ENSP00000329623.3|ENST00000333681.5|ENSG00000171791.14|OTTHUMG00000132791.5|OTTHUMT00000450052.2|BCL2-201|BCL2|239"
CD47 = "ENSP00000355361.5|ENST00000361309.6|ENSG00000196776.18|OTTHUMG00000044216.4|OTTHUMT00000102793.1|CD47-202|CD47|323"
PAIRS = {"Ced9": BCL2, "P66": CD47}

# Class partitions, copied from kmerseek src/rust/alphabets.rs and checked to cover the
# 20 canonical residues exactly once.
CLUSTERS: dict[str, list[str]] = {
    "protein20": list("ACDEFGHIKLMNPQRSTVWY"),
    "dayhoff6": ["C", "AGPST", "DENQ", "HKR", "ILMV", "FWY"],
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
    "hp_thomas_dill2": ["ACFILMVWY", "DEGHKNPQRST"],
    "hp_kyte_doolittle2": ["ACFILMV", "DEGHKNPQRSTWY"],
    "hp_thomas_dill_no_c2": ["AFILMVWY", "CDEGHKNPQRST"],
    "hp_lehninger_c_nonpolar2": ["ACFGILMPVWY", "DEHKNQRST"],
    "hp_lehninger_hpc3": ["AFGILMPVWY", "DEHKNQRST", "C"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "wass14": ["WM", "DI", "P", "C", "AV", "K", "T", "RE", "G", "L", "Y", "SH", "F", "NQ"],
    "hsdm17": ["A", "D", "KE", "R", "N", "T", "S", "Q", "Y", "F", "LIV", "M", "C", "W", "H", "G", "P"],
    "uniprot18": ["A", "R", "N", "D", "C", "Q", "EP", "G", "HL", "I", "K", "M", "F", "S", "T", "W", "Y", "V"],
}
ALPHABETS = list(CLUSTERS)

BITS_TARGETS = [16, 20, 24, 28, 32, 36, 40, 44]
# Arms that earlier work used, kept so their numbers reproduce here.
EXTRA_KS = {
    "hp_lehninger2": [17, 19, 24],  # 17-19: BCL2/Ced9 core (nb 220); 24: BHF's arm
    "protein20": [10, 13],  # P66 controls
    "dayhoff6": [15],  # P66 control
}
K_MIN, K_MAX = 5, 50
# Penalty for the one alphabet with no measured kappa: the value every kmerseek #54
# benchmark used, X-drop 8.
FALLBACK_PENALTY, FALLBACK_XDROP = "2", "8"
# When the 200-query fit at index time is refused, retry with this many queries and
# this many shuffles of each. Above ~30 bits both curves are short; 1000 x 8 rescued
# hp_lehninger2 k=32 in 8 seconds.
RETRY_QUERIES, RETRY_SHUFFLES = 1000, 8


def fitted(survival_csv: Path) -> bool | None:
    """The `fitted` flag from a ka_survival CSV; None when the file is missing or empty."""
    if not survival_csv.exists():
        return None
    with open(survival_csv) as fh:
        header = fh.readline().rstrip("\n").split(",")
        row = fh.readline().rstrip("\n").split(",")
    if "fitted" not in header or len(row) < len(header):
        return None
    return row[header.index("fitted")].strip().lower() == "true"


def residue_counts() -> collections.Counter:
    cnt: collections.Counter = collections.Counter()
    with open(HUMAN) as fh:
        for line in fh:
            if not line.startswith(">"):
                cnt.update(line.strip())
    return collections.Counter({r: n for r, n in cnt.items() if r in "ACDEFGHIKLMNPQRSTVWY"})


def bits_per_position(cnt: collections.Counter) -> dict[str, float]:
    tot = sum(cnt.values())
    out = {}
    for a, cl in CLUSTERS.items():
        assert sorted("".join(cl)) == sorted("ACDEFGHIKLMNPQRSTVWY"), a
        shares = [sum(cnt[r] for r in c) / tot for c in cl]
        out[a] = -sum(p * math.log2(p) for p in shares if p > 0)
    return out


def ladder(alphabet: str, h: float) -> list[int]:
    ks = {round(b / h) for b in BITS_TARGETS} | set(EXTRA_KS.get(alphabet, []))
    return sorted(k for k in ks if K_MIN <= k <= K_MAX)


def read_kappa() -> dict[str, tuple[str, str]]:
    """alphabet -> (c_opt, xdrop_4c) as the strings the pipeline puts on the command line."""
    out = {}
    with open(KAPPA_TSV) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("alphabet") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            out[f[0]] = (f[6], f[7])
    return out


def penalty_for(alphabet: str, kappa: dict[str, tuple[str, str]]) -> tuple[str, str]:
    return kappa.get(alphabet, (FALLBACK_PENALTY, FALLBACK_XDROP))


def run(cmd: list[str], log: Path) -> tuple[int, float]:
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def one_arm(alphabet: str, k: int, c: str, x: str, dry: bool) -> dict:
    tag = f"{alphabet}.k{k}"
    idx = OUT / "idx" / f"human.{tag}.rocksdb"
    search_csv = OUT / "search" / f"{tag}.csv"
    survival = OUT / "ka_survival" / f"{tag}.csv"
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    rec = {"alphabet": alphabet, "ksize": k, "penalty": c, "xdrop": x, "tag": tag}

    idx_cmd = [
        str(KMERSEEK), "index", "-i", str(HUMAN), "-o", str(idx), "-k", str(k), "-s", "1",
        "-a", alphabet, "--remove-low-complexity",
        "--extend-mismatch-penalty", c, "--extend-xdrop", x,
        "--ka-queries", "200", "--ka-survival-out", str(survival),
    ]
    search_cmd = [
        str(KMERSEEK), "search", "-q", str(QUERIES), "-t", str(idx), "-k", str(k), "-a", alphabet,
        "--threshold", "0", "--min-shared-kmers", "1",
        "--max-query-pvalue", "1", "--min-region-score", "0",
        "--extend-mismatch-penalty", c, "--extend-xdrop", x,
        "-o", str(search_csv),
    ]
    if dry:
        print(" ".join(idx_cmd))
        print(" ".join(search_cmd))
        return rec

    if not (idx / "CURRENT").exists():
        rc, dt = run(idx_cmd, logs / f"{tag}.index.log")
        rec["index_rc"], rec["index_s"] = rc, round(dt, 1)
        if rc != 0:
            return rec
    else:
        rec["index_rc"], rec["index_s"] = 0, 0.0

    # The Karlin-Altschul fit is refused when either curve runs out of score bins, which
    # happens above about 30 bits per seed. First retry with more calibration queries
    # and more shuffles of each (the two knobs that fill the two curves); if still
    # refused, search with exact regions only so every other metric is still computed,
    # and mark the arm as having no E-value.
    fit = fitted(survival)
    rec["fit_first"] = fit
    if fit is False:
        retry_out = OUT / "ka_survival" / f"{tag}.retry.csv"
        cal_cmd = [
            str(KMERSEEK), "calibrate", "-t", str(idx),
            "--extend-mismatch-penalty", c, "--extend-xdrop", x,
            "--ka-queries", str(RETRY_QUERIES), "--ka-reference-shuffles", str(RETRY_SHUFFLES),
            "--ka-survival-out", str(retry_out),
        ]
        rc, dt = run(cal_cmd, logs / f"{tag}.calibrate.log")
        rec["calibrate_rc"], rec["calibrate_s"] = rc, round(dt, 1)
        fit = fitted(retry_out)
    rec["fitted"] = fit
    if fit is not True:
        # Exact regions only: no extension, so no E-value, but IDF, tf-idf, enrichment
        # and the Poisson score are all still written.
        search_cmd = [a for a in search_cmd if a not in ("--extend-mismatch-penalty", "--extend-xdrop", c, x)]
        search_cmd += ["--extend-mismatch-penalty", "0"]
        (OUT / "search" / f"{tag}.nofit").write_text("no Karlin-Altschul fit; searched with exact regions\n")

    if not search_csv.exists():
        rc, dt = run(search_cmd, logs / f"{tag}.search.log")
        rec["search_rc"], rec["search_s"] = rc, round(dt, 1)
    else:
        rec["search_rc"], rec["search_s"] = 0, 0.0

    # Pairwise layer: no database, just the shared k-mers and regions between the two.
    for qname, target in PAIRS.items():
        pj = OUT / "pair" / f"{tag}.{qname}.json"
        if pj.exists():
            continue
        cmd = [
            str(KMERSEEK), "pair", "-q", str(QUERIES), "--query-name", qname,
            "-t", str(HUMAN), "--target-name", target, "-k", str(k), "-a", alphabet,
            "-o", str(pj),
        ]
        rc, _ = run(cmd, logs / f"{tag}.pair.{qname}.log")
        rec[f"pair_{qname}_rc"] = rc
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabets", default=",".join(ALPHABETS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="print the ladder and exit")
    args = ap.parse_args()

    cnt = residue_counts()
    hbits = bits_per_position(cnt)
    kappa = read_kappa()
    alphabets = [a for a in args.alphabets.split(",") if a]

    plan = []
    for a in alphabets:
        c, x = penalty_for(a, kappa)
        for k in ladder(a, hbits[a]):
            plan.append((a, k, c, x))
    (OUT / "plan.json").write_text(json.dumps(
        [{"alphabet": a, "ksize": k, "bits": round(k * hbits[a], 2), "penalty": c, "xdrop": x}
         for a, k, c, x in plan], indent=1))
    (OUT / "bits_per_position.json").write_text(json.dumps(hbits, indent=1))
    print(f"{len(plan)} arms over {len(alphabets)} alphabets", file=sys.stderr)
    if args.plan_only:
        for a, k, c, x in plan:
            print(f"{a:26s} k={k:2d}  bits={k * hbits[a]:5.1f}  C={c} X={x}")
        return

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_arm, a, k, c, x, args.dry_run): (a, k) for a, k, c, x in plan}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            results.append(rec)
            print(json.dumps(rec), file=sys.stderr, flush=True)
            with open(OUT / "runs.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(f"done: {len(results)} arms", file=sys.stderr)


if __name__ == "__main__":
    main()
