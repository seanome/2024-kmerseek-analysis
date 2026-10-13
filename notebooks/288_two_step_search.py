#!/usr/bin/env python3
"""Notebook 288: Ced9 -> BCL2 and P66 -> CD47 as pairwise questions, and a two-step search.

kmerseek 982a055 computes a region's E-value as (src/rust/search.rs:1708)

    E = K * m * n * exp(-lambda * S)

with m the query's length in residues (search.rs:1690), n the database's k-mer count
(search.rs:1691, `db_n_kmers`, which search.rs:1240 sets to the sum over every indexed k-mer
of the number of targets holding it, so the sum over targets of each target's distinct
k-mers), K and the lambda scale from the index's Karlin-Altschul fit, and S = matches -
penalty * mismatches over the region. Nothing else in E depends on the database, and the
region itself (start, end, mismatches) comes from the query and that one target only. So the
same region scored against a smaller database has E smaller by n_small / n_human, as long as
K and the lambda scale are kept from the human fit.

Two questions, with the decision rules fixed before any of this ran (also in the notebook's
first cell):

Q1, pairwise. Ced9 and BCL2 are a textbook pair, so "is Ced9 related to BCL2?" can be asked
before any search. Search Ced9 against an index holding BCL2 alone, and against one holding
the 14 human BCL-2 family proteins (HGNC gene group 1057). Same for P66 against CD47 alone.
Each of the 70 settings of notebook 241 with a human Karlin-Altschul fit. The real query's
best E-value over the 70 settings is compared with the same number for 100 dipeptide
shuffles of the query (same length, same count of every adjacent residue pair).
  Pass: best pairwise E < 0.05, and empirical p = (1 + shuffles with best E <= the real
  query's) / 101 < 0.05.

Q2, two-step. Step 1 picks candidates from the proteome search, without the answer:
  rule A50: the 50 human proteins with the lowest best E-value over the 70 settings;
  rule B:   every human protein with a region at E < 10 (BLAST's default reporting cut)
            at 3 or more of the 70 settings.
Step 2 re-scores only the candidates, with n set to the candidates' k-mer count at each
setting. The shuffles go through the same two steps.
  Pass: the partner is a candidate, its best step-2 E < 0.05, and at most 4 of 100 shuffles
  (under 5%) have any candidate with step-2 E < 0.05.

Steps (each resumes; run in a Slurm job, not on a login node):
  pairwise   small indexes (BCL2, CD47, BCL-2 family), 202 queries searched (minutes)
  indexes    the 70 human indexes, no fit (--ka-queries 0); hours
  search     202 queries against the 70 human indexes, reduced per chunk; hours
  twostep    step 1, k-mer counts, step 2, and kmerseek on Ced9's and P66's real candidate
             sets as a check of the step-2 arithmetic (minutes)

Paths come from the environment:
  NB288_DIR       output folder (default $SCRATCH/288-two-step-search)
  NB241_KMERSEEK  kmerseek 982a055 binary
  NB241_DIR       notebook 241's run folder: queries.fa, search/, ka_survival/, and
                  null_bhf_dipeptide/arms.json (notebook 256's copy of 241's 152 settings)
  NB241_HUMAN     GENCODE v49 canonical translations (md5 d0774e68fbf8aa38cc4c2bfbfe6280df)

Usage: 288_two_step_search.py pairwise|indexes|search|twostep [--workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ENV = os.environ
SCRATCH = Path(ENV.get("SCRATCH", "/tmp"))
OUT = Path(ENV.get("NB288_DIR", SCRATCH / "288-two-step-search"))
NB241 = Path(ENV.get("NB241_DIR", SCRATCH / "241-alphabet-ranking"))
KMERSEEK = Path(
    ENV.get(
        "NB241_KMERSEEK",
        SCRATCH / "241-alphabet-ranking-kmerseek/target/release/kmerseek",
    )
)
HUMAN = Path(
    ENV.get(
        "NB241_HUMAN",
        SCRATCH / "data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
ARMS = NB241 / "null_bhf_dipeptide/arms.json"
FAMILY_JSON = HERE.parent / "data/288/hgnc_gene_group_1057_bcl2_family.json"

PARTNER = {"Ced9": "BCL2", "P66": "CD47"}
N_SHUFFLES = 100
SEED = 288
TOP_N = 50  # rule A
CALL_E = 10.0  # rule B: a "call" is a region at E < 10
MIN_SETTINGS = 3  # rule B: calls at this many settings or more
PASS_E = 0.05
# The proteome search keeps, per query, setting and target, the best region when its
# E-value is at most this, so rule B (E < 10) and the top 50 are always inside it.
KEEP_E = 1e4
HEAVY_BITS = (
    20.5  # settings at or below this many seed bits search 25 queries at a time
)
CHUNK = 25

# The class partitions of notebook 241's driver (copied there from kmerseek
# src/rust/alphabets.rs), used only to count each protein's distinct k-mers for step 2.
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
    "wass14": [
        "WM",
        "DI",
        "P",
        "C",
        "AV",
        "K",
        "T",
        "RE",
        "G",
        "L",
        "Y",
        "SH",
        "F",
        "NQ",
    ],
    "hsdm17": [
        "A",
        "D",
        "KE",
        "R",
        "N",
        "T",
        "S",
        "Q",
        "Y",
        "F",
        "LIV",
        "M",
        "C",
        "W",
        "H",
        "G",
        "P",
    ],
    "uniprot18": [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "EP",
        "G",
        "HL",
        "I",
        "K",
        "M",
        "F",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ],
}
# kmerseek drops a k-mer whose encoded letters are all the same for every alphabet with a
# class table, and for hp_lehninger2 (sketch.rs:430); protein20 and dayhoff6 drop only raw
# runs of one residue (sketch.rs:483).
ENCODED_RUNS_DROPPED = set(CLUSTERS) - {"protein20", "dayhoff6"}

REGION_COLS = [
    "query_name",
    "target_name",
    "region_search_space",
    "db_n_targets",
    "db_n_kmers",
    "region_start",
    "region_end",
    "region_subseq",
    "target_start",
    "target_end",
    "target_subseq",
    "moltype_seq",
    "region_length",
    "region_n_mismatches",
    "region_ka_u",
    "region_ka_lambda",
    "region_ka_bits",
    "region_evalue",
]


# ---------------------------------------------------------------- inputs


def read_fasta(p: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = []
        elif name is not None:
            out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def write_fasta(p: Path, recs: list[tuple[str, str]]) -> None:
    p.write_text("".join(f">{h}\n{s}\n" for h, s in recs))


def gene_of(header: str) -> str:
    return header.split("|")[6]


def load_dipeptide_shuffle():
    """Notebook 256's dipeptide shuffle, as committed in notebook 284's script."""
    spec = importlib.util.spec_from_file_location(
        "nb284", HERE / "284_shuffle_agreement.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dipeptide_shuffle


def make_queries() -> list[tuple[str, str]]:
    """Ced9 and P66 from notebook 241's queries.fa, each followed by 100 dipeptide shuffles."""
    real = read_fasta(NB241 / "queries.fa")
    shuffle = load_dipeptide_shuffle()
    rng = random.Random(SEED)
    qs = []
    for case in PARTNER:
        qs.append((case, real[case]))
        for i in range(N_SHUFFLES):
            s = shuffle(real[case], rng)
            assert Counter(zip(s, s[1:])) == Counter(zip(real[case], real[case][1:]))
            qs.append((f"{case}_shuf{i:03d}", s))
    return qs


def settings() -> list[dict]:
    """The 70 of notebook 241's 152 settings that have a human Karlin-Altschul fit, with
    that fit's K and lambda scale and the human index's k-mer count n_human."""
    path = OUT / "settings.json"
    if path.exists():
        return json.loads(path.read_text())
    out = []
    for a in json.loads(ARMS.read_text()):
        if a["nofit"]:
            continue
        tag = f"{a['alphabet']}.k{a['k']}"
        fit = None
        # notebook 241 retried a refused fit with kmerseek calibrate; the retry is the fit
        # its search used when the first one was refused
        for name in (f"{tag}.retry.csv", f"{tag}.csv"):
            f = NB241 / "ka_survival" / name
            if not f.exists():
                continue
            row = next(csv.DictReader(open(f)))
            if row["fitted"].strip().lower() == "true":
                fit = row
                break
        assert fit is not None, f"{tag}: arms.json says fitted, no fitted survival file"
        with open(NB241 / "search" / f"{tag}.csv") as fh:
            row = next(csv.DictReader(fh))
        out.append(
            dict(
                tag=tag,
                alphabet=a["alphabet"],
                k=a["k"],
                bits=a["bits"],
                penalty=a["penalty"],
                xdrop=a["xdrop"],
                ka_k=float(fit["k"]),
                lambda_scale=float(fit["slope"]),
                fit_file=f.name,
                n_human=int(row["db_n_kmers"]),
                n_human_targets=int(row["db_n_targets"]),
            )
        )
    assert len(out) == 70, len(out)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- kmerseek


def run(cmd: list[str], log: Path) -> int:
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def build_index(fa: Path, idx: Path, s: dict, fit: bool, log: Path) -> int:
    """Notebook 241's index command; with fit=False, --ka-queries 0 (no fit)."""
    cmd = [
        str(KMERSEEK),
        "index",
        "-i",
        str(fa),
        "-o",
        str(idx),
        "-k",
        str(s["k"]),
        "-s",
        "1",
        "-a",
        s["alphabet"],
        "--remove-low-complexity",
        "--extend-mismatch-penalty",
        s["penalty"],
        "--extend-xdrop",
        s["xdrop"],
    ]
    if fit:
        cmd += [
            "--ka-queries",
            "200",
            "--ka-survival-out",
            str(idx) + ".ka_survival.csv",
        ]
    else:
        cmd += ["--ka-queries", "0"]
    return run(cmd, log)


def search(qfa: Path, idx: Path, s: dict, out_csv: Path, log: Path) -> int:
    """Notebook 241's search command, with the human fit's K given as --ka-k."""
    cmd = [
        str(KMERSEEK),
        "search",
        "-q",
        str(qfa),
        "-t",
        str(idx),
        "-k",
        str(s["k"]),
        "-a",
        s["alphabet"],
        "--threshold",
        "0",
        "--min-shared-kmers",
        "1",
        "--max-query-pvalue",
        "1",
        "--min-region-score",
        "0",
        "--extend-mismatch-penalty",
        s["penalty"],
        "--extend-xdrop",
        s["xdrop"],
        "--ka-k",
        repr(s["ka_k"]),
        "-o",
        str(out_csv),
    ]
    return run(cmd, log)


def read_regions(out_csv: Path, s: dict, scale: float | None = None) -> pl.DataFrame:
    """kmerseek's region rows with the E-value under the human fit.

    --ka-k sets the lambda scale to 1 (search.rs, resolve_ka), so region_ka_lambda here is
    the closed-form lambda. e_value_human_fit puts the human fit's lambda scale back:
    K * m * n * exp(-scale * lambda * S), with n this index's own k-mer count. Rows from
    notebook 241's own search already carry the scaled lambda: pass scale=1.0 for those.
    """
    scale = s["lambda_scale"] if scale is None else scale
    if out_csv.stat().st_size == 0:
        return pl.DataFrame()
    c = float(s["penalty"])
    d = (
        pl.read_csv(out_csv, columns=REGION_COLS, infer_schema_length=0)
        .filter(pl.col("region_start").is_not_null() & (pl.col("region_start") != ""))
        .with_columns(
            pl.col("query_name").str.strip_chars(),
            *[
                pl.col(x).cast(pl.Float64)
                for x in REGION_COLS
                if x.startswith(("region_", "target_s", "target_e", "db_"))
                and not x.endswith("subseq")
            ],
        )
        .with_columns(
            gene=pl.col("target_name").str.split("|").list.get(6),
            target_length=pl.col("target_name")
            .str.split("|")
            .list.get(7)
            .cast(pl.Int64),
            query_length=pl.col("region_search_space") + s["k"] - 1,
            score=(pl.col("region_length") - pl.col("region_n_mismatches"))
            - c * pl.col("region_n_mismatches"),
        )
        .with_columns(
            e_value_human_fit=pl.when(pl.col("region_ka_lambda") > 0)
            .then(
                s["ka_k"]
                * pl.col("query_length")
                * pl.col("db_n_kmers")
                * (-scale * pl.col("region_ka_lambda") * pl.col("score")).exp()
            )
            .otherwise(float("inf")),
            setting=pl.lit(s["tag"]),
            alphabet=pl.lit(s["alphabet"]),
            k=pl.lit(s["k"]),
        )
    )
    return d


# ---------------------------------------------------------------- step: pairwise


def pairwise_one(s: dict, db: str) -> str:
    done = OUT / "pairwise" / f"{db}.{s['tag']}.parquet"
    if done.exists():
        return f"{db} {s['tag']} cached"
    work = OUT / "pairwise_work"
    idx = work / f"{db}.{s['tag']}.rocksdb"
    fa = OUT / "db" / f"{db}.fa"
    fit_note = "fit attempted"
    if not (idx / "CURRENT").exists():
        rc = build_index(fa, idx, s, True, work / f"{db}.{s['tag']}.index.log")
        if rc != 0:
            fit_note = f"index with fit failed (rc {rc}); rebuilt with --ka-queries 0"
            shutil.rmtree(idx, ignore_errors=True)
            rc = build_index(
                fa, idx, s, False, work / f"{db}.{s['tag']}.index_nofit.log"
            )
            if rc != 0:
                return f"{db} {s['tag']} FAILED index rc={rc}"
    surv = Path(str(idx) + ".ka_survival.csv")
    fitted_here = None
    if surv.exists() and surv.stat().st_size:
        row = next(csv.DictReader(open(surv)), None)
        fitted_here = None if row is None else row["fitted"].strip().lower() == "true"
    out_csv = work / f"{db}.{s['tag']}.csv"
    rc = search(
        OUT / "queries.fa", idx, s, out_csv, work / f"{db}.{s['tag']}.search.log"
    )
    if rc != 0:
        return f"{db} {s['tag']} FAILED search rc={rc}"
    d = read_regions(out_csv, s).with_columns(
        database=pl.lit(db), fitted_here=pl.lit(fitted_here), fit_note=pl.lit(fit_note)
    )
    d.write_parquet(done)
    out_csv.unlink()
    return f"{db} {s['tag']} ok {d.height} regions, small-index fit: {fitted_here}"


def step_pairwise(workers: int) -> None:
    for x in ("pairwise", "pairwise_work", "db"):
        (OUT / x).mkdir(parents=True, exist_ok=True)
    sets = settings()
    qs = make_queries()
    write_fasta(OUT / "queries.fa", qs)
    human = read_fasta(HUMAN)
    by_gene: dict[str, list[str]] = {}
    for h in human:
        by_gene.setdefault(gene_of(h), []).append(h)
    family = sorted(
        d["symbol"] for d in json.loads(FAMILY_JSON.read_text())["response"]["docs"]
    )
    dbs = {"BCL2": ["BCL2"], "CD47": ["CD47"], "BCL2_family": family}
    for db, genes in dbs.items():
        recs = []
        for g in genes:
            hs = by_gene.get(g, [])
            assert len(hs) == 1, f"{g}: {len(hs)} canonical entries in {HUMAN}"
            recs.append((hs[0], human[hs[0]]))
        write_fasta(OUT / "db" / f"{db}.fa", recs)
    jobs = [(s, db) for s in sets for db in dbs]
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(pairwise_one, *zip(*jobs)):
            print(msg, file=sys.stderr, flush=True)
            if "FAILED" in msg:
                failed.append(msg)
    if failed:
        sys.exit(f"{len(failed)} pairwise jobs failed; rerun to retry them")
    summarise_pairwise(sets)


def human_rows(sets: list[dict]) -> pl.DataFrame:
    """Ced9's and P66's region rows on the BCL-2 family and CD47 from notebook 241's
    proteome search, to check the pairwise numbers against."""
    family = {
        d["symbol"] for d in json.loads(FAMILY_JSON.read_text())["response"]["docs"]
    }
    genes = family | {"CD47"}
    frames = []
    for s in sets:
        d = pl.read_csv(
            NB241 / "search" / f"{s['tag']}.csv",
            columns=REGION_COLS,
            infer_schema_length=0,
        ).filter(pl.col("target_name").str.split("|").list.get(6).is_in(list(genes)))
        p = OUT / "tmp_241.csv"
        d.write_csv(p)
        frames.append(
            read_regions(p, s, scale=1.0).filter(
                pl.col("query_name").is_in(list(PARTNER))
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


def summarise_pairwise(sets: list[dict]) -> None:
    d = pl.read_parquet(sorted((OUT / "pairwise").glob("*.parquet")))
    d.write_parquet(OUT / "288_pairwise_regions.parquet")
    keys = [
        "query_name",
        "gene",
        "setting",
        "region_start",
        "region_end",
        "target_start",
        "target_end",
    ]
    h = human_rows(sets).select(
        *keys,
        pl.col("region_evalue").alias("e_value_241"),
        pl.col("db_n_kmers").alias("n_human"),
        pl.col("e_value_human_fit").alias("e_value_241_recomputed"),
    )
    real = d.filter(pl.col("query_name").is_in(list(PARTNER))).join(
        h, on=keys, how="left"
    )
    real.drop("region_subseq", "target_subseq", "moltype_seq").write_csv(
        OUT / "288_pairwise_vs_241.csv"
    )
    n_unmatched = real.filter(pl.col("e_value_241").is_null()).height
    print(
        f"pairwise rows of Ced9/P66 with no same region in 241: {n_unmatched}",
        file=sys.stderr,
    )
    # Best region per query, database and partner gene, over the 70 settings.
    best = (
        d.with_columns(case=pl.col("query_name").str.split("_shuf").list.get(0))
        .filter(
            (
                (pl.col("case") == "Ced9")
                & pl.col("database").is_in(["BCL2", "BCL2_family"])
            )
            | ((pl.col("case") == "P66") & (pl.col("database") == "CD47"))
        )
        .sort("e_value_human_fit", "setting", "gene", "region_start", "target_start")
        .group_by("query_name", "database", maintain_order=True)
        .first()
    )
    best.drop("moltype_seq").write_csv(OUT / "288_pairwise_best.csv")


# ---------------------------------------------------------------- step: indexes


def human_index(s: dict) -> str:
    idx = OUT / "idx" / f"human.{s['tag']}.rocksdb"
    if (idx / "CURRENT").exists():
        return f"{s['tag']} cached"
    shutil.rmtree(idx, ignore_errors=True)
    rc = build_index(HUMAN, idx, s, False, OUT / "logs" / f"{s['tag']}.index.log")
    return f"{s['tag']} {'ok' if rc == 0 else f'FAILED rc={rc}'}"


def step_indexes(workers: int) -> None:
    for x in ("idx", "logs"):
        (OUT / x).mkdir(parents=True, exist_ok=True)
    sets = sorted(settings(), key=lambda s: s["bits"])  # slowest (fewest bits) first
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(human_index, sets):
            print(msg, file=sys.stderr, flush=True)
            if "FAILED" in msg:
                failed.append(msg)
    if failed:
        sys.exit(f"{len(failed)} indexes failed; rerun to retry them")


# ---------------------------------------------------------------- step: search


def search_chunk(tag: str, chunk: list[tuple[str, str]], s: dict) -> str:
    done = OUT / "proteome" / f"{tag}.parquet"
    if done.exists():
        return f"{tag} cached"
    fa, out_csv = OUT / "tmp" / f"{tag}.fa", OUT / "tmp" / f"{tag}.csv"
    write_fasta(fa, chunk)
    idx = OUT / "idx" / f"human.{s['tag']}.rocksdb"
    rc = search(fa, idx, s, out_csv, OUT / "logs" / f"{tag}.search.log")
    if rc != 0:
        return f"{tag} FAILED rc={rc}"
    family = {
        d["symbol"] for d in json.loads(FAMILY_JSON.read_text())["response"]["docs"]
    }
    d = read_regions(out_csv, s)
    # Best region per query and target; kept when E <= KEEP_E, or the target is a BCL-2
    # family protein or CD47 (the partners, kept at any E).
    best = (
        d.sort("e_value_human_fit", "region_start", "target_start")
        .group_by("query_name", "target_name", maintain_order=True)
        .first()
        .filter(
            (pl.col("e_value_human_fit") <= KEEP_E)
            | pl.col("gene").is_in(list(family | {"CD47"}))
        )
        .select(
            "query_name",
            "target_name",
            "gene",
            "target_length",
            "setting",
            "alphabet",
            "k",
            "db_n_kmers",
            "query_length",
            "region_start",
            "region_end",
            "target_start",
            "target_end",
            "region_n_mismatches",
            "region_evalue",
            "e_value_human_fit",
            "region_subseq",
            "target_subseq",
        )
    )
    best.write_parquet(done)
    out_csv.unlink()
    fa.unlink()
    return f"{tag} ok {d.height} regions, {best.height} kept"


def step_search(workers: int) -> None:
    for x in ("proteome", "tmp", "logs"):
        (OUT / x).mkdir(parents=True, exist_ok=True)
    sets = settings()
    qs = list(read_fasta(OUT / "queries.fa").items())  # written by the pairwise step
    assert len(qs) == 2 * (N_SHUFFLES + 1), len(qs)
    missing = [
        s["tag"]
        for s in sets
        if not (OUT / "idx" / f"human.{s['tag']}.rocksdb" / "CURRENT").exists()
    ]
    if missing:
        sys.exit(
            f"{len(missing)} human indexes missing (first {missing[0]}); run `indexes`"
        )
    jobs = []
    for s in sorted(sets, key=lambda s: s["bits"]):
        if s["bits"] <= HEAVY_BITS:
            jobs += [
                (f"{s['tag']}.c{i:02d}", qs[i * CHUNK : (i + 1) * CHUNK], s)
                for i in range(-(-len(qs) // CHUNK))
            ]
        else:
            jobs.append((f"{s['tag']}.all", qs, s))
    print(
        f"{len(qs)} queries, {len(jobs)} chunks over {len(sets)} settings",
        file=sys.stderr,
    )
    failed = []
    with cf.ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(search_chunk, *zip(*jobs)):
            print(msg, file=sys.stderr, flush=True)
            if "FAILED" in msg:
                failed.append(msg)
    if failed:
        sys.exit(f"{len(failed)} chunks failed; rerun to retry them")


# ---------------------------------------------------------------- step: twostep


def encoder(alphabet: str) -> dict[str, str]:
    return {
        aa: chr(ord("a") + i) for i, cls in enumerate(CLUSTERS[alphabet]) for aa in cls
    }


def distinct_kmers(seq: str, alphabet: str, k: int) -> int:
    """A protein's distinct k-mers as kmerseek indexes them with --remove-low-complexity:
    windows that are one residue repeated are dropped, and for alphabets with a class
    table (and hp_lehninger2) so are windows whose encoded letters are all the same.
    Residues outside the 20 keep their own letter, as in kmerseek's table lookup."""
    enc = encoder(alphabet)
    e = "".join(enc.get(c, c) for c in seq.upper())
    drop_encoded = alphabet in ENCODED_RUNS_DROPPED
    seen = set()
    for i in range(len(seq) - k + 1):
        raw = seq[i : i + k]
        if raw.count(raw[0]) == k:
            continue
        w = e[i : i + k]
        if drop_encoded and w.count(w[0]) == k:
            continue
        seen.add(w)
    return len(seen)


def kmer_counts(s: dict) -> pl.DataFrame:
    human = read_fasta(HUMAN)
    return pl.DataFrame(
        {
            "target_name": list(human),
            "setting": s["tag"],
            "n_kmers": [
                distinct_kmers(q, s["alphabet"], s["k"]) for q in human.values()
            ],
        }
    )


def step_twostep(workers: int) -> None:
    sets = settings()
    by_tag = {s["tag"]: s for s in sets}
    counts_path = OUT / "288_human_kmer_counts.parquet"
    if not counts_path.exists():
        with cf.ProcessPoolExecutor(workers) as ex:
            counts = pl.concat(list(ex.map(kmer_counts, sets)))
        counts.write_parquet(counts_path)
    counts = pl.read_parquet(counts_path)
    # Check: the counts summed over the proteome against each human index's n.
    check = (
        counts.group_by("setting")
        .agg(pl.col("n_kmers").sum().alias("n_counted"))
        .join(
            pl.DataFrame(
                [{"setting": s["tag"], "n_human": s["n_human"]} for s in sets]
            ),
            on="setting",
        )
        .with_columns(
            rel_diff=(pl.col("n_counted") - pl.col("n_human")) / pl.col("n_human")
        )
    )
    check.sort("setting").write_csv(OUT / "288_kmer_count_check.csv")
    print(check.select(pl.col("rel_diff").abs().max()), file=sys.stderr)

    p = pl.read_parquet(sorted((OUT / "proteome").glob("*.parquet")))
    p = p.with_columns(case=pl.col("query_name").str.split("_shuf").list.get(0))
    # Step 1, best E per query and protein over the 70 settings
    per_prot = p.group_by("query_name", "case", "target_name", "gene").agg(
        best_e=pl.col("e_value_human_fit").min(),
        n_settings_called=(pl.col("e_value_human_fit") < CALL_E).sum(),
    )
    rule_a = (
        per_prot.sort("best_e", "target_name")
        .with_columns(rank=pl.col("best_e").rank("ordinal").over("query_name"))
        .filter(pl.col("rank") <= TOP_N)
        .with_columns(rule=pl.lit(f"A{TOP_N}"))
    )
    rule_b = per_prot.filter(pl.col("n_settings_called") >= MIN_SETTINGS).with_columns(
        rule=pl.lit("B"), rank=pl.lit(None, pl.UInt32)
    )
    cand = pl.concat([rule_a, rule_b], how="diagonal_relaxed")
    cand.drop("best_e").write_parquet(OUT / "288_candidates.parquet")
    # n of each query's candidate set at each setting
    n_cand = (
        cand.select("query_name", "rule", "target_name")
        .join(counts, on="target_name")
        .group_by("query_name", "rule", "setting")
        .agg(n_candidates=pl.col("n_kmers").sum(), n_proteins=pl.len())
    )
    # Step 2: each candidate's region at each setting, E scaled from n_human to n_candidates
    step2 = (
        p.join(
            cand.select("query_name", "rule", "target_name"),
            on=["query_name", "target_name"],
        )
        .join(n_cand, on=["query_name", "rule", "setting"])
        .with_columns(
            e_step2=pl.col("e_value_human_fit")
            * pl.col("n_candidates")
            / pl.col("db_n_kmers")
        )
    )
    step2.write_parquet(OUT / "288_step2_regions.parquet")
    best2 = (
        step2.sort("e_step2", "setting", "target_name")
        .group_by("query_name", "case", "rule", maintain_order=True)
        .first()
    )
    partner2 = (
        step2.filter(pl.col("gene") == pl.col("case").replace_strict(PARTNER))
        .sort("e_step2", "setting")
        .group_by("query_name", "case", "rule", maintain_order=True)
        .first()
    )
    sizes = cand.group_by("query_name", "rule").agg(n_candidate_proteins=pl.len())
    partner_in = (
        cand.with_columns(case=pl.col("query_name").str.split("_shuf").list.get(0))
        .filter(pl.col("gene") == pl.col("case").replace_strict(PARTNER))
        .select(
            "query_name",
            "rule",
            partner_rank_step1=pl.col("rank"),
            partner_is_candidate=pl.lit(True),
        )
    )
    summ = (
        best2.select(
            "query_name",
            "case",
            "rule",
            best_gene="gene",
            best_setting="setting",
            best_e_step2="e_step2",
            best_e_step1="e_value_human_fit",
        )
        .join(
            partner2.select(
                "query_name",
                "rule",
                partner_setting="setting",
                partner_e_step2="e_step2",
                partner_e_step1="e_value_human_fit",
            ),
            on=["query_name", "rule"],
            how="left",
        )
        .join(sizes, on=["query_name", "rule"], how="left")
        .join(partner_in, on=["query_name", "rule"], how="left")
        .with_columns(pl.col("partner_is_candidate").fill_null(False))
        .sort("case", "rule", "query_name")
    )
    summ.write_csv(OUT / "288_twostep_summary.csv")
    # Partner rank in step 1 for every query (the partner may sit below the top 50)
    ranks = per_prot.with_columns(
        rank=pl.col("best_e").rank("min").over("query_name"),
        n_ranked=pl.len().over("query_name"),
    ).filter(pl.col("gene") == pl.col("case").replace_strict(PARTNER))
    ranks.write_csv(OUT / "288_partner_rank_step1.csv")

    # Check of the step-2 arithmetic: kmerseek itself on the real queries' candidate sets.
    human = read_fasta(HUMAN)
    rows = []
    work = OUT / "candidate_check"
    work.mkdir(exist_ok=True)
    qfa = work / "real.fa"
    allq = read_fasta(OUT / "queries.fa")
    write_fasta(qfa, [(q, allq[q]) for q in PARTNER])
    for (q, rule), g in cand.filter(pl.col("query_name").is_in(list(PARTNER))).group_by(
        "query_name", "rule"
    ):
        fa = work / f"{q}.{rule}.fa"
        write_fasta(fa, [(t, human[t]) for t in g["target_name"].sort()])
        for s in sets:
            idx = work / f"{q}.{rule}.{s['tag']}.rocksdb"
            if not (idx / "CURRENT").exists():
                assert build_index(fa, idx, s, False, work / f"{idx.name}.log") == 0
            out_csv = work / f"{q}.{rule}.{s['tag']}.csv"
            assert search(qfa, idx, s, out_csv, work / f"{idx.name}.search.log") == 0
            d = read_regions(out_csv, s).filter(pl.col("query_name") == q)
            if d.height:
                rows.append(
                    d.sort("e_value_human_fit")
                    .group_by("target_name", maintain_order=True)
                    .first()
                    .select(
                        "query_name",
                        "target_name",
                        "setting",
                        "db_n_kmers",
                        "region_start",
                        "target_start",
                        e_kmerseek="e_value_human_fit",
                    )
                    .with_columns(rule=pl.lit(rule))
                )
            shutil.rmtree(idx)
    chk = pl.concat(rows).join(
        step2.select(
            "query_name",
            "rule",
            "target_name",
            "setting",
            "region_start",
            "target_start",
            "e_step2",
            "n_candidates",
        ),
        on=["query_name", "rule", "target_name", "setting"],
        how="left",
        suffix="_step2",
    )
    chk.write_csv(OUT / "288_step2_check.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairwise", "indexes", "search", "twostep"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {
        "pairwise": step_pairwise,
        "indexes": step_indexes,
        "search": step_search,
        "twostep": step_twostep,
    }[args.step](args.workers)


if __name__ == "__main__":
    main()
