#!/usr/bin/env python3
"""Notebook 286: does kmerseek reach BCL2 from Ced9 through a human protein in between?

PSI-BLAST and jackhmmer reach a distant homolog by searching again with what the first
search found. Here the same two steps are done with kmerseek, at notebook 241's 152
settings (one alphabet at one k-mer size each), against the GENCODE v49 canonical human
proteome:

  step 1   Ced9 (and P66) against the human proteome. Every human protein X with a region
           at E < 100 at any setting is an intermediate.
  step 2   every human protein against the human proteome, so that each X's own hits are
           known. A region X -> Y with E < 100 at any setting continues the path.
  path     Ced9 -> X -> Y counts only when the Ced9 stretch on X and the X stretch that hits
           Y share at least one residue of X. Its score is the larger of its two E-values
           (a path is as weak as its weaker step). Each Y takes its best path, and the
           human proteins are ranked by that score.
  null     the same two steps from 20 dipeptide-shuffled copies of each query, made by
           notebook 284's code with seed 284, so they are 284's copies.

Step 2 searches the whole proteome rather than only the intermediates: each query or copy
has about 2_000 intermediates, and 42 such sets together cover almost every human protein.

Steps (each resumes where it stopped):
  step1          42 queries at the 152 settings. Keeps regions at E < 100 and every region
                 on BCL2 and CD47. <OUT>/step1/<setting>.parquet
  check          Ced9's and P66's best E-value on each human protein must equal notebook
                 241's (regions.parquet) to 1e-9 relative, or the run is not 241's search.
  pilot          step 2 on a fixed sample of human proteins at every setting, with the
                 wall time and peak memory of every kmerseek process. <OUT>/pilot.jsonl
  plan           predicted time and memory per setting for the whole proteome, from the
                 pilot. Writes <OUT>/step2_plan.csv; --submit also submits the step 2 arrays
                 and the score job.
  step2 --task i one setting (row i of arms.json) over the whole proteome, the 241
                 intermediates of Ced9 and P66 first. <OUT>/step2/<setting>.parquet
  score          the paths, BCL2's and CD47's ranks, and the null. <OUT>/results/
                 Reads every finished step 2 chunk, so it can run before step 2 ends;
                 results/286_settings_done.json says which settings were complete.

Paths come from the environment:
  NB286_DIR       output folder (default $SCRATCH/286-intermediate-sequence)
  NB241_KMERSEEK  kmerseek 982a055 binary (notebook 241's build)
  NB241_IDX       folder of the 152 rebuilt human indexes, human.<alphabet>.k<k>.rocksdb
                  (notebook 283's rebuild: $SCRATCH/283-bhf-low-complexity/indexes/idx)
  NB241_HUMAN     gencode.v49.pc_translations.canonical.fa (the indexed proteome)
  NB256_ARMS      notebook 256's arms.json (alphabet, k, bits, penalty, xdrop, nofit)
  NB241_QUERIES   notebook 241's queries.fa (Ced9, P66, BHF)
  NB241_REGIONS   notebook 241's regions.parquet, for `check`

Usage: 286_intermediate_sequence.py step1|check|pilot|plan|step2|score [--workers 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ENV = os.environ
SCRATCH = Path(ENV.get("SCRATCH", "/tmp"))
OUT = Path(ENV.get("NB286_DIR", SCRATCH / "286-intermediate-sequence"))
KMERSEEK = Path(
    ENV.get(
        "NB241_KMERSEEK",
        SCRATCH / "241-alphabet-ranking-kmerseek/target/release/kmerseek",
    )
)
IDX = Path(ENV.get("NB241_IDX", SCRATCH / "283-bhf-low-complexity/indexes/idx"))
HUMAN = Path(
    ENV.get(
        "NB241_HUMAN",
        SCRATCH / "data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)
ARMS = Path(
    ENV.get("NB256_ARMS", SCRATCH / "241-alphabet-ranking/null_bhf_dipeptide/arms.json")
)
REGIONS_241 = Path(
    ENV.get("NB241_REGIONS", SCRATCH / "241-alphabet-ranking/regions.parquet")
)

# The two cases and the protein each should reach.
PARTNERS = {"Ced9": "BCL2", "P66": "CD47"}
N_SHUFFLES = 20  # notebook 284's copies: make_queries(n=20, seed=284)
SHUFFLE_SEED = 284
# A region continues a path, and a human protein is an intermediate, at E below this.
E_INCLUDE = 100.0
# Settings at or below this many seed bits are searched in small chunks (notebook 256's
# HEAVY_BITS, there 25 queries of 252 aa; here the same number of residues).
HEAVY_BITS = 20.5
HEAVY_CHUNK_RESIDUES = 25 * 252
LIGHT_CHUNK_RESIDUES = 301 * 252  # 256 searched its 301 queries in one file
# Pilot sample for sizing step 2: random proteins, plus the longest ones, which set the
# largest chunk's memory.
PILOT_N_RANDOM = 150
PILOT_N_LONGEST = 3
PILOT_SEED = 286
KEEP = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_n_mismatches",
    "region_evalue",
]
# What is written per region; a protein is its full FASTA header (gene symbols repeat on
# the X and Y pseudoautosomal genes), and the gene symbol is the header's 7th field.
SCHEMA = {
    "query_name": pl.String,
    "target_name": pl.String,
    "region_start": pl.Int32,
    "region_end": pl.Int32,
    "target_start": pl.Int32,
    "target_end": pl.Int32,
    "region_n_mismatches": pl.Int32,
    "region_evalue": pl.Float64,
    "alphabet": pl.String,
    "ksize": pl.Int32,
}


def read_fasta(p: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = ""
        else:
            out[name] += line.strip()
    return out


def gene_of(header: pl.Expr) -> pl.Expr:
    return header.str.split("|").list.get(6, null_on_oob=True)


def arm_tag(arm: dict) -> str:
    return f"{arm['alphabet']}.k{arm['k']}"


def load_arms() -> list[dict]:
    return json.loads(ARMS.read_text())


def queries() -> list[tuple[str, str]]:
    """Ced9 and P66, each followed by notebook 284's 20 dipeptide-shuffled copies.

    Imported from 284_shuffle_agreement.py, so the copies are 284's: its make_queries draws
    Ced9's, P66's and BHF's copies from one random stream in that order, and BHF's are
    dropped here after drawing.
    """
    spec = importlib.util.spec_from_file_location(
        "nb284", HERE / "284_shuffle_agreement.py"
    )
    nb284 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb284)
    qs = nb284.make_queries(N_SHUFFLES, SHUFFLE_SEED)
    return [(h, s) for h, s in qs if h.split("_shuf")[0] in PARTNERS]


def chunks(records: list[tuple[str, str]], cap: int) -> list[list[tuple[str, str]]]:
    """Consecutive records, at most `cap` residues per chunk; a longer protein alone."""
    out, cur, n = [], [], 0
    for h, s in records:
        if cur and n + len(s) > cap:
            out.append(cur)
            cur, n = [], 0
        cur.append((h, s))
        n += len(s)
    if cur:
        out.append(cur)
    return out


def kmerseek_cmd(fa: Path, csv: Path, arm: dict) -> list[str]:
    """Notebook 284's search command (one_chunk in 284_shuffle_agreement.py), unchanged."""
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
    return [
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


def run_measured(cmd: list[str], log: Path) -> tuple[int, float, float]:
    """Run one kmerseek process; return its exit code, wall seconds and peak memory (GB)."""
    t0 = time.monotonic()
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    # ru_maxrss is kilobytes on Linux and bytes on macOS.
    rss_gb = ru.ru_maxrss / (2**20 if sys.platform.startswith("linux") else 2**30)
    return p.returncode, time.monotonic() - t0, rss_gb


def read_regions(csv: Path, arm: dict, keep_genes: tuple[str, ...]) -> pl.DataFrame:
    """kmerseek's CSV reduced to regions with E < E_INCLUDE, plus every region on
    `keep_genes`. Regions without a start (whole-protein rows) are dropped."""
    if csv.stat().st_size == 0:
        return pl.DataFrame(schema=SCHEMA)
    lf = (
        pl.scan_csv(csv, infer_schema_length=0)
        .select(KEEP)
        .filter(pl.col("region_start").is_not_null() & (pl.col("region_start") != ""))
        .with_columns(
            pl.col("query_name").str.strip_chars(),
            pl.col("region_evalue").cast(pl.Float64, strict=False),
            *[
                pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int32)
                for c in KEEP[2:7]
            ],
        )
    )
    keep = pl.col("region_evalue") < E_INCLUDE
    if keep_genes:
        keep = keep | gene_of(pl.col("target_name")).is_in(list(keep_genes))
    return (
        lf.filter(keep)
        .with_columns(
            alphabet=pl.lit(arm["alphabet"]), ksize=pl.lit(arm["k"], pl.Int32)
        )
        .select(list(SCHEMA))
        .collect()
    )


def one_chunk(
    tag: str,
    recs: list[tuple[str, str]],
    arm: dict,
    folder: Path,
    keep_genes: tuple[str, ...] = (),
) -> dict:
    """Search one chunk at one setting; write <folder>/<tag>.parquet. Returns a timing row."""
    done = folder / f"{tag}.parquet"
    row = dict(
        tag=tag,
        setting=arm_tag(arm),
        bits=arm["bits"],
        n_proteins=len(recs),
        n_residues=sum(len(s) for _, s in recs),
        longest=max(len(s) for _, s in recs),
    )
    if done.exists():
        return row | dict(status="cached")
    tmp = OUT / "tmp"
    fa, csv = tmp / f"{tag}.fa", tmp / f"{tag}.csv"
    fa.write_text("".join(f">{h}\n{s}\n" for h, s in recs))
    rc, wall, rss = run_measured(
        kmerseek_cmd(fa, csv, arm), OUT / "logs" / f"{tag}.log"
    )
    row |= dict(rc=rc, wall_s=round(wall, 1), peak_rss_gb=round(rss, 2))
    if rc != 0 or not csv.exists():
        return row | dict(status="FAILED")
    df = read_regions(csv, arm, keep_genes)
    df.write_parquet(done.with_suffix(".tmp"))
    done.with_suffix(".tmp").rename(done)
    row["n_regions_kept"] = df.height
    csv.unlink(missing_ok=True)
    fa.unlink(missing_ok=True)
    return row | dict(status="ok")


def run_jobs(jobs: list[tuple], workers: int, timing: Path) -> int:
    """Run (tag, recs, arm, folder, keep_genes) jobs `workers` at a time; return the number
    that failed.

    Each finished chunk's timing row is appended to `timing` (JSON lines) as it arrives, so
    a job stopped part way keeps the measurements it made. Workers start with spawn, not
    fork: a pool forked after the parent has used polars can hang on Linux (Sherlock) with
    no error, and macOS, which spawns by default, never shows it.
    """
    failed = 0
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex, open(
        timing, "a"
    ) as log:
        for row in ex.map(one_chunk, *zip(*jobs)):
            print(json.dumps(row), file=sys.stderr, flush=True)
            failed += row["status"] == "FAILED"
            if row["status"] != "cached":
                log.write(json.dumps(row) + "\n")
                log.flush()
    return failed


def setup() -> None:
    for d in ("tmp", "logs", "step1", "step2", "results"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    missing = [
        a for a in load_arms() if not (IDX / f"human.{arm_tag(a)}.rocksdb").exists()
    ]
    if missing:
        sys.exit(
            f"{len(missing)} of 152 indexes missing in {IDX} (first: {arm_tag(missing[0])}); "
            "run notebook 283's `indexes` step first"
        )
    if not os.environ.get("RAYON_NUM_THREADS"):
        sys.exit(
            "set RAYON_NUM_THREADS (CPUs / workers): each kmerseek process otherwise "
            "starts one thread per CPU of the node"
        )


def step1(workers: int) -> None:
    setup()
    done = OUT / "step1" / "regions.parquet"
    if done.exists():
        print(f"step1: done before ({done})", file=sys.stderr, flush=True)
        return
    qs = queries()
    (OUT / "queries.fa").write_text("".join(f">{h}\n{s}\n" for h, s in qs))
    folder = OUT / "step1" / "chunks"
    folder.mkdir(exist_ok=True)
    jobs = []
    for arm in load_arms():
        cap = (
            HEAVY_CHUNK_RESIDUES if arm["bits"] <= HEAVY_BITS else LIGHT_CHUNK_RESIDUES
        )
        for i, c in enumerate(chunks(qs, cap)):
            jobs.append(
                (f"{arm_tag(arm)}.c{i:02d}", c, arm, folder, tuple(PARTNERS.values()))
            )
    print(f"step1: {len(qs)} queries, {len(jobs)} chunks", file=sys.stderr, flush=True)
    if run_jobs(jobs, workers, OUT / "step1_timing.jsonl"):
        sys.exit("step1: some chunks failed; rerun to retry them")
    pl.scan_parquet(folder / "*.parquet").sink_parquet(done.with_suffix(".tmp"))
    done.with_suffix(".tmp").rename(done)
    shutil.rmtree(folder)


def best_per_target(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.group_by("query_name", "target_name").agg(pl.col("region_evalue").min())


# Settings left out of the check, because notebook 241's own table at them was not made
# with the flags 241 recorded. hp_lehninger_hpc3 k21: 241's runs.jsonl and its .nofit
# file say "no fit, exact regions", but its search log (2026-09-23 06:30) shows mismatch
# penalty 0.87 / X-drop 3.46, and the driver kept that output (search_s 0.0). Notebook
# 256's arms.json, 283's index rebuild (10-13) and this script all search it exact, and
# 283's rebuild gives E = inf on the 4 pairs it changes here (Ced9 on GABRP and FAT1, P66
# on VEZF1 and TEX101). The other 81 unfitted settings' logs all say "extension off".
CHECK_SKIP_241 = {("hp_lehninger_hpc3", 21)}


def check() -> None:
    """Ced9's and P66's best E-value per human protein (E < 100) must equal notebook 241's,
    over every setting except CHECK_SKIP_241."""

    def keep(alphabet: str, ksize: str) -> pl.Expr:
        skip = pl.lit(False)
        for a, k in CHECK_SKIP_241:
            skip = skip | ((pl.col(alphabet) == a) & (pl.col(ksize) == k))
        return (
            pl.col("query_name").is_in(list(PARTNERS))
            & (pl.col("region_evalue") < E_INCLUDE)
            & ~skip
        )

    ours = best_per_target(
        pl.scan_parquet(OUT / "step1" / "regions.parquet").filter(
            keep("alphabet", "ksize")
        )
    )
    theirs = best_per_target(
        pl.scan_parquet(REGIONS_241)
        .filter(keep("alphabet", "ksize_arm"))
        .select("query_name", "target_name", "region_evalue")
    )
    n_skipped = (
        pl.scan_parquet(REGIONS_241)
        .filter(
            pl.col("query_name").is_in(list(PARTNERS))
            & (pl.col("region_evalue") < E_INCLUDE)
            & ~keep("alphabet", "ksize_arm")
        )
        .select(pl.len())
        .collect()
        .item()
    )
    print(
        f"check: left out {sorted(CHECK_SKIP_241)}, "
        f"{n_skipped} of notebook 241's Ced9/P66 regions at E < {E_INCLUDE:g}",
        file=sys.stderr,
    )
    j = ours.join(
        theirs,
        on=["query_name", "target_name"],
        how="full",
        coalesce=True,
        suffix="_241",
    ).collect()
    bad = j.filter(
        pl.col("region_evalue").is_null()
        | pl.col("region_evalue_241").is_null()
        | (
            (pl.col("region_evalue") - pl.col("region_evalue_241")).abs()
            > 1e-9 * pl.col("region_evalue_241").abs()
        )
    )
    print(
        f"check: {j.height} (query, human protein) pairs with E < {E_INCLUDE:g}; "
        f"{bad.height} differ from notebook 241",
        file=sys.stderr,
    )
    if bad.height:
        print(bad.head(20), file=sys.stderr)
        sys.exit("check failed: this search is not notebook 241's")
    j.write_csv(OUT / "results" / "286_check_against_241.csv")


def human_records() -> list[tuple[str, str]]:
    return list(read_fasta(HUMAN).items())


def pilot_sample() -> list[tuple[str, str]]:
    recs = human_records()
    longest = sorted(recs, key=lambda r: -len(r[1]))[:PILOT_N_LONGEST]
    rest = [r for r in recs if r not in longest]
    return random.Random(PILOT_SEED).sample(rest, PILOT_N_RANDOM) + longest


def pilot(workers: int) -> None:
    setup()
    sample = pilot_sample()
    folder = OUT / "pilot"
    folder.mkdir(exist_ok=True)
    longest = {h for h, _ in sample[-PILOT_N_LONGEST:]}
    jobs = []
    for arm in load_arms():
        cap = (
            HEAVY_CHUNK_RESIDUES if arm["bits"] <= HEAVY_BITS else LIGHT_CHUNK_RESIDUES
        )
        rnd = chunks([r for r in sample if r[0] not in longest], cap)
        big = chunks([r for r in sample if r[0] in longest], cap)
        jobs += [
            (f"{arm_tag(arm)}.random{i:02d}", c, arm, folder) for i, c in enumerate(rnd)
        ]
        jobs += [
            (f"{arm_tag(arm)}.longest{i:02d}", c, arm, folder)
            for i, c in enumerate(big)
        ]
    print(
        f"pilot: {len(sample)} proteins, {len(jobs)} chunks",
        file=sys.stderr,
        flush=True,
    )
    run_jobs(jobs, workers, OUT / "pilot.jsonl")


# Sizing margins: the pilot's sample is 1.3% of the proteome, so its rate per residue is
# taken 1.5x; memory is the largest pilot process, 1.3x, times the processes per task.
TIME_MARGIN = 1.5
MEM_MARGIN = 1.3
TASK_OVERHEAD_GB = 8  # python, polars merging the chunk tables
MAX_TASK_HOURS = 48
NODE_MEM_GB = 500  # largest hns memory most nodes have; above it a setting is not run


def plan(workers: int, cpus: int, submit: bool) -> pl.DataFrame:
    """Per setting: predicted wall time and memory for step 2 over the whole proteome."""
    p = pl.read_ndjson(OUT / "pilot.jsonl", infer_schema_length=None)
    failed = p.filter(pl.col("status") == "FAILED")
    if failed.height:
        print(f"plan: {failed.height} pilot chunks failed:\n{failed}", file=sys.stderr)
    total_res = sum(len(s) for _, s in human_records())
    rnd = p.filter(pl.col("tag").str.contains(r"\.random"))
    big = p.filter(pl.col("tag").str.contains(r"\.longest"))
    rate = rnd.group_by("setting").agg(
        s_per_res=pl.col("wall_s").sum() / pl.col("n_residues").sum()
    )
    plan = (
        rate.join(
            p.group_by("setting").agg(
                bits=pl.col("bits").first(),
                pilot_max_rss_gb=pl.col("peak_rss_gb").max(),
                pilot_failed=(pl.col("status") == "FAILED").sum(),
            ),
            on="setting",
        )
        .join(
            big.group_by("setting").agg(longest_chunk_s=pl.col("wall_s").max()),
            on="setting",
            how="left",
        )
        .with_columns(
            pred_hours=(
                pl.max_horizontal(
                    pl.col("s_per_res") * total_res / workers,
                    pl.col("longest_chunk_s").fill_null(0),
                )
                * TIME_MARGIN
                / 3600
            ),
            pred_mem_gb=pl.col("pilot_max_rss_gb") * MEM_MARGIN * workers
            + TASK_OVERHEAD_GB,
        )
        .with_columns(
            ask_hours=pl.col("pred_hours")
            .ceil()
            .clip(2, MAX_TASK_HOURS)
            .cast(pl.Int32),
            ask_mem_gb=(pl.col("pred_mem_gb") / 16).ceil().cast(pl.Int32) * 16,
        )
        .sort("pred_hours", descending=True)
    )
    arms = {arm_tag(a): i for i, a in enumerate(load_arms())}
    plan = plan.with_columns(
        task=pl.col("setting").replace_strict(arms, return_dtype=pl.Int32)
    )
    plan.write_csv(OUT / "step2_plan.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=200):
        print(
            plan.select(
                "setting",
                "bits",
                "pilot_max_rss_gb",
                "pilot_failed",
                "pred_hours",
                "pred_mem_gb",
                "ask_hours",
                "ask_mem_gb",
            )
        )
    over = plan.filter(
        (pl.col("ask_mem_gb") > NODE_MEM_GB)
        | (pl.col("pred_hours") > MAX_TASK_HOURS)
        | (pl.col("pilot_failed") > 0)
    )
    print(
        f"plan: {plan.height} settings, {over.height} over the limits "
        f"({NODE_MEM_GB} GB, {MAX_TASK_HOURS} h) or failed in the pilot; those are not submitted",
        file=sys.stderr,
    )
    if submit:
        submit_step2(
            plan.join(over.select("setting"), on="setting", how="anti"), cpus, workers
        )
    return plan


def submit_step2(plan: pl.DataFrame, cpus: int, workers: int) -> None:
    """One array per (hours, memory) ask, then the score job after all of them."""
    sb = HERE / "286_step2.sbatch"
    ids = []
    for (h, m), g in plan.group_by(["ask_hours", "ask_mem_gb"]):
        tasks = ",".join(str(t) for t in sorted(g["task"]))
        cmd = [
            "sbatch",
            "--parsable",
            f"--array={tasks}",
            f"--time={h}:00:00",
            f"--mem={m}G",
            f"--cpus-per-task={cpus}",
            f"--export=ALL,NB286_WORKERS={workers}",
            str(sb),
        ]
        jid = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        ).stdout.strip()
        print(f"submitted {jid}: {g.height} settings, {h} h, {m} GB", file=sys.stderr)
        ids.append(jid.split(";")[0])
    score = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{':'.join(ids)}",
            str(HERE / "286_score.sbatch"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"submitted score job {score}, after {' '.join(ids)}", file=sys.stderr)
    (OUT / "submitted.json").write_text(json.dumps(dict(step2=ids, score=score)))


def priority_proteins() -> set[str]:
    """Ced9's and P66's intermediates in this run's step 1: searched first in step 2, so
    the real queries' answer comes before the null's."""
    s1 = pl.scan_parquet(OUT / "step1" / "regions.parquet")
    return set(
        s1.filter(
            pl.col("query_name").is_in(list(PARTNERS))
            & (pl.col("region_evalue") < E_INCLUDE)
        )
        .select("target_name")
        .unique()
        .collect()["target_name"]
    )


def step2(task: int, workers: int) -> None:
    setup()
    arm = load_arms()[task]
    tag = arm_tag(arm)
    final = OUT / "step2" / f"{tag}.parquet"
    if final.exists():
        print(f"step2 {tag}: done before", file=sys.stderr)
        return
    first = priority_proteins()
    recs = human_records()
    recs = [r for r in recs if r[0] in first] + [r for r in recs if r[0] not in first]
    cap = HEAVY_CHUNK_RESIDUES if arm["bits"] <= HEAVY_BITS else LIGHT_CHUNK_RESIDUES
    folder = OUT / "step2" / f"{tag}.chunks"
    folder.mkdir(exist_ok=True)
    jobs = [
        (f"{tag}.c{i:04d}", c, arm, folder) for i, c in enumerate(chunks(recs, cap))
    ]
    print(
        f"step2 {tag}: {len(recs)} proteins ({len(first)} first), {len(jobs)} chunks",
        file=sys.stderr,
        flush=True,
    )
    if run_jobs(jobs, workers, OUT / "step2" / f"{tag}.timing.jsonl"):
        sys.exit(f"step2 {tag}: some chunks failed; rerun to retry them")
    (
        pl.scan_parquet(folder / "*.parquet")
        .filter(pl.col("query_name") != pl.col("target_name"))
        .sink_parquet(final.with_suffix(".tmp"))
    )
    final.with_suffix(".tmp").rename(final)
    shutil.rmtree(folder)


def bh_motif(gene: str, start: int, end: int) -> str:
    """UniProt BH motifs (notebook 284) a 1-based inclusive stretch overlaps."""
    motifs = {
        "BCL2": {
            "BH4": (10, 30),
            "BH3": (93, 107),
            "BH1": (136, 155),
            "BH2": (187, 202),
        }
    }
    return ";".join(
        m for m, (a, b) in motifs.get(gene, {}).items() if start <= b and end >= a
    )


def score() -> None:
    """Every path query -> X -> Y with both E < E_INCLUDE and an X residue in common; each Y
    (gene) keeps its best path; the partner's rank among the genes reached."""
    s1 = pl.scan_parquet(OUT / "step1" / "regions.parquet").filter(
        pl.col("region_evalue") < E_INCLUDE
    )
    # Finished settings, plus the finished chunks of settings still running (their first
    # chunks hold Ced9's and P66's intermediates), so the real queries can be scored early.
    step2_files = sorted((OUT / "step2").glob("*.parquet"))
    settings_done = [f.stem for f in step2_files]
    partial = sorted(
        f
        for d in (OUT / "step2").glob("*.chunks")
        if d.name.removesuffix(".chunks") not in settings_done
        for f in d.glob("*.parquet")
    )
    settings_partial = sorted({f.parent.name.removesuffix(".chunks") for f in partial})
    s2 = pl.scan_parquet(step2_files + partial).filter(
        pl.col("query_name") != pl.col("target_name")
    )
    names = list(read_fasta(OUT / "queries.fa"))
    best_rows, partner_rows = [], []
    for q in names:
        case = q.split("_shuf")[0]
        partner = PARTNERS[case]
        a = (
            s1.filter(pl.col("query_name") == q)
            .select(
                x=pl.col("target_name"),
                q_start=pl.col("region_start"),
                q_end=pl.col("region_end"),
                x1_start=pl.col("target_start"),
                x1_end=pl.col("target_end"),
                e1=pl.col("region_evalue"),
                alphabet1=pl.col("alphabet"),
                k1=pl.col("ksize"),
                mm1=pl.col("region_n_mismatches"),
            )
            .collect()
        )
        b = (
            s2.filter(pl.col("query_name").is_in(a["x"].unique().implode()))
            .select(
                x=pl.col("query_name"),
                y=pl.col("target_name"),
                x2_start=pl.col("region_start"),
                x2_end=pl.col("region_end"),
                y_start=pl.col("target_start"),
                y_end=pl.col("target_end"),
                e2=pl.col("region_evalue"),
                alphabet2=pl.col("alphabet"),
                k2=pl.col("ksize"),
                mm2=pl.col("region_n_mismatches"),
            )
            .collect()
        )
        paths = (
            a.join(b, on="x")
            .with_columns(
                overlap=pl.min_horizontal("x1_end", "x2_end")
                - pl.max_horizontal("x1_start", "x2_start")
            )
            .filter(pl.col("overlap") > 0)
            .with_columns(
                score=pl.max_horizontal("e1", "e2"),
                product=pl.col("e1") * pl.col("e2"),
                y_gene=gene_of(pl.col("y")),
                x_gene=gene_of(pl.col("x")),
            )
        )
        # Best path per gene: lowest score, then lowest product, then names, so ties are
        # broken the same way on every run.
        best = (
            paths.sort(
                "score",
                "product",
                "x",
                "alphabet1",
                "k1",
                "alphabet2",
                "k2",
                "x2_start",
                "y_start",
            )
            .group_by("y_gene", maintain_order=True)
            .first()
            .with_columns(
                rank=pl.col("score").rank("min").cast(pl.Int32),
                query_name=pl.lit(q),
                case=pl.lit(case),
                real=pl.lit(q == case),
            )
        )
        best_rows.append(best)
        p = best.filter(pl.col("y_gene") == partner)
        direct = (
            pl.scan_parquet(OUT / "step1" / "regions.parquet")
            .filter(
                (pl.col("query_name") == q)
                & (gene_of(pl.col("target_name")) == partner)
            )
            .select(pl.col("region_evalue").min())
            .collect()
            .item()
        )
        row = dict(
            query_name=q,
            case=case,
            real=q == case,
            partner=partner,
            n_intermediates=a["x"].n_unique(),
            n_genes_reached=best.height,
            direct_best_e=direct,
            reached=p.height > 0,
        )
        row |= dict.fromkeys(
            [
                "rank",
                "n_tied",
                "score",
                "e1",
                "e2",
                "x_gene",
                "setting1",
                "setting2",
                "q_start",
                "q_end",
                "x1_start",
                "x1_end",
                "x2_start",
                "x2_end",
                "y_start",
                "y_end",
                "overlap",
                "partner_motifs",
            ]
        )
        if p.height:
            r = p.row(0, named=True)
            row |= dict(
                rank=r["rank"],
                n_tied=int((best["score"] == r["score"]).sum()) - 1,
                score=r["score"],
                e1=r["e1"],
                e2=r["e2"],
                x_gene=r["x_gene"],
                setting1=f"{r['alphabet1']}.k{r['k1']}",
                setting2=f"{r['alphabet2']}.k{r['k2']}",
                q_start=r["q_start"],
                q_end=r["q_end"],
                x1_start=r["x1_start"],
                x1_end=r["x1_end"],
                x2_start=r["x2_start"],
                x2_end=r["x2_end"],
                y_start=r["y_start"],
                y_end=r["y_end"],
                overlap=r["overlap"],
                partner_motifs=bh_motif(partner, r["y_start"] + 1, r["y_end"]),
            )
        partner_rows.append(row)
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "query_name",
                        "n_intermediates",
                        "n_genes_reached",
                        "rank",
                        "x_gene",
                    )
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    res = OUT / "results"
    pl.concat(best_rows, how="diagonal_relaxed").write_parquet(
        res / "286_best_path_per_gene.parquet"
    )
    pr = pl.DataFrame(partner_rows, infer_schema_length=None)
    pr.write_csv(res / "286_partner_rank.csv")
    (res / "286_settings_done.json").write_text(
        json.dumps(
            dict(
                n_complete=len(settings_done),
                complete=settings_done,
                n_partial=len(settings_partial),
                partial=settings_partial,
            )
        )
    )
    # Small tables for the notebook: each real query's 100 best genes reached, with paths.
    top = (
        pl.concat(best_rows, how="diagonal_relaxed")
        .filter(pl.col("real") & (pl.col("rank") <= 100))
        .sort("query_name", "rank")
    )
    top.write_csv(res / "286_real_top100_paths.csv")
    print(pr, file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "step", choices=["step1", "check", "pilot", "plan", "step2", "score"]
    )
    ap.add_argument("--workers", type=int, default=int(ENV.get("NB286_WORKERS", 4)))
    ap.add_argument("--cpus", type=int, default=int(ENV.get("SLURM_CPUS_PER_TASK", 32)))
    ap.add_argument("--task", type=int, default=int(ENV.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    if args.step == "step1":
        step1(args.workers)
    elif args.step == "check":
        check()
    elif args.step == "pilot":
        pilot(args.workers)
    elif args.step == "plan":
        plan(args.workers, args.cpus, args.submit)
    elif args.step == "step2":
        if args.task < 0:
            sys.exit("step2 needs --task or SLURM_ARRAY_TASK_ID")
        step2(args.task, args.workers)
    else:
        score()


if __name__ == "__main__":
    main()
