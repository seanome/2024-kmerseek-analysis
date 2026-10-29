#!/usr/bin/env python3
"""Notebook 352: kmerseek index build and query time at 1, 10 and up to 100 billion residues.

The rules are in notebooks/352_data/preregistered.yaml (committed before any data). This
file holds every step the Sherlock jobs run, plus the arithmetic the notebook uses:

    check-cpu        stop if a GPU is visible or a GPU library is loaded
    verify-parts     check every downloaded UniParc part against RELEASE.metalink's md5
    histogram        pass 1: residues per bin of the subset hash, over all parts
    cutoffs          from the histogram, the hash cut-off of each subset size
    split            pass 2: write each sequence into the slice of the smallest size it
                     belongs to, and keep query candidates from the smallest size
    pick-queries     the 1,000 queries, the same for every size
    run-size         copy one subset to local disk, build the index, run the 1,000 queries
    gate             choose the third size from the measured 10 billion residue step
    budget-add       add a job's wall time to the running total and print it

Only the Python standard library and numpy are used, so the same file runs inside the
kmerseek image on Sherlock and in the notebook environment on the laptop.

Words used below: a "slice" is the set of sequences whose hash falls between the cut-offs
of two neighbouring sizes; the subset of a size is all slices up to it. "u" is a
sequence's subset hash, a number in [0, 1).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Keys of the two hashes, fixed in the preregistration.
SUBSET_HASH_KEY = b"nb352-subset"
QUERY_HASH_KEY = b"nb352-queries"
# Number of histogram bins for the subset hash (preregistration: 2^24).
HISTOGRAM_BITS = 24
N_BINS = 1 << HISTOGRAM_BITS
# Sizes in residues. The first three are the preregistered sizes; 20, 30 and 50 billion are
# the gate's fallbacks for the third size.
SIZES_RESIDUES = (
    1_000_000_000,
    10_000_000_000,
    20_000_000_000,
    30_000_000_000,
    50_000_000_000,
    100_000_000_000,
)
THIRD_SIZE_CHOICES = (
    100_000_000_000,
    50_000_000_000,
    30_000_000_000,
    20_000_000_000,
)
N_QUERIES = 1_000
# A size counts (enters the fit and the gate) when at least 99% of its queries finish:
# 990 of 1,000. The preregistration did not say how many must finish. Olga set this on
# 2026-10-20, after the preregistration and before any measurement; the notebook lists it
# as a stated change. `complete` still records whether all of them finished.
MIN_PERCENT_OK = 99
KSIZE = 19

# The kmerseek setting (preregistration, kmerseek_setting).
SETTING_NAME = "hp_pbotc_1st_ed2.k19.s1.extend-c1.59.lcfalse"
INDEX_ARGS = [
    "--alphabet", "hp_pbotc_1st_ed2",
    "--ksize", str(KSIZE),
    "--scaled", "1",
    "--extend-mismatch-penalty", "1.59",
    "--extend-xdrop", "6.36",
    "--ka-queries", "0",
]  # fmt: skip
# K fitted for this alphabet, k and penalty on Swiss-Prot minus Chromadorea (dark-set index
# minus_Chromadorea.hp_pbotc_1st_ed2.k19.lcfalse, ka_survival.C1.59.csv, column k).
KA_K = "0.00020260135043837733"
SEARCH_ARGS = [
    "--alphabet", "hp_pbotc_1st_ed2",
    "--ksize", str(KSIZE),
    "--extend-mismatch-penalty", "1.59",
    "--extend-xdrop", "6.36",
    "--ka-k", KA_K,
    "--ka-queries", "0",
    "--chain-max-gap", "30",
    "--chain-max-shift", "10",
    "--threshold", "0.0",
    "--min-shared-kmers", "2",
    "--max-query-pvalue", "0.05",
    "--min-region-score", "1.3",
]  # fmt: skip
# Query layout (preregistration, measurements.queries): 16 processes at a time, 8 threads each.
QUERY_PROCESSES = 16
THREADS_PER_QUERY = 8
INDEX_THREADS = QUERY_PROCESSES * THREADS_PER_QUERY

# Gate limits (preregistration, budget.gate).
BUDGET_HOURS = 48.0
GATE_MARGIN = 1.2
GATE_MEMORY_GB = 950.0
GATE_LOCAL_DISK_TB = 2.8

# Time kept free before the job's Slurm end time: the longest query at the anchor run took
# far less, and this leaves room to write the tables.
JOB_END_MARGIN_S = 1_800

GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda", "pycuda")


# ---------------------------------------------------------------- CPU only


def check_cpu_only() -> None:
    """Stop with an error if a GPU could be used.

    Checks that CUDA_VISIBLE_DEVICES is empty or unset to "", that no NVIDIA device file
    exists, and that no GPU library has been imported into this Python process.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise SystemExit(f"CPU only: CUDA_VISIBLE_DEVICES is {visible!r}; it must be empty")
    nvidia = [p for p in Path("/dev").glob("nvidia[0-9]*")]
    if nvidia:
        raise SystemExit(f"CPU only: GPU devices are visible: {nvidia}")
    loaded = [m for m in GPU_MODULES if m in sys.modules]
    if loaded:
        raise SystemExit(f"CPU only: GPU libraries are loaded: {loaded}")


# ---------------------------------------------------------------- hashes


def hash_first_8_bytes(accession: str, key: bytes) -> int:
    """The first 8 bytes of keyed BLAKE2b (default 64-byte digest) as an unsigned integer.

    This is the hash the preregistration names. hashlib.blake2b(digest_size=8) would be a
    different hash, not a prefix of this one.
    """
    digest = hashlib.blake2b(accession.encode(), key=key).digest()
    return int.from_bytes(digest[:8], "big")


def hash_unit(accession: str, key: bytes) -> float:
    """Return a number in [0, 1): hash_first_8_bytes divided by 2^64."""
    return hash_first_8_bytes(accession, key) / 2.0**64


def hash_bin(accession: str) -> int:
    """Histogram bin of a sequence's subset hash: the top HISTOGRAM_BITS bits."""
    return hash_first_8_bytes(accession, SUBSET_HASH_KEY) >> (64 - HISTOGRAM_BITS)


# ---------------------------------------------------------------- FASTA


def read_fasta(path: Path):
    """Yield (accession, sequence) from a FASTA file, gzipped or not.

    The accession is the first word of the header, without the ">".
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    accession, chunks = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if accession is not None:
                    yield accession, "".join(chunks)
                accession, chunks = line[1:].split(None, 1)[0], []
            else:
                chunks.append(line.strip())
    if accession is not None:
        yield accession, "".join(chunks)


def write_record(handle, accession: str, sequence: str, width: int = 60) -> None:
    """Write one FASTA record with a ">accession" header and lines of `width` residues."""
    handle.write(f">{accession}\n")
    for start in range(0, len(sequence), width):
        handle.write(sequence[start : start + width] + "\n")


# ---------------------------------------------------------------- download check


def parse_metalink(path: Path) -> dict[str, tuple[int, str]]:
    """Return {file name: (size in bytes, md5)} from a UniProt RELEASE.metalink file."""
    text = Path(path).read_text()
    found = re.findall(
        r'<file name="([^"]+)">\s*<size>(\d+)</size>.*?<hash type="md5">(\w+)</hash>',
        text,
        re.S,
    )
    return {name: (int(size), md5) for name, size, md5 in found}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify_parts(metalink: Path, part_dir: Path, out_tsv: Path, workers: int) -> int:
    """Check each .fasta.gz listed in the metalink; write name, bytes, md5, ok. Return failures."""
    expected = {n: v for n, v in parse_metalink(metalink).items() if n.endswith(".fasta.gz")}
    names = sorted(expected)

    def one(name):
        path = part_dir / name
        if not path.exists():
            return name, -1, "missing", False
        size = path.stat().st_size
        md5 = md5_of(path)
        return name, size, md5, size == expected[name][0] and md5 == expected[name][1]

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, names))
    with open(out_tsv, "w") as out:
        out.write("file\tbytes\tmd5\texpected_bytes\texpected_md5\tok\n")
        for name, size, md5, ok in rows:
            out.write(f"{name}\t{size}\t{md5}\t{expected[name][0]}\t{expected[name][1]}\t{ok}\n")
    failures = sum(not r[3] for r in rows)
    print(f"{len(rows) - failures} of {len(rows)} parts match the metalink size and md5")
    return failures


# ---------------------------------------------------------------- pass 1: histogram


def histogram_of_parts(parts: list[Path]) -> tuple[np.ndarray, int, int]:
    """Residues per subset-hash bin over `parts`; also the sequence and residue totals."""
    hist = np.zeros(N_BINS, dtype=np.int64)
    n_seq = n_res = 0
    for part in parts:
        bins, lengths = [], []
        for accession, sequence in read_fasta(part):
            bins.append(hash_bin(accession))
            lengths.append(len(sequence))
        # bincount sums in float64, exact below 2**53; cast back to int64.
        hist += np.bincount(
            np.asarray(bins, dtype=np.int64), weights=np.asarray(lengths, dtype=np.float64), minlength=N_BINS
        ).astype(np.int64)
        n_seq += len(lengths)
        n_res += int(sum(lengths))
    return hist, n_seq, n_res


def _histogram_job(args):
    parts, out = args
    hist, n_seq, n_res = histogram_of_parts([Path(p) for p in parts])
    np.save(out, hist)
    return n_seq, n_res


def run_histogram(part_dir: Path, out_dir: Path, workers: int) -> None:
    """Pass 1 over every part, `workers` processes, one histogram file per process."""
    parts = sorted(part_dir.glob("uniparc_active_p*.fasta.gz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [parts[i::workers] for i in range(workers)]
    jobs = [([str(p) for p in g], str(out_dir / f"hist_{i:03d}.npy")) for i, g in enumerate(groups) if g]
    with concurrent.futures.ProcessPoolExecutor(len(jobs)) as pool:
        totals = list(pool.map(_histogram_job, jobs))
    total = np.zeros(N_BINS, dtype=np.int64)
    for _, out in jobs:
        total += np.load(out)
    np.save(out_dir / "histogram.npy", total)
    n_seq = sum(t[0] for t in totals)
    n_res = sum(t[1] for t in totals)
    stats = {"n_parts": len(parts), "n_sequences": n_seq, "n_residues": n_res}
    (out_dir / "histogram_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


def cutoffs_from_histogram(hist: np.ndarray, sizes=SIZES_RESIDUES) -> list[dict]:
    """For each size, the first bin whose cumulative residues reach it.

    A sequence is in the subset of that size when its bin is at most `last_bin`. The
    residue count is exact (the histogram counts residues, not an estimate). A size larger
    than the whole source gets `last_bin` None.
    """
    cumulative = np.cumsum(hist)
    rows = []
    for size in sizes:
        if cumulative[-1] < size:
            rows.append({"size_residues": size, "last_bin": None, "residues": None})
            continue
        last = int(np.searchsorted(cumulative, size, side="left"))
        rows.append({"size_residues": size, "last_bin": last, "residues": int(cumulative[last])})
    return rows


# ---------------------------------------------------------------- pass 2: slices


def slice_index(bin_number: int, last_bins: list[int]) -> int | None:
    """Index of the smallest size whose subset holds this bin, or None if none does."""
    for i, last in enumerate(last_bins):
        if bin_number <= last:
            return i
    return None


def split_part(part: Path, last_bins: list[int], out_dir: Path) -> dict:
    """Write each sequence of one part into its slice file; keep query candidates.

    Candidates: sequences of the smallest size's slice with at least KSIZE residues, the
    N_QUERIES with the lowest query hash (only these can be among the overall lowest).
    """
    handles = [gzip.open(out_dir / f"slice{i}" / f"{part.name}", "wt", compresslevel=1) for i in range(len(last_bins))]
    counts = [[0, 0] for _ in last_bins]
    best: list[tuple[float, str, str]] = []  # max-heap by negated hash
    for accession, sequence in read_fasta(part):
        i = slice_index(hash_bin(accession), last_bins)
        if i is None:
            continue
        write_record(handles[i], accession, sequence)
        counts[i][0] += 1
        counts[i][1] += len(sequence)
        if i == 0 and len(sequence) >= KSIZE:
            q = hash_unit(accession, QUERY_HASH_KEY)
            if len(best) < N_QUERIES:
                heapq.heappush(best, (-q, accession, sequence))
            elif -best[0][0] > q:
                heapq.heapreplace(best, (-q, accession, sequence))
    for handle in handles:
        handle.close()
    with open(out_dir / "candidates" / f"{part.name}.tsv", "w") as out:
        for negq, accession, sequence in sorted(best, reverse=True):
            out.write(f"{-negq!r}\t{accession}\t{sequence}\n")
    return {"part": part.name, "counts": counts}


def _split_job(args):
    part, last_bins, out_dir = args
    return split_part(Path(part), last_bins, Path(out_dir))


def run_split(part_dir: Path, cutoffs_json: Path, out_dir: Path, workers: int) -> None:
    rows = [r for r in json.loads(cutoffs_json.read_text()) if r["last_bin"] is not None]
    last_bins = [r["last_bin"] for r in rows]
    for i in range(len(last_bins)):
        (out_dir / f"slice{i}").mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    parts = sorted(part_dir.glob("uniparc_active_p*.fasta.gz"))
    jobs = [(str(p), last_bins, str(out_dir)) for p in parts]
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_split_job, jobs))
    totals = [[0, 0] for _ in last_bins]
    for r in results:
        for i, (n, res) in enumerate(r["counts"]):
            totals[i][0] += n
            totals[i][1] += res
    summary = []
    n_cum = res_cum = 0
    for i, row in enumerate(rows):
        n_cum += totals[i][0]
        res_cum += totals[i][1]
        summary.append(
            {
                "size_residues": row["size_residues"],
                "slices": list(range(i + 1)),
                "n_sequences": n_cum,
                "n_residues": res_cum,
                "histogram_residues": row["residues"],
            }
        )
    (out_dir / "subsets.json").write_text(json.dumps(summary, indent=2))
    for s in summary:
        assert s["n_residues"] == s["histogram_residues"], s
    print(json.dumps(summary, indent=2))


def pick_queries(candidate_dir: Path, out_fasta: Path, out_tsv: Path) -> None:
    """The N_QUERIES candidates with the lowest query hash, in that order."""
    rows = []
    for path in sorted(candidate_dir.glob("*.tsv")):
        for line in path.read_text().splitlines():
            q, accession, sequence = line.split("\t")
            rows.append((float(q), accession, sequence))
    rows.sort()
    rows = rows[:N_QUERIES]
    if len(rows) < N_QUERIES:
        raise SystemExit(f"only {len(rows)} query candidates")
    with open(out_fasta, "w") as fa, open(out_tsv, "w") as tsv:
        tsv.write("order\taccession\tlength\tquery_hash\n")
        for order, (q, accession, sequence) in enumerate(rows):
            write_record(fa, accession, sequence)
            tsv.write(f"{order}\t{accession}\t{len(sequence)}\t{q!r}\n")
    print(f"{len(rows)} queries written to {out_fasta}")


# ---------------------------------------------------------------- timing


def maxrss_bytes(ru_maxrss: int, platform: str = sys.platform) -> int:
    """ru_maxrss in bytes: Linux reports KiB (1,024 bytes), macOS reports bytes."""
    return ru_maxrss if platform == "darwin" else ru_maxrss * 1024


def timed_run(
    cmd: list[str], env: dict, stderr_path: Path, count_stdout: bool,
    timeout_s: float | None = None,
) -> dict:
    """Run one process; return wall seconds, peak RSS (GB), exit status and output counts.

    Wall time is from just before the process starts to its exit, read with
    time.perf_counter. Peak RSS is ru_maxrss from os.wait4, converted to GB (1e9 bytes). When
    count_stdout is true, stdout is read in 4 MB blocks and only its bytes and line breaks
    are counted, then discarded. With timeout_s, a process still running at the
    deadline is killed and recorded with exit 124, including partial output counts.
    """
    with open(stderr_path, "wb") as err:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE if count_stdout else subprocess.DEVNULL, stderr=err)
        timed_out = threading.Event()

        def stop_process():
            try:
                # Avoid Popen.poll(): wait4 below owns child reaping for peak RSS.
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            timed_out.set()

        timer = None
        if timeout_s is not None:
            timer = threading.Timer(max(0.0, timeout_s), stop_process)
            timer.daemon = True
            timer.start()
        n_bytes = n_lines = 0
        if count_stdout:
            for block in iter(lambda: proc.stdout.read(1 << 22), b""):
                n_bytes += len(block)
                n_lines += block.count(b"\n")
            proc.stdout.close()
        _, status, usage = os.wait4(proc.pid, 0)
        if timer is not None:
            timer.cancel()
            timer.join()
        wall = time.perf_counter() - start
    proc.returncode = os.waitstatus_to_exitcode(status)
    return {
        "wall_s": wall,
        "peak_rss_gb": maxrss_bytes(usage.ru_maxrss) / 1e9,
        "exit": 124 if timed_out.is_set() else proc.returncode,
        "stdout_bytes": n_bytes,
        "stdout_lines": n_lines,
    }


def dir_bytes(path: Path) -> tuple[int, int]:
    """Apparent and allocated bytes of the files in a directory.

    Apparent: file sizes summed (`du -sb` gives this plus a few kB for the directories
    themselves). Allocated: 512 x st_blocks summed, the space on disk. Symbolic links are
    not followed.
    """
    apparent = allocated = 0
    for root, _, files in os.walk(path):
        for name in files:
            st = os.lstat(os.path.join(root, name))
            apparent += st.st_size
            allocated += st.st_blocks * 512
    return apparent, allocated


def split_queries(queries_fasta: Path, out_dir: Path) -> list[tuple[int, str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for order, (accession, sequence) in enumerate(read_fasta(queries_fasta)):
        path = out_dir / f"q{order:04d}.fasta"
        with open(path, "w") as out:
            write_record(out, accession, sequence)
        rows.append((order, accession, path))
    return rows


def min_queries_ok() -> int:
    """Fewest finished queries for a size to count: MIN_PERCENT_OK % of N_QUERIES, rounded up."""
    return -(-N_QUERIES * MIN_PERCENT_OK // 100)


def run_size(args) -> None:
    """One size: copy its subset to local disk, build the index, run the queries.

    Stops starting queries at the deadline: the earlier of the remaining 48 h budget and
    the job's Slurm end time less JOB_END_MARGIN_S, so queries.tsv and summary.json are
    written before Slurm stops the job. Queries cut off by it are recorded with exit
    "not_run". The size counts only if at least MIN_PERCENT_OK % of its queries finish.
    """
    check_cpu_only()
    local = Path(args.local_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.budget_left_h * 3600.0
    if args.job_end_epoch:
        deadline = min(deadline, args.job_end_epoch - JOB_END_MARGIN_S)
    info = {
        "setting": SETTING_NAME,
        "size_residues": args.size_residues,
        "host": os.uname().nodename,
        "local_dir": str(local),
        "kmerseek": args.kmerseek,
        "index_threads": INDEX_THREADS,
        "query_processes": QUERY_PROCESSES,
        "threads_per_query": THREADS_PER_QUERY,
        "budget_left_h_at_start": args.budget_left_h,
        "job_end_epoch": args.job_end_epoch,
        "deadline_epoch": deadline,
    }
    subsets = {s["size_residues"]: s for s in json.loads(Path(args.subsets_json).read_text())}
    subset = subsets[args.size_residues]
    info["n_sequences"] = subset["n_sequences"]
    info["n_residues"] = subset["n_residues"]

    # 1. Copy the slices into one FASTA on local disk (not timed as part of the build).
    fasta = local / "database.fasta"
    t0 = time.perf_counter()
    with open(fasta, "wb") as dst:
        for i in subset["slices"]:
            for part in sorted(Path(args.slice_root, f"slice{i}").glob("*.fasta.gz")):
                with gzip.open(part, "rb") as src:
                    shutil.copyfileobj(src, dst, 1 << 24)
    info["copy_to_local_s"] = time.perf_counter() - t0
    info["database_fasta_bytes"] = fasta.stat().st_size

    # 2. Build the index on local disk, every core.
    index = local / f"database.{SETTING_NAME}.kmerseek.rocksdb"
    # Temporary files kmerseek or RocksDB may write go to the node's own disk too.
    (local / "tmp").mkdir(exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(INDEX_THREADS), CUDA_VISIBLE_DEVICES="", TMPDIR=str(local / "tmp"))
    cmd = [args.kmerseek, "index", *INDEX_ARGS, "--input", str(fasta), "--output", str(index)]
    info["index_command"] = " ".join(cmd)
    build = timed_run(cmd, env, out / "index.stderr.log", count_stdout=False)
    info["index_build_wall_s"] = build["wall_s"]
    info["index_build_peak_rss_gb"] = build["peak_rss_gb"]
    info["index_build_exit"] = build["exit"]
    (out / "summary.json").write_text(json.dumps(info, indent=2))
    if build["exit"] != 0:
        raise SystemExit(f"kmerseek index failed with exit {build['exit']}; see {out / 'index.stderr.log'}")
    info["index_bytes_apparent"], info["index_bytes_allocated"] = dir_bytes(index)
    fasta.unlink()  # free local disk; the index is what is searched
    (out / "summary.json").write_text(json.dumps(info, indent=2))

    # 3. The queries: one process each, QUERY_PROCESSES at a time.
    queries = split_queries(Path(args.queries_fasta), local / "queries")
    qenv = dict(os.environ, RAYON_NUM_THREADS=str(THREADS_PER_QUERY), CUDA_VISIBLE_DEVICES="", TMPDIR=str(local / "tmp"))
    (out / "query_stderr").mkdir(exist_ok=True)
    rows: dict[int, dict] = {}
    lock = threading.Lock()
    stop = threading.Event()
    n_done = [0]

    def one(order, accession, path):
        if stop.is_set() or time.time() > deadline:
            stop.set()
            return order, {"accession": accession, "exit": "not_run"}
        qcmd = [args.kmerseek, "search", *SEARCH_ARGS, "--query", str(path), "--target", str(index)]
        result = timed_run(
            qcmd, qenv, out / "query_stderr" / f"q{order:04d}.log",
            count_stdout=True, timeout_s=max(0.0, deadline - time.time()),
        )
        result["accession"] = accession
        with lock:
            n_done[0] += 1
            done = n_done[0]
        if done % 50 == 0:
            print(f"{time.strftime('%F %T')} {done} queries done", flush=True)
        return order, result

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(QUERY_PROCESSES) as pool:
        for order, result in pool.map(lambda r: one(*r), queries):
            with lock:
                rows[order] = result
    info["all_queries_wall_s"] = time.perf_counter() - t0
    with open(out / "queries.tsv", "w") as tsv:
        cols = ["order", "accession", "exit", "wall_s", "peak_rss_gb", "stdout_bytes", "stdout_lines"]
        tsv.write("\t".join(cols) + "\n")
        for order in sorted(rows):
            r = dict(rows[order], order=order)
            tsv.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    ok = [r["wall_s"] for r in rows.values() if r["exit"] == 0]
    info["n_queries_ok"] = len(ok)
    info["n_queries_failed"] = sum(1 for r in rows.values() if r["exit"] not in (0, "not_run"))
    info["n_queries_not_run"] = sum(1 for r in rows.values() if r["exit"] == "not_run")
    info["complete"] = len(ok) == N_QUERIES
    info["counted"] = len(ok) >= min_queries_ok()
    if ok:
        info["query_wall_s_median"] = float(np.median(ok))
        info["query_wall_s_p95"] = float(np.percentile(ok, 95))
        info["query_peak_rss_gb_max"] = max(r["peak_rss_gb"] for r in rows.values() if r["exit"] == 0)
        info["query_stdout_bytes_total"] = sum(r["stdout_bytes"] for r in rows.values() if r["exit"] == 0)
    (out / "summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------- budget and gate


def budget_total_h(budget_tsv: Path) -> float:
    """Sum of the wall hours recorded in the budget file (job, job id, start, end, seconds)."""
    if not budget_tsv.exists():
        return 0.0
    total = 0.0
    for line in budget_tsv.read_text().splitlines()[1:]:
        total += float(line.split("\t")[4]) / 3600.0
    return total


def budget_add(budget_tsv: Path, job: str, job_id: str, start: float, end: float) -> float:
    new = not budget_tsv.exists()
    with open(budget_tsv, "a") as out:
        if new:
            out.write("job\tjob_id\tstart_epoch\tend_epoch\twall_s\n")
        out.write(f"{job}\t{job_id}\t{start:.0f}\t{end:.0f}\t{end - start:.0f}\n")
    total = budget_total_h(budget_tsv)
    print(f"running total: {total:.2f} of {BUDGET_HOURS:.0f} h wall time used ({job} {job_id} took {(end - start) / 3600:.2f} h)")
    return total


def job_hours(budget_tsv: Path, job: str) -> float:
    """Wall hours of the named job in the budget file (the last row with that name)."""
    hours = None
    for line in budget_tsv.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if fields[0] == job:
            hours = float(fields[4]) / 3600.0
    if hours is None:
        raise SystemExit(f"no budget row for {job}")
    return hours


def choose_third_size(measured_10: dict, job_h: float, used_h: float, available: list[int]) -> dict:
    """The gate: the largest third size whose linear prediction fits, or none.

    Prediction for size S from the measured 10 billion residue step (all PREDICTIONS, not
    measurements): wall hours = measured job hours x S / 10e9 x GATE_MARGIN; build memory =
    measured build peak x ratio; query memory = 16 x measured largest query peak x ratio;
    local disk = (index bytes + FASTA bytes) x ratio.
    """
    base = 10_000_000_000
    rows = []
    chosen = None
    for size in THIRD_SIZE_CHOICES:
        ratio = size / base
        pred = {
            "size_residues": size,
            "in_source": size in available,
            "predicted_job_h": job_h * ratio * GATE_MARGIN,
            "predicted_build_memory_gb": measured_10["index_build_peak_rss_gb"] * ratio,
            "predicted_query_memory_gb": QUERY_PROCESSES * measured_10["query_peak_rss_gb_max"] * ratio,
            "predicted_local_disk_tb": (measured_10["index_bytes_allocated"] + measured_10["database_fasta_bytes"]) * ratio / 1e12,
        }
        pred["fits"] = bool(
            pred["in_source"]
            and used_h + pred["predicted_job_h"] <= BUDGET_HOURS
            and pred["predicted_build_memory_gb"] <= GATE_MEMORY_GB
            and pred["predicted_query_memory_gb"] <= GATE_MEMORY_GB
            and pred["predicted_local_disk_tb"] <= GATE_LOCAL_DISK_TB
        )
        rows.append(pred)
        if chosen is None and pred["fits"]:
            chosen = size
    return {
        "measured_10_job_h": job_h,
        "used_h": used_h,
        "chosen_size_residues": chosen,
        "predictions": rows,
    }


# ---------------------------------------------------------------- fit (notebook)


def t_quantile_975(df: int) -> float:
    """Two-sided 95% t quantile. For df = 1 the t distribution is the Cauchy, so it is tan(0.475 pi)."""
    if df == 1:
        return math.tan(math.pi * 0.475)
    from scipy import stats  # only needed for more than three points

    return float(stats.t.ppf(0.975, df))


def loglog_fit(x, y) -> dict:
    """Least-squares line through (log10 x, log10 y): slope b, its SE and 95% interval.

    Needs at least three points; with three, df = 1.
    """
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    n = len(lx)
    if n < 3:
        raise ValueError("a slope with an interval needs at least three points")
    xm = lx.mean()
    sxx = float(((lx - xm) ** 2).sum())
    slope = float(((lx - xm) * (ly - ly.mean())).sum() / sxx)
    intercept = float(ly.mean() - slope * xm)
    resid = ly - (intercept + slope * lx)
    df = n - 2
    se = math.sqrt(float((resid**2).sum()) / df / sxx)
    t = t_quantile_975(df)
    return {
        "n_points": n,
        "slope": slope,
        "intercept": intercept,
        "se": se,
        "df": df,
        "t_975": t,
        "ci_low": slope - t * se,
        "ci_high": slope + t * se,
    }


def rule_fires(fit: dict) -> bool:
    """Preregistered rule: the lower bound of the 95% interval of the exponent is above 1."""
    return fit["ci_low"] > 1.0


# ---------------------------------------------------------------- command line


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-cpu")
    v = sub.add_parser("verify-parts")
    v.add_argument("metalink", type=Path)
    v.add_argument("part_dir", type=Path)
    v.add_argument("out_tsv", type=Path)
    v.add_argument("--workers", type=int, default=16)
    h = sub.add_parser("histogram")
    h.add_argument("part_dir", type=Path)
    h.add_argument("out_dir", type=Path)
    h.add_argument("--workers", type=int, default=32)
    c = sub.add_parser("cutoffs")
    c.add_argument("histogram", type=Path)
    c.add_argument("out_json", type=Path)
    s = sub.add_parser("split")
    s.add_argument("part_dir", type=Path)
    s.add_argument("cutoffs_json", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--workers", type=int, default=32)
    q = sub.add_parser("pick-queries")
    q.add_argument("candidate_dir", type=Path)
    q.add_argument("out_fasta", type=Path)
    q.add_argument("out_tsv", type=Path)
    r = sub.add_parser("run-size")
    r.add_argument("--size-residues", type=int, required=True)
    r.add_argument("--subsets-json", required=True)
    r.add_argument("--slice-root", required=True)
    r.add_argument("--queries-fasta", required=True)
    r.add_argument("--local-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--kmerseek", default="kmerseek")
    r.add_argument("--budget-left-h", type=float, required=True)
    r.add_argument("--job-end-epoch", type=float, default=0.0, help="Slurm end time of this job (0: none)")
    g = sub.add_parser("gate")
    g.add_argument("summary_10", type=Path)
    g.add_argument("budget_tsv", type=Path)
    g.add_argument("subsets_json", type=Path)
    g.add_argument("out_json", type=Path)
    b = sub.add_parser("budget-add")
    b.add_argument("budget_tsv", type=Path)
    b.add_argument("job")
    b.add_argument("job_id")
    b.add_argument("start_epoch", type=float)
    b.add_argument("end_epoch", type=float)
    bl = sub.add_parser("budget-left")
    bl.add_argument("budget_tsv", type=Path)
    a = p.parse_args(argv)

    check_cpu_only()
    if a.cmd == "check-cpu":
        print("CPU only: no GPU visible, no GPU library loaded")
    elif a.cmd == "verify-parts":
        sys.exit(1 if verify_parts(a.metalink, a.part_dir, a.out_tsv, a.workers) else 0)
    elif a.cmd == "histogram":
        run_histogram(a.part_dir, a.out_dir, a.workers)
    elif a.cmd == "cutoffs":
        rows = cutoffs_from_histogram(np.load(a.histogram))
        a.out_json.write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2))
    elif a.cmd == "split":
        run_split(a.part_dir, a.cutoffs_json, a.out_dir, a.workers)
    elif a.cmd == "pick-queries":
        pick_queries(a.candidate_dir, a.out_fasta, a.out_tsv)
    elif a.cmd == "run-size":
        run_size(a)
    elif a.cmd == "gate":
        measured = json.loads(a.summary_10.read_text())
        if not measured.get("counted"):
            decision = {"chosen_size_residues": None, "reason": f"fewer than {min_queries_ok()} of the 10 billion residue step's {N_QUERIES} queries finished"}
        else:
            available = [s["size_residues"] for s in json.loads(a.subsets_json.read_text())]
            job_h = job_hours(a.budget_tsv, "size-10000000000")
            decision = choose_third_size(measured, job_h, budget_total_h(a.budget_tsv), available)
        a.out_json.write_text(json.dumps(decision, indent=2))
        print(json.dumps(decision, indent=2))
        print(f"THIRD_SIZE={decision['chosen_size_residues'] or 0}")
    elif a.cmd == "budget-add":
        budget_add(a.budget_tsv, a.job, a.job_id, a.start_epoch, a.end_epoch)
    elif a.cmd == "budget-left":
        print(f"{BUDGET_HOURS - budget_total_h(a.budget_tsv):.3f}")


if __name__ == "__main__":
    main()
