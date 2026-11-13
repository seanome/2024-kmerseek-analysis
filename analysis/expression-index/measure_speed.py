"""Measure download and translation speed on the first runs in processing order.

Downloads each run's FASTQ from ENA with a resumable curl (retry, 60 s stall limit),
translates every read with translate.py, and writes speed.tsv. Also times a fixed byte
range of the largest run's first FASTQ, since tiny files measure connection set-up, not
bandwidth. Deletes each FASTQ after it is read.

    python measure_speed.py runs.tsv N_RUNS WORK_DIR OUT_TSV
"""

import gzip
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from translate import kept_translations  # noqa: E402

CURL = ["curl", "-sS", "-L", "-C", "-", "--retry", "10", "--retry-all-errors",
        "--speed-limit", "1", "--speed-time", "60", "-o"]


def fetch(url, dest):
    t = time.time()
    for attempt in range(5):
        if subprocess.run(CURL + [str(dest), url]).returncode == 0:
            return time.time() - t
    raise RuntimeError(f"download failed: {url}")


def read_fastq(path):
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                yield line.strip()


def main(runs_tsv, n, work, out):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(runs_tsv, sep="\t").sort_values("size_gb").head(int(n))
    rows = []
    for _, r in runs.iterrows():
        urls = [u for c in ("ena_fastq_http_1", "ena_fastq_http_2", "ena_fastq_http")
                for u in [r.get(c)] if isinstance(u, str) and u.startswith("http")][:2]
        dl_s = nbytes = reads = kept_reads = frames = 0
        codons = Counter()
        tr_s = 0.0
        for u in urls:
            dest = work / Path(u).name
            dl_s += fetch(u, dest)
            nbytes += dest.stat().st_size
            t = time.time()
            for seq in read_fastq(dest):
                reads += 1
                k = kept_translations(seq, codons)
                frames += len(k)
                kept_reads += bool(k)
            tr_s += time.time() - t
            dest.unlink()
        rows.append({"accession": r["accession"], "files": len(urls), "bytes": nbytes,
                     "download_s": round(dl_s, 2), "reads": reads, "reads_with_kept_frame": kept_reads,
                     "kept_translations": frames, "translate_s": round(tr_s, 2),
                     "reads_per_s": round(reads / tr_s, 1) if tr_s else None, **dict(codons)})
    # bandwidth: first 200 MB of the largest run's first file
    big = pd.read_csv(runs_tsv, sep="\t").sort_values("size_gb").iloc[-1]
    url = next(big[c] for c in ("ena_fastq_http_1", "ena_fastq_http") if isinstance(big.get(c), str))
    t = time.time()
    # stop after 120 s: the bytes received in that time give the rate
    subprocess.run(["curl", "-sS", "-L", "-r", "0-209715199", "--max-time", "120",
                    "-o", str(work / "range.bin"), url])
    s = time.time() - t
    b = (work / "range.bin").stat().st_size
    (work / "range.bin").unlink()
    rows.append({"accession": f"{big['accession']} (first 200 MB, bandwidth test)", "bytes": b,
                 "download_s": round(s, 2)})
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
