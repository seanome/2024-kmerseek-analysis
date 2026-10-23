#!/usr/bin/env python3
"""Fetch both FASTQ files of one ENA run, verify them, and write a manifest row.

Usage: download_run.py <ena_filereport.tsv> <run_accession> <reads_dir> [reuse_dir]

A file counts as complete only when its MD5 equals the MD5 in the ENA file report,
it decompresses without error, and both mates hold the same number of records as
ENA's read_count. If `reuse_dir/<run>/<run>_<mate>.fastq.gz` exists and its MD5
matches, it is hard-linked instead of downloaded (same filesystem, so deleting the
other copy later does not remove ours). Otherwise curl downloads it from ENA.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def count_records(path):
    """Records and bases, from a full decompression (pigz exit status is checked)."""
    p1 = subprocess.Popen(
        [shutil.which("pigz") or "gzip", "-dc", str(path)], stdout=subprocess.PIPE
    )
    p2 = subprocess.run(
        [
            "awk",
            'NR%4==1{if(substr($0,1,1)!="@")bad=1} '
            "NR%4==2{n=length($0);b+=n} "
            'NR%4==3{if(substr($0,1,1)!="+")bad=1} '
            "NR%4==0{if(length($0)!=n)bad=1} "
            'END{if(NR%4||bad)exit 2; printf "%d %d\\n", NR/4,b}',
        ],
        stdin=p1.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    p1.stdout.close()
    if p1.wait() != 0:
        raise RuntimeError(f"pigz failed on {path}")
    p2.check_returncode()
    n, b = p2.stdout.split()
    return int(n), int(b)


def main():
    report, run, reads_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    reuse_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    rows = {}
    with open(report) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(header, line.rstrip("\n").split("\t")))
            rows[r["run_accession"]] = r
    r = rows[run]
    files = list(
        zip(
            r["fastq_ftp"].split(";"),
            r["fastq_md5"].split(";"),
            r["fastq_bytes"].split(";"),
        )
    )
    # ENA may also list an unpaired file; keep only _1 and _2, in that order
    files = [next(f for f in files if f[0].endswith(f"_{m}.fastq.gz")) for m in (1, 2)]
    urls, md5s, sizes = (
        [f[0] for f in files],
        [f[1] for f in files],
        [int(f[2]) for f in files],
    )
    reads_dir.mkdir(parents=True, exist_ok=True)
    (reads_dir / f"{run}.done").unlink(missing_ok=True)
    out_rows = []
    for mate, (url, want_md5, want_bytes) in enumerate(zip(urls, md5s, sizes), start=1):
        dest = reads_dir / f"{run}_{mate}.fastq.gz"
        done = reads_dir / f"{run}_{mate}.json"
        done.unlink(missing_ok=True)
        t0 = time.time()
        source = None
        valid_existing = (
            dest.exists()
            and dest.stat().st_size == want_bytes
            and md5(dest) == want_md5
        )
        if dest.exists() and not valid_existing:
            dest.unlink()
        cand = reuse_dir / run / f"{run}_{mate}.fastq.gz" if reuse_dir else None
        if valid_existing:
            source = "existing destination reverified against ENA"
        elif (
            cand
            and cand.exists()
            and cand.stat().st_size == want_bytes
            and md5(cand) == want_md5
        ):
            os.link(cand, dest)
            source = f"hard link of {cand} (MD5 matched ENA)"
        else:
            part = dest.with_name(dest.name + ".part")
            if part.exists():
                part.unlink()
            subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--retry",
                    "8",
                    "--retry-delay",
                    "20",
                    "--retry-all-errors",
                    "-o",
                    str(part),
                    "https://" + url,
                ],
                check=True,
            )
            got = md5(part)
            if got != want_md5:
                part.rename(dest.with_name(dest.name + ".badmd5"))
                raise SystemExit(f"{run}_{mate}: MD5 {got} != ENA {want_md5}")
            part.rename(dest)
            source = "https://" + url
        n, b = count_records(dest)
        row = dict(
            run=run,
            mate=mate,
            file=str(dest),
            source=source,
            bytes=dest.stat().st_size,
            ena_bytes=want_bytes,
            md5=want_md5,
            records=n,
            bases=b,
            ena_read_count=int(r["read_count"]),
            seconds=round(time.time() - t0, 1),
        )
        marker = done.with_suffix(".json.tmp")
        marker.write_text(json.dumps(row, indent=1))
        marker.replace(done)
        out_rows.append(row)
    n1, n2 = out_rows[0]["records"], out_rows[1]["records"]
    if not (n1 == n2 == int(r["read_count"])):
        raise SystemExit(
            f"{run}: records R1 {n1}, R2 {n2}, ENA read_count {r['read_count']}"
        )
    (reads_dir / f"{run}.done").write_text("ok\n")
    print(json.dumps(out_rows))


if __name__ == "__main__":
    main()
