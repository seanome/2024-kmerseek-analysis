"""Recover complete ENA paired runs without replacing completed inputs."""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def transfer(args):
    """Resume interrupted transfers on older cluster curl versions."""
    for attempt in range(6):
        result = subprocess.run(args, check=False)
        if result.returncode == 0:
            return
        if result.returncode not in {18, 28, 52, 55, 56} or attempt == 5:
            raise subprocess.CalledProcessError(result.returncode, args)
        time.sleep(min(60, 2**attempt))


def recover(row, root, validator):
    start = time.monotonic()
    run = row["run_accession"]
    out = root / "raw" / run
    out.mkdir(parents=True, exist_ok=True)
    # A directory lock prevents two submissions from writing the same run.
    lock = out / ".active"
    lock.mkdir()
    try:
        marker = out / "validated.json"
        expected = dict(run=run, md5=row["fastq_md5"], bytes=row["fastq_bytes"])
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["archive"] != expected:
                raise ValueError(
                    f"{run}: completion marker belongs to different inputs"
                )
            for f in old["files"]:
                if digest(Path(f["path"]), "md5") != f["md5"]:
                    raise ValueError(f"{run}: previously validated input changed")
            print(run, "reused", flush=True)
            return
        urls = row["fastq_ftp"].split(";")
        md5s = row["fastq_md5"].split(";")
        sizes = list(map(int, row["fastq_bytes"].split(";")))
        if len(urls) != 2 or len(md5s) != 2 or len(sizes) != 2:
            raise ValueError(f"{run}: expected two archive mate files")
        files = []
        for mate, (url, md5, size) in enumerate(zip(urls, md5s, sizes), 1):
            final = out / f"{run}_{mate}.fastq.gz"
            if not final.exists():
                partial = final.with_suffix(".gz.partial")
                # Preserve a complete but corrupt partial before downloading afresh.
                if partial.exists() and partial.stat().st_size == size:
                    actual_md5 = digest(partial, "md5")
                    if actual_md5 != md5:
                        quarantine = partial.with_name(
                            partial.name + ".invalid-md5-" + actual_md5[:12]
                        )
                        if quarantine.exists():
                            raise ValueError(
                                "Quarantine destination already exists; preserved"
                            )
                        partial.rename(quarantine)
                        quarantine.with_name(quarantine.name + ".json").write_text(
                            json.dumps(
                                {
                                    "reason": "Full-sized partial failed unchanged ENA MD5",
                                    "expected_md5": md5,
                                    "actual_md5": actual_md5,
                                    "bytes": size,
                                    "preserved_path": str(quarantine),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                # curl resumes only its own incomplete file; it never writes final.
                if not partial.exists() or partial.stat().st_size < size:
                    transfer(
                        [
                            "curl",
                            "--fail",
                            "--location",
                            "--retry",
                            "5",
                            "--connect-timeout",
                            "30",
                            "--speed-limit",
                            "1024",
                            "--speed-time",
                            "120",
                            "--continue-at",
                            "-",
                            "--output",
                            str(partial),
                            "https://" + url,
                        ],
                    )
                if partial.stat().st_size != size or digest(partial, "md5") != md5:
                    raise ValueError(
                        f"{run} mate {mate}: checksum or size mismatch; partial preserved"
                    )
                partial.rename(final)
            if final.stat().st_size != size or digest(final, "md5") != md5:
                raise ValueError(f"{run} mate {mate}: existing file differs; preserved")
            files.append(dict(path=str(final), md5=md5, bytes=size))
        result = json.loads(
            subprocess.check_output(
                [str(validator), *[f["path"] for f in files]], text=True
            )
        )
        # ENA run read_count is checked against the number of paired records.
        if result["pairs"] != int(row["read_count"]) or result["bases"] != int(
            row["base_count"]
        ):
            raise ValueError(
                f"{run}: FASTQ totals disagree with archive metadata: {result}"
            )
        record = dict(
            archive=expected,
            files=files,
            **result,
            wall_seconds=time.monotonic() - start,
            job_id=__import__("os").environ.get("SLURM_JOB_ID"),
        )
        tmp = marker.with_suffix(".json.partial")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.rename(marker)
        print(run, result, flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("validator", type=Path)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--batches", type=int, default=8)
    a = p.parse_args()
    rows = sorted(
        csv.DictReader(a.manifest.open(), delimiter="\t"),
        key=lambda r: r["run_accession"],
    )
    if not 0 <= a.batch < a.batches:
        p.error("batch must be within batches")
    for row in rows[a.batch :: a.batches]:
        recover(row, a.root, a.validator)
