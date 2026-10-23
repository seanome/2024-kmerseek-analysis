#!/usr/bin/env python3
"""Download the first N read pairs of each selected library from ENA.

This is a deterministic prefix (the first N records of each FASTQ), not a random
subsample. The cap exists because this machine has 30 GB of disk. Files that
already exist with a matching `.done` marker are not downloaded again.
"""
import csv, json, os, subprocess, sys, time, hashlib
from pathlib import Path

N_PAIRS = int(os.environ.get("N_PAIRS", "5000000"))
HERE = Path(__file__).resolve().parent
READS = HERE / "reads"; READS.mkdir(exist_ok=True)
timepoints = sys.argv[1].split(",") if len(sys.argv) > 1 else ["2h", "24h"]

rows = [r for r in csv.DictReader(open(HERE / "samples_all64.tsv"), delimiter="\t")
        if r["timepoint_inferred"] in timepoints and r["usable"] == "yes"]
log = open(HERE / "download.log", "a")
def say(*a):
    print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)

for r in rows:
    for mate in ("1", "2"):
        url = r["r" + mate]
        out = READS / f"{r['sample']}_{mate}.fastq.gz"
        done = out.with_suffix(".gz.done")
        if done.exists() or Path(f"{out}.part").exists():  # done, or another worker is on it
            continue
        say("start", r["sample"], mate, url)
        t0 = time.time()
        cmd = (f"curl -sS --retry 5 --retry-delay 10 '{url}' | gzip -dc | head -n {4*N_PAIRS} | gzip -1 > '{out}.part'")
        p = subprocess.run(cmd, shell=True, executable="/bin/bash")
        # head closes the pipe early; curl/gzip then exit non-zero on purpose. Validate by content instead.
        awk = "awk 'NR%4==1{if(substr($0,1,1)!=\"@\")bad++} NR%4==3{if(substr($0,1,1)!=\"+\")bad++} END{print NR/4, bad+0}'"
        chk = subprocess.run(f"gzip -dc '{out}.part' | " + awk,
                             shell=True, capture_output=True, text=True, executable="/bin/bash").stdout.split()
        n, bad = int(float(chk[0])), int(chk[1])
        expected = min(N_PAIRS, int(r["read_pairs"]))
        if bad or n != expected:
            say("FAILED", r["sample"], mate, "records", n, "expected", expected, "bad", bad)
            os.rename(f"{out}.part", f"{out}.failed")
            continue
        os.rename(f"{out}.part", out)
        json.dump(dict(sample=r["sample"], run=r["run"], mate=mate, url=url, records=n, n_pairs_cap=N_PAIRS,
                       full_library_pairs=int(r["read_pairs"]), seconds=round(time.time()-t0,1),
                       bytes=out.stat().st_size, note="deterministic prefix of the public FASTQ; not a random subsample",
                       full_file_md5_from_ena=r["md5_" + mate]), open(done, "w"), indent=1)
        say("done", r["sample"], mate, n, "records", round(time.time()-t0), "s")
say("ALL DONE for", timepoints)
