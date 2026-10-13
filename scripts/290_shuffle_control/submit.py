#!/usr/bin/env python3
"""Submit notebook 290's searches as SLURM arrays.

    submit.py <pilot|full> <base> <stage> <img> [--dry-run]

pilot  one chunk per species (Botryllus chunk_0004, BHF's chunk; chunk_0000 elsewhere),
       both shuffle kinds, all six settings (60 searches); the real chunk searched again at
       POSITIVE_CONTROL (5 searches); and the run's own region files of those chunks (30
       reads, no search).
full   every chunk of every species, both kinds, six settings, and every run region file.
       Tasks already done (a .check.tsv exists) are skipped by the task itself.

One array of searches per setting, all species together, asking the largest memory and time
any species needs for that setting in plan/resources.tsv. Each runs at most LIMIT[setting]
tasks at once, and the setting's run-file reads wait for its searches (afterany), so at
most 20 of this experiment's jobs run at any time.
Python 3.6, standard library only.
"""
import csv
import os
import subprocess
import sys

SIX_SETTINGS = [("hp_lehninger2", "17"), ("hp_lehninger2", "25"), ("hp_lehninger_c_nonpolar2", "19"),
                ("hp_pbotc_1st_ed2", "19"), ("hp_pbotc_1st_ed2", "21"), ("hp_pbotc_1st_ed2", "23")]
# Slots per setting, in proportion to the run's summed search hours for that setting
# (hp_lehninger2 k17 69 h, the hp_lehninger_c_nonpolar2 and hp_pbotc_1st_ed2 k19 26 h each,
# k21 12 h, k23 8 h, hp_lehninger2 k25 7 h), so the six arrays finish at about the same time.
LIMIT = {("hp_lehninger2", "17"): 9, ("hp_lehninger2", "25"): 1, ("hp_lehninger_c_nonpolar2", "19"): 3,
         ("hp_pbotc_1st_ed2", "19"): 3, ("hp_pbotc_1st_ed2", "21"): 2, ("hp_pbotc_1st_ed2", "23"): 2}
assert sum(LIMIT.values()) == 20
SPECIES = ["human", "worm", "yeast", "borrelia", "botryllus"]
PILOT_CHUNK = {"human": "chunk_0000", "worm": "chunk_0000", "yeast": "chunk_0000",
               "borrelia": "chunk_0000", "botryllus": "chunk_0004"}
POSITIVE_CONTROL = ("hp_pbotc_1st_ed2", "23")
COLS = ["species", "clade", "chunk", "kind", "alphabet", "ksize", "penalty", "xdrop", "index", "run_regions"]


def hhmm(hours):
    m = int(round(hours * 60))
    return "{:02d}:{:02d}:00".format(m // 60, m % 60)


def main(mode, base, stage, img, dry):
    tasks = list(csv.DictReader(open(os.path.join(base, "plan", "tasks.tsv")), delimiter="\t"))
    res = {(r["species"], r["alphabet"], r["ksize"]): r
           for r in csv.DictReader(open(os.path.join(base, "plan", "resources.tsv")), delimiter="\t")}
    if mode == "pilot":
        keep = [t for t in tasks if t["chunk"] == PILOT_CHUNK[t["species"]]
                and (t["kind"] != "real" or (t["alphabet"], t["ksize"]) == POSITIVE_CONTROL)]
    elif mode == "full":
        keep = [t for t in tasks if t["kind"] != "real"]
    else:
        sys.exit(__doc__)
    reads = {}
    for t in keep:
        r = dict(t, kind="realfile")
        reads[r["run_regions"]] = r
    lists = os.path.join(base, "lists", mode)
    logs = os.path.join(base, "logs", mode)
    for d in (lists, logs):
        os.makedirs(d, exist_ok=True)
    n_search = n_read = 0
    for a, k in SIX_SETTINGS:
        prev = None
        searches = [t for t in keep if (t["alphabet"], t["ksize"]) == (a, k)]
        runfiles = [r for r in reads.values() if (r["alphabet"], r["ksize"]) == (a, k)]
        for part, rows in (("search", searches), ("runfiles", runfiles)):
            if not rows:
                continue
            name = "290-{}-{}.k{}-{}".format(mode, a, k, part)
            path = os.path.join(lists, name + ".tsv")
            with open(path, "w") as fh:
                for t in rows:
                    fh.write("\t".join(t[c] for c in COLS) + "\n")
            if part == "runfiles":
                mem, time, cpus = "4G", "01:00:00", "2"
                n_read += len(rows)
            else:
                need = [res[(sp, a, k)] for sp in sorted(set(t["species"] for t in rows))]
                mem = str(max(int(r["mem_gb"]) for r in need)) + "G"
                time, cpus = hhmm(max(float(r["time_h"]) for r in need)), "8"
                n_search += len(rows)
            cmd = ["sbatch", "--parsable", "-J", name, "-c", cpus, "--mem", mem, "-t", time,
                   "--array=0-{}%{}".format(len(rows) - 1, LIMIT[(a, k)]),
                   "-o", os.path.join(logs, "%x.%a.%A.log"),
                   "--export=ALL,LIST={},BASE={},STAGE={},IMG={}".format(path, base, stage, img)]
            if prev:
                cmd.append("--dependency=afterany:" + prev)
            cmd.append(os.path.join(stage, "task.sbatch"))
            if dry:
                print(" ".join(cmd))
                prev = "<{}>".format(name)
            else:
                prev = subprocess.check_output(cmd, universal_newlines=True).strip()
                print("{}: {} tasks, {} {} each, job {}".format(name, len(rows), mem, time, prev))
    print("{} searches and {} run-file reads {}".format(n_search, n_read, "planned" if dry else "submitted"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if len(args) != 4:
        sys.exit(__doc__)
    main(*args, dry="--dry-run" in sys.argv)
