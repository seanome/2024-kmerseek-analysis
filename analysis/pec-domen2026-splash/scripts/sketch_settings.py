#!/usr/bin/env python3
"""File size and time of each k-mer sketch setting, Botryllus routes (notebook 361).

Usage: sketch_settings.py <pec_root> <out_dir> <threads>

Each setting is built alone with `sourmash scripts manysketch --singleton`, one sketch per peptide (one
per contig for DNA), into its own zip, for SPLASH compactors and KaMRaT contigs (t-test p < 0.05) in
each of the six contrasts. The zip's size on disk is reported, which holds the zip's index and manifest
as well as the sketches. route_sizes.py summed sketch members only, so it gave less.
Time and peak memory come from /usr/bin/time -v. Peptides are already translated, so translation is left
out of the time.

Writes <out_dir>/sketch_settings.tsv after every run. Zips stay in <out_dir>/zips.
"""
import csv
import glob
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

root, out, threads = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
(out / "zips").mkdir(parents=True, exist_ok=True)
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
SEARCH = {"splash": "search_compactors", "kamrat": "search_kamrat_p05"}
SETTINGS = ["dna,k=31,scaled=5", "protein,k=7,scaled=5", "protein,k=7,scaled=20", "dayhoff,k=12,scaled=5", "dayhoff,k=12,scaled=10",
            "hp,k=30,scaled=5"]


def time_file(p):
    t = open(p).read()
    h = re.search(r"Elapsed.*: (\S+)\n", t).group(1).split(":")
    return dict(cpu_s=sum(float(x) for x in re.findall(r"(?:User|System) time \(seconds\): (\S+)", t)),
                wall_s=sum(float(x) * 60 ** i for i, x in enumerate(reversed(h))),
                peak_gb=int(re.search(r"Maximum resident.*: (\d+)", t).group(1)) / 1e6)


def manifest(z):
    with zipfile.ZipFile(z) as zf, zf.open("SOURMASH-MANIFEST.csv") as m:
        lines = (b.decode() for b in m)
        next(lines)
        rows = list(csv.DictReader(lines))
    return len(rows), sum(int(r["n_hashes"]) for r in rows)


rows = []
for c in CONTRASTS:
    for route, search in SEARCH.items():
        s = root / "contrasts" / c / search
        peptides = sorted(glob.glob(str(s / "pep.peptides.faa"))) or sorted(glob.glob(str(s / "chunks" / "*.faa")))
        assert peptides, s
        pep = out / "zips" / f"{c}.{route}.faa"
        if not pep.exists():
            with open(pep, "w") as fh:
                for f in peptides:
                    fh.write(open(f).read())
        dna = root / "sizes" / "dna" / f"{c}.{route}.fna"
        for setting in SETTINGS:
            tag = f"{c}.{route}." + setting.replace(",", ".").replace("=", "")
            man, z, tf = out / "zips" / f"{tag}.csv", out / "zips" / f"{tag}.zip", out / "zips" / f"{tag}.time"
            is_dna = setting.startswith("dna")
            man.write_text(f"name,genome_filename,protein_filename\n{tag},{dna if is_dna else ''},{'' if is_dna else pep}\n")
            subprocess.run(["/usr/bin/time", "-v", "-o", str(tf), "sourmash", "scripts", "manysketch", str(man), "--singleton", "-p", setting,
                            "-c", threads, "-o", str(z)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            n, hashes = manifest(z)
            rows.append(dict(contrast=c, route=route, setting=setting, zip_bytes=z.stat().st_size, sketches=n, hashes=hashes, **time_file(tf)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out / "sketch_settings.tsv", sep="\t", index=False)
