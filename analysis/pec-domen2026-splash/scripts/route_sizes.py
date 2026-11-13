#!/usr/bin/env python3
"""Size of the data at each step: raw FASTQ, kept contigs, peptides, k-mer sketches (notebook 361).

Usage: route_sizes.py <pec_root> <out_dir> <threads>

Botryllus PEC samples, six contrasts, the two routes with sketches on disk: SPLASH compactors
(corrected p <= 0.05) and KaMRaT contigs (t-test p < 0.05). Protein k=7, Dayhoff k=12 and H/P k=30
sketches are the search step's own, scaled=5, one sketch per peptide, split by type using the zip's
manifest. No DNA sketch existed, so one is made here with the same settings: DNA k=31, scaled=5, one
sketch per contig, in <out_dir>/dna.

Human: raw FASTQ per series, contigs and peptides per contrast. The search step's sketches were not
kept, so no sketch sizes.

Sizes are bytes on disk. Contig bases equal the bytes of an uncompressed FASTA, leaving out headers.
Raw FASTQ is gzip-compressed.
Writes <out_dir>/raw_fastq.tsv and <out_dir>/route_sizes.tsv.
"""
import csv
import glob
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl

root, out, threads = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
(out / "dna").mkdir(parents=True, exist_ok=True)
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
ROUTES = {"splash": ("compactors/compactors.tsv", "compactor", "search_compactors"),
          "kamrat": ("kamrat_p05/contigs.tsv", "contig", "search_kamrat_p05")}
HUMAN_ROUTES = {"splash_corrected": ("compactors/compactors.tsv", "compactor", "search_compactors"),
                "splash_p05": ("compactors_p05/compactors.tsv", "compactor", "search_compactors_p05"),
                "kamrat_ttest_corrected": ("kamrat_ttest/contigs.tsv", "contig", "search_kamrat_ttest"),
                "kamrat_ttest_p05": ("kamrat_p05/contigs.tsv", "contig", "search_kamrat_p05")}


def contig_stats(tsv, col):
    s = pl.scan_csv(tsv, separator="\t", low_memory=True).select(pl.col(col)).unique().collect()[col]
    return s, s.len(), int(s.str.len_bytes().sum())


def peptide_stats(search):
    n, residues = 0, 0
    for f in glob.glob(str(search / "chunks" / "*.faa")):
        with open(f) as fh:
            for line in fh:
                if line[0] == ">":
                    n += 1
                else:
                    residues += len(line) - 1
    return n, residues


def zip_by_moltype(search):
    """{moltype: (bytes, sketches, hashes)} over the search step's zips."""
    size, count, hashes = {}, {}, {}
    for z in glob.glob(str(search / "*.zip")):
        with zipfile.ZipFile(z) as zf:
            by_name = {i.filename: i.compress_size for i in zf.infolist()}
            with zf.open("SOURMASH-MANIFEST.csv") as m:
                lines = (b.decode() for b in m)
                next(lines)
                for r in csv.DictReader(lines):
                    t = r["moltype"]
                    size[t] = size.get(t, 0) + by_name[r["internal_location"]]
                    count[t] = count.get(t, 0) + 1
                    hashes[t] = hashes.get(t, 0) + int(r["n_hashes"])
    return {t: (size[t], count[t], hashes[t]) for t in size}


def dna_sketch(seqs, stem):
    fa = out / "dna" / f"{stem}.fna"
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">c{i}\n{s}\n")
    z = out / "dna" / f"{stem}.zip"
    man = out / "dna" / f"{stem}.csv"
    man.write_text(f"name,genome_filename,protein_filename\n{stem},{fa},\n")
    subprocess.run(["sourmash", "scripts", "manysketch", str(man), "--singleton", "-p", "dna,k=31,scaled=5",
                    "-c", threads, "-o", str(z)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return z.stat().st_size


ena = pl.read_csv(root / "meta/ena_filereport.tsv", separator="\t").with_columns(
    pl.col("sample_alias").str.extract(r"(JA\d+)").alias("sample"),
    pl.col("fastq_bytes").map_elements(lambda s: sum(int(x) for x in s.split(";")), return_dtype=pl.Int64).alias("bytes"))
raw, rows = [], []
for c in CONTRASTS:
    samples = (root / "contrasts" / c / "samples.txt").read_text().split()
    e = ena.filter(pl.col("sample").is_in(samples))
    assert set(e["sample"]) == set(samples), c
    raw.append(dict(data="Botryllus", set=c, samples=len(samples), read_pairs=int(e["read_count"].sum()),
                    bases=int(e["base_count"].sum()), fastq_gz_bytes=int(e["bytes"].sum())))
    for route, (tsv, col, search) in ROUTES.items():
        seqs, n, bases = contig_stats(root / "contrasts" / c / tsv, col)
        s = root / "contrasts" / c / search
        peps, residues = peptide_stats(s)
        row = dict(data="Botryllus", contrast=c, route=route, contigs=n, contig_bases=bases, peptides=peps, residues=residues)
        for t, (b, k, h) in zip_by_moltype(s).items():
            row |= {f"{t}_bytes": b, f"{t}_sketches": k, f"{t}_hashes": h}
        row["dna_bytes"] = dna_sketch(seqs, f"{c}.{route}") if n else 0
        rows.append(row)
        print(row, flush=True)
raw.append(dict(data="Botryllus", set="all 47 samples", samples=sum(r["samples"] for r in raw),
                read_pairs=sum(r["read_pairs"] for r in raw), bases=sum(r["bases"] for r in raw),
                fastq_gz_bytes=sum(r["fastq_gz_bytes"] for r in raw)))
for series in sorted(p.name for p in (root / "human/reads").iterdir() if p.is_dir()):
    fq = glob.glob(str(root / "human/reads" / series / "**/*.fastq.gz"), recursive=True)
    raw.append(dict(data="human", set=series, samples=None, read_pairs=None, bases=None,
                    fastq_gz_bytes=sum(Path(f).stat().st_size for f in fq)))
for d in sorted((root / "human/contrasts").iterdir()):
    for route, (tsv, col, search) in HUMAN_ROUTES.items():
        if not (d / tsv).exists():
            continue
        _, n, bases = contig_stats(d / tsv, col)
        peps, residues = peptide_stats(d / search)
        rows.append(dict(data="human", contrast=d.name, route=route, contigs=n, contig_bases=bases, peptides=peps, residues=residues))
pl.DataFrame(raw, infer_schema_length=None).write_csv(out / "raw_fastq.tsv", separator="\t")
pl.DataFrame(rows, infer_schema_length=None).write_csv(out / "route_sizes.tsv", separator="\t")
print("wrote", out)
