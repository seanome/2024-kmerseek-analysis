#!/usr/bin/env python3
"""Inputs for counting reads per gene in the human Fig. 4 series (notebook 361).

  fig4_reads_inputs.py gene-map <gencode.id_map.tsv> <out.tsv>
      Protein ID and gene symbol (upper case) for every GENCODE protein, no header: the gene map
      that `pec-tools gene-counts` reads. The symbol is the 7th field of the GENCODE FASTA header.

  fig4_reads_inputs.py manifest <trimmed_series_dir> <share> <out.tsv>
      One line per run: run, trimmed FASTQ, "-" (single-end), and how many reads to count.
      share 1 counts every read (written as 0, which gene-counts reads as "all"); share N counts
      the first 1/N of the reads fastp kept, at least one.
"""

import csv
import json
import sys
from pathlib import Path


def gene_map(id_map, out):
    n = 0
    with open(id_map, newline="") as f, open(out, "w") as o:
        for row in csv.DictReader(f, delimiter="\t"):
            fields = row["original_id"].split("|")
            if len(fields) < 8 or not fields[6]:
                raise SystemExit(f"no gene symbol in the header of {row['stable_id']}: {row['original_id']}")
            o.write(f"{row['stable_id']}\t{fields[6].upper()}\n")
            n += 1
    if n == 0:
        raise SystemExit(f"{id_map} has no proteins")
    print(f"{n} proteins", file=sys.stderr)


def manifest(trimmed, share, out):
    share = int(share)
    if share < 1:
        raise SystemExit("share must be 1 or more")
    runs = sorted(Path(trimmed).glob("*.fastq.gz"))
    if not runs:
        raise SystemExit(f"no *.fastq.gz in {trimmed}")
    with open(out, "w") as o:
        for fq in runs:
            run = fq.name.removesuffix(".fastq.gz")
            kept = json.loads((fq.parent / f"{run}.fastp.json").read_text())["summary"]["after_filtering"]["total_reads"]
            if kept < 1:
                raise SystemExit(f"{run}: fastp kept no reads")
            n = 0 if share == 1 else max(1, kept // share)
            o.write(f"{run}\t{fq.resolve()}\t-\t{n}\n")
    print(f"{len(runs)} runs", file=sys.stderr)


if __name__ == "__main__":
    commands = {"gene-map": gene_map, "manifest": manifest}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:])
