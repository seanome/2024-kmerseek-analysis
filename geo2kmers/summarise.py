"""Join every sketch report of a run into one table, and describe the probes.

Usage: summarise.py <sketches folder> <probes.fasta> <probe_sequence_source.tsv> <out folder>
Writes sketch_summary.tsv and probe_summary.tsv to the out folder.
"""

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

COLUMNS = [
    "sample", "channel", "alphabet", "k", "mode", "rule", "background", "threshold",
    "probes_in_intensity_table", "probes_with_a_sequence", "probes_with_intensity_zero", "probes_used",
    "probes_with_no_kmer", "probes_with_zero_usable_frames", "distinct_kmers", "sum_of_weight",
    "signature_written",
]  # fmt: skip

sketches, fasta, source_table, out = sys.argv[1:]
reports = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{sketches}/*.report.json"))]
with open(f"{out}/sketch_summary.tsv", "w") as f:
    f.write("\t".join(COLUMNS) + "\n")
    f.writelines("\t".join(str(r.get(c, "")) for c in COLUMNS) + "\n" for r in reports)

with open(source_table) as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
lengths = [int(r["length_nt"]) for r in rows if r["source"] in ("gpl_sequence", "genbank_fetch")]
with open(fasta) as f:
    sequences = [line.strip() for line in f if not line.startswith(">")]
summary = {
    "probes_in_platform": len(rows),
    "probes_gpl_sequence": sum(r["source"] == "gpl_sequence" for r in rows),
    "probes_genbank_fetch": sum(r["source"] == "genbank_fetch" for r in rows),
    "probes_record_too_long": sum(r["source"] == "record_too_long" for r in rows),
    "probes_missing": sum(r["source"] == "missing" for r in rows),
    "distinct_sequences": len(set(sequences)),
    "length_nt_min": min(lengths),
    "length_nt_median": statistics.median(lengths),
    "length_nt_max": max(lengths),
}
with open(f"{out}/probe_summary.tsv", "w") as f:
    f.write("measure\tvalue\n")
    f.writelines(f"{key}\t{value}\n" for key, value in summary.items())
print("wrote", out, len(reports), "sketch rows")
