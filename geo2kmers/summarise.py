"""Join every sketch report of a run into one table, and describe the probes.

Usage: summarise.py <sketches folder> <probes.fasta> <probe_sequence_source.tsv> <out folder> [sample tables folder]
Writes sketch_summary.tsv and probe_summary.tsv to the out folder. With a folder of GSM tables it
also writes longest_probes_share.tsv: per channel, the share of the DNA k = 31 weight sum that comes
from probes longer than 800 nt.
"""

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = [
    "sample", "channel", "alphabet", "k", "mode", "rule", "background", "threshold",
    "probes_in_intensity_table", "probes_with_a_sequence", "probes_with_intensity_zero", "probes_used",
    "probes_with_no_kmer", "probes_with_zero_usable_frames", "distinct_kmers", "sum_of_weight",
    "signature_written",
]  # fmt: skip

sketches, fasta, source_table, out, *gsm_folder = sys.argv[1:]
reports = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{sketches}/*.report.json"))]
with open(f"{out}/sketch_summary.tsv", "w") as f:
    f.write("\t".join(COLUMNS) + "\n")
    f.writelines("\t".join(str(r.get(c, "")) for c in COLUMNS) + "\n" for r in reports)

with open(source_table) as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
lengths = [int(r["length_nt"]) for r in rows if r["source"] != "missing"]
with open(fasta) as f:
    sequences = [line.strip() for line in f if not line.startswith(">")]
summary = {
    "probes_in_platform": len(rows),
    "probes_gpl_sequence": sum(r["source"] == "gpl_sequence" for r in rows),
    "probes_genbank_fetch": sum(r["source"] == "genbank_fetch" for r in rows),
    "probes_missing": sum(r["source"] == "missing" for r in rows),
    "distinct_sequences": len(set(sequences)),
    "length_nt_min": min(lengths),
    "length_nt_median": statistics.median(lengths),
    "length_nt_max": max(lengths),
}
with open(f"{out}/probe_summary.tsv", "w") as f:
    f.write("measure\tvalue\n")
    f.writelines(f"{key}\t{value}\n" for key, value in summary.items())
if gsm_folder:
    from geo2kmers.sketch import dna_kmers, load_intensities, read_fasta

    probes = read_fasta(fasta)
    windows = {p: len(dna_kmers(s, 31)) for p, s in probes.items()}
    long_probes = {p for p, s in probes.items() if len(s) > 800}
    with open(f"{out}/longest_probes_share.tsv", "w") as f:
        f.write("sample\tchannel\tprobes_over_800_nt\tshare_of_kmer_windows\tshare_of_weight_sum\n")
        for r in reports:
            if (r["alphabet"], r["mode"]) != ("dna", "weighted"):
                continue
            values = load_intensities(gsm_folder[0], r["sample"], r["channel"])[0]
            weight = {p: max(0.0, v - r["background"]) * windows[p] for p, v in values.items() if p in probes}
            share_windows = sum(windows[p] for p in long_probes) / sum(windows.values())
            share = sum(weight[p] for p in long_probes) / sum(weight.values())
            row = [r["sample"], r["channel"], len(long_probes), f"{share_windows:.3f}", f"{share:.3f}"]
            f.write("\t".join(map(str, row)) + "\n")
print("wrote", out, len(reports), "sketch rows")
