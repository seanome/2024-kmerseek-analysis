"""Write the tables that describe the frame test arrays.

Usage: array_tables.py <folder> <out folder>
The folder holds GPL1211.txt, calls_regress.tsv (mussel), calls_GPL738.tsv and calls_GPL1211.tsv from
frame_test.py, and seq1211, seq738 and real/mussel.fasta from `sequences`. The out folder holds
cds_GPL1211.tsv from fetch_coding_regions.py. Writes array_summary.tsv, longest_stretch_GPL1211.tsv
and longest_vs_proteome.tsv to the out folder.
"""

import csv
import statistics
import sys

from geo2kmers.sketch import (
    MIN_STRETCH_AA,
    longest_stretch,
    read_fasta,
    translate_frames,
)

folder, out = sys.argv[1:]


def has_stop_free_frame(sequence, k=10):
    return any(len(frame) >= k and "*" not in frame for frame in translate_frames(sequence))


with open(f"{out}/array_summary.tsv", "w") as f:
    columns = "platform probe_rows rows_with_sequence distinct_sequences length_nt_min length_nt_median"
    f.write("\t".join(columns.split()) + "\tlength_nt_max\tdistinct_with_a_stop_free_frame_k10\trows_over_5000_nt\n")
    for platform, name in (("GPL1211", "seq1211"), ("GPL738", "seq738")):
        with open(f"{folder}/{name}/probe_sequence_source.tsv") as source:
            probe_rows = sum(1 for _ in source) - 1
        sequences = list(read_fasta(f"{folder}/{name}/probes.fasta").values())
        lengths = [len(s) for s in sequences]
        distinct = set(sequences)
        row = [platform, probe_rows, len(sequences), len(distinct), min(lengths), statistics.median(lengths)]
        row += [max(lengths), sum(map(has_stop_free_frame, distinct)), sum(n > 5000 for n in lengths)]
        f.write("\t".join(map(str, row)) + "\n")

accession = {}
with open(f"{folder}/GPL1211.txt") as f:
    for line in f:
        cells = line.rstrip("\n").split("\t")
        if len(cells) > 1 and cells[0].isdigit():
            accession[cells[0]] = cells[1].split(".")[0]
with open(f"{out}/cds_GPL1211.tsv") as f:
    # the frame of the coding region's first whole codon, on the record's own strand
    true_frame = {
        r["accession"]: (int(r["start"]) + int(r["codon_start"]) - 2) % 3 for r in csv.DictReader(f, delimiter="\t")
    }
first_probe = {}
for probe, sequence in read_fasta(f"{folder}/seq1211/probes.fasta").items():
    first_probe.setdefault(sequence, probe)
annotated = []
for sequence, probe in first_probe.items():
    if accession[probe] in true_frame:
        frame, _, stretch, _ = longest_stretch(sequence)
        annotated.append((len(stretch), frame, true_frame[accession[probe]]))
with open(f"{out}/longest_stretch_GPL1211.tsv", "w") as f:
    f.write("min_longest_stretch_aa\tsequences\tlongest_in_true_frame\tshare\n")
    for floor in (0, 30, 50, 100, 150):
        kept = [(frame, truth) for length, frame, truth in annotated if length >= floor]
        right = sum(frame == truth for frame, truth in kept)
        f.write(f"{floor}\t{len(kept)}\t{right}\t{right / len(kept):.3f}\n")

with open(f"{out}/longest_vs_proteome.tsv", "w") as f:
    f.write("array\tproteome\talphabet\tk\tsequences_with_passing_stretch_100aa\tlongest_stretch_same_frame\tshare\n")
    for name, fasta, calls in (
        ("GPL1799 mussel", f"{folder}/real/mussel.fasta", f"{folder}/calls_regress.tsv"),
        ("GPL738 mouse", f"{folder}/seq738/probes.fasta", f"{folder}/calls_GPL738.tsv"),
        ("GPL1211 human", f"{folder}/seq1211/probes.fasta", f"{folder}/calls_GPL1211.tsv"),
    ):
        sequences = read_fasta(fasta)
        best = {}
        with open(calls) as c:
            for r in csv.DictReader(c, delimiter="\t"):
                if (r["alphabet"], r["k"]) == ("protein", "10"):
                    length, frame = int(r["stretch_length_aa"]), int(r["frame"])
                    start = int(r["stretch_start_aa"])
                    stretch = translate_frames(sequences[r["probe_id"]])[frame][start : start + length]
                    if (length, frame) > best.get(r["probe_id"], (0, 0, ""))[:2]:
                        best[r["probe_id"]] = (length, frame, stretch)
        kept = {p: v[1] for p, v in best.items() if len(v[2]) - v[2].count("X") >= MIN_STRETCH_AA}
        same = sum(longest_stretch(sequences[p])[0] == frame for p, frame in kept.items())
        row = [name, "same species", "protein", 10, len(kept), same, f"{same / len(kept):.3f}"]
        f.write("\t".join(map(str, row)) + "\n")
