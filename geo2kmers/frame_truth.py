"""Check frames picked by frame_test.py against the coding region NCBI annotates on each record.

Usage: frame_truth.py <GPL table> <probes.fasta> <cds.tsv> <calls.tsv> <out folder>
cds.tsv is made by fetch_coding_regions.py and calls.tsv by frame_test.py. A call is right if it is in
the annotated reading frame, which is the coding start plus codon_start. Writes frame_truth.tsv and
frame_truth_by_length.tsv: the second keeps only sequences whose longest passing stretch is long enough.
"""

import collections
import csv
import sys

gpl, probes, cds_table, call_table, out = sys.argv[1:6]
SETTINGS = [("protein", 7), ("protein", 10), ("hp", 30)]
accession = {}
with open(gpl) as f:
    for line in f:
        cells = line.rstrip("\n").split("\t")
        if len(cells) > 1 and cells[0].isdigit():
            accession[cells[0]] = cells[1].split(".")[0]
with open(cds_table) as f:
    frame_of = {
        r["accession"]: (int(r["start"]) + int(r["codon_start"]) - 2) % 3 for r in csv.DictReader(f, delimiter="\t")
    }
first_probe = {}
with open(probes) as f:
    name = None
    sequence = []
    for line in [*f, ">end\n"]:
        if line[0] == ">":
            if name:
                first_probe.setdefault("".join(sequence).upper(), name)
            name, sequence = line[1:].split()[0], []
        else:
            sequence.append(line.strip())
calls = collections.defaultdict(list)
with open(call_table) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        calls[r["alphabet"], int(r["k"]), r["probe_id"]].append((int(r["stretch_length_aa"]), int(r["frame"])))

with open(f"{out}/frame_truth.tsv", "w") as t, open(f"{out}/frame_truth_by_length.tsv", "w") as by_length:
    t.write("alphabet\tk\tannotated\twith_passing_stretch\tlongest_in_true_frame\tshare\tall_in_true_frame\n")
    by_length.write("alphabet\tk\tmin_longest_stretch_aa\tsequences\tlongest_in_true_frame\tshare\n")
    for alphabet, k in SETTINGS:
        annotated = called = longest_right = all_right = 0
        kept = collections.Counter()
        right = collections.Counter()
        for probe in first_probe.values():
            frame = frame_of.get(accession[probe])
            if frame is None:
                continue
            annotated += 1
            found = calls.get((alphabet, k, probe))
            if not found:
                continue
            called += 1
            length, best = max(found)
            longest_right += best == frame
            all_right += all(c[1] == frame for c in found)
            for cut in (0, 30, 50, 100):
                if length >= cut:
                    kept[cut] += 1
                    right[cut] += best == frame
        t.write(f"{alphabet}\t{k}\t{annotated}\t{called}\t{longest_right}\t{longest_right / called:.3f}\t{all_right}\n")
        for cut in (0, 30, 50, 100):
            by_length.write(f"{alphabet}\t{k}\t{cut}\t{kept[cut]}\t{right[cut]}\t{right[cut] / kept[cut]:.3f}\n")
