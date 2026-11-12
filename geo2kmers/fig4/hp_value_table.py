"""Write one series' HP k-mer values, one row for all k-mers held by the same probe sets.

Usage: hp_value_table.py <array>.<alphabet>.kmers.npz <series>.probe_set_values.tsv.gz <out.tsv.gz>
Both inputs are hp_kmer_table.py's. K-mers held by the same probe sets have the same values in every sample, so one
row stands for all of them: its `kmers` column says how many, and `probe_sets` names the probe sets. A row's value is
the mean over its probe sets, as in the tables. Row names depend only on the .npz file, so two series on one array
share them. Rows with no value in some sample, or the same value in every sample, are left out. The rules are in
hp_tests_plan.md.
"""

import gzip
import sys

import numpy as np

kmers_file, values_file, out = sys.argv[1:]
data = np.load(kmers_file)
starts, sharing, owners, names = (data["start"], data["probe_sets"], data["owner"], data["names"])

# A k-mer in one probe set belongs to that probe set's row; rows for k-mers in several follow, in order of first use
groups = {}
row_of = np.empty(len(starts), dtype=np.int64)
single = sharing == 1
row_of[single] = owners[starts[single]]
for i in np.flatnonzero(~single):
    key = owners[starts[i] : starts[i] + sharing[i]].tobytes()
    row_of[i] = groups.setdefault(key, len(names) + len(groups))
kmers_per_row = np.bincount(row_of, minlength=len(names) + len(groups))
members = [np.array([i]) for i in range(len(names))] + [np.frombuffer(key, dtype=owners.dtype) for key in groups]

with gzip.open(values_file, "rt") as f:
    samples = f.readline().rstrip("\n").split("\t")[1:]
    by_name = {line.split("\t", 1)[0]: np.array(line.rstrip("\n").split("\t")[1:], dtype=float) for line in f}
missing = np.full(len(samples), np.nan)
table = np.array([by_name.get(n, missing) for n in names])

written = no_value = constant = 0
with gzip.open(out, "wt") as f:
    f.write("feature\tkmers\tprobe_sets\t" + "\t".join(samples) + "\n")
    for row, held in enumerate(members):
        if not kmers_per_row[row]:
            continue
        part = table[held]
        present = np.isfinite(part).sum(axis=0)
        mean = np.where(present > 0, np.nansum(part, axis=0) / np.maximum(present, 1), np.nan)
        if not np.isfinite(mean).all():
            no_value += 1
            continue
        if mean.max() == mean.min():
            constant += 1
            continue
        written += 1
        f.write(f"r{row}\t{kmers_per_row[row]}\t{','.join(names[held])}\t" + "\t".join(f"{v:.4f}" for v in mean) + "\n")
print(
    f"{written} rows written for {int(kmers_per_row.sum())} k-mers; {no_value} rows with a missing value and "
    f"{constant} constant rows left out"
)
