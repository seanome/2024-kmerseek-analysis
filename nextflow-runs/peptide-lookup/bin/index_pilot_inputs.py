"""Make nested, reproducible protein subsets for measuring index growth."""

import argparse
import hashlib
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument("database")
p.add_argument("out_dir")
p.add_argument("--sizes", type=int, nargs="+", default=[20000, 100000])
a = p.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
sizes = tuple(sorted(set(a.sizes)))
assert sizes and min(sizes) > 0
# Lowest SHA256 values give a uniform sample without using catalogue order.
import heapq

chosen = []
with open(a.database) as fh:
    for line in fh:
        if line.startswith(">"):
            acc = line.split()[0][1:]
            key = int.from_bytes(
                hashlib.sha256(("300:" + acc).encode()).digest(), "big"
            )
            if len(chosen) < max(sizes):
                heapq.heappush(chosen, (-key, acc))
            elif key < -chosen[0][0]:
                heapq.heapreplace(chosen, (-key, acc))
ordered = [acc for _, acc in sorted(chosen, reverse=True)]
assert len(ordered) == max(sizes)
selected = {n: set(ordered[:n]) for n in sizes}
handles = {n: open(out / f"db_{n}.fasta", "w") for n in sizes}
counts = {n: [0, 0] for n in sizes}
keep = []
with open(a.database) as fh:
    for line in fh:
        if line.startswith(">"):
            acc = line.split()[0][1:]
            keep = [n for n in sizes if acc in selected[n]]
            for n in keep:
                counts[n][0] += 1
        else:
            for n in keep:
                counts[n][1] += len(line.strip())
        for n in keep:
            handles[n].write(line)
for h in handles.values():
    h.close()
with open(out / "input_sizes.tsv", "w") as fh:
    fh.write("n_sequences\tn_residues\n")
    for n in sizes:
        assert counts[n][0] == n
        fh.write(f"{n}\t{counts[n][1]}\n")
