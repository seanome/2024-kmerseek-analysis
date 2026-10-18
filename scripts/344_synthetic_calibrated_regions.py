#!/usr/bin/env python3
"""Made-up region files whose E-values are right by construction, to test notebook 344.

    344_synthetic_calibrated_regions.py <dir>

<dir> holds decoys/ (make_decoys.py's output on the human proteome), human.fasta, and
results/decoys and results/seg filled the way the pipeline fills them. This writes
results/regions/<kind>.regions.tsv.gz and .summary.tsv. Each decoy gets a Poisson(1)
number of regions; each region's region_run_evalue and region_ka_evalue are drawn
uniformly from 0 to 1 (region_ka_evalue left empty for 10% of rows). So over Q decoys
about Q x E regions have a value at or below E, which is what a calibrated E-value means.
Target and residues are random stretches of real proteins. Not a result: it only shows
what the notebook and the pre-registered rule do when the E-values are right.
"""

import csv, glob, gzip, random, sys

S = sys.argv[1]
rng = random.Random(1)
human = {}
name = None
for line in open(f"{S}/human.fasta"):
    if line[0] == ">":
        name = line[1:].split()[0]
        human[name] = []
    else:
        human[name].append(line.strip())
human = {k: "".join(v) for k, v in human.items()}
names = list(human)
cols = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_subseq",
    "target_subseq",
    "moltype_seq",
    "region_length",
    "region_n_mismatches",
    "region_n_shared_kmers",
    "region_n_chained",
    "region_run_length",
    "region_pr_same",
    "region_run_evalue",
    "region_ka_bits",
    "region_ka_evalue",
    "region_evalue",
    "region_evalue_source",
    "db_n_targets",
    "region_search_space",
]


def poisson(m):
    import math

    L, k, p = math.exp(-m), 0, 1.0
    while True:
        p *= rng.random()
        if p < L:
            return k
        k += 1


for kind in ("composition", "dipeptide", "window10"):
    out = gzip.open(f"{S}/results/regions/{kind}.regions.tsv.gz", "wt")
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(cols)
    n = 0
    for f in sorted(glob.glob(f"{S}/decoys/{kind}.chunk_*.fasta")):
        lines = open(f).read().split("\n")
        for h, seq in zip(lines[0::2], lines[1::2]):
            q = h[1:]
            for _ in range(poisson(1.0)):
                t = rng.choice(names)
                ts = human[t]
                L = min(30, len(seq), len(ts))
                a, b = rng.randrange(len(seq) - L + 1), rng.randrange(len(ts) - L + 1)
                run = rng.random()
                ka = "" if rng.random() < 0.1 else rng.random()
                w.writerow(
                    [
                        q,
                        t,
                        a,
                        a + L,
                        b,
                        b + L,
                        seq[a : a + L],
                        ts[b : b + L],
                        "",
                        L,
                        3,
                        2,
                        1,
                        19,
                        0.5,
                        run,
                        30,
                        ka,
                        ka if ka != "" else run,
                        "ka",
                        20600,
                        0,
                    ]
                )
                n += 1
    out.close()
    with open(f"{S}/results/regions/{kind}.summary.tsv", "w") as fh:
        fh.write("label\trows_in\trows_kept\trows_ka_empty\tqueries_with_rows\n")
        fh.write(f"{kind}\t{n}\t{n}\t0\t0\n")
    print(kind, n)
