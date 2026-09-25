#!/usr/bin/env python3
"""Shortest k per alphabet at which a seed from a random position of the human proteome
(QfO 2020_04) is contained in fewer than 100 human proteins, measured with
`kmerseek index --stats-only --kmer-stats-out` (kmerseek-region:0.4.0-rc5-arm64):
proteins per seed = sum(occurrences^2 * n_kmers) / sum(occurrences * n_kmers), where
occurrences is the number of proteins containing a seed.

Usage: python scripts/elm_seed_floor.py <out_dir> <human.fasta>
"""

from pathlib import Path

W = Path(sys.argv[1])
H = sys.argv[2]
ALPHA = [
    ("protein20", 5),
    ("uniprot18", 5),
    ("hsdm17", 5),
    ("wass14", 5),
    ("mmseqs12", 5),
    ("sdm12", 6),
    ("dayhoff6", 8),
    ("wwmj5", 8),
    ("gbmr7", 9),
    ("gbmr4", 12),
    ("hp_lehninger_hpc3", 16),
    ("hp_lehninger2", 18),
    ("hp_lehninger_c_nonpolar2", 18),
    ("hp_pbotc_1st_ed2", 18),
    ("hp_thomas_dill2", 19),
    ("hp_thomas_dill_no_c2", 19),
    ("hp_kyte_doolittle2", 19),
    ("polarity4", 10),
    ("funcgroups8", 7),
]


def stat(a, k):
    out = W / f"human.{a}.k{k}.csv"
    if not out.exists():
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                "/Users/olga/data:/Users/olga/data",
                "--entrypoint",
                "kmerseek",
                "kmerseek-region:0.4.0-rc5-arm64",
                "index",
                "--alphabet",
                a,
                "--ksize",
                str(k),
                "--input",
                H,
                "--output",
                str(W / f"tmp_{a}_{k}"),
                "--stats-only",
                "--kmer-stats-out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    s1 = s2 = 0
    for r in csv.DictReader(l for l in open(out) if not l.startswith("#")):
        o, n = int(r["occurrences"]), int(r["n_kmers"])
        s1 += o * n
        s2 += o * o * n
    return s2 / s1


print("alphabet,pipeline_kmin,k,proteins_per_seed")
for a, kmin in ALPHA:
    k = max(3, kmin - 6)
    while True:
        v = stat(a, k)
        print(f"{a},{kmin},{k},{v:.1f}", flush=True)
        if v < 100 or k > kmin + 12:
            break
        k += 1
