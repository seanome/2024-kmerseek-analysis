#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
format_kmerseek_results.py

Reformat raw kmerseek CSV output into the 4-column TSV used by the
benchmark evaluator, keeping the best max_containment per (query, target).

Usage:
    format_kmerseek_results.py <raw.csv.gz> <out.tsv.gz>
"""

import csv
import gzip
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    best = {}  # (query, target) -> (max_containment, poisson_pvalue)
    opener = gzip.open if in_path.endswith(".gz") else open
    with opener(in_path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["query_name"], row["target_name"])
            mc  = float(row.get("max_containment", 0) or 0)
            pv  = float(row.get("poisson_pvalue",  1) or 1)
            if key not in best or mc > best[key][0]:
                best[key] = (mc, pv)

    with gzip.open(out_path, "wt") as out:
        for (q, t), (mc, pv) in best.items():
            out.write(f"{q}\t{t}\t{mc}\t{pv}\n")

    print(f"Wrote {len(best)} pairs to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
