"""Write one comparison's design file, as kamrat_table.py does, without building a k-mer table.

Usage: write_design.py <groups.tsv> <dataset> <comparison> <design out>
"""

import csv
import sys

groups_file, dataset, comparison, design_out = sys.argv[1:]
with open(groups_file) as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["dataset"] == dataset and r["comparison"] == comparison]
if not rows:
    sys.exit(f"{groups_file} has no samples for {dataset} {comparison}")
with open(design_out, "w") as f:
    f.writelines(f"{r['sample']}\t{'post' if r['group'] == '1' else 'pre'}\n" for r in rows)
