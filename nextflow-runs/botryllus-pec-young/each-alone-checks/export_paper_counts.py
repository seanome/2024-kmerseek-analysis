"""Write the paper's own counts and its four young-colony DESeq2 groups from Dataset S1.

Outputs, in the current folder: paper_counts.tsv (genes x 38 vials), paper_groups.tsv
(group, vial, treat, colony) and paper_group<k>.tsv (the paper's DESeq2 results for group
k). paper_design.R reads them.

Usage: export_paper_counts.py <pnas.2610968123.sd01.xlsx>
"""

import re
import sys

import pandas as pd

SHEETS = {
    1: "Group1_2Hr_young_DEseqres",
    2: "group2_24Hr_Young_DEseqresult",
    4: "Group4_13and49Days_Young_DEseq",
    5: "group5_132Days_young_DEseq",
}


def main(xlsx):
    counts = pd.read_excel(xlsx, sheet_name="Pacemaker_Counts_Table")
    rows, group = [], None
    groups = pd.read_excel(xlsx, sheet_name="DEseq group annotation", header=None)
    for v in groups[0].dropna().astype(str):
        if v.startswith("Group"):
            group = int(v.split()[1])
            continue
        treat, vial, colony = re.search(r"PaceMaker(Con|Tre)_(JA\d+)_([^_]+)_", v).groups()
        lib = v.replace(".out.counts.txt", "")
        (col,) = [c for c in counts.columns if lib in str(c)]
        rows.append((group, vial, treat, colony, col))
    g = pd.DataFrame(rows, columns=["group", "vial", "treat", "colony", "col"])
    g = g[g.group.isin(SHEETS)]
    out = counts.set_index(counts.columns[0])[g.col.tolist()]
    out.columns = g.vial.tolist()
    out.to_csv("paper_counts.tsv", sep="\t")
    g.drop(columns="col").to_csv("paper_groups.tsv", sep="\t", index=False)
    for k, sheet in SHEETS.items():
        d = pd.read_excel(xlsx, sheet_name=sheet)
        cols = ["Gene_ID", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
        d[cols].to_csv(f"paper_group{k}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
