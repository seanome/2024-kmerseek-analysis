"""Draw the share of each array's probe sets that get a protein long enough for a 30-amino-acid HP k-mer.

Usage: plot_protein_coverage.py <coverage.tsv> <path to the repository's notebooks folder>
One bar per array, split by what the probe set's annotation leads to: a protein of 30 amino acids or more, a nucleotide
sequence only (EST or genomic clone, untranslated), an Ensembl protein not yet fetched, or no protein of 30 amino
acids or more: no protein-coding source named, or a named source with no sequence or a shorter protein.
Writes protein_coverage.png and .svg next to the table.
"""

import csv
import sys
from collections import defaultdict

table, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

NONE = "No protein of 30 aa or more (mostly non-coding RNA)"
GROUPS = [
    (
        "Protein of 30 aa or more",
        pf.OKABE_ITO["blue"],
        {"refseq_mrna", "refseq_model", "genbank_mrna", "cdna_with_protein"},
    ),
    ("Nucleotide only: EST or genomic clone", pf.OKABE_ITO["orange"], {"est", "genomic_clone"}),
    ("Ensembl protein, not yet fetched", pf.OKABE_ITO["sky_blue"], {"ensembl_coding"}),
    (NONE, "#D9D9D9", {"other_only", "none"}),
]
sums = defaultdict(lambda: defaultdict(int))
totals = defaultdict(int)
with open(table) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        group = next(g for g, _, sources in GROUPS if r["source"] in sources)
        n = int(r["long_enough"]) if group.startswith(("Protein", "Nucleotide")) else int(r["probe_sets"])
        sums[r["array"]][group] += n
        sums[r["array"]][NONE] += int(r["probe_sets"]) - n
        totals[r["array"]] += int(r["probe_sets"])
arrays = list(totals)[::-1]
pf.use_style()
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 40)
for name, colour, _ in GROUPS:
    ax.barh([0], [0], color=colour, label=name)
for y, array in enumerate(arrays):
    left = 0
    for name, colour, _ in GROUPS:
        share = sums[array][name] / totals[array]
        ax.barh(y, share, left=left, color=colour, height=0.6)
        if share > 0.06:
            text_colour = "white" if colour == pf.OKABE_ITO["blue"] else "black"
            ax.text(
                left + share / 2,
                y,
                f"{sums[array][name]:,}\n({share:.0%})",
                ha="center",
                va="center",
                color=text_colour,
            )
        left += share
ax.set_yticks(range(len(arrays)), [f"{a}\n{totals[a]:,} probe sets" for a in arrays])
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the array's probe sets")
pf.shared_legend(fig, ncol=2)
pf.save(fig, table.rsplit("/", 1)[0] + "/protein_coverage")
