#!/usr/bin/env python3
"""Which genes hold a KaMRaT k-mer at all, in the table and after the test (notebook 361).

Usage: kamrat_kmers_in_genes.py <pec_root> <manifest.tsv> <out.tsv>

The old gene models exist here only as proteins, so a contig is translated in six frames and each
window of 10 residues (31 bases) is looked up in those proteins, exact match. This sees only the
protein-coding part of a gene, and misses a k-mer with a sequence difference from the gene model.

One row per label set, route, gene group and direction:
  genes                 genes in the group
  in_table              hold a k-mer of any contig KaMRaT merged, before the test
  kept_either_way       hold a k-mer of a kept contig
  kept_same_way         hold a k-mer of a kept contig going the same way as the list
Gene group `list` is the paper's list; `all` is every gene with a protein in that contrast's DESeq2
sheet, direction `any`.
"""
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

root, manifest, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
K, CONTRASTS = 10, ["young_13_49d", "young_132d", "old_41_93d"]
CODE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
NT = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGT", "TGCA")

ids = (pl.read_csv(root / "db/botznik.id_map.tsv", separator="\t").filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
       .select("stable_id", pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
gene_of = dict(ids.iter_rows())
windows = defaultdict(set)
name, seq = None, []


def add(name, seq):
    g = gene_of.get(name)
    if g:
        s = "".join(seq)
        for i in range(len(s) - K + 1):
            windows[s[i : i + K]].add(g)


for line in open(root / "db/botznik.faa"):
    if line[0] == ">":
        if name:
            add(name, seq)
        name, seq = line[1:].split()[0], []
    else:
        seq.append(line.strip())
add(name, seq)
print("10-residue windows", len(windows), flush=True)


def translate(s):
    out = []
    for i in range(0, len(s) - 2, 3):
        a, b, c = NT.get(s[i]), NT.get(s[i + 1]), NT.get(s[i + 2])
        out.append("X" if None in (a, b, c) else CODE[16 * a + 4 * b + c])
    return "".join(out)


def genes_in(contig):
    found = set()
    for s in (contig, contig.translate(COMP)[::-1]):
        for f in range(3):
            for part in translate(s[f:]).split("*"):
                for i in range(len(part) - K + 1):
                    found |= windows.get(part[i : i + K], set())
    return found


def holders(contigs):
    found = set()
    for c in contigs:
        found |= genes_in(c)
    return found


deseq = pl.read_parquet(root / "paper/paper_deseq_all.parquet", columns=["Gene_ID", "baseMean", "contrast"])
lists = pl.read_csv(root / "paper/paper_named_lists.tsv", separator="\t")
have = set(gene_of.values())
rows = []
man = pl.read_csv(manifest, separator="\t").filter(pl.col("contrast").is_in(CONTRASTS))
for c in CONTRASTS:
    with gzip.open(root / "contrasts" / c / "kamrat_ttest/contigs_all.tsv.gz", "rt") as f:
        col = f.readline().rstrip("\n").split("\t").index("contig")
        table = holders(line.split("\t")[col] for line in f)
    print(c, "genes holding a table k-mer", len(table), flush=True)
    pool = set(deseq.filter((pl.col("contrast") == c) & pl.col("baseMean").is_not_null())["Gene_ID"]) & have
    sets = [("real", "kamrat_ttest_p05", root / "contrasts" / c / "kamrat_p05"), ("real", "kamrat_deseq2_p05", root / "contrasts" / c / "kamrat_deseq2_p05")]
    sets += [(f, "kamrat_deseq2_p05", root / "label_swap" / c / f / "kamrat_deseq2_p05") for f in man.filter(pl.col("contrast") == c)["flip"]]
    for run, route, d in sets:
        kept = pl.read_csv(d / "contigs.tsv", separator="\t", columns=["contig", "direction"])
        got = {w: holders(kept.filter(pl.col("direction") == w)["contig"]) for w in ("up", "down")}
        either = got["up"] | got["down"]
        for w in ("up", "down"):
            g = set(lists.filter((pl.col("contrast") == c) & (pl.col("direction") == w))["gene_id"]) & pool
            rows.append(dict(contrast=c, run=run, route=route, group="list", direction=w, genes=len(g), in_table=len(g & table),
                             kept_either_way=len(g & either), kept_same_way=len(g & got[w])))
        rows.append(dict(contrast=c, run=run, route=route, group="all", direction="any", genes=len(pool), in_table=len(pool & table),
                         kept_either_way=len(pool & either), kept_same_way=len(pool & either)))
        print(rows[-1], flush=True)
pl.DataFrame(rows).write_csv(out, separator="\t")
print("wrote", out)
