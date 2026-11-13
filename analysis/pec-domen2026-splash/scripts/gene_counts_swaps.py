#!/usr/bin/env python3
"""gene-counts sample_mean rule under swapped labels, against lists and against DESeq2 (notebook 361).

Usage: gene_counts_swaps.py <new_counts.tsv> <old_counts.tsv> <analysis_dir> <nfcore_deseq_dir>
       <nfcore_design_dir> <paper_named_lists.tsv> <botznik.id_map.tsv> <out_dir> [n_cpus]

Rules: exports/cheap_route/README.md, "Thinned reads and swapped labels". One row for each real
design and each of the 14 designs in design/label_swap_kamrat/manifest.tsv.

A. reached_paper / reached_nfcore: list genes the rule gives the list's direction, using the design's
   labels. Lists always come from the real labels.
B. PyDESeq2 on the nf-core gene counts with the same labels and the same pair or genotype term, Wald
   test PEC vs control, poscounts size factors. deseq2_genes: p <= 0.05 and 2-fold. agree: those genes
   the rule gives the same direction, on new-proteome counts.

Writes <out_dir>/swaps.tsv after every row, so a stopped run keeps what it finished.
"""
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

new_counts, old_counts, analysis, deseq_dir, nf_design, lists_tsv, id_map, out = sys.argv[1:9]
n_cpus = int(sys.argv[9]) if len(sys.argv) > 9 else 8
analysis, out = Path(analysis), Path(out)
(out / "directions").mkdir(parents=True, exist_ok=True)
SCRIPT = Path(__file__).resolve().parent / "gene_directions.py"
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
warnings.filterwarnings("ignore", category=UserWarning)


def unit(contrast):
    return "genotype" if contrast == "young_132d" else "pair"


def directions(counts, design, tag):
    tsv = out / "directions" / f"{tag}.tsv"
    subprocess.run([sys.executable, str(SCRIPT), counts, str(design), str(tsv), "sample_mean"], check=True, stdout=subprocess.DEVNULL)
    d = pd.read_csv(tsv, sep="\t", keep_default_na=False)
    return dict(zip(d.gene, d.direction))


def deseq(contrast, design):
    meta = pd.read_csv(design, sep="\t", dtype=str).set_index("sample")
    c = pd.read_csv(f"{nf_design}/{contrast}/rsem_counts_whole.tsv", sep="\t", index_col=0)[list(meta.index)].T
    c = c.loc[:, c.sum() > 0].round().astype(int)
    meta["unit"] = meta[unit(contrast)].map(lambda s: "u" + re.sub(r"\W", "_", s))
    dds = DeseqDataSet(counts=c, metadata=meta[["unit", "condition"]], design="~unit + condition", refit_cooks=True,
                       size_factors_fit_type="poscounts", inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "PEC", "control"], inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    st.summary()
    return st.results_df


lists = pd.read_csv(lists_tsv, sep="\t")
proteins = pd.read_csv(id_map, sep="\t").original_id
with_protein = set(proteins[proteins.str.fullmatch(r"g\d+\.t\d+")].str.replace(r"\.t\d+$", "", regex=True))
nf = {c: pd.read_csv(f"{deseq_dir}/{c}/tables/differential/rnaseq/{c}_pec_domen2026.deseq2.results.tsv", sep="\t") for c in CONTRASTS}
runs = [dict(key=f"{c}/real", contrast=c, design=analysis / f"design/contrast_{c}.design.tsv") for c in CONTRASTS]
m = pd.read_csv(analysis / "design/label_swap_kamrat/manifest.tsv", sep="\t")
runs += [dict(key=r.key, contrast=r.contrast, design=analysis / r.design) for r in m.itertuples()]
real_lists = {}
rows = []
for r in runs:
    c, tag = r["contrast"], r["key"].replace("/", ".")
    new, old = directions(new_counts, r["design"], f"{tag}.new"), directions(old_counts, r["design"], f"{tag}.old")
    p = lists[(lists.contrast == c) & lists.gene_id.isin(with_protein)]
    n = nf[c][(nf[c].pvalue <= 0.05) & (nf[c].log2FoldChange.abs() > 1)]
    want = ["up" if x > 0 else "down" for x in n.log2FoldChange]
    res = deseq(c, r["design"])
    hit = res[(res.pvalue <= 0.05) & (res.log2FoldChange.abs() > 1)]
    dirs = ["up" if x > 0 else "down" for x in hit.log2FoldChange]
    row = dict(key=r["key"], contrast=c, swapped=not r["key"].endswith("/real"), paper_genes=len(p),
               reached_paper=sum(old.get(g) == d for g, d in zip(p.gene_id, p.direction)), nfcore_genes=len(n),
               reached_nfcore=sum(new.get(g) == d for g, d in zip(n.gene_id, want)), deseq2_genes=len(hit),
               agree=sum(new.get(g) == d for g, d in zip(hit.index, dirs)))
    if not row["swapped"]:
        real_lists[c] = set(hit.index)
    row["deseq2_genes_also_on_real_pydeseq2_list"] = len(real_lists[c] & set(hit.index))
    row["deseq2_genes_also_on_nfcore_list"] = len(set(n.gene_id) & set(hit.index))
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(out / "swaps.tsv", sep="\t", index=False)
