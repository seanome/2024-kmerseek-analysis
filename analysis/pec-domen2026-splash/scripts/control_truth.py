#!/usr/bin/env python3
"""Score the known-answer control: which planted changes each channel finds.

Usage: control_truth.py <control_transcripts.fa> <truth_design.tsv> <contrast_dir> <out_dir>

A gene is the transcript ID without its `_isoA` / `_isoB` suffix. An anchor (31 nt) is placed in
every gene that contains it on either strand. An anchor placed in exactly one gene counts for that
gene; anchors in two or more genes, or in none (sequencing errors), are counted separately.

Channels:
  splash      SPLASH supervised channel: Cjs_0_pval_corrected <= 0.05 (anchors in scores.tsv)
  splash_opt  SPLASH unsupervised pval_opt, BH over all tested anchors, <= 0.05
  abundance   anchor counts, PyDESeq2 ~ pair + condition, BH q <= 0.05 and |log2FC| > 1
  abundance_condition_only   same with ~ condition
  dekupl      DE-kupl's differential k-mers (its own adjusted p <= 0.05 and |log2FC| >= 1;
              rows of diff-counts.tsv.gz), placed in genes the same way as anchors

A gene is found by a channel if at least one of its anchors is significant in that channel.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

K = 31
COMP = str.maketrans("ACGT", "TGCA")


def read_fasta(p):
    seqs, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        else:
            seqs[name].append(line.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def gene_of(t):
    return t.rsplit("_iso", 1)[0] if "_iso" in t else t


def bh(p):
    from pec_analysis_utils import bh_over_fitted_anchors

    return bh_over_fitted_anchors(p)


def main():
    fa, truth_p, cdir, out = (
        sys.argv[1],
        sys.argv[2],
        Path(sys.argv[3]),
        Path(sys.argv[4]),
    )
    out.mkdir(parents=True, exist_ok=True)
    tx = read_fasta(fa)
    truth = pl.read_csv(truth_p, separator="\t")
    gene_class = {}
    for t, c in zip(truth["transcript"], truth["class"]):
        g = gene_of(t)
        if gene_class.get(g, c) != c:
            raise SystemExit(f"gene {g} has two classes")
        gene_class[g] = c
    # transcripts that the simulator could not sample (shorter than 170 nt) are listed apart
    expressed = {gene_of(t) for t in truth["transcript"]}

    all_anchors = pl.read_csv(
        cdir / "splash_out/result.after_correction.all_anchors.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    anchors = set(all_anchors["anchor"].to_list())
    dk_path = cdir / "dekupl/DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz"
    dekupl_kmers = set()
    if dk_path.exists():
        dekupl_kmers = set(pl.read_csv(dk_path, separator="\t", columns=["kmer"])["kmer"].to_list())
    wanted = anchors | dekupl_kmers
    kmer_genes = defaultdict(set)
    for t, s in tx.items():
        g = gene_of(t)
        rc = s.translate(COMP)[::-1]
        for seq in (s, rc):
            for i in range(len(seq) - K + 1):
                k = seq[i : i + K]
                if k in wanted:
                    kmer_genes[k].add(g)

    def place(a):
        gs = kmer_genes.get(a, set())
        return next(iter(gs)) if len(gs) == 1 else ("multi" if gs else "none")

    cols = all_anchors.columns
    sup_col = "Cjs_0_pval"
    if sup_col not in cols:
        raise SystemExit(f"{sup_col} not in all_anchors columns: {cols}")
    corr = [c for c in cols if c.startswith(sup_col) and "corrected" in c]
    tab = all_anchors.select(["anchor", sup_col, "pval_opt", *corr]).with_columns(
        pl.col(sup_col).cast(pl.Float64), pl.col("pval_opt").cast(pl.Float64)
    )
    if corr:
        sup_q = tab[corr[0]].cast(pl.Float64).to_numpy()
    else:
        sup_q = bh(tab[sup_col].to_numpy())
    sig = {
        "splash": set(np.array(tab["anchor"].to_list())[sup_q <= 0.05]),
        "splash_opt": set(
            np.array(tab["anchor"].to_list())[bh(tab["pval_opt"].to_numpy()) <= 0.05]
        ),
    }
    scores_n = pl.read_csv(
        cdir / "splash_out/result.after_correction.scores.tsv",
        separator="\t",
        infer_schema_length=0,
    ).height
    for name, design in (
        ("abundance", "pair_condition"),
        ("abundance_condition_only", "condition"),
    ):
        d = pl.read_parquet(cdir / f"deseq/deseq_{design}.parquet")
        hit = d.filter(
            (pl.col("q_bh_all") <= 0.05) & (pl.col("log2FoldChange").abs() > 1)
        )
        sig[name] = set(hit["anchor"].to_list())
    if dk_path.exists():
        sig["dekupl"] = dekupl_kmers

    rows, summary = [], dict(
        tested_anchors=len(anchors),
        splash_scores_rows=scores_n,
        supervised_q_column=corr[0] if corr else "BH(Cjs_0_pval) computed here",
        channels={},
    )
    placement = Counter(
        place(a) if place(a) in ("multi", "none") else "one_gene" for a in anchors
    )
    summary["tested_anchor_placement"] = dict(placement)
    classes = sorted(set(gene_class.values()))
    class_n = Counter(gene_class.values())
    for ch, s in sig.items():
        found = Counter()
        genes_found = set()
        place_counts = Counter()
        for a in s:
            p = place(a)
            if p in ("multi", "none"):
                place_counts[p] += 1
                continue
            place_counts["one_gene"] += 1
            genes_found.add(p)
        for g in genes_found:
            found[gene_class[g]] += 1
        anchors_by_class = Counter(
            gene_class[place(a)] for a in s if place(a) not in ("multi", "none")
        )
        summary["channels"][ch] = dict(
            significant_anchors=len(s),
            placement=dict(place_counts),
            genes_found={c: found[c] for c in classes},
            genes_in_class={c: class_n[c] for c in classes},
            recall={c: round(found[c] / class_n[c], 4) for c in classes},
            significant_anchors_by_class={c: anchors_by_class[c] for c in classes},
        )
        for g in genes_found:
            rows.append(dict(channel=ch, gene=g, truth_class=gene_class[g]))
    pl.DataFrame(
        rows, schema={"channel": pl.Utf8, "gene": pl.Utf8, "truth_class": pl.Utf8}
    ).write_csv(out / "control_genes_found.tsv", separator="\t")
    (out / "control_truth_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
