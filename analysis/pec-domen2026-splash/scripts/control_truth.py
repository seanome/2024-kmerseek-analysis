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
  dekupl_limma, dekupl_DESeq2   DE-kupl's differential k-mers with each test (column `tag`; every row of diff-counts.tsv.gz is taken,
              so any p-value and fold-change cut is the one set in the DE-kupl config, not applied here;
              a missing output is listed under `channels_not_run`, not scored as zero hits), placed in genes the same way as anchors
  kamrat_ttest   KaMRaT t-test on DE-kupl's k-mer counts: k-mers with BH-corrected p <= 0.05
              (kamrat_ttest/kmers_padj05.tsv.gz, column `tag`), placed in genes the same way
  kamrat_ttest_p05   KaMRaT contigs, the same t-test recomputed per contig, uncorrected p < 0.05,
              no fold-change cutoff (kamrat_p05/contigs.tsv; select_kamrat_nominal.py ... 0)
  kamrat_deseq2_p05, kamrat_deseq2_q05   KaMRaT contigs tested with PyDESeq2 ~ pair + condition
              (kamrat_deseq2/deseq/deseq_pair_condition.parquet): uncorrected p < 0.05, or BH q <= 0.05;
              no fold-change cutoff. Every 31-mer of a kept contig counts as kept and is placed in
              genes the same way as an anchor (added 2026-11-02).
  dekupl_limma_p05, dekupl_DESeq2_p05   every k-mer DE-kupl tested with uncorrected p < 0.05, no fold-change
              cutoff (raw_pvals.txt.gz), placed in genes the same way (added 2026-11-02)

A gene is found by a channel if at least one of its anchors is significant in that channel.
Anchors shared by two or more genes (paralogs in the old gene models) are not given to any gene;
they are split into those whose gene set holds a planted change and those in unchanged genes only,
and planted genes reached only through shared anchors are counted apart.
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
    dekupl_kmers = {}
    dekupl_not_run = []
    for method in ("limma", "DESeq2"):
        dk_path = cdir / f"dekupl_{method}/DEkupl_result/control_vs_PEC_kmer_counts/diff-counts.tsv.gz"
        if dk_path.exists():
            dekupl_kmers[method] = set(pl.read_csv(dk_path, separator="\t", columns=["tag"])["tag"].to_list())
        else:
            dekupl_not_run.append(method)
            print(f"WARNING: {dk_path} is missing; dekupl_{method} is reported as not run", file=sys.stderr)
    km_path = cdir / "kamrat_ttest/kmers_padj05.tsv.gz"
    if km_path.exists():
        dekupl_kmers["kamrat"] = set(pl.read_csv(km_path, separator="\t", columns=["tag"])["tag"].to_list())
    else:
        dekupl_not_run.append("kamrat")
        print(f"WARNING: {km_path} is missing; kamrat_ttest is reported as not run", file=sys.stderr)
    # KaMRaT contigs (added 2026-11-02): every 31-mer of a kept contig is taken as kept, so a
    # contig is placed in genes the same way as an anchor
    contig_sets = {}
    for name, path, col, cut in (
        ("kamrat_ttest_p05", cdir / "kamrat_p05/contigs.tsv", "pvalue", 0.05),
        ("kamrat_deseq2_p05", cdir / "kamrat_deseq2/deseq/deseq_pair_condition.parquet", "pvalue", 0.05),
        ("kamrat_deseq2_q05", cdir / "kamrat_deseq2/deseq/deseq_pair_condition.parquet", "q_bh_all", 0.05),
    ):
        if not path.exists():
            dekupl_not_run.append(name)
            print(f"WARNING: {path} is missing; {name} is reported as not run", file=sys.stderr)
            continue
        t = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path, separator="\t")
        seq = "anchor" if "anchor" in t.columns else "contig"
        kept = t.filter(pl.col(col) < cut if col == "pvalue" else pl.col(col) <= cut)[seq].to_list()
        contig_sets[name] = {c[i : i + K] for c in kept for i in range(len(c) - K + 1)}
        print(f"{name}: {len(kept)} contigs, {len(contig_sets[name])} 31-mers", file=sys.stderr)
    # DE-kupl limma, every tested k-mer with uncorrected p < 0.05 (raw_pvals.txt.gz, no header)
    for method in ("limma", "DESeq2"):
        rp = cdir / f"dekupl_{method}/DEkupl_result/control_vs_PEC_kmer_counts/raw_pvals.txt.gz"
        if rp.exists():
            t = pl.read_csv(rp, separator="\t", has_header=False, new_columns=["tag", "pvalue"],
                            schema_overrides={"pvalue": pl.Float64}, null_values=["NA"])
            contig_sets[f"dekupl_{method}_p05"] = set(t.filter(pl.col("pvalue") < 0.05)["tag"].to_list())
        else:
            dekupl_not_run.append(f"dekupl_{method}_p05")
    wanted = anchors.union(*dekupl_kmers.values(), *contig_sets.values())
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
        dq = cdir / f"deseq/deseq_{design}.parquet"
        if not dq.exists():  # the equal-depth runs (slurm/20_equal_depth.sbatch) have SPLASH only
            dekupl_not_run.append(name)
            print(f"WARNING: {dq} is missing; {name} is reported as not run", file=sys.stderr)
            continue
        d = pl.read_parquet(dq)
        hit = d.filter(
            (pl.col("q_bh_all") <= 0.05) & (pl.col("log2FoldChange").abs() > 1)
        )
        sig[name] = set(hit["anchor"].to_list())
    for method, kmers in dekupl_kmers.items():
        sig["kamrat_ttest" if method == "kamrat" else f"dekupl_{method}"] = kmers
    sig.update(contig_sets)

    rows, summary = [], dict(
        tested_anchors=len(anchors),
        splash_scores_rows=scores_n,
        supervised_q_column=corr[0] if corr else "BH(Cjs_0_pval) computed here",
        channels={},
        channels_not_run=["kamrat_ttest" if m == "kamrat" else m if m.startswith("abundance") or m.startswith("kamrat_") or m.endswith("_p05") else f"dekupl_{m}"
                          for m in dekupl_not_run],
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
        # anchors in two or more genes: does the gene set hold a planted change, or only unchanged genes?
        multi_split = Counter()
        multi_genes_changed = Counter()
        for a in s:
            gs = kmer_genes.get(a, set())
            if len(gs) < 2:
                continue
            cls = {gene_class[g] for g in gs}
            if cls == {"unchanged"}:
                multi_split["all_unchanged"] += 1
            else:
                multi_split["includes_planted_change"] += 1
                for g in gs:
                    if gene_class[g] != "unchanged":
                        multi_genes_changed[g] += 1
        found_shared = Counter(gene_class[g] for g in multi_genes_changed if g not in genes_found)
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
            shared_anchor_split=dict(multi_split),
            planted_genes_found_only_through_shared_anchors={c: found_shared[c] for c in classes},
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
