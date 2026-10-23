"""Per-depth comparison of SPLASH, DE-kupl and the kallisto reference route.

Reads the tables made by prepare_methods.py, grade_genes.py, annotate.py and
05_baseline.sbatch, and writes comparison.json plus TSVs for the demo page.

Rules (fixed before results):
  * Genes are matched only through strict 31-mer gene grading (grade_genes.py)
    or through local alignment of the contigs/compactors; never by anchor or
    k-mer string. Several contigs/anchors from one gene count once.
  * Each method keeps its own statistics; no combined p-value.
  * Published DE (Zhang 2019) needs the same direction. DE-kupl direction is
    its log2FC sign; SPLASH direction is descriptive (prepare_methods.py);
    kallisto's is the PyDESeq2 log2FoldChange sign.
  * Agreement between methods is corroboration on the same reads, not
    independent validation.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

SAMPLES = ["DW1", "DW2", "DW3", "ZW1", "ZW2", "ZW3"]
K = 31
COMP = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s.translate(COMP)[::-1]


def sign_word(x):
    return "higher in breed" if x > 0 else "higher in wild" if x < 0 else "equal"


def gene_direction(dirs):
    s = set(dirs)
    return s.pop() if len(s) == 1 else "mixed"


def describe(values):
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return dict(n=0)
    return dict(n=int(len(v)), min=float(v.min()), q1=float(v.quantile(.25)), median=float(v.median()),
                q3=float(v.quantile(.75)), max=float(v.max()))


def align_pairs(dk_seqs, sp_seqs, min_identity=0.95, min_length=40):
    """DE-kupl contig vs SPLASH compactor pairs that share a 31-mer seed,
    confirmed by local alignment (Biopython, match 1, mismatch -2, gaps -3/-1)."""
    index = defaultdict(set)
    for sid, dna in sp_seqs.items():
        for i in range(len(dna) - K + 1):
            k = dna[i:i + K]
            index[min(k, rc(k))].add(sid)
    aligner = Align.PairwiseAligner(mode="local", match_score=1, mismatch_score=-2,
                                    open_gap_score=-3, extend_gap_score=-1)
    pairs = []
    for cid, dna in dk_seqs.items():
        cands = set()
        for i in range(len(dna) - K + 1):
            k = dna[i:i + K]
            cands |= index.get(min(k, rc(k)), set())
        for sid in sorted(cands):
            best = None
            for strand, target in (("+", sp_seqs[sid]), ("-", rc(sp_seqs[sid]))):
                aln = aligner.align(dna, target)[0]
                c = aln.counts()
                length = c.identities + c.mismatches + c.gaps
                ident = c.identities / length if length else 0
                if best is None or c.identities > best["identities"]:
                    best = dict(dekupl=cid, splash=sid, strand=strand, identities=c.identities,
                                aligned_length=length, identity=round(ident, 4))
            if best and best["aligned_length"] >= min_length and best["identity"] >= min_identity:
                pairs.append(best)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="per-depth analysis directory")
    p.add_argument("--baseline", required=True, help="baseline/<depth> directory")
    p.add_argument("--published", required=True)
    a = p.parse_args()
    w = Path(a.work)
    counts = json.loads((w / "method_counts.json").read_text())
    grades = pd.read_csv(w / "grades.tsv", sep="\t", dtype={"gene": str})
    anchors = pd.read_csv(w / "splash_anchors.tsv", sep="\t")
    sp_seqs = pd.read_csv(w / "splash_sequences.tsv", sep="\t")
    contigs = pd.read_csv(w / "dekupl_contigs.tsv", sep="\t")
    published = pd.DataFrame(json.loads(Path(a.published).read_text()))
    published["gene_id"] = published["gene_id"].astype(str)
    published["direction"] = published["log2fc"].map(sign_word)
    pub_dir = dict(zip(published.gene_id, published.direction))

    g = grades.set_index("id")
    anchors["group_id"] = "splash:" + anchors["anchor"]
    anchors = anchors.join(g[g.level == "group"][["status", "gene", "kmers_in_reference", "genes_hit"]],
                           on="group_id")
    anchors["status"] = anchors["status"].fillna("no compactor")
    contigs = contigs.join(g[g.level == "sequence"][["status", "gene", "kmers_in_reference", "genes_hit"]],
                           on="contig_id")

    # Baseline
    base = Path(a.baseline)
    de = pd.read_csv(base / "de" / "de_results.tsv", sep="\t", dtype={"feature": str})
    agg = json.loads((base / "aggregation.json").read_text())
    kal_sig = de[de["significant"]]
    kal_dir = dict(zip(kal_sig.feature, kal_sig.log2FoldChange.map(sign_word)))

    # Gene sets per discovery method (strict only; one gene counts once).
    dk_strict = contigs[contigs.status == "strict"]
    dk_genes = {gene: gene_direction(grp.direction) for gene, grp in dk_strict.groupby("gene")}
    sp_strict = anchors[anchors.status == "strict"]
    sp_genes = {gene: gene_direction(grp.direction_descriptive) for gene, grp in sp_strict.groupby("gene")}

    def recovered(genes):
        return sorted(gn for gn, d in genes.items() if gn in pub_dir and d == pub_dir[gn])

    rec = dict(dekupl=recovered(dk_genes), splash=recovered(sp_genes), kallisto=recovered(kal_dir))
    seq_route = set(rec["dekupl"]) | set(rec["splash"])
    only_seq = sorted(seq_route - set(rec["kallisto"]))

    # Sequence-level agreement for findings that have no strict gene.
    dk_unassigned = contigs[contigs.status != "strict"]
    sp_unassigned_ids = set(anchors.loc[anchors.status != "strict", "group_id"])
    sp_un = sp_seqs[sp_seqs.group_id.isin(sp_unassigned_ids)]
    pairs = align_pairs(dict(zip(dk_unassigned.contig_id, dk_unassigned.contig)),
                        dict(zip(sp_un.seq_id, sp_un.dna)))
    pd.DataFrame(pairs).to_csv(w / "aligned_pairs.tsv", sep="\t", index=False)
    seq_to_group = dict(zip(sp_seqs.seq_id, sp_seqs.group_id))

    status_counts = dict(
        dekupl_contigs=dict(Counter(contigs.status)),
        splash_anchors=dict(Counter(anchors.status)))
    absent = dict(
        dekupl=dict(contigs=int((contigs.status == "absent").sum()),
                    abs_log2fc=describe(contigs.loc[contigs.status == "absent", "log2FC"].abs()),
                    fold_change_median=float(2 ** contigs.loc[contigs.status == "absent", "log2FC"].abs().median())
                    if (contigs.status == "absent").any() else None),
        splash=dict(anchors=int((anchors.status == "absent").sum()),
                    abs_log2_descriptive=describe(anchors.loc[anchors.status == "absent",
                                                              "log2_breed_over_wild_descriptive"].abs())))
    summary = dict(
        depth=counts["depth"], method_counts=counts, status_counts=status_counts,
        strict_genes=dict(dekupl=len(dk_genes), splash=len(sp_genes), kallisto_significant=len(kal_dir),
                          dekupl_and_splash=len(set(dk_genes) & set(sp_genes))),
        published_total=len(published),
        published_recovered_same_direction={k: len(v) for k, v in rec.items()},
        published_recovered_by_sequence_route_not_kallisto=len(only_seq),
        aligned_pairs_without_strict_gene=dict(
            pairs=len(pairs), dekupl_contigs=len({p["dekupl"] for p in pairs}),
            splash_anchors=len({seq_to_group[p["splash"]] for p in pairs})),
        absent_from_reference=absent,
        kallisto=dict(tested_genes=int(len(de)), significant=int(len(kal_sig)),
                      p_pseudoaligned={s: agg["sample_counts"][s]["p_pseudoaligned"] for s in SAMPLES},
                      not_pseudoaligned_percent={s: round(100 - agg["sample_counts"][s]["p_pseudoaligned"], 2)
                                                 for s in SAMPLES}),
        lengths=dict(dekupl_contig_nt=describe(contigs.contig_length),
                     splash_compactor_nt=describe(sp_seqs.dna.str.len())),
    )
    (w / "comparison.json").write_text(json.dumps(summary, indent=2))
    gene_rows = []
    for gene in sorted(set(dk_genes) | set(sp_genes) | set(kal_dir) | set(pub_dir)):
        gene_rows.append(dict(gene=gene, dekupl=dk_genes.get(gene, ""), splash_descriptive=sp_genes.get(gene, ""),
                              kallisto=kal_dir.get(gene, ""), published=pub_dir.get(gene, ""),
                              published_recovered_by=",".join(k for k, v in rec.items() if gene in v)))
    pd.DataFrame(gene_rows).to_csv(w / "gene_table.tsv", sep="\t", index=False)
    contigs.to_csv(w / "dekupl_contigs_graded.tsv", sep="\t", index=False)
    anchors.to_csv(w / "splash_anchors_graded.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
