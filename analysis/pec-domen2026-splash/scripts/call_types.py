#!/usr/bin/env python3
"""What kind of sequence difference each SPLASH call is, where reached peptides sit in the protein,
and how many reached genes rest only on peptides shared with other genes (notebook 361).

Usage: call_types.py <contrasts_dir> <botznik.id_map.tsv> <out_dir> [control/truth_design.tsv]

Reads existing outputs only (no new SPLASH runs).

1. Target distances (H3, allele or isoform). For every anchor SPLASH keeps (Cjs_0_pval_corrected
   <= 0.05, result.after_correction.scores.tsv), take its two most abundant targets (the 31 bases
   right after the anchor, pooled over samples) and count the positions where they differ
   (Hamming distance) and the fewest single-base edits that turn one into the other (edit
   distance). A one-base change in the target is a single-base allele or a sequencing error; a
   long stretch of differences is a different exon (splicing) or a similar gene (paralog). A
   Hamming distance of 4 or more with an edit distance of 3 or less is a small insertion or
   deletion that shifts the rest of the target. The same is done for the anchors kept by DESeq2
   on anchor counts (compactors_anchor_nominal/anchors_nominal.tsv, targets looked up in
   all_anchors.tsv) and, as a reference for "any anchor", for a random 300,000 of all tested
   anchors with two or more targets.
2. Position in the protein (H4). For every peptide with a supported match to an old gene-model
   protein (E <= 1e-5, query coverage >= 50 %), take its best match by bit score and place the
   middle of the aligned stretch on the protein: (target_start - 1 + target_end) / 2 / target_len,
   0 at the start (N-terminus) and 1 at the end (C-terminus). Coordinates from pair_align.py are
   1-based and inclusive.
3. Gene-unique peptides (H5). A peptide is gene-unique when all its supported matches fall in one
   gene (g<N>; all isoforms g<N>.t<M> count as one gene). Genes reached with all peptides (same
   rule as reached_genes.py) against genes reached with gene-unique peptides only.

Outputs in <out_dir>: target_distance_bins.tsv, target_distance_summary.tsv,
protein_position_bins.tsv, protein_position_summary.tsv, gene_unique_reach.tsv,
control_reach_truth.tsv (with the truth file: the control's reached genes by planted class), and
target_distances.parquet (one row per kept anchor, not committed).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

contrasts_dir, id_map, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
truth = sys.argv[4] if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
CONTRASTS = ["control", "young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NULL_CONTRASTS = ["control", "young_2h", "young_24h"]  # all_anchors.tsv is 1-6 GB each
ROUTES = {"splash": "search_compactors", "anchor_nominal": "search_compactors_anchor_nominal"}
HAM_EDGES = [(1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 31, ">10")]
TCOLS = ["anchor", "most_freq_target_1", "cnt_most_freq_target_1", "most_freq_target_2",
         "cnt_most_freq_target_2", "Cjs_0_pval_corrected"]
N_NULL = 300_000
POS_BINS = 20


def as_codes(s):
    """Targets as an (N, 31) uint8 array; every SPLASH target here is 31 bases."""
    lens = s.str.len_bytes().unique().to_list()
    assert lens == [31], f"target lengths {lens}"
    return np.frombuffer("".join(s.to_list()).encode(), dtype=np.uint8).reshape(-1, 31)


def edit_distance(a, b):
    """Levenshtein distance row by row for two (N, L) arrays, vectorised over N."""
    n, L = a.shape
    prev = np.tile(np.arange(L + 1, dtype=np.int16), (n, 1))
    for i in range(1, L + 1):
        cur = np.empty_like(prev)
        cur[:, 0] = i
        for j in range(1, L + 1):
            cur[:, j] = np.minimum(np.minimum(prev[:, j] + 1, cur[:, j - 1] + 1),
                                   prev[:, j - 1] + (a[:, i - 1] != b[:, j - 1]))
        prev = cur
    return prev[:, L]


def distances(df):
    """Add hamming and edit distances; drop anchors with one target (logged by the caller)."""
    df = df.filter(pl.col("most_freq_target_2").is_not_null() & (pl.col("most_freq_target_2") != ""))
    a, b = as_codes(df["most_freq_target_1"]), as_codes(df["most_freq_target_2"])
    return df.with_columns(pl.Series("hamming", (a != b).sum(axis=1).astype(np.int16)),
                           pl.Series("edit", edit_distance(a, b)))


def ham_bin(col="hamming"):
    e = pl.when(pl.col(col) <= 0).then(pl.lit("0"))
    for lo, hi, lab in HAM_EDGES:
        e = e.when(pl.col(col).is_between(lo, hi)).then(pl.lit(lab))
    return e.otherwise(pl.lit("?")).alias("hamming_bin")


def read_targets(path, lazy_filter=None):
    q = pl.scan_csv(path, separator="\t", infer_schema=False).select(TCOLS)
    if lazy_filter is not None:
        q = lazy_filter(q)
    return q.collect(engine="streaming").with_columns(
        pl.col("cnt_most_freq_target_1", "cnt_most_freq_target_2").cast(pl.Float64),
        pl.col("Cjs_0_pval_corrected").cast(pl.Float64))


sets = []
for c in CONTRASTS:
    so = contrasts_dir / c / "splash_out"
    kept = read_targets(so / "result.after_correction.scores.tsv").filter(pl.col("Cjs_0_pval_corrected") <= 0.05)
    d = distances(kept)
    print(c, "splash kept", kept.height, "with 2+ targets", d.height, flush=True)
    sets.append(d.with_columns(pl.lit(c).alias("contrast"), pl.lit("splash").alias("anchor_set")))
    nom = contrasts_dir / c / "compactors_anchor_nominal" / "anchors_nominal.tsv"
    big = so / "result.after_correction.all_anchors.tsv"
    if nom.exists():
        names = pl.read_csv(nom, separator="\t", columns=["anchor"])
        t = read_targets(big, lambda q: q.join(names.lazy(), on="anchor", how="semi"))
        d = distances(t)
        print(c, "anchor_nominal", names.height, "found in all_anchors", t.height, "with 2+ targets", d.height, flush=True)
        sets.append(d.with_columns(pl.lit(c).alias("contrast"), pl.lit("anchor_nominal").alias("anchor_set")))
    if c in NULL_CONTRASTS:
        t = read_targets(big, lambda q: q.filter(pl.col("most_freq_target_2").is_not_null()
                                                 & (pl.col("most_freq_target_2") != "")))
        n_tested = t.height
        t = t.sample(n=min(N_NULL, t.height), seed=361)
        d = distances(t)
        print(c, "all tested anchors with 2+ targets", n_tested, "sampled", d.height, flush=True)
        sets.append(d.with_columns(pl.lit(c).alias("contrast"), pl.lit("all_tested_sample").alias("anchor_set")))

dist = pl.concat(sets).with_columns(ham_bin(), (pl.col("cnt_most_freq_target_2") / pl.col("cnt_most_freq_target_1")).alias("count_ratio"))
dist.write_parquet(out / "target_distances.parquet")
bins = (dist.group_by("contrast", "anchor_set", "hamming_bin")
        .agg(pl.len().alias("anchors"),
             (pl.col("edit") <= 3).sum().alias("edit_distance_le_3"),
             pl.col("count_ratio").median().alias("median_second_over_first_count"))
        .join(dist.group_by("contrast", "anchor_set").agg(pl.len().alias("total")), on=["contrast", "anchor_set"])
        .with_columns((pl.col("anchors") / pl.col("total")).alias("fraction"))
        .sort("contrast", "anchor_set", "hamming_bin"))
bins.write_csv(out / "target_distance_bins.tsv", separator="\t")
summ = (dist.group_by("contrast", "anchor_set")
        .agg(pl.len().alias("anchors"),
             pl.col("hamming").median().alias("median_hamming"),
             (pl.col("hamming") == 1).mean().alias("frac_hamming_1"),
             (pl.col("hamming") <= 3).mean().alias("frac_hamming_le_3"),
             ((pl.col("hamming") >= 4) & (pl.col("edit") <= 3)).mean().alias("frac_small_indel"),
             ((pl.col("hamming") >= 4) & (pl.col("edit") > 3)).mean().alias("frac_long_difference"),
             (pl.col("hamming") > 10).mean().alias("frac_hamming_gt_10"),
             pl.col("count_ratio").filter(pl.col("hamming") == 1).median().alias("median_count_ratio_hamming_1"),
             (pl.col("count_ratio").filter(pl.col("hamming") == 1) < 0.05).mean().alias("frac_hamming_1_ratio_lt_0.05"))
        .sort("contrast", "anchor_set"))
summ.write_csv(out / "target_distance_summary.tsv", separator="\t")
with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
    print(bins)
    print(summ)

# ---- 2 and 3: peptides on the old gene models
genes = (pl.read_csv(id_map, separator="\t", columns=["stable_id", "original_id"])
         .with_columns(pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
pos_rows, pos_summ, uniq = [], [], []
edges = np.linspace(0, 1, POS_BINS + 1)
for c in CONTRASTS:
    for route, sub in ROUTES.items():
        f = contrasts_dir / c / sub / "align.botznik.parquet"
        if not f.exists():
            print(c, route, "no alignment file", flush=True)
            continue
        a = (pl.read_parquet(f, columns=["query", "target", "target_len", "target_start", "target_end", "bit_score", "supported"])
             .filter(pl.col("supported")).join(genes, left_on="target", right_on="stable_id"))
        ng = a.group_by("query").agg(pl.col("gene").n_unique().alias("n_genes"))
        all_genes = a.select("gene").unique().height
        uq = a.join(ng.filter(pl.col("n_genes") == 1), on="query", how="semi")
        if c == "control" and truth:
            t = (pl.read_csv(truth, separator="\t").with_columns(pl.col("transcript").str.replace("_iso[AB]$", "").alias("gene"))
                 .unique("gene").select("gene", "class"))
            pl.concat([x.select("gene").unique().join(t, on="gene", how="left")
                       .with_columns(pl.col("class").fill_null("not in control"))
                       .group_by("class").len().rename({"len": "genes"}).with_columns(pl.lit(rule).alias("peptides_used"))
                       for rule, x in [("all", a), ("gene_unique", uq)]]).sort("peptides_used", "class") \
                .select("peptides_used", "class", "genes").write_csv(out / "control_reach_truth.tsv", separator="\t")
            print(pl.read_csv(out / "control_reach_truth.tsv", separator="\t"), flush=True)
        uniq.append(dict(contrast=c, route=route, peptides=ng.height,
                         gene_unique_peptides=int((ng["n_genes"] == 1).sum()),
                         genes_all_peptides=all_genes, genes_gene_unique_peptides=uq.select("gene").unique().height))
        best = (a.sort("bit_score", descending=True).unique("query", keep="first")
                .with_columns(((pl.col("target_start") - 1 + pl.col("target_end")) / 2 / pl.col("target_len")).alias("rel")))
        best = best.with_columns((1.0 / pl.len().over("gene")).alias("gene_weight"))
        rel, w = best["rel"].to_numpy(), best["gene_weight"].to_numpy()
        h, _ = np.histogram(rel, edges)
        hw, _ = np.histogram(rel, edges, weights=w)
        for i in range(POS_BINS):
            pos_rows.append(dict(contrast=c, route=route, bin_start=edges[i], bin_end=edges[i + 1], peptides=int(h[i]),
                                 fraction=h[i] / h.sum(), gene_weighted_fraction=hw[i] / hw.sum()))
        pos_summ.append(dict(contrast=c, route=route, peptides=len(rel), genes=best["gene"].n_unique(),
                             median=float(np.median(rel)), frac_first_fifth=float((rel < 0.2).mean()),
                             frac_last_fifth=float((rel >= 0.8).mean()),
                             gene_weighted_first_fifth=float(w[rel < 0.2].sum() / w.sum()),
                             gene_weighted_last_fifth=float(w[rel >= 0.8].sum() / w.sum())))
        print(c, route, pos_summ[-1], uniq[-1], flush=True)
pl.DataFrame(pos_rows).write_csv(out / "protein_position_bins.tsv", separator="\t")
pl.DataFrame(pos_summ).write_csv(out / "protein_position_summary.tsv", separator="\t")
pl.DataFrame(uniq).with_columns((pl.col("genes_all_peptides") - pl.col("genes_gene_unique_peptides")).alias("genes_only_shared_peptides"),
                                (1 - pl.col("genes_gene_unique_peptides") / pl.col("genes_all_peptides")).alias("fraction_only_shared")) \
    .write_csv(out / "gene_unique_reach.tsv", separator="\t")
with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
    print(pl.DataFrame(pos_summ))
    print(pl.read_csv(out / "gene_unique_reach.tsv", separator="\t"))
print("wrote", out)
