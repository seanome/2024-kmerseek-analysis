#!/usr/bin/env python3
"""The Fig. 4C-D score of Domen et al. 2026 from reads per gene counted with no genome (notebook 361).

Usage: fig4_score_from_reads.py <counts_root> <analysis_dir> <out_dir>

counts_root holds <series>/share_<N>/gene.counts.tsv, gene.summary.tsv and gene_counts.time, as
nextflow-runs/pec-domen2026-fig4-score writes them; share_<N> is the first 1/N of each run's reads.
analysis_dir is analysis/pec-domen2026-splash. Plan and rules, written before any count:
design/fig4_score_from_reads/README.md.

Score, the formula of Voskoboynik et al. 2024 as in scripts/exercise_score.py (PR 212):
  expression = log2(CPM + 1), CPM = a gene's reads per million reads that went to any gene
  value of a gene in a sample = (expression - (StepMiner threshold + 0.5)) / 3 / sd,
    threshold and sd over every sample of the series
  score = -1 x sum over cluster 13 + 1 x sum over cluster 14 + 2 x sum over cluster 3

Writes fig4_reads_comparisons.tsv (one row per series and share of reads, with the three rules),
fig4_reads_samples.tsv (one row per sample) and fig4_reads_random.tsv.gz (the random gene sets).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score

WEIGHTS = {13: -1, 14: 1, 3: 2}  # cluster: weight
# series: (column of design/human/samples.tsv, first group, second group); the authors' groups
GROUPS = {
    "GSE120862": ("training", "untrained", "trained"),
    "GSE99963": ("time", "Pre", "Post"),
    "GSE144304": ("group", "Young", "Fit"),
}
AUC_WITHIN = 0.15  # rule 1
MIN_SPEARMAN = 0.8  # rule 2
MAX_RANDOM_SHARE = 0.05  # rule 3
N_RANDOM = 1000
SEED = 361


def stepminer_threshold(x):
    """Halfway between the two means of the best one-step fit of the sorted values."""
    a = np.sort(x)
    n = len(a)
    if a[0] == a[-1]:
        return float(a[0])
    s, s2 = np.cumsum(a), np.cumsum(a * a)
    k = np.arange(1, n)  # the left part holds the k smallest values
    left = s2[k - 1] - s[k - 1] ** 2 / k
    right = s2[-1] - s2[k - 1] - (s[-1] - s[k - 1]) ** 2 / (n - k)
    b = int(np.argmin(left + right)) + 1
    return float((a[:b].mean() + a[b:].mean()) / 2)


def gene_values(expr):
    """(expression - (threshold + 0.5)) / 3 / sd for every gene (rows) and sample (columns)."""
    x = expr.to_numpy(float)
    thr = np.array([stepminer_threshold(row) for row in x]) + 0.5
    sd = x.std(axis=1)
    v = (x - thr[:, None]) / 3
    v = np.divide(v, sd[:, None], out=v.copy(), where=sd[:, None] > 0)
    return pd.DataFrame(v, index=expr.index, columns=expr.columns)


def log_cpm(counts):
    return np.log2(counts / counts.sum() * 1e6 + 1)


def auc_of(score, first, second):
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    return float(roc_auc_score(labels, np.r_[score[first], score[second]]))


def read_time(path):
    """CPU seconds (user + system), wall seconds and peak memory in GB from /usr/bin/time -v."""
    t = Path(path).read_text()
    user = float(re.search(r"User time \(seconds\): ([\d.]+)", t).group(1))
    system = float(re.search(r"System time \(seconds\): ([\d.]+)", t).group(1))
    parts = [float(p) for p in re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", t).group(1).split(":")]
    wall = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    peak_gb = float(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1e6
    return user + system, wall, peak_gb


def random_aucs(values, expr, listed, first, second, rng):
    """AUC of N_RANDOM scores in which each listed gene is swapped for a gene from the same tenth of
    mean expression. Clusters and weights are kept; a draw never uses a gene twice or a listed gene."""
    tenth = pd.qcut(expr.mean(axis=1).rank(method="first"), 10, labels=False)
    pool = {t: np.array([g for g in tenth.index[tenth == t] if g not in listed]) for t in range(10)}
    weights = np.array([WEIGHTS[listed[g]] for g in listed], float)
    tenths = np.array([tenth[g] for g in listed])
    v = values.to_numpy(float)
    row = {g: i for i, g in enumerate(values.index)}
    cols = [values.columns.get_loc(s) for s in first + second]
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    out = np.empty(N_RANDOM)
    for d in range(N_RANDOM):
        picks = np.empty(len(tenths), dtype=int)
        for t in np.unique(tenths):
            where = np.flatnonzero(tenths == t)
            picks[where] = [row[g] for g in rng.choice(pool[t], size=len(where), replace=False)]
        out[d] = roc_auc_score(labels, (weights[:, None] * v[picks]).sum(axis=0)[cols])
    return out


def main(counts_root, analysis, out):
    counts_root, analysis, out = Path(counts_root), Path(analysis), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clusters = {
        c: [g.strip().upper() for g in open(analysis / f"design/exercise_score/cluster_{c}.genes.txt") if g.strip()]
        for c in WEIGHTS
    }
    assert [len(clusters[c]) for c in WEIGHTS] == [48, 137, 153], "the authors' lists are 48 + 137 + 153"
    design = pd.read_csv(analysis / "design/human/samples.tsv", sep="\t")
    printed = pd.read_csv(analysis / "design/exercise_score/published_fig4.tsv", sep="\t")
    printed = printed[printed.comparison == "main"].set_index("dataset")
    table = pd.read_csv(analysis / "exports/exercise_score/exercise_score_samples.tsv", sep="\t")
    table = table[(table.comparison == "main") & (table.grouping == "authors")]
    rng = np.random.default_rng(SEED)

    rows, samples, randoms = [], [], []
    for series, (column, name0, name1) in GROUPS.items():
        d = design[design.series == series].set_index("run")
        # PR 212 names GSE144304 samples by the first part of the GEO title
        d["sample"] = d.title.str.split("_").str[0] if series == "GSE144304" else d.title
        first, second = list(d.index[d[column] == name0]), list(d.index[d[column] == name1])
        want = printed.loc[series]
        assert (len(first), len(second)) == (want.n0, want.n1), f"{series}: group sizes differ from the printed ones"
        from_table = table[table.dataset == series].set_index("sample").score
        shares = sorted(int(p.name.split("_")[1]) for p in (counts_root / series).glob("share_*"))
        for share in shares:
            folder = counts_root / series / f"share_{share}"
            counts = pd.read_csv(folder / "gene.counts.tsv", sep="\t", index_col=0)
            assert set(counts.columns) == set(d.index), f"{series} share {share}: runs differ from the design"
            summary = pd.read_csv(folder / "gene.summary.tsv", sep="\t").set_index("sample")
            assert (counts.sum() == summary.assigned[counts.columns]).all(), "counts do not add up to the summary"
            counts = counts[list(d.index)]
            expr = log_cpm(counts)
            values = gene_values(expr)
            listed = {g: c for c, genes in clusters.items() for g in genes if g in values.index}
            score = sum(WEIGHTS[c] * values.loc[[g for g in listed if listed[g] == c]].sum() for c in WEIGHTS)
            auc = auc_of(score, first, second)
            both = d.loc[first + second]
            rho = float(spearmanr(score[both.index], from_table[both["sample"]].to_numpy()).statistic)
            draws = random_aucs(values, expr, listed, first, second, rng)
            # added after the plan (see the results README): every gene with a read, each with weight 1
            auc_all = auc_of(values.sum(), first, second)
            low_side = want.auc < 0.5
            as_far = float(((draws <= auc) if low_side else (draws >= auc)).mean())
            cpu_s, wall_s, peak_gb = read_time(folder / "gene_counts.time")
            rule1 = bool((auc < 0.5) == low_side and abs(auc - want.auc) <= AUC_WITHIN)
            rows.append(
                dict(
                    series=series,
                    share_of_reads=f"1/{share}" if share > 1 else "all",
                    share=share,
                    group0=name0,
                    group1=name1,
                    n0=len(first),
                    n1=len(second),
                    auc_from_reads=auc,
                    printed_auc=want.auc,
                    rule1_printed_auc_comes_back=rule1,
                    welch_p=float(ttest_ind(score[first], score[second], equal_var=False).pvalue),
                    printed_p=want.p,
                    spearman_with_gene_table=rho,
                    rule2_same_samples_score_the_same=bool(rho >= MIN_SPEARMAN),
                    random_sets_as_far=as_far,
                    random_auc_median=float(np.median(draws)),
                    rule3_listed_genes_matter=bool(as_far <= MAX_RANDOM_SHARE),
                    auc_all_genes_weight_1=auc_all,
                    **{f"listed_genes_found_cluster_{c}": sum(v == c for v in listed.values()) for c in WEIGHTS},
                    listed_genes_found=len(listed),
                    genes_with_a_read=len(counts),
                    reads_counted=int(summary.pairs.sum()),
                    reads_assigned_to_a_gene=int(summary.assigned.sum()),
                    median_reads_assigned_per_sample=float(summary.assigned.median()),
                    cpu_seconds=cpu_s,
                    wall_seconds=wall_s,
                    peak_memory_gb=peak_gb,
                )
            )
            samples.append(
                pd.DataFrame(
                    {
                        "series": series,
                        "share": share,
                        "run": both.index,
                        "sample": both["sample"].to_numpy(),
                        "group": [name0] * len(first) + [name1] * len(second),
                        "score_from_reads": score[both.index].to_numpy(),
                        "score_from_gene_table": from_table[both["sample"]].to_numpy(),
                    }
                )
            )
            randoms.append(pd.DataFrame({"series": series, "share": share, "draw": range(N_RANDOM), "auc": draws}))
            print(series, share, f"AUC {auc:.3f} printed {want.auc}", f"rho {rho:.3f}", f"random as far {as_far:.3f}")
    pd.DataFrame(rows).to_csv(out / "fig4_reads_comparisons.tsv", sep="\t", index=False)
    pd.concat(samples).to_csv(out / "fig4_reads_samples.tsv", sep="\t", index=False)
    pd.concat(randoms).to_csv(out / "fig4_reads_random.tsv.gz", sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
