#!/usr/bin/env python3
"""Redo the exercise score of Domen et al. 2026 Fig. 4C-D from GEO tables (notebook 361).

Usage: exercise_score.py <geo_dir> <gpl_dir> <design_exercise_score_dir> <out_dir>

geo_dir holds the GEO series matrices and the count tables named in LOADERS; gpl_dir holds the GEO
platform tables, <GPL>.txt. The method is Voskoboynik et al. 2024 (bone.py, getRanks2 and
mergeRanks in their repository):

  for each probe of each listed gene, all probes kept, genes matched on the upper-case name:
    value in a sample = (expression - (StepMiner threshold + 0.5)) / 3 / sd
    sd and threshold use every sample of the series, not only the two groups compared
  cluster value = sum over its probes
  score = -1 x cluster 13 + 1 x cluster 14 + 2 x cluster 3
  GSE27536 in Fig. 4C uses -1 x cluster 13 alone

StepMiner fits one step to the sorted values of a probe, at the split with the smallest squared
error, and the threshold is halfway between the two means. Each group pair is compared by Welch's
t-test and by the area under the ROC curve (AUC), with the second group as positive.

Expression, as their methods state: GEO's normalised values on a log2 scale, log2(FPKM + 1) for
GSE120862, log2(CPM + 1) for the two count tables, with reads in no gene left out of the total. CPM
is counts per million reads. GSE27536 and GSE28998 get log2(x + 1) on top of GEO's log2 values, found
after the first run because it gives the printed p values; the README lists the first-run numbers.

A comparison counts as recovered, a rule fixed before the first run, when the AUC rounds to the
printed AUC and p is within a factor of 2 of the printed p (AUC alone for GSE83578 24 h, which
prints no p). The group sizes must match too.

"authors" groups are those in the authors' notebook. "as_labelled" groups follow the words on the
published panel, where those words and the notebook disagree.

Writes to out_dir: exercise_score_comparisons.tsv, exercise_score_samples.tsv and
exercise_score_probe_counts.tsv.
"""

import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

geo, gpl_dir, design, out = (Path(p) for p in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
CLUSTERS = {13: -1, 14: 1, 3: 2}
genes = {c: [g.strip().upper() for g in open(design / f"cluster_{c}.genes.txt") if g.strip()] for c in CLUSTERS}
assert [len(genes[c]) for c in CLUSTERS] == [48, 137, 153], "the authors' lists are 48 + 137 + 153"


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


def read_matrix(name):
    """Series matrix: (expression table with probes as rows, sample table)."""
    meta, table = defaultdict(list), []
    with gzip.open(geo / name, "rt", errors="replace") as f:
        inside = False
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                inside = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif inside:
                table.append(line.rstrip("\n").replace('"', "").split("\t"))
            elif line.startswith("!Sample_"):
                k, *v = line.rstrip("\n").split("\t")
                meta[k].append([x.strip('"') for x in v])
    expr = pd.DataFrame(table[1:], columns=table[0]).set_index("ID_REF").apply(pd.to_numeric, errors="coerce")
    s = pd.DataFrame(
        {"gsm": table[0][1:], "title": meta["!Sample_title"][0], "source": meta["!Sample_source_name_ch1"][0]}
    )
    for c in meta["!Sample_characteristics_ch1"]:
        key = c[0].split(":")[0].strip()
        s[key] = [x.split(": ", 1)[-1] for x in c]
    return expr, s.set_index("gsm")


def probe_genes(gpl, column, split):
    """Probe id -> gene names (upper case), from a GEO platform table."""
    lines = open(gpl_dir / f"{gpl}.txt", errors="replace").read().split("!platform_table_begin\n")[1]
    t = pd.read_csv(pd.io.common.StringIO(lines.split("!platform_table_end")[0]), sep="\t", dtype=str, low_memory=False)
    out_map = {}
    for p, v in zip(t["ID"], t[column]):
        if isinstance(v, str):
            out_map[p] = split(v)
    return out_map


def gpl17586_names(v):
    # gene_assignment: "accession // SYMBOL // title // location // id /// next accession ..."
    return {x.split(" // ")[1].strip().upper() for x in v.split(" /// ") if len(x.split(" // ")) > 1}


def slashes(v):
    return {x.strip().upper() for x in v.split("///") if x.strip()}


def log2_if_linear(expr):
    return np.log2(expr.clip(lower=0) + 1) if expr.max().max() > 100 else expr


def rna_gene_names():
    t = pd.read_csv(geo / "GSE120862_FPKM.txt.gz", sep="\t", usecols=["gene_id", "gene_name"], dtype=str)
    t["gene_id"] = t["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    return {i: {n.upper()} for i, n in zip(t["gene_id"], t["gene_name"])}


def cpm_log(c):
    c = c[c.index.str.startswith("ENSG")]  # GSE144304 ends in HTSeq's __no_feature style rows, which are not genes
    return np.log2(c / c.sum() * 1e6 + 1)


def load(dataset):
    """Expression (log2), probe -> genes, sample table with the fields used for groups."""
    if dataset == "GSE104079":
        e, s = read_matrix("GSE104079_series_matrix.txt.gz")
        return e, probe_genes("GPL13912", "GENE_SYMBOL", slashes), s
    if dataset in ("GSE27536", "GSE28998"):
        e, s = read_matrix(f"{dataset}_series_matrix.txt.gz")
        # already log2 in GEO; a second log2(x + 1) is what gives the printed p values (see README)
        return np.log2(e + 1), probe_genes("GPL570", "Gene Symbol", slashes), s
    if dataset == "GSE83578":
        e, s = read_matrix("GSE83578_series_matrix.txt.gz")
        return e, probe_genes("GPL13667", "Gene Symbol", slashes), s
    if dataset == "GSE111555":
        e, s = read_matrix("GSE111555_series_matrix.txt.gz")
        return e, probe_genes("GPL17586", "gene_assignment", gpl17586_names), s
    if dataset == "GSE120862":
        t = pd.read_csv(geo / "GSE120862_FPKM.txt.gz", sep="\t").set_index("gene_id").drop(columns="gene_name")
        t.index = t.index.str.replace(r"\.\d+$", "", regex=True)
        parts = t.columns.str.extract(
            r"(?P<training>\w+?)_(?P<time>baseline|\+1h|\+4h)_(?P<leg>NE|Ex)_(?P<subject>\d+)"
        )
        parts.index = t.columns
        return np.log2(t + 1), rna_gene_names(), parts
    if dataset == "GSE99963":
        t = pd.read_csv(geo / "GSE99963_GeneCoverage.tsv.gz", sep="\t").set_index("GeneID")
        _, m = read_matrix("GSE99963_series_matrix.txt.gz")
        s = m.set_index("title")
        assert set(s.index) == set(t.columns)
        return cpm_log(t), rna_gene_names(), s
    if dataset == "GSE144304":
        t = pd.read_csv(geo / "GSE144304_JoinedCounts.txt.gz", sep="\t", index_col=0)
        t.index = t.index.str.replace(r"\.\d+$", "", regex=True)
        _, m = read_matrix("GSE144304_series_matrix.txt.gz")
        m["key"] = m["title"].str.split("_").str[0]
        m = m.set_index("key")
        t.columns = t.columns.str.split("_").str[0]
        assert set(m.index) == set(t.columns)
        return cpm_log(t), rna_gene_names(), m
    raise ValueError(dataset)


def cluster_values(expr, p2g, drop=()):
    """One row per sample, one column per cluster, and the probes counted per cluster."""
    expr = log2_if_linear(expr)
    res, counts = {}, {}
    for c, lst in genes.items():
        want = set(lst) - set(drop)
        ids = [p for p in expr.index if p in p2g and p2g[p] & want]
        # a probe that names two listed genes counts once per gene, as a lookup by gene does
        mult = [len(p2g[p] & want) for p in ids]
        X = expr.loc[ids].to_numpy(float)
        keep = ~np.isnan(X[:, -1])  # the authors skip a probe whose last sample is blank
        X, mult = X[keep], np.array(mult)[keep]
        total = np.zeros(X.shape[1])
        for row, m in zip(X, mult):
            ok = ~np.isnan(row)
            thr = stepminer_threshold(row[ok]) + 0.5
            sd = np.nan_to_num(row).std()
            v = (np.nan_to_num(row) - thr) / 3
            v = v / sd if sd > 0 else v
            total += m * v
        res[c], counts[c] = total, int(mult.sum())
    return pd.DataFrame(res, index=expr.columns), counts


def score(cv, model):
    return -cv[13] if model == "m1_only" else sum(w * cv[c] for c, w in CLUSTERS.items())


def compare(sc, a, b):
    x, y = sc.loc[a].to_numpy(), sc.loc[b].to_numpy()
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    return dict(
        n0=len(x),
        n1=len(y),
        mean0=x.mean(),
        mean1=y.mean(),
        our_auc=roc_auc_score(lab, np.r_[x, y]),
        our_p=ttest_ind(x, y, equal_var=False).pvalue,
    )


def where(s, **kw):
    m = pd.Series(True, index=s.index)
    for k, v in kw.items():
        m &= s[k.replace("_", " ")].isin(v if isinstance(v, list) else [v])
    return list(s.index[m])


def comparisons(d, s):
    """(comparison, grouping, words, samples of group 0, samples of group 1, model)."""
    if d == "GSE104079":
        yield (
            "main",
            "authors",
            "alternate-day fasting vs exercised + fasting, liver and soleus pooled",
            where(s, treatment="Alternate day fasting"),
            where(s, treatment="Exercised_Alternate day fasting"),
            "full",
        )
        yield (
            "main",
            "as_labelled",
            "soleus muscle, control vs exercised; GEO has no pre and post of one animal",
            where(s, treatment="Control", tissue="Soleus"),
            where(s, treatment="Exercised", tissue="Soleus"),
            "full",
        )
    if d == "GSE27536":
        yield (
            "main",
            "authors",
            "sedentary vs trained, healthy and COPD pooled, cluster 13 alone",
            where(s, exercise_state="sedentary"),
            where(s, exercise_state="trained"),
            "m1_only",
        )
        copd = "chronic obstructive pulmonary disease (COPD)"
        yield (
            "main",
            "as_labelled",
            "COPD only, sedentary vs trained, cluster 13 alone",
            where(s, exercise_state="sedentary", disease_state=copd),
            where(s, exercise_state="trained", disease_state=copd),
            "m1_only",
        )
        yield (
            "full_model",
            "authors",
            "sedentary vs trained, all three clusters (their Table S1)",
            where(s, exercise_state="sedentary"),
            where(s, exercise_state="trained"),
            "full",
        )
    if d == "GSE120862":
        yield (
            "main",
            "authors",
            "all 42 untrained vs all 42 trained samples, by training status",
            where(s, training="untrained"),
            where(s, training="trained"),
            "full",
        )
        yield (
            "main",
            "as_labelled",
            "untrained, exercised leg, before vs 1 h and 4 h after",
            where(s, training="untrained", leg="Ex", time="baseline"),
            where(s, training="untrained", leg="Ex", time=["+1h", "+4h"]),
            "full",
        )
        yield (
            "leg_after_bout",
            "as_labelled",
            "untrained, 1 h and 4 h after, rested vs exercised leg",
            where(s, training="untrained", leg="NE", time=["+1h", "+4h"]),
            where(s, training="untrained", leg="Ex", time=["+1h", "+4h"]),
            "full",
        )
        yield (
            "rest_after_training",
            "as_labelled",
            "rested leg at rest, before vs after 2 months of training",
            where(s, training="untrained", leg="NE", time="baseline"),
            where(s, training="trained", leg="NE", time="baseline"),
            "full",
        )
    if d == "GSE83578":
        t = "time (pre-exercise and post-exercise 30 min, 3h, 24h)"
        for name, v in (("pre_vs_30min", "1"), ("pre_vs_3h", "2"), ("pre_vs_24h", "3")):
            pre = list(s.index[s[t] == "0"])
            post = list(s.index[s[t] == v])
            yield (name, "authors", f"time 0 vs time {v}, control and LPS blood pooled", pre, post, "full")
            ctrl = s["treatment"] == "control"
            yield (
                name,
                "as_labelled",
                f"time 0 vs time {v}, control blood only, no LPS",
                list(s.index[(s[t] == "0") & ctrl]),
                list(s.index[(s[t] == v) & ctrl]),
                "full",
            )
        yield (
            "early_vs_rest",
            "authors",
            "times 0 and 3 vs times 1 and 2, control and LPS blood pooled (their Table S1)",
            list(s.index[s[t].isin(["0", "3"])]),
            list(s.index[s[t].isin(["1", "2"])]),
            "full",
        )
    if d == "GSE111555":
        yield (
            "main",
            "authors",
            "PBMC, before vs after training",
            where(s, source="PBMCs, pretraining, resting state"),
            where(s, source="PBMCs, posttraining, resting state"),
            "full",
        )
    if d == "GSE99963":
        arm = s["source"].str.split("_").str
        yield (
            "main",
            "authors",
            "before vs after, sedentary and exercise arms pooled",
            list(s.index[arm[1] == "Pre"]),
            list(s.index[arm[1] == "Post"]),
            "full",
        )
        yield (
            "main",
            "as_labelled",
            "exercise arm only, before vs after",
            list(s.index[(arm[1] == "Pre") & (arm[0] == "Exercise")]),
            list(s.index[(arm[1] == "Post") & (arm[0] == "Exercise")]),
            "full",
        )
    if d == "GSE144304":
        yield (
            "main",
            "authors",
            "young adults (panel: Sed) vs fit old adults (panel: Ex)",
            where(s, group="Young"),
            where(s, group="Fit"),
            "full",
        )
        yield (
            "main",
            "as_labelled",
            "frail old adults vs fit old adults",
            where(s, group="Frail"),
            where(s, group="Fit"),
            "full",
        )
    if d == "GSE28998":
        yield (
            "main",
            "authors",
            "untrained arm vs trained arm, both 4 h after one bout",
            where(s, training_status="untrained"),
            where(s, training_status="trained"),
            "full",
        )


pub = pd.read_csv(design / "published_fig4.tsv", sep="\t").set_index(["dataset", "comparison"])
exp_counts = pd.read_csv(design / "expected_probe_counts.tsv", sep="\t").set_index("dataset")
rows, samples, counts = [], [], []
for d in ["GSE104079", "GSE27536", "GSE120862", "GSE83578", "GSE111555", "GSE99963", "GSE144304", "GSE28998"]:
    expr, p2g, s = load(d)
    expr = expr[[c for c in expr.columns if c in s.index]]
    cv, n = cluster_values(expr, p2g)
    print(
        d,
        expr.shape,
        "probes per cluster",
        n,
        "authors",
        exp_counts.loc[d].to_dict(),
        "max expr",
        round(float(expr.max().max()), 1),
    )
    counts.append(
        dict(
            dataset=d,
            samples=expr.shape[1],
            **{f"ours_{c}": n[c] for c in CLUSTERS},
            **{f"authors_{c}": exp_counts.loc[d, f"cluster{c}"] for c in CLUSTERS},
        )
    )
    # adding a variant with SDCCAG3 removed: the authors' name table seems to lack it (see README)
    no_gene = cluster_values(expr, p2g, drop=["SDCCAG3"])[0] if d in ("GSE99963", "GSE144304") else None
    for name, grouping, words, a, b, model in comparisons(d, s):
        sc = score(cv, model)
        r = dict(dataset=d, comparison=name, grouping=grouping, groups=words, model=model, **compare(sc, a, b))
        key = (d, name)
        if key in pub.index and grouping == "authors":
            q = pub.loc[key]
            r.update(published_n0=q.n0, published_n1=q.n1, published_auc=q.auc, published_p=q.p)
            same_n = (r["n0"], r["n1"]) == (q.n0, q.n1)
            auc_ok = f"{r['our_auc']:.2f}" == f"{q.auc:.2f}"
            p_ok = np.isnan(q.p) or 0.5 <= r["our_p"] / q.p <= 2
            r.update(same_n=same_n, auc_recovered=auc_ok, p_recovered=p_ok, recovered=same_n and auc_ok and p_ok)
        rows.append(r)
        if no_gene is not None and name == "main" and grouping == "authors":
            r.update(
                {
                    "our_auc_without_SDCCAG3": compare(score(no_gene, model), a, b)["our_auc"],
                    "our_p_without_SDCCAG3": compare(score(no_gene, model), a, b)["our_p"],
                }
            )
        print(
            "  ",
            name,
            grouping,
            r["n0"],
            r["n1"],
            round(r["our_auc"], 3),
            f"{r['our_p']:.3g}",
            r.get("published_auc", ""),
            r.get("published_p", ""),
            r.get("recovered", ""),
        )
        samples.append(
            pd.DataFrame(
                {
                    "dataset": d,
                    "comparison": name,
                    "grouping": grouping,
                    "group": [0] * len(a) + [1] * len(b),
                    "sample": a + b,
                    "score": sc.loc[a + b].to_numpy(),
                }
            )
        )
pd.DataFrame(rows).to_csv(out / "exercise_score_comparisons.tsv", sep="\t", index=False)
pd.concat(samples).to_csv(out / "exercise_score_samples.tsv", sep="\t", index=False)
pd.DataFrame(counts).to_csv(out / "exercise_score_probe_counts.tsv", sep="\t", index=False)
