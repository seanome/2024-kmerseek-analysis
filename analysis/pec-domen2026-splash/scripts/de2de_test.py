#!/usr/bin/env python3
"""Do Botryllus PEC k-mers move the same way in human muscle? (notebook 361, design/de2de/README.md)

Usage: de2de_test.py <de2de dir> <design/human dir> <human_runs.tsv> <out dir> [<botryllus set,...>]

The optional last argument names the Botryllus sets under <de2de dir>/botryllus; the default is the
three real contrasts. The label-swap control passes the 7 swap sets (young_13_49d__pairflip1, ...).

<de2de dir> holds botryllus/<contrast>/{up,down}.codes.tsv and human/<run>.counts.tsv.gz.
Writes <out dir>/de2de_results.tsv (one row per Botryllus contrast, setting and human contrast),
<out dir>/de2de_null.tsv.gz (D under every shuffle) and prints the results.

Runs are read one at a time, twice: first to total each k-mer's count per human contrast (which
k-mers are kept), then to sum each run's log2 CPM over the kept up and down k-mers. A k-mer absent
from a run has count 0, so log2(0 + 0.5) = -1; it is added as such without being stored.
"""
import gzip
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

root, design_dir, runs_tsv, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
BOT = sys.argv[5].split(",") if len(sys.argv) > 5 else ["young_13_49d", "young_132d", "old_41_93d"]
SETTINGS = [("protein", 7), ("dayhoff", 13), ("hp", 25)]
HUMAN = {"GSE99963": ("gse99963_fig4_post_vs_pre", True), "GSE144304": ("gse144304_fig4_fit_vs_young", False),
         "GSE120862": ("gse120862_fig4_trained_vs_untrained", True)}
MIN_COUNT, N_SHUFFLE, ALPHA, ABSENT = 10, 10000, 0.05 / 27, np.log2(0.5)
rng = np.random.default_rng(361)
runs = pl.read_csv(runs_tsv, separator="\t")


def weights(design, paired):
    """Rows of run weights, real labels first, then every shuffle. D = weights @ v."""
    samples = design["sample"].to_list()
    pec = (design["condition"] == "PEC").to_numpy()
    if not paired:
        def w(lab):
            return np.where(lab, 1 / lab.sum(), -1 / (~lab).sum())
        return samples, np.array([w(pec)] + [w(rng.permutation(pec)) for _ in range(N_SHUFFLE)]), False
    subj = design["subject"].to_list()
    subjects = sorted(set(subj))
    # one subject's difference: mean of its second-group runs minus mean of its first-group runs
    base = np.zeros((len(subjects), len(samples)))
    for i, s in enumerate(subjects):
        m = np.array([x == s for x in subj])
        base[i] = np.where(m & pec, 1 / (m & pec).sum(), 0) - np.where(m & ~pec, 1 / (m & ~pec).sum(), 0)
    signs = np.array(list(itertools.product([1, -1], repeat=len(subjects))))  # first row: all +1, the real labels
    return samples, signs @ base / len(subjects), True


human = {}
for series, (contrast, paired) in HUMAN.items():
    design = pl.read_csv(design_dir / f"contrast_{contrast}.design.tsv", separator="\t")
    samples, W, exact = weights(design, paired)
    human[series] = dict(contrast=contrast, samples=samples, W=W, exact=exact)
series_of = {r: s for s, h in human.items() for r in h["samples"]}
assert set(series_of) == set(runs["run"]), "human_runs.tsv and the designs list different runs"

# Query k-mers per setting, and each Botryllus contrast's label: 1 up, -1 down, 0 neither or both
q, lab, n_codes = {}, {}, {}
for a, k in SETTINGS:
    sets = {}
    for b in BOT:
        for d in ("up", "down"):
            sets[b, d] = (pl.read_csv(root / "botryllus" / b / f"{d}.codes.tsv", separator="\t",
                                      schema_overrides={"code": pl.Int64})
                          .filter((pl.col("alphabet") == a) & (pl.col("k") == k))["code"].to_numpy())
    q[a, k] = np.unique(np.concatenate(list(sets.values())))
    for b in BOT:
        up = np.isin(q[a, k], sets[b, "up"])
        down = np.isin(q[a, k], sets[b, "down"])
        lab[a, k, b] = np.where(up & ~down, 1, np.where(down & ~up, -1, 0)).astype(np.int8)
        n_codes[a, k, b] = (len(sets[b, "up"]), len(sets[b, "down"]), int((up & down).sum()))
    print(a, k, "query k-mers", len(q[a, k]), flush=True)


def read_run(run):
    d = pl.read_csv(root / "human" / f"{run}.counts.tsv.gz", separator="\t",
                    schema_overrides={"code": pl.Int64, "count": pl.Int64})
    for a, k in SETTINGS:
        s = d.filter((pl.col("alphabet") == a) & (pl.col("k") == k))
        total = s.filter(pl.col("code") == -1)["count"].item()
        s = s.filter(pl.col("code") >= 0)
        idx = np.searchsorted(q[a, k], s["code"].to_numpy())
        assert (q[a, k][idx] == s["code"].to_numpy()).all(), f"{run}: a counted k-mer is not a query k-mer"
        yield a, k, idx, s["count"].to_numpy(), total


# Pass 1: each k-mer's total count over each human contrast's runs
tot = {(a, k, s): np.zeros(len(q[a, k]), np.int64) for a, k in SETTINGS for s in human}
for run in runs["run"]:
    for a, k, idx, cnt, _ in read_run(run):
        tot[a, k, series_of[run]][idx] += cnt
print("pass 1 done", flush=True)

# Pass 2: per run, sums of log2 CPM over the kept up and down k-mers; per k-mer, its own human change
sums = {}  # (a, k, series, b) -> {run: (sum_up, present_up, sum_down, present_down)}
e = {(a, k, s): np.zeros(len(q[a, k])) for a, k in SETTINGS for s in human}
for run in runs["run"]:
    s = series_of[run]
    w_real = human[s]["W"][0][human[s]["samples"].index(run)]
    for a, k, idx, cnt, total in read_run(run):
        x = np.log2(cnt / total * 1e6 + 0.5)
        e[a, k, s][idx] += w_real * (x - ABSENT)  # the weights sum to 0, so absent k-mers add nothing
        kept = tot[a, k, s][idx] >= MIN_COUNT
        for b in BOT:
            lb = lab[a, k, b][idx]
            sums.setdefault((a, k, s, b), {})[run] = (x[kept & (lb == 1)].sum(), (kept & (lb == 1)).sum(),
                                                      x[kept & (lb == -1)].sum(), (kept & (lb == -1)).sum())
print("pass 2 done", flush=True)

rows, nulls = [], []
for (a, k), b, (s, h) in itertools.product(SETTINGS, BOT, human.items()):
    keep = tot[a, k, s] >= MIN_COUNT
    up_k, down_k = keep & (lab[a, k, b] == 1), keep & (lab[a, k, b] == -1)
    nu, nd, nb = n_codes[a, k, b]
    row = dict(botryllus=b, alphabet=a, k=k, human=s, human_contrast=h["contrast"], up_codes=nu, down_codes=nd,
               in_both=nb, up_kept=int(up_k.sum()), down_kept=int(down_k.sum()))
    if row["up_kept"] == 0 or row["down_kept"] == 0:
        rows.append(row | dict(verdict="no k-mers kept"))
        continue
    v = []
    for run in h["samples"]:
        su, pu, sd, pd = sums[a, k, s, b][run]
        up_mean = (su + (row["up_kept"] - pu) * ABSENT) / row["up_kept"]
        down_mean = (sd + (row["down_kept"] - pd) * ABSENT) / row["down_kept"]
        v.append(up_mean - down_mean)
    D = h["W"] @ np.array(v)
    p = float((np.abs(D) >= np.abs(D[0]) - 1e-12).mean())
    row |= dict(D=float(D[0]), p=p, shuffles=len(D) - 1, exact=h["exact"],
                smallest_p=(2 / len(D)) if h["exact"] else 1 / len(D),
                up_share_rising=float((e[a, k, s][up_k] > 0).mean()),
                down_share_rising=float((e[a, k, s][down_k] > 0).mean()))
    row["verdict"] = ("too few subjects to decide" if row["smallest_p"] > ALPHA
                      else "pass" if p <= ALPHA else "fail")
    rows.append(row)
    nulls.append(pl.DataFrame({"botryllus": b, "alphabet": a, "k": k, "human": s,
                               "real": [True] + [False] * (len(D) - 1), "D": D}))

res = pl.DataFrame(rows)
res.write_csv(out / "de2de_results.tsv", separator="\t")
with gzip.open(out / "de2de_null.tsv.gz", "wb") as f:
    pl.concat(nulls).write_csv(f, separator="\t")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res.select("botryllus", "alphabet", "k", "human", "up_kept", "down_kept", "D", "p", "smallest_p",
                     "up_share_rising", "down_share_rising", "verdict"))
