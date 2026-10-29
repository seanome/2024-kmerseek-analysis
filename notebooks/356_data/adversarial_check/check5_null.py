"""Check 5: random-protein null, re-implemented, plus variants that try to break the excess.

For every verdict exon e and every pair p of the same species, M[e, p] = 1 if any kmerseek
call (any setting) of pair p lands on e's interval [a, b) placed on p's target; NaN if p's
target is shorter than b. A null draw picks, for every exon, one allowed decoy pair at
random (never the exon's own pair) and counts landings. Variants change the allowed set.
A second family of nulls keeps the exon's own pair and moves the window along the same target.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
from common import *

R = 1000
rng = np.random.default_rng(356)
v = verdict_exons().with_row_index("eid")
km = load_calls("kmerseek")
pr = pairs()
tl = exons().group_by("species", "target_acc").agg(pl.col("target_len").first())
m = models()
TAG = ""
if os.environ.get("COLLINEAR"):
    # Keep only calls whose relative position on the query matches that on the target
    # (|qmid/qlen - tmid/tlen| <= COLLINEAR). Query length = residues in plddt.parquet.
    tol = float(os.environ["COLLINEAR"])
    ql = m.filter(pl.col("side") == "query").select(
        pl.col("accession").alias("query_acc"), pl.col("n_res").alias("qlen")
    )
    km = (
        km.join(ql, on="query_acc", how="left")
        .join(tl, on=["species", "target_acc"], how="left")
        .filter(
            (
                (pl.col("qstart") + pl.col("qend")) / 2 / pl.col("qlen")
                - (pl.col("tstart") + pl.col("tend")) / 2 / pl.col("target_len")
            ).abs()
            <= tol
        )
        .drop("qlen", "target_len")
    )
    TAG = f"_collinear{tol}"
    print("COLLINEAR filter", tol, "calls kept:", km.height)
tm = m.filter(pl.col("side") == "target").select(
    "species",
    pl.col("accession").alias("target_acc"),
    pl.col("has_model").alias("t_model"),
)
qm = m.filter(pl.col("side") == "query").select(
    pl.col("accession").alias("query_acc"), pl.col("has_model").alias("q_model")
)
P = (
    pr.join(tl, on=["species", "target_acc"], how="left")
    .join(tm, on=["species", "target_acc"], how="left")
    .join(qm, on="query_acc", how="left")
    .with_columns(
        pl.col("t_model").fill_null(False), pl.col("q_model").fill_null(False)
    )
)
ncalls = (
    km.group_by("species", "query_acc", "target_acc").len().rename({"len": "n_calls"})
)
P = P.join(ncalls, on=["species", "query_acc", "target_acc"], how="left").with_columns(
    pl.col("n_calls").fill_null(0)
)
P = P.with_row_index("pid")
print(
    "pairs:",
    P.height,
    " with no target_len (no exon table entry):",
    P["target_len"].null_count(),
    " with >=1 kmerseek call:",
    (P["n_calls"] > 0).sum(),
)
calls_by_pair = {}
for k, g in km.group_by("species", "query_acc", "target_acc"):
    calls_by_pair[k] = (g["tstart"].to_numpy(), g["tend"].to_numpy())

ve = v.join(
    P.select("species", "query_acc", "target_acc", "pid", "n_calls"),
    on=["species", "query_acc", "target_acc"],
)
a = ve["prot_start"].to_numpy()
b = ve["prot_end"].to_numpy()
E = b - a
own = ve["pid"].to_numpy()
esp = ve["species"].to_numpy()
own_len = ve["target_len"].to_numpy()
own_nc = ve["n_calls"].to_numpy()
plen = P["target_len"].fill_null(-1).to_numpy()
psp = P["species"].to_numpy()
pnc = P["n_calls"].to_numpy()
pmod = (P["t_model"] & P["q_model"]).to_numpy()


def land_matrix_for(ts, te, a, b):
    """bool[len(a)] : does any call [ts,te) land on [a,b)."""
    out = np.zeros(len(a), bool)
    if len(ts) == 0:
        return out
    L = te - ts
    for i in range(0, len(a), 200):
        aa, bb = a[i : i + 200, None], b[i : i + 200, None]
        ov = np.minimum(te[None], bb) - np.maximum(ts[None], aa)
        ok = (ov > 0) & (ov >= 0.8 * L[None]) & (ov >= 0.3 * (bb - aa))
        out[i : i + 200] = ok.any(1)
    return out


M = np.full((len(a), P.height), np.nan)
for p in range(P.height):
    key = (psp[p], P["query_acc"][p], P["target_acc"][p])
    rows = np.where((esp == psp[p]) & (plen[p] >= b))[0]
    ts, te = calls_by_pair.get(key, (np.array([], int), np.array([], int)))
    M[rows, p] = land_matrix_for(ts, te, a[rows], b[rows])
real = M[np.arange(len(a)), own]
print("real landings from matrix:", int(np.nansum(real)), "of", len(a))


def run_null(allowed, name):
    """allowed: bool[n_exon, n_pair]."""
    allowed = allowed & ~np.isnan(M)
    allowed[np.arange(len(a)), own] = False
    n_ok = allowed.sum(1)
    has = n_ok > 0
    # P(land) for each exon under uniform choice among allowed decoys
    pl_ = np.where(
        has, np.nansum(np.where(allowed, M, 0), 1) / np.maximum(n_ok, 1), np.nan
    )
    draws = (rng.random((R, has.sum())) < pl_[has][None]).sum(1)
    obs = int(real[has].sum())
    n = int(has.sum())
    res = dict(
        null=name,
        n_exons=n,
        observed=obs,
        observed_pct=round(100 * obs / n, 2),
        null_median_pct=round(100 * np.median(draws) / n, 2),
        null_max_pct=round(100 * draws.max() / n, 2),
        null_mean_exons=round(float(pl_[has].sum()), 1),
        excess_exons=round(obs - float(pl_[has].sum()), 1),
        frac_draws_ge_obs=float((draws >= obs).mean()),
        min_decoys_per_exon=int(n_ok[has].min()) if n else 0,
    )
    print(json.dumps(res))
    return res


same_sp = esp[:, None] == psp[None]
out = []
out.append(run_null(same_sp, "A: any other pair, same species, target len >= exon end"))
out.append(
    run_null(same_sp & pmod[None], "B: decoy pair must also have both AlphaFold models")
)
out.append(run_null(same_sp & (pnc[None] > 0), "C: decoy pair has >=1 kmerseek call"))
out.append(
    run_null(
        same_sp & (np.abs(plen[None] - own_len[:, None]) <= 0.2 * own_len[:, None]),
        "D: decoy target length within +-20%",
    )
)
ncr = pnc[None] / np.maximum(own_nc[:, None], 1)
out.append(
    run_null(
        same_sp & (ncr >= 0.8) & (ncr <= 1.25),
        "E: decoy pair's kmerseek call count within 0.8x-1.25x",
    )
)
dens_own = own_nc / own_len
dens_p = pnc / np.where(plen > 0, plen, 1)
dr = dens_p[None] / np.maximum(dens_own[:, None], 1e-9)
out.append(
    run_null(
        same_sp
        & (dr >= 0.8)
        & (dr <= 1.25)
        & (np.abs(plen[None] - own_len[:, None]) <= 0.2 * own_len[:, None]),
        "F: decoy length +-20% AND calls per residue 0.8x-1.25x",
    )
)


# Same-target window nulls: keep the exon's own pair and calls, slide the window.
def window_null(exclude_overlap, name):
    probs = np.full(len(a), np.nan)
    for i in range(len(a)):
        key = (esp[i], ve["query_acc"][i], ve["target_acc"][i])
        ts, te = calls_by_pair.get(key, (np.array([], int), np.array([], int)))
        N = own_len[i]
        s = np.arange(0, N - E[i] + 1)
        if exclude_overlap:
            s = s[(s + E[i] <= a[i]) | (s >= b[i])]
        if len(s) == 0:
            continue
        probs[i] = land_matrix_for(ts, te, s, s + E[i]).mean() if len(ts) else 0.0
    has = ~np.isnan(probs)
    draws = (rng.random((R, has.sum())) < probs[has][None]).sum(1)
    obs = int(real[has].sum())
    n = int(has.sum())
    res = dict(
        null=name,
        n_exons=n,
        observed=obs,
        observed_pct=round(100 * obs / n, 2),
        null_median_pct=round(100 * np.median(draws) / n, 2),
        null_max_pct=round(100 * draws.max() / n, 2),
        null_mean_exons=round(float(probs[has].sum()), 1),
        excess_exons=round(obs - float(probs[has].sum()), 1),
        frac_draws_ge_obs=float((draws >= obs).mean()),
    )
    print(json.dumps(res))
    return res, probs


r1, _ = window_null(
    False, "G: same pair, same-length window anywhere on the same target"
)
r2, probs_g = window_null(
    True, "H: same pair, same-length window not touching the exon"
)
out += [r1, r2]
pl.DataFrame(out).write_csv(f"{HERE}/my_null_variants{TAG}.csv")


# Exons on one protein are not independent (one pair's call density sets them all), so the
# per-exon Bernoulli draws above understate the spread. Bootstrap pairs instead: resample the
# 261 pairs with replacement and recompute (observed - null expectation).
def cluster_boot(pexp, name, B=5000):
    has = ~np.isnan(pexp)
    df = pl.DataFrame(
        {"pid": own[has], "obs": real[has].astype(float), "exp": pexp[has]}
    )
    g = df.group_by("pid").agg(pl.col("obs").sum(), pl.col("exp").sum())
    d = (g["obs"] - g["exp"]).to_numpy()
    bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(
        f"pair-bootstrap excess, {name}: {d.sum():.1f} exons, 95% CI [{lo:.1f}, {hi:.1f}], "
        f"P(excess<=0)={(bs <= 0).mean():.4f}, pairs={len(d)}"
    )


def pexp_for(allowed):
    allowed = allowed & ~np.isnan(M)
    allowed[np.arange(len(a)), own] = False
    n_ok = allowed.sum(1)
    return np.where(
        n_ok > 0, np.nansum(np.where(allowed, M, 0), 1) / np.maximum(n_ok, 1), np.nan
    )


cluster_boot(pexp_for(same_sp), "A")
cluster_boot(
    pexp_for(
        same_sp
        & (dr >= 0.8)
        & (dr <= 1.25)
        & (np.abs(plen[None] - own_len[:, None]) <= 0.2 * own_len[:, None])
    ),
    "F",
)
cluster_boot(probs_g, "H")
