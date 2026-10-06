"""Classification metrics for kmerseek's ranking scores, judged against human Pfam domains.

Used by notebook 255 (`255_build_notebook.py`) and by `255_ranking_metrics_per_arm.py`.

The unit is one independent domain match: a region pair between two of the 998
Pfam-annotated human proteins, counted once whichever protein was the query. A match is
correct when both of its spans overlap a domain of the same Pfam family, under one of
four overlap rules (`RULES`). `label_regions` reproduces the labels stored in
`labeled_pairs_overlap_rules.parquet` exactly (checked in the notebook).
"""

from __future__ import annotations

import contextlib

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

LABELS = Path(
    "/Users/olga/data/botryllus/ranking-metrics/labeled_pairs_overlap_rules.parquet"
)
TRUTH = Path(
    "/Users/olga/data/qfo-pfam-region-benchmark/midi-plus/truth/human_domain_truth.parquet"
)
PF998 = Path("/Users/olga/data/alphabet-logreg-pfam998")
PLAN = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases/plan.json")
PER_ARM = Path("/Users/olga/data/botryllus/ranking-metrics/per_arm_metrics.parquet")
FIG = Path(__file__).resolve().parent.parent / "figures"

# (column, lower value is better, plain name)
METRICS = [
    ("region_evalue", True, "E-value"),
    ("region_ka_bits", False, "bit score"),
    ("region_mean_idf", False, "mean IDF"),
    ("region_tfidf", False, "tf-idf"),
    ("region_enrichment", False, "enrichment"),
    ("region_poisson_score", False, "Poisson score"),
    ("region_tail_probability", True, "Poisson p-value"),
    ("region_n_shared_kmers", False, "shared k-mers"),
    ("containment", False, "containment"),
    ("query_enrichment", False, "protein enrichment"),
    ("query_poisson_pvalue", True, "protein Poisson p-value"),
]
LENGTH = ("region_length", False, "region length")
ALL = METRICS + [LENGTH]
NAME = {c: n for c, _, n in ALL}
LOWER = {c: lo for c, lo, _ in ALL}

RULES = [
    ("correct_any_overlap", "any overlap"),
    ("correct_frac20_of_region", "≥20% of the matched region inside the domain"),
    ("correct_frac20_of_domain", "≥20% of the domain covered by the region"),
    ("correct_iou20", "IoU ≥ 0.2"),
]
RULE_NAME = dict(RULES)

LENGTH_BINS = [
    (0, 30, "<30 aa"),
    (30, 60, "30-59 aa"),
    (60, 120, "60-119 aa"),
    (120, 10**9, "≥120 aa"),
]

# One colour per metric, grouped by what the metric is built from: Karlin-Altschul
# statistics (blues), k-mer rarity (greens), counts of shared k-mers in the region
# (oranges), whole-protein scores (purples). Region length is the grey control.
_C = {
    "region_evalue": "#08519c",
    "region_ka_bits": "#6baed6",
    "region_mean_idf": "#006d2c",
    "region_tfidf": "#74c476",
    "region_enrichment": "#a63603",
    "region_poisson_score": "#e6550d",
    "region_tail_probability": "#fd8d3c",
    "region_n_shared_kmers": "#fdbe85",
    "containment": "#54278f",
    "query_enrichment": "#807dba",
    "query_poisson_pvalue": "#bcbddc",
    "region_length": "#b0b0b0",
}
COLOR = _C
LENGTH_BAND = dict(color="#d9d9d9", lw=9, solid_capstyle="butt", zorder=1)


# ----------------------------------------------------------------------------- labels
def label_regions(df: pl.DataFrame, truth: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add the four rule columns to a kmerseek region table.

    kmerseek reports region_start / target_start 0-based and the ends inclusive of the
    last residue in 1-based terms, so a span is [start + 1, end] in the truth's 1-based
    inclusive coordinates. Both spans must overlap a domain of the same Pfam family.
    """
    truth = pl.read_parquet(TRUTH) if truth is None else truth
    dom = truth.select(
        "accession",
        "pfam_id",
        pl.col("domain_start").cast(pl.Int64).alias("ds"),
        pl.col("domain_end").cast(pl.Int64).alias("de"),
    )
    fams = dom.select("accession", "pfam_id").unique()
    shared = (
        fams.join(fams, on="pfam_id", suffix="_t")
        .select(
            pl.col("accession").alias("query_name"),
            pl.col("accession_t").alias("target_name"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("_share"))
    )
    df = df.with_row_index("_rid").join(
        shared, on=["query_name", "target_name"], how="left"
    )
    cand = df.filter(pl.col("_share").fill_null(False))

    def side(name, s, e):
        x = cand.select(
            "_rid",
            pl.col(name).alias("accession"),
            (pl.col(s).cast(pl.Int64) + 1).alias("s"),
            pl.col(e).cast(pl.Int64).alias("e"),
        ).join(dom, on="accession")
        ov = (pl.min_horizontal("e", "de") - pl.max_horizontal("s", "ds") + 1).clip(
            lower_bound=0
        )
        return x.with_columns(
            ov.alias("ov"),
            (pl.col("e") - pl.col("s") + 1).alias("rl"),
            (pl.col("de") - pl.col("ds") + 1).alias("dl"),
        ).drop("accession", "s", "e", "ds", "de")

    j = side("query_name", "region_start", "region_end").join(
        side("target_name", "target_start", "target_end"),
        on=["_rid", "pfam_id"],
        suffix="_t",
    )
    pos = (pl.col("ov") > 0) & (pl.col("ov_t") > 0)
    iou = lambda o, r, d: pl.col(o) / (pl.col(r) + pl.col(d) - pl.col(o))
    j = j.with_columns(
        correct_any_overlap=pos,
        correct_frac20_of_region=pos
        & (pl.col("ov") >= 0.2 * pl.col("rl"))
        & (pl.col("ov_t") >= 0.2 * pl.col("rl_t")),
        correct_frac20_of_domain=pos
        & (pl.col("ov") >= 0.2 * pl.col("dl"))
        & (pl.col("ov_t") >= 0.2 * pl.col("dl_t")),
        correct_iou20=(iou("ov", "rl", "dl") >= 0.2)
        & (iou("ov_t", "rl_t", "dl_t") >= 0.2),
    )
    rules = [r for r, _ in RULES]
    g = j.group_by("_rid").agg([pl.col(r).any() for r in rules])
    return (
        df.join(g, on="_rid", how="left")
        .with_columns([pl.col(r).fill_null(False) for r in rules])
        .sort("_rid")
        .drop("_rid", "_share")
    )


def independent_matches(df: pl.DataFrame) -> pl.DataFrame:
    """One row per unordered pair of region spans. kmerseek scores a region from the
    query side, so A->B and B->A carry different values; keep the direction with the
    lower E-value, as labeled_pairs_overlap_rules.parquet does (checked: in all 3_547
    pairs whose two E-values differ, the kept row has the lower one). Ties, including
    two infinite E-values, keep the row whose query name sorts first."""
    swap = pl.col("query_name") > pl.col("target_name")
    a = [
        pl.when(swap).then(pl.col(t)).otherwise(pl.col(q)).alias(f"_k{i}")
        for i, (q, t) in enumerate(
            [
                ("query_name", "target_name"),
                ("region_start", "target_start"),
                ("region_end", "target_end"),
            ]
        )
    ]
    b = [
        pl.when(swap).then(pl.col(q)).otherwise(pl.col(t)).alias(f"_k{i + 3}")
        for i, (q, t) in enumerate(
            [
                ("query_name", "target_name"),
                ("region_start", "target_start"),
                ("region_end", "target_end"),
            ]
        )
    ]
    keys = [f"_k{i}" for i in range(6)]
    ev = pl.col("region_evalue").fill_nan(float("inf")).fill_null(float("inf"))
    return (
        df.with_columns(a + b + [ev.alias("_ev")])
        .sort(["_ev", "query_name"])
        .unique(subset=keys, keep="first", maintain_order=True)
        .drop(keys + ["_ev"])
    )


# ----------------------------------------------------------------------------- scores
def score(df: pl.DataFrame, col: str) -> np.ndarray:
    """Higher = ranked as more likely correct. NaN where the metric has no value: null,
    NaN or infinite, and for the E-value and bit score also where the region has no
    Karlin-Altschul lambda (kmerseek writes E = inf and bits = 0 there)."""
    x = df[col].cast(pl.Float64).to_numpy().copy()
    bad = ~np.isfinite(x)
    if col in ("region_evalue", "region_ka_bits") and "region_ka_lambda" in df.columns:
        bad |= df["region_ka_lambda"].cast(pl.Float64).fill_null(0).to_numpy() == 0
    x[bad] = np.nan
    if LOWER[col]:
        with np.errstate(divide="ignore"):
            x = -np.log10(x)  # 0 -> +inf: a p-value of 0 ties at the top
        x[np.isposinf(x)] = np.finfo(float).max
    return x


def auc(s: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC as the chance that a random correct match scores above a random incorrect
    one, ties counted half (Mann-Whitney U / n1 n0)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(s: np.ndarray, y: np.ndarray) -> float:
    """Average precision, the same sum as sklearn's average_precision_score (precision at
    each distinct score, weighted by the recall gained there), in numpy for the bootstrap.
    """
    n1 = int(y.sum())
    if n1 == 0 or n1 == len(y):
        return np.nan
    o = np.argsort(-s, kind="stable")
    ss, yy = s[o], y[o]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(yy)[last]
    n = (np.arange(1, len(yy) + 1))[last]
    rec = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * tp / n))


def ap_sklearn(s: np.ndarray, y: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _query_groups(q: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(q, kind="stable")
    _, starts = np.unique(q[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap(
    df: pl.DataFrame,
    rule: str,
    cols: list[str],
    n_boot: int = 1_000,
    seed: int = 0,
    mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """AUC and AP per metric with 95% intervals from resampling query proteins (a query's
    matches move together). Each metric is scored on its own scored rows; `length_auc`
    and `length_ap` are region length on the same rows."""
    mask = np.ones(df.height, bool) if mask is None else mask
    y_all = df[rule].to_numpy()
    groups = _query_groups(df["query_name"].to_numpy())
    rng = np.random.default_rng(seed)
    draws = [
        np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for _ in range(n_boot)
    ]
    L = score(df, "region_length")
    rows = []
    for c in cols:
        s = score(df, c)
        ok = np.isfinite(s) & mask
        est = dict(
            auc=auc(s[ok], y_all[ok]),
            ap=ap(s[ok], y_all[ok]),
            length_auc=auc(L[ok], y_all[ok]),
            length_ap=ap(L[ok], y_all[ok]),
        )
        boots = {k: [] for k in est}
        for idx in draws:
            idx = idx[ok[idx]]
            yb = y_all[idx]
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            boots["auc"].append(auc(s[idx], yb))
            boots["ap"].append(ap(s[idx], yb))
            boots["length_auc"].append(auc(L[idx], yb))
            boots["length_ap"].append(ap(L[idx], yb))
        # the same metric over every match in the mask, unscored matches tied at the bottom
        s_all = np.where(np.isfinite(s), s, -np.finfo(float).max)[mask]
        row = dict(
            metric=NAME[c],
            column=c,
            rule=RULE_NAME[rule],
            n_scored=int(ok.sum()),
            n_correct=int(y_all[ok].sum()),
            base_rate=float(y_all[ok].mean()) if ok.any() else np.nan,
            auc_unscored_last=auc(s_all, y_all[mask]),
            ap_unscored_last=ap(s_all, y_all[mask]),
        )
        for k, v in est.items():
            lo, hi = (
                np.nanpercentile(boots[k], [2.5, 97.5])
                if boots[k]
                else (np.nan, np.nan)
            )
            row |= {k: v, f"{k}_lo": float(lo), f"{k}_hi": float(hi)}
        # paired: metric minus length within the same resample
        for k in ("auc", "ap"):
            dd = np.array(boots[k]) - np.array(boots[f"length_{k}"])
            lo, hi = np.nanpercentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan)
            row |= {
                f"{k}_minus_length": est[k] - est[f"length_{k}"],
                f"{k}_minus_length_lo": float(lo),
                f"{k}_minus_length_hi": float(hi),
            }
        rows.append(row)
    return pl.DataFrame(rows)


def confusion(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=(tp + tn) / len(y),
        precision=prec,
        recall=rec,
        f1=f1,
        mcc=mcc,
    )


def best_f1(s: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """The cutoff that maximises F1 over all matches; a match with no value is never
    called. Returns the cutoff (in score units: higher = better) and the confusion counts.
    """
    ok = np.isfinite(s)
    order = np.argsort(-s[ok], kind="stable")
    ss, yy = s[ok][order], y[ok][order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    last = np.r_[
        ss[1:] != ss[:-1], True
    ]  # a cutoff can only fall between distinct values
    tp, fp, cut = tp[last], fp[last], ss[last]
    f1 = 2 * tp / (tp + fp + y.sum())
    i = int(np.argmax(f1))
    return float(cut[i]), confusion(np.where(ok, s, -np.inf) >= cut[i], y)


def thresholds(df: pl.DataFrame, rule: str, cols: list[str]) -> pl.DataFrame:
    """Accuracy, precision, recall, F1 and MCC over every match of the table (a match
    the metric cannot score counts as not called)."""
    y = df[rule].to_numpy()
    rows = []
    for c in cols:
        s = score(df, c)
        cut, m = best_f1(s, y)
        raw = 10 ** -cut if LOWER[c] else cut
        rows.append(
            dict(
                metric=NAME[c],
                rule=RULE_NAME[rule],
                cutoff="best F1",
                value_at_cutoff=raw,
                **m,
            )
        )
    if "region_evalue" in cols:
        ev = score(df, "region_evalue")
        for e in (1.0, 0.01):
            called = np.isfinite(ev) & (ev >= -np.log10(e))
            rows.append(
                dict(
                    metric="E-value",
                    rule=RULE_NAME[rule],
                    cutoff=f"E ≤ {e:g}",
                    value_at_cutoff=e,
                    **confusion(called, y),
                )
            )
    return pl.DataFrame(rows)


def per_query(df: pl.DataFrame, rule: str, cols: list[str]) -> pl.DataFrame:
    """Rank each query's matches by the metric (unscored matches last). Over the queries
    with at least one correct match: the chance the top match is correct, and the mean
    of 1 / (expected rank of the first correct match). Ties are resolved by averaging over
    a random order of the tied matches, so a tie is never an arbitrary win."""
    y = df[rule].to_numpy()
    groups = [g for g in _query_groups(df["query_name"].to_numpy()) if y[g].any()]
    rows = []
    for c in cols + ["_random"]:
        s = (
            np.zeros(df.height)
            if c == "_random"
            else np.nan_to_num(score(df, c), nan=-np.inf)
        )
        p1, rr = [], []
        for g in groups:
            sg, yg = s[g], y[g]
            top = sg == sg.max()
            p1.append(yg[top].mean())
            best = sg[yg].max()
            above = int((~yg & (sg > best)).sum())
            tied_wrong = int((~yg & (sg == best)).sum())
            tied_right = int((yg & (sg == best)).sum())
            rr.append(1 / (1 + above + tied_wrong / (tied_right + 1)))
        rows.append(
            dict(
                metric="random order" if c == "_random" else NAME[c],
                rule=RULE_NAME[rule],
                n_queries=len(groups),
                precision_at_1=float(np.mean(p1)),
                mrr=float(np.mean(rr)),
            )
        )
    return pl.DataFrame(rows)


def ties_and_missing(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        raw = df[c].cast(pl.Float64)
        s = score(df, c)
        ok = np.isfinite(s)
        top = s[ok].max()
        rows.append(
            dict(
                metric=NAME[c],
                column=c,
                n_matches=df.height,
                n_scored=int(ok.sum()),
                n_null_or_inf=int(
                    raw.is_null().sum() + raw.is_nan().sum() + raw.is_infinite().sum()
                ),
                n_no_lambda=int(
                    (~ok).sum()
                    - (
                        raw.is_null().sum()
                        + raw.is_nan().sum()
                        + raw.is_infinite().sum()
                    )
                ),
                frac_unscored=float(1 - ok.mean()),
                n_tied_at_top=int((s[ok] == top).sum()),
                n_distinct_values=int(len(np.unique(s[ok]))),
            )
        )
    return pl.DataFrame(rows)


def length_bins(
    df: pl.DataFrame, rule: str, cols: list[str], n_boot: int = 1_000, seed: int = 0
) -> pl.DataFrame:
    L = df["region_length"].to_numpy()
    out = []
    for lo, hi, lab in LENGTH_BINS:
        b = bootstrap(
            df, rule, cols, n_boot=n_boot, seed=seed, mask=(L >= lo) & (L < hi)
        )
        out.append(b.with_columns(pl.lit(lab).alias("length_bin")))
    return pl.concat(out)


# ----------------------------------------------------------------------------- figures
def setup_style():
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 150,
            "figure.dpi": 100,
        }
    )


def legend_row(fig, hs, y=0.968):
    fig.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(hs),
        frameon=False,
        fontsize=9,
        handlelength=2.2,
    )


def forest(
    ax,
    tab: pl.DataFrame,
    value: str,
    lo: str | None,
    hi: str | None,
    control: str | None,
    tick: str | None = None,
    xlabel: str = "",
    xlim=(0, 1),
    labels: bool = True,
):
    """One row per metric (top = first in `tab`). The control (region length on the same
    matches) is drawn first as a wide pale bar so a metric equal to it stays visible on
    top; the metric is a dot with its 95% interval in the metric's colour; `tick` is a
    black vertical tick (base rate, or a random order)."""
    n = tab.height
    fin = lambda v: v is not None and np.isfinite(v)
    for i, r in enumerate(tab.iter_rows(named=True)):
        yv = n - 1 - i
        if control and fin(r.get(control)):
            ax.plot([r[control]] * 2, [yv - 0.36, yv + 0.36], **LENGTH_BAND)
        if tick and fin(r.get(tick)):
            ax.plot(
                [r[tick]] * 2, [yv - 0.3, yv + 0.3], color="black", lw=1.2, zorder=2
            )
        v = r[value]
        if not fin(v):
            ax.text(
                xlim[0] + 0.01 * (xlim[1] - xlim[0]),
                yv,
                "no value",
                va="center",
                fontsize=8,
                color="#555555",
            )
            continue
        col = COLOR.get(r["column"], "#333333")
        if lo and fin(r.get(lo)):
            ax.plot([r[lo], r[hi]], [yv, yv], color=col, lw=1.6, zorder=3)
        ax.plot([v], [yv], "o", color=col, ms=6, mec="black", mew=0.5, zorder=4)
    ax.set_yticks(range(n))
    names = tab["label" if "label" in tab.columns else "metric"].to_list()[::-1]
    if labels:
        ax.set_yticklabels(names)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#eeeeee", zorder=0)


def handles(
    control_label: str,
    tick_label: str | None = None,
    dot_label: str = "kmerseek metric (one colour each), 95% interval",
):
    from matplotlib.lines import Line2D

    h = [
        Line2D(
            [],
            [],
            color="#666666",
            marker="o",
            mec="black",
            mew=0.5,
            lw=1.6,
            label=dot_label,
        ),
        Line2D(
            [],
            [],
            label=control_label,
            **{k: v for k, v in LENGTH_BAND.items() if k != "zorder"},
        ),
    ]
    if tick_label:
        h.append(
            Line2D(
                [],
                [],
                color="black",
                lw=0,
                marker="|",
                ms=12,
                mew=1.2,
                label=tick_label,
            )
        )
    return h


# ---------------------------------------------------------------- every alphabet at once
GRID_METRICS = [
    "E-value",
    "bit score",
    "tf-idf",
    "shared k-mers",
    "Poisson p-value",
    "mean IDF",
    "enrichment",
]


def gain_over_length(
    per_pair: pl.DataFrame, rule: str, stat: str = "ap"
) -> pl.DataFrame:
    """One row per (alphabet, metric): the median over that alphabet's k-mer sizes of the
    metric's AUC or AP minus region length's on the same matches, and how many
    alphabet-ksize pairs went into it (the E-value and bit score exist only where the
    index has a Karlin-Altschul fit)."""
    d = per_pair.filter(
        (pl.col("rule") == rule)
        & (pl.col("metric") != "region length")
        & pl.col(stat).is_not_nan()
    )
    return (
        d.with_columns((pl.col(stat) - pl.col(f"length_{stat}")).alias("gain"))
        .group_by("alphabet", "metric")
        .agg(
            pl.col("gain").median().alias("median_gain"),
            pl.len().alias("n_pairs"),
            (pl.col("gain") > 0).sum().alias("n_pairs_above_length"),
        )
        .sort("alphabet", "metric")
    )


def fig_gain_grid(g: pl.DataFrame, path: Path, title: str, stat_label: str):
    """Rows: the 19 alphabets. Columns: ranking metrics. Cell colour and number: median
    over k-mer sizes of (metric - region length). Blue: the metric ranks correct matches
    higher than length alone; red: lower; white: the same. Small grey number: how many
    alphabet-ksize pairs the median is over."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    alphas = sorted(g["alphabet"].unique().to_list())
    M = np.full((len(alphas), len(GRID_METRICS)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for r in g.iter_rows(named=True):
        if r["metric"] in GRID_METRICS:
            i, j = alphas.index(r["alphabet"]), GRID_METRICS.index(r["metric"])
            M[i, j], N[i, j] = r["median_gain"], r["n_pairs"]
    lim = float(np.nanmax(np.abs(M)))
    fig = plt.figure(figsize=(10.5, 10.0))
    cax = fig.add_axes([0.30, 0.855, 0.45, 0.016])
    ax = fig.add_axes([0.30, 0.04, 0.68, 0.74])
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(
        f"{stat_label}: metric minus region length (median over k-mer sizes)\n"
        "blue = ranks correct matches higher than length alone, red = lower",
        fontsize=9,
    )
    cax.xaxis.set_label_position("top")
    for i in range(len(alphas)):
        for j in range(len(GRID_METRICS)):
            if np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    "no E-value",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#777777",
                )
                continue
            dark = abs(M[i, j]) > 0.6 * lim
            ax.text(
                j,
                i - 0.12,
                f"{M[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if dark else "black",
            )
            ax.text(
                j,
                i + 0.28,
                f"n={N[i, j]}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#f0f0f0" if dark else "#555555",
            )
    ax.set_xticks(range(len(GRID_METRICS)))
    ax.set_xticklabels(GRID_METRICS, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels(alphas, fontsize=9)
    ax.tick_params(length=0)
    fig.text(0.01, 0.99, title, fontsize=9.5, va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ------------------------------------------------- figures for sections 1, 2 and 8
# Drawn with the paper style (pubfig.py) inside a style context, so the other figures
# of the notebook keep their own settings.
RULE_SHORT = {
    "any overlap": "any overlap",
    "≥20% of the matched region inside the domain": "≥20% of region in domain",
    "≥20% of the domain covered by the region": "≥20% of domain covered",
    "IoU ≥ 0.2": "IoU ≥ 0.2",
}
NO_VALUE = "#d9d9d9"  # a match the metric gives no value to


@contextlib.contextmanager
def paper_style():
    """Draw with pubfig's nature.mplstyle, then put back every setting but the fonts.

    The fonts stay (Arial, embedded as TrueType), because a font is looked up when the
    figure is saved or shown, after this block has ended; restoring them would write the
    figure in DejaVu Sans. Not plt.style.context: in a Jupyter kernel (matplotlib 3.10,
    matplotlib-inline) the figures of every later cell stop being shown once it exits.
    """
    import pubfig as pf

    keep = {
        k: v
        for k, v in mpl.rcParams.items()
        if k not in ("backend", "backend_fallback", "interactive")
        and not k.startswith(("font.family", "font.sans-serif", "font.monospace"))
        and not k.startswith(
            ("mathtext.", "pdf.fonttype", "ps.fonttype", "svg.fonttype")
        )
    }
    try:
        mpl.rcParams.update(
            mpl.rc_params_from_file(pf.STYLE, use_default_template=False)
        )
        yield
    finally:
        mpl.rcParams.update(keep)


CORRECT = "#009E73"
INCORRECT = "#999999"


def fig_labels_rebuilt(check: pl.DataFrame, n_matches: int):
    """Correct matches per rule: stored label (wide pale bar), rebuilt label (thin bar)."""
    import pubfig as pf

    with paper_style():
        fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
        y = np.arange(check.height)
        ax.barh(
            y,
            check["stored correct"],
            height=0.75,
            color="#bdd7e7",
            label="stored label",
        )
        ax.barh(
            y,
            check["rebuilt correct"],
            height=0.3,
            color="#08519c",
            label="rebuilt by label_regions",
        )
        for yi, s, d in zip(y, check["stored correct"], check["rows that disagree"]):
            ax.text(s + 15, yi, f"{s}, {d} disagree", va="center", fontsize=6)
        ax.set_yticks(y, [RULE_SHORT[r] for r in check["rule"]])
        ax.invert_yaxis()
        ax.set_xlim(0, max(check["stored correct"]) * 1.45)
        ax.set_xlabel(f"Correct matches (n, of {n_matches:_})")
        pf.shared_legend(fig)
    return fig


def fig_ties_and_missing(ties: pl.DataFrame, by_lambda: pl.DataFrame):
    """a: matches each metric scores. b: matches tied at the top score.
    c: share correct among matches with and without a Karlin-Altschul lambda, per rule.
    """
    import pubfig as pf

    with paper_style():
        fig, (a, b, c) = pf.figure(
            pf.TWO_COLUMN_MM, 70, ncols=3, width_ratios=[1.2, 0.8, 1]
        )
        y = np.arange(ties.height)
        cols = [COLOR[c_] for c_ in ties["column"]]
        no_value = ties["n_null_or_inf"] + ties["n_no_lambda"]
        a.barh(y, ties["n_scored"], color=cols, height=0.75)
        a.barh(
            y,
            no_value,
            left=ties["n_scored"],
            color=NO_VALUE,
            height=0.75,
            label="no value (no lambda: E = inf, bits = 0)",
        )
        a.set_yticks(y, ties["metric"])
        a.invert_yaxis()
        a.set_xlabel("Matches (n)")
        b.scatter(ties["n_tied_at_top"], y, color=cols, s=14, zorder=3)
        b.set_yticks(y, [])
        b.invert_yaxis()
        b.set_xlabel("Matches tied at the\nbest score (n)")
        for yi in y:
            b.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
        ry = np.arange(by_lambda.height)
        c.scatter(
            100 * by_lambda["share_correct_no_lambda"],
            ry,
            marker="x",
            color="#000000",
            s=14,
            label="no lambda (E = inf)",
        )
        c.scatter(
            100 * by_lambda["share_correct_lambda"],
            ry,
            marker="o",
            facecolor="none",
            edgecolor="#000000",
            s=14,
            label="lambda > 0",
        )
        c.set_yticks(ry, [RULE_SHORT[r] for r in by_lambda["rule"]])
        c.invert_yaxis()
        c.set_xlim(0, None)
        c.set_xlabel("Correct matches (%)")
        for yi in ry:
            c.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
        pf.shared_legend(fig, ncol=3)
        for ax, letter, dx in [(a, "a", -80), (b, "b", -12), (c, "c", -75)]:
            pf.panel_label(ax, letter, dx_pt=dx)
    return fig


def fig_checks(auc_check: pl.DataFrame, df: pl.DataFrame, rule: str):
    """a: ROC AUC from scikit-learn against the notebook's table. b: tf-idf against region
    length for every match, coloured by the label under `rule`."""
    import pubfig as pf

    with paper_style():
        fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 65, ncols=2, width_ratios=[0.8, 1.2])
        col = {
            "tf-idf": "region_tfidf",
            "E-value (lambda > 0 only)": "region_evalue",
            "region length": "region_length",
        }
        lo = (
            min(auc_check["sklearn on the raw column"].min(), auc_check["table"].min())
            - 0.01
        )
        a.plot([lo, 1], [lo, 1], color="#000000", lw=0.5, ls="--", label="same value")
        for m, sk, tb in auc_check.select(
            "metric", "sklearn on the raw column", "table"
        ).iter_rows():
            a.scatter(tb, sk, color=COLOR[col[m]], s=20, zorder=3, label=m)
        a.set_xlim(lo, 1)
        a.set_ylim(lo, 1)
        a.set_aspect("equal")
        a.set_xlabel("ROC AUC, notebook table")
        a.set_ylabel("ROC AUC, scikit-learn")
        y = df[rule].to_numpy()
        for flag, color, marker, lab in [
            (False, INCORRECT, "o", "incorrect match"),
            (True, CORRECT, "^", "correct match"),
        ]:
            s = df.filter(pl.col(rule) == flag)
            b.scatter(
                s["region_length"],
                s["region_tfidf"],
                s=3,
                alpha=0.4,
                linewidths=0,
                color=color,
                marker=marker,
                label=f"{lab} ({RULE_NAME[rule]})",
            )
        b.set_xscale("log")
        b.set_yscale("log")
        for axis in (b.xaxis, b.yaxis):
            axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        b.set_xlabel("Region length (aa)")
        b.set_ylabel("tf-idf")
        pf.shared_legend(fig, ncol=3)
        pf.panel_label(a, "a", dx_pt=-35)
        pf.panel_label(b, "b", dx_pt=-35)
    return fig
