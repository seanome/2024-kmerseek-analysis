#!/usr/bin/env python3
"""CAFA-style and domain-boundary metrics, adapted honestly to Pfam domain finding.

What carries over from CAFA cleanly, and what does not:

  Fmax        Carries over. CAFA's Fmax is *protein-centric*: precision is averaged over
              proteins that made at least one prediction, recall over all proteins with a
              true annotation, and the max is taken over thresholds. That macro-average is
              a different number from the micro-averaged `best_f1` this pipeline also
              reports -- a handful of domain-dense proteins cannot dominate it. Both are
              kept because they answer different questions.

  Smin/wFmax  Carry over only in weakened form, and the docs should say so. CAFA weights
              GO terms by *information accretion*, which is defined against the ontology
              DAG: a term's IA is its information content conditioned on its parents.
              **Pfam is flat.** Clans are a shallow grouping, not a subsumption hierarchy,
              so there are no parents to condition on and information accretion degenerates
              to plain information content, IC(family) = -log2 P(family). That is still a
              real and useful weighting -- recovering a rare family should count for more
              than recovering a ubiquitous one -- but it is not the CAFA quantity, and
              calling it "information accretion" would overclaim.

  AUPR        Already computed by the caller.

Domain-specific metrics (CASP/Chainsaw/Merizo lineage) that CAFA has no analogue for:

  NDO         Normalized domain overlap, residue level.
  DBD         Domain boundary distance, in residues.
  IoU >= 0.8  The strict "correctly parsed" criterion used by structure parsers, alongside
              this pipeline's looser default.
  domain count Whether a protein was called single- vs multi-domain, scored by MCC.

Every threshold sweep runs on one shared numpy grid rather than re-filtering the frame per
threshold, so cost is O(calls + proteins x thresholds) instead of O(calls x thresholds).
"""

import numpy as np
import polars as pl

DEFAULT_N_THRESHOLDS = 101


def information_content(truth: pl.DataFrame) -> pl.DataFrame:
    """IC(family) = -log2(proteins carrying it / proteins total).

    Estimated from the query-side answer key, which is the population the metric is
    reported over. A family on every protein carries IC 0; a singleton carries the most.
    """
    n_proteins = truth["accession"].n_unique()
    return (
        truth.group_by("pfam_id")
        .agg(pl.col("accession").n_unique().alias("n_with"))
        .with_columns(
            (-(pl.col("n_with") / n_proteins).log(base=2)).alias("ic")
        )
        .select("pfam_id", "ic")
    )


def _suffix_counts(protein_idx, bin_idx, weights, n_proteins, n_bins):
    """Per-protein counts at every threshold, in one pass.

    A call passes descending thresholds 0..bin_idx, so the count at threshold j is the
    suffix sum over bins >= j. Accumulate into a (protein, bin) histogram, then suffix-sum
    along the bin axis -- one O(n) scatter plus one cumsum, rather than a filter per
    threshold.
    """
    hist = np.zeros((n_proteins, n_bins), dtype=np.float64)
    np.add.at(hist, (protein_idx, bin_idx), weights)
    return np.cumsum(hist[:, ::-1], axis=1)[:, ::-1]


def protein_centric_curve(calls: pl.DataFrame, truth: pl.DataFrame, ic: pl.DataFrame,
                          n_thresholds: int = DEFAULT_N_THRESHOLDS) -> pl.DataFrame:
    """Fmax / wFmax / Smin operating points, protein-centric and interval-aware.

    A prediction counts only when it is placed correctly (is_tp, i.e. IoU past the
    caller's cutoff), so this measures domain *finding*, not family naming. Recall is
    over distinct true instances: several regions hitting one domain is one recovery.
    """
    proteins = truth["accession"].unique().sort()
    if proteins.len() == 0:
        return pl.DataFrame()
    pidx = {a: i for i, a in enumerate(proteins.to_list())}
    n_proteins = len(pidx)

    ic_map = dict(zip(ic["pfam_id"].to_list(), ic["ic"].to_list()))

    # Per-protein denominators: instance count and summed IC of the true set.
    t = truth.with_columns(
        pl.col("pfam_id").replace_strict(ic_map, default=0.0).alias("ic")
    )
    n_true = np.zeros(n_proteins)
    w_true = np.zeros(n_proteins)
    for acc, n, w in (
        t.group_by("accession")
        .agg(pl.len().alias("n"), pl.col("ic").sum().alias("w"))
        .iter_rows()
    ):
        if acc in pidx:
            n_true[pidx[acc]] = n
            w_true[pidx[acc]] = w

    if calls.height == 0:
        thresholds = np.array([0.0])
        zeros = np.zeros(1)
        return pl.DataFrame({
            "threshold": thresholds, "pr": zeros, "rc": zeros, "f": zeros,
            "wpr": zeros, "wrc": zeros, "wf": zeros,
            "ru": np.array([w_true.mean() if n_proteins else 0.0]), "mi": zeros,
            "s": np.array([w_true.mean() if n_proteins else 0.0]),
        })

    scores = calls["score"].fill_null(float("-inf")).to_numpy().astype(float)
    # Quantile grid, not a linear one: region scores and bitscores are both heavily
    # right-skewed, so evenly spaced cutoffs would spend most of the grid on a tail
    # where nothing changes and skip the range where the curve actually turns.
    qs = np.linspace(0, 1, n_thresholds)
    t_asc = np.unique(np.quantile(scores, qs))
    n_bins = len(t_asc)

    # Number of thresholds a call clears; it contributes to descending bins 0..k-1.
    k = np.searchsorted(t_asc, scores, side="right")
    valid = k > 0
    bin_idx = (k - 1)[valid]

    call_p = np.array([pidx.get(a, -1) for a in calls["query_acc"].to_list()])
    call_ic = np.array([ic_map.get(f, 0.0) for f in calls["pfam_id"].to_list()])
    is_tp = calls["is_tp"].to_numpy()

    keep = valid & (call_p >= 0)
    kb = (k - 1)[keep]
    kp = call_p[keep]

    n_pred = _suffix_counts(kp, kb, np.ones(keep.sum()), n_proteins, n_bins)
    w_pred = _suffix_counts(kp, kb, call_ic[keep], n_proteins, n_bins)

    # Detections: one row per true instance, at its best-scoring correct call, so recall
    # counts instances rather than rewarding many regions hitting the same domain.
    det = (
        calls.filter("is_tp")
        .group_by("query_acc", "pfam_id", "true_start", "true_end")
        .agg(pl.col("score").max())
    )
    if det.height:
        d_scores = det["score"].fill_null(float("-inf")).to_numpy().astype(float)
        d_k = np.searchsorted(t_asc, d_scores, side="right")
        d_p = np.array([pidx.get(a, -1) for a in det["query_acc"].to_list()])
        d_ic = np.array([ic_map.get(f, 0.0) for f in det["pfam_id"].to_list()])
        d_keep = (d_k > 0) & (d_p >= 0)
        n_tp = _suffix_counts(d_p[d_keep], (d_k - 1)[d_keep], np.ones(d_keep.sum()),
                              n_proteins, n_bins)
        w_tp = _suffix_counts(d_p[d_keep], (d_k - 1)[d_keep], d_ic[d_keep],
                              n_proteins, n_bins)
    else:
        n_tp = np.zeros((n_proteins, n_bins))
        w_tp = np.zeros((n_proteins, n_bins))

    has_truth = n_true > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        # CAFA's asymmetry, kept deliberately: precision is averaged only over proteins
        # that predicted something at this threshold, recall over every protein with a
        # true annotation. A tool that stays silent on hard proteins keeps its precision
        # but pays in recall, which is the intended incentive.
        prec_i = np.where(n_pred > 0, np.minimum(n_tp / np.maximum(n_pred, 1), 1.0), np.nan)
        rec_i = np.where(has_truth[:, None], n_tp / np.maximum(n_true[:, None], 1), np.nan)
        wprec_i = np.where(w_pred > 0, np.minimum(w_tp / np.maximum(w_pred, 1e-12), 1.0), np.nan)
        wrec_i = np.where(
            (w_true > 0)[:, None], w_tp / np.maximum(w_true[:, None], 1e-12), np.nan
        )

    pr = np.nanmean(np.where(np.isnan(prec_i), np.nan, prec_i), axis=0)
    rc = np.nanmean(rec_i, axis=0)
    wpr = np.nanmean(wprec_i, axis=0)
    wrc = np.nanmean(wrec_i, axis=0)
    pr, rc, wpr, wrc = (np.nan_to_num(x) for x in (pr, rc, wpr, wrc))

    def harmonic(p, r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(2 * p * r / (p + r))

    ru = np.nanmean(np.maximum(w_true[:, None] - w_tp, 0.0), axis=0)
    mi = np.nanmean(np.maximum(w_pred - w_tp, 0.0), axis=0)

    return pl.DataFrame({
        "threshold": t_asc[::-1],
        "pr": pr[::-1], "rc": rc[::-1], "f": harmonic(pr, rc)[::-1],
        "wpr": wpr[::-1], "wrc": wrc[::-1], "wf": harmonic(wpr, wrc)[::-1],
        "ru": ru[::-1], "mi": mi[::-1], "s": np.sqrt(ru**2 + mi**2)[::-1],
    })


def cafa_scalars(curve: pl.DataFrame) -> dict:
    if curve.height == 0:
        return {"fmax": 0.0, "fmax_threshold": None, "fmax_precision": 0.0,
                "fmax_recall": 0.0, "wfmax": 0.0, "smin": None, "smin_threshold": None,
                "smin_ru": None, "smin_mi": None}
    best_f = curve.sort("f", descending=True).head(1).to_dicts()[0]
    best_wf = curve.sort("wf", descending=True).head(1).to_dicts()[0]
    best_s = curve.sort("s", descending=False).head(1).to_dicts()[0]
    return {
        "fmax": best_f["f"],
        "fmax_threshold": best_f["threshold"],
        "fmax_precision": best_f["pr"],
        "fmax_recall": best_f["rc"],
        "wfmax": best_wf["wf"],
        # Lower is better: total information missed and information invented, in bits.
        "smin": best_s["s"],
        "smin_threshold": best_s["threshold"],
        "smin_ru": best_s["ru"],
        "smin_mi": best_s["mi"],
    }


def boundary_metrics(calls: pl.DataFrame, truth: pl.DataFrame,
                     strict_iou: float = 0.8) -> dict:
    """Residue-level overlap and boundary accuracy.

    NDO here is the residue-level normalized domain overlap: correctly-labelled residues
    over true domain residues. It is the quantity CASP's NDO score is built from, not
    CASP's full scoring matrix, and is reported under that plainer definition.

    DBD is the distance in residues between a predicted boundary and the true one,
    reported as a median over correctly identified domains. Only correct calls have a
    meaningful boundary error -- the distance from a wrong domain to a right one is not a
    boundary measurement.
    """
    out = {
        "ndo": 0.0, "residue_precision": 0.0, "residue_recall": 0.0, "residue_f1": 0.0,
        "dbd_median": None, "dbd_mean": None,
        f"precision_iou{int(strict_iou * 100)}": 0.0,
        f"recall_iou{int(strict_iou * 100)}": 0.0,
        "n_tp_strict": 0,
    }
    if calls.height == 0 or truth.height == 0:
        return out

    tp = calls.filter("is_tp")
    total_true_residues = int(
        (truth["domain_end"] - truth["domain_start"]).sum()
    )
    total_pred_residues = int((calls["qend"] - calls["qstart"]).sum())

    if tp.height:
        # Overlap of each correct call with the instance it matched. Summed over calls
        # deduped to one per instance, so overlapping regions on one domain are not
        # double-counted into more than that domain's own length.
        best = (
            tp.group_by("query_acc", "pfam_id", "true_start", "true_end")
            .agg(
                pl.col("qstart").sort_by("iou", descending=True).first(),
                pl.col("qend").sort_by("iou", descending=True).first(),
                pl.col("iou").max(),
            )
            .with_columns(
                (
                    pl.min_horizontal("qend", "true_end")
                    - pl.max_horizontal("qstart", "true_start")
                ).clip(lower_bound=0).alias("ov")
            )
        )
        correct_residues = int(best["ov"].sum())
        out["ndo"] = correct_residues / total_true_residues if total_true_residues else 0.0
        out["residue_recall"] = out["ndo"]
        out["residue_precision"] = (
            correct_residues / total_pred_residues if total_pred_residues else 0.0
        )
        p, r = out["residue_precision"], out["residue_recall"]
        out["residue_f1"] = 2 * p * r / (p + r) if (p + r) else 0.0

        dbd = best.with_columns(
            (
                (pl.col("qstart") - pl.col("true_start")).abs()
                + (pl.col("qend") - pl.col("true_end")).abs()
            ).truediv(2).alias("dbd")
        )["dbd"]
        out["dbd_median"] = float(dbd.median())
        out["dbd_mean"] = float(dbd.mean())

    strict = calls.filter(pl.col("iou") >= strict_iou)
    n_strict_inst = (
        strict.select("query_acc", "pfam_id", "true_start", "true_end").unique().height
    )
    key = int(strict_iou * 100)
    out["n_tp_strict"] = strict.height
    out[f"precision_iou{key}"] = strict.height / calls.height if calls.height else 0.0
    out[f"recall_iou{key}"] = n_strict_inst / truth.height if truth.height else 0.0
    return out


def domain_count_metrics(calls: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Single- vs multi-domain call, scored by MCC.

    Scored only over proteins the tool actually made a call on. Counting silent proteins
    as "predicted single-domain" would reward a tool for saying nothing, since most
    proteins are single-domain.
    """
    out = {"domain_count_accuracy": None, "domain_count_mcc": None, "n_proteins_scored": 0}
    if calls.height == 0:
        return out

    true_n = truth.group_by("accession").agg(pl.len().alias("n_true"))
    pred_n = (
        calls.select("query_acc", "pfam_id", "qstart", "qend").unique()
        .group_by("query_acc").agg(pl.len().alias("n_pred"))
        .rename({"query_acc": "accession"})
    )
    j = pred_n.join(true_n, on="accession", how="inner")
    if j.height == 0:
        return out

    j = j.with_columns(
        (pl.col("n_true") > 1).alias("true_multi"),
        (pl.col("n_pred") > 1).alias("pred_multi"),
    )
    tp = int((j["true_multi"] & j["pred_multi"]).sum())
    tn = int((~j["true_multi"] & ~j["pred_multi"]).sum())
    fp = int((~j["true_multi"] & j["pred_multi"]).sum())
    fn = int((j["true_multi"] & ~j["pred_multi"]).sum())

    out["n_proteins_scored"] = j.height
    out["domain_count_accuracy"] = (tp + tn) / j.height
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # Undefined, not zero, when a whole row or column of the confusion matrix is empty:
    # MCC has no value there and 0.0 would read as "no better than chance".
    out["domain_count_mcc"] = ((tp * tn - fp * fn) / denom) if denom > 0 else None
    return out
