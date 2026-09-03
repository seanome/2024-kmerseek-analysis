#!/usr/bin/env python3
"""CAFA-style and domain-boundary metrics, adapted honestly to Pfam domain finding.

What carries over from CAFA, and what does not:

  Fmax        Carries over. CAFA's Fmax is *protein-centric*: precision is averaged over
              proteins that made at least one prediction, recall over all proteins with a
              true annotation, and the max is taken over thresholds. That macro-average is
              a different number from the micro-averaged `best_f1` this pipeline also
              reports -- a handful of domain-dense proteins cannot dominate it. Both are
              kept because they answer different questions.

              Reported at two levels, on the same row. `fmax` is interval-aware: a call
              counts only where it also lands on the annotated interval. `family_fmax` is
              the CAFA-classic reading -- the SET of families called on a protein against
              the set truly present, placement ignored. A tool that names the right family
              and draws the boundary wrong scores zero on the first and full marks on the
              second, so the pair separates recognition from delineation, which one number
              cannot.

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


def family_view(calls: pl.DataFrame, truth: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recast calls and truth as SETS of (protein, family), dropping interval placement.

    The unit becomes the family called on a protein, so a family predicted ten times on one
    protein is one prediction and a family annotated three times on one protein is one
    annotation. Deduping to the best score per (query_acc, pfam_id) is what makes the
    metric independent of how many redundant copies of a call a tool emitted -- which is
    the point. The interval-aware curve deliberately penalises that redundancy
    (assign_instances turns the copies into false positives); this one deliberately does
    not, because it is measuring recognition rather than delineation.

    `is_tp` is recomputed from scratch rather than carried over. A call that named the
    right family in the wrong place has is_tp False at the interval level and is a correct
    family call here -- that difference is the whole quantity this view exists to expose.
    Membership is tested against the truth it is handed, so a stratum's cut is respected
    without any extra restriction step.
    """
    truth_fam = truth.select("accession", "pfam_id").unique()
    present = truth_fam.select(
        pl.col("accession").alias("query_acc"), "pfam_id"
    ).with_columns(pl.lit(True).alias("_present"))
    calls_fam = (
        calls.group_by("query_acc", "pfam_id")
        .agg(pl.col("score").max())
        .join(present, on=["query_acc", "pfam_id"], how="left")
        .with_columns(pl.col("_present").fill_null(False).alias("is_tp"))
        .drop("_present")
    )
    return calls_fam, truth_fam


def family_level_counts(calls: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Denominators the family-level Fmax is computed over, so no cell reports a rate alone.

    n_family_calls is after the dedup, which is why it can be far below n_calls: a tool
    emitting fifty regions that all transfer PF00069 onto one protein has made one family
    prediction.
    """
    calls_fam, truth_fam = family_view(calls, truth)
    return {
        "n_family_truth": truth_fam.height,
        "n_family_calls": calls_fam.height,
        "n_family_found": (
            int(calls_fam["is_tp"].sum()) if calls_fam.height else 0
        ),
    }


def protein_centric_curve(calls: pl.DataFrame, truth: pl.DataFrame, ic: pl.DataFrame,
                          n_thresholds: int = DEFAULT_N_THRESHOLDS,
                          level: str = "interval") -> pl.DataFrame:
    """Fmax / wFmax / Smin operating points, protein-centric.

    `level` picks what a prediction and an annotation ARE, and nothing else changes:

      interval  the default. A prediction counts only when it is placed correctly (is_tp,
                i.e. IoU past the caller's cutoff), so this measures domain *finding*.
                Recall is over distinct true instances: several regions hitting one domain
                is one recovery.
      family    CAFA-classic. The prediction is the SET of families called on a protein
                against the set truly present, placement ignored entirely. Reported
                alongside rather than instead of the interval reading: the pair separates
                "did not recognise the family" from "recognised it but drew the boundary
                wrong", which the interval number alone scores identically at zero.

    The two share this whole function on purpose. A second near-copy of the threshold
    sweep is a thing that drifts, and the drift would be silent -- both numbers would keep
    coming out plausible.

    family_fmax >= fmax is USUALLY true and is NOT an identity. Ignoring placement can only
    help precision, but the family reading also swaps the recall denominator from instances
    to families, and a per-protein macro-average is not invariant under that swap. One
    protein with three instances of family A, all found, and one instance of family B,
    missed: interval recall is 3/4, family recall is 1/2. Verified on the mini run at
    3 rows out of 2762, every one of them a cell where a protein carries many instances of
    one family (an IGSF decoy at 51 instances per family is the worst, fmax 0.831 against
    family_fmax 0.727). Real, not a bug, so nothing clamps it -- but a reader taking the
    gap as a recognition-minus-delineation reading needs to know it can go the other way
    where tandem arrays dominate a cut.

    The family grid is derived from the DEDUPED scores rather than from the incoming call
    scores. That makes family_fmax exactly invariant to how many redundant copies of a call
    a tool emitted, which is the property the whole level exists for: a family's best score
    does not move when a worse-scoring copy of it is added, so neither does the grid, so
    neither does any number here. Sharing the interval grid instead would give the two
    readings identical thresholds -- worth ~0.001 on one mini cell out of 912 -- at the cost
    of that invariance, which is the more meaningful of the two.
    """
    if level not in ("interval", "family"):
        raise ValueError(f"level must be 'interval' or 'family', not {level!r}")
    if level == "family":
        calls, truth = family_view(calls, truth)
    # What makes two correct calls the SAME recovery. At the interval level a true instance
    # is identified by its coordinates; at the family level the family on the protein is
    # the whole identity, and there are no coordinates left to group on.
    det_keys = (["query_acc", "pfam_id"] if level == "family"
                else ["query_acc", "pfam_id", "true_start", "true_end"])

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
    # where nothing changes and skip the range where the curve turns.
    qs = np.linspace(0, 1, n_thresholds)
    t_asc = np.unique(np.quantile(scores, qs))
    n_bins = len(t_asc)

    # Number of thresholds a call clears; it contributes to descending bins 0..k-1.
    k = np.searchsorted(t_asc, scores, side="right")
    valid = k > 0
    bin_idx = (k - 1)[valid]

    call_p = np.array([pidx.get(a, -1) for a in calls["query_acc"].to_list()])
    call_ic = np.array([ic_map.get(f, 0.0) for f in calls["pfam_id"].to_list()])

    keep = valid & (call_p >= 0)
    kb = (k - 1)[keep]
    kp = call_p[keep]

    n_pred = _suffix_counts(kp, kb, np.ones(keep.sum()), n_proteins, n_bins)
    w_pred = _suffix_counts(kp, kb, call_ic[keep], n_proteins, n_bins)

    # Detections: one row per true instance, at its best-scoring correct call, so recall
    # counts instances rather than rewarding many regions hitting the same domain.
    det = (
        calls.filter("is_tp")
        .group_by(det_keys)
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


def cafa_scalars(curve: pl.DataFrame, prefix: str = "") -> dict:
    """Pick the operating points off a curve. `prefix` namespaces the keys.

    The interval and family readings are emitted on the SAME metrics row, so the second one
    is asked for as cafa_scalars(family_curve, prefix="family_") and every column it
    produces -- threshold, precision, recall, weighted and semantic-distance terms
    included -- lands beside its interval twin instead of in a parallel table.
    """
    if curve.height == 0:
        out = {"fmax": 0.0, "fmax_threshold": None, "fmax_precision": 0.0,
               "fmax_recall": 0.0, "wfmax": 0.0, "smin": None, "smin_threshold": None,
               "smin_ru": None, "smin_mi": None}
        return {f"{prefix}{k}": v for k, v in out.items()}
    best_f = curve.sort("f", descending=True).head(1).to_dicts()[0]
    best_wf = curve.sort("wf", descending=True).head(1).to_dicts()[0]
    best_s = curve.sort("s", descending=False).head(1).to_dicts()[0]
    out = {
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
    return {f"{prefix}{k}": v for k, v in out.items()}


def boundary_metrics(calls: pl.DataFrame, truth: pl.DataFrame,
                     strict_iou: float = 0.8, exclude_points: bool = True) -> dict:
    """Residue-level overlap and boundary accuracy.

    There is no `ndo` key. There used to be, and it was assigned the value of
    residue_recall on the line after it was computed -- one quantity under two names.
    Every report table carrying both showed them identical to every printed decimal across
    every arm, which reads as a corrupted column rather than as a duplicate. The
    residue-level quantity is real and is kept; the CASP name is not, because what CASP
    scores is a domain DECOMPOSITION of a chain -- an overlap matrix between a predicted
    partition and a reference partition -- and calls here are per family and may overlap
    each other, so a partition is not what this benchmark produces. Reporting a partition
    metric's name over a plain residue recall claims a comparison to CASP that the data
    cannot support.

    DBD is the distance in residues between a predicted boundary and the true one,
    reported as a median over correctly identified domains. Only correct calls have a
    meaningful boundary error -- the distance from a wrong domain to a right one is not a
    boundary measurement.

    `exclude_points` drops truth intervals flagged is_point, and the calls that matched
    them, from every number below. A point feature is a single annotated residue -- a
    catalytic site, a metal ligand -- that build_swissprot_truth widens by one and
    build_mcsa_truth widens by a window purely so an interval exists at all. There is no
    boundary to be right or wrong about at that length, so the residue rates, DBD and the
    terminal
    offsets would be measuring the widening rather than the prediction. Both sides are cut,
    truth and calls, so numerator and denominator keep describing the same set. Truth sets
    with no is_point column -- Pfam, Pfam-N -- are untouched.
    """
    out = {
        "residue_precision": 0.0, "residue_recall": 0.0, "residue_f1": 0.0,
        "dbd_median": None, "dbd_mean": None,
        "nterm_offset_median": None, "nterm_offset_mean": None, "nterm_offset_iqr": None,
        "cterm_offset_median": None, "cterm_offset_mean": None, "cterm_offset_iqr": None,
        f"precision_iou{int(strict_iou * 100)}": 0.0,
        f"recall_iou{int(strict_iou * 100)}": 0.0,
        "n_tp_strict": 0,
        # Always present, so a cell that excluded nothing and a cell that excluded
        # everything are told apart in the table rather than by a missing column.
        "n_point_instances_excluded": 0,
    }
    if exclude_points and "is_point" in truth.columns:
        points = truth.filter("is_point").select(
            pl.col("accession").alias("query_acc"), "pfam_id",
            pl.col("domain_start").alias("true_start"),
            pl.col("domain_end").alias("true_end"),
        ).unique()
        truth = truth.filter(~pl.col("is_point"))
        # Anti-join, not a semi-join on the kept instances: a call that matched nothing has
        # null true_start/true_end, polars does not match null join keys, and a semi-join
        # would therefore delete every false positive and hand back a perfect precision.
        calls = calls.join(
            points, on=["query_acc", "pfam_id", "true_start", "true_end"], how="anti"
        )
        out["n_point_instances_excluded"] = points.height

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
        out["residue_recall"] = (
            correct_residues / total_true_residues if total_true_residues else 0.0
        )
        out["residue_precision"] = (
            correct_residues / total_pred_residues if total_pred_residues else 0.0
        )
        p, r = out["residue_precision"], out["residue_recall"]
        out["residue_f1"] = 2 * p * r / (p + r) if (p + r) else 0.0

        # N- and C-terminal offsets are kept SEPARATE, and signed. Boundary methods
        # usually fail asymmetrically -- a k-mer method loses the first k-1 residues at the
        # N terminus for a structural reason, not a random one -- and averaging the two
        # ends into one number destroys the interpretable part. Sign convention:
        # positive means the call starts/ends LATER than the true domain, so a k-mer method
        # that trims the front shows n_offset > 0.
        bnd = best.with_columns(
            (pl.col("qstart") - pl.col("true_start")).alias("n_offset"),
            (pl.col("qend") - pl.col("true_end")).alias("c_offset"),
        ).with_columns(
            ((pl.col("n_offset").abs() + pl.col("c_offset").abs()) / 2).alias("dbd")
        )
        out["dbd_median"] = float(bnd["dbd"].median())
        out["dbd_mean"] = float(bnd["dbd"].mean())
        for end in ("n", "c"):
            col = bnd[f"{end}_offset"]
            out[f"{end}term_offset_median"] = float(col.median())
            out[f"{end}term_offset_mean"] = float(col.mean())
            out[f"{end}term_offset_iqr"] = float(col.quantile(0.75) - col.quantile(0.25))

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

    Scored only over proteins the tool made a call on. Counting silent proteins
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


def sensitivity_to_first_fp(calls: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Fraction of a query's true domains recovered before its first false positive.

    The metric Foldseek and Folddisco report on SCOPe, carried over so this pipeline's
    numbers are in the same units as the NBT literature rather than needing a conversion
    the reader has to trust. Same definition as
    notebooks/sensitivity_until_first_false_positive.py: rank a query's calls by score,
    walk down until the first incorrect one, and count what was recovered above it.

    Averaged over query proteins that produced at least one call. A protein a tool stayed
    silent on has no ranking to evaluate and is excluded rather than scored 0 -- scoring it
    would conflate "ranked badly" with "said nothing", which are different failures.
    """
    out = {"sens_first_fp_mean": None, "sens_first_fp_median": None,
           "n_proteins_ranked": 0}
    if calls.height == 0 or truth.height == 0:
        return out

    n_true = truth.group_by("accession").agg(pl.len().alias("n_true")).rename(
        {"accession": "query_acc"}
    )
    ranked = calls.sort("score", descending=True, nulls_last=True)

    per = (
        ranked.group_by("query_acc", maintain_order=True)
        .agg(pl.col("is_tp").alias("hits"))
        .join(n_true, on="query_acc", how="inner")
    )
    if per.height == 0:
        return out

    vals = []
    for hits, nt in zip(per["hits"].to_list(), per["n_true"].to_list()):
        if not nt:
            continue
        tp_before = 0
        for h in hits:
            if not h:
                break
            tp_before += 1
        vals.append(min(tp_before / nt, 1.0))
    if not vals:
        return out

    arr = np.array(vals, dtype=float)
    out["sens_first_fp_mean"] = float(arr.mean())
    out["sens_first_fp_median"] = float(np.median(arr))
    out["n_proteins_ranked"] = len(vals)
    return out
