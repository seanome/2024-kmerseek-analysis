#!/usr/bin/env python3
"""Turn aligned regions into Pfam domain calls and score them against the answer key.

The question is domain finding, not orthology: for a human query protein, which stretch of
it is a given Pfam family, and did the tool put the region in the right place?

Pipeline for a search-based tool:

  1. Read the tool's regions: (query, target, qstart, qend, tstart, tend, score).
  2. Transfer. Look up the target protein's Pfam domains. A region claims a family when it
     covers at least --min-overlap of that domain on the *target* side. The region's query
     interval, carrying that family label, is now a domain call.
  3. Score. A call is a true positive when the query protein really does have that family
     AND the call's interval reciprocally overlaps (IoU) a real instance of it by at least
     --min-overlap. Right family in the wrong place is a false positive, not a hit -- that
     distinction is the whole reason for scoring regions instead of protein pairs.

With --direct-annotation (hmmscan against Pfam-A) step 2 is skipped: the tool already names
the family, so its intervals are query-side calls as they stand.

Two overlap criteria, deliberately different:
  transfer (target side)  overlap / domain_length -- did the region land on that domain
  scoring  (query side)   IoU = overlap / union   -- is the call in the right place
Transfer is the looser of the two on purpose. Being strict there would throw away correct
families over target-side boundary noise, which is not what is being measured.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# bin/ is on PATH under Nextflow but not on PYTHONPATH, so make the sibling module
# importable regardless of the working directory the task runs in.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cafa_metrics as cm  # noqa: E402


# Covariate axes the results get cut by. Continuous ones are binned; HGNC gene group is
# already categorical. Bin edges are fixed rather than data-derived so a stratum means
# the same thing across every tool, species and combo in the sweep.
STRATA = {
    "plddt": ("mean_plddt", [0, 50, 70, 90, 100]),
    "disorder": ("disorder_fraction_plddt", [0.0, 0.1, 0.3, 0.6, 1.01]),
    "omega": ("omega", [0.0, 0.1, 0.25, 0.5, 10.0]),
}
# Cutting on every one of ~4200 HGNC groups would produce mostly single-protein strata
# where no metric is stable. Only groups with at least this many query proteins are cut.
MIN_STRATUM_PROTEINS = 30


def extract_accession(col: pl.Expr) -> pl.Expr:
    """UniProt FASTA names are sp|P12345|NAME_SPECIES; annotations key on the accession.
    Names already bare (Foldseek, after its filename cleanup) pass through untouched."""
    return (
        pl.when(col.str.contains(r"\|"))
        .then(col.str.split("|").list.get(1, null_on_oob=True))
        .otherwise(col)
    )


def load_regions(path: Path, direct: bool) -> pl.LazyFrame | None:
    """Normalize any tool's output to one schema. Returns None for an empty result, which
    is a real outcome (a combo that found nothing), not an error.

    An empty gzip or zstd stream is NOT a zero-byte file -- both carry a frame header --
    so the size check alone cannot catch a tool that legitimately found nothing. polars
    raises NoDataError on those, which is caught here rather than at each call site.
    """
    try:
        return _load_regions(path, direct)
    except pl.exceptions.NoDataError:
        return None


def _load_regions(path: Path, direct: bool) -> pl.LazyFrame | None:
    if path.stat().st_size == 0:
        return None

    if path.suffix == ".parquet":
        # kmerseek. region_start/region_end are query-side; target_start/target_end are
        # target-side. region_poisson_score is -log10 of the region's Poisson tail, so
        # bigger is better and it sorts the same direction as a bitscore.
        lf = pl.scan_parquet(path)
        names = lf.collect_schema().names()
        if not names:
            return None
        return lf.select(
            extract_accession(pl.col("query_name")).alias("query_acc"),
            extract_accession(pl.col("target_name")).alias("target_acc"),
            pl.col("region_start").cast(pl.Int64).alias("qstart"),
            pl.col("region_end").cast(pl.Int64).alias("qend"),
            pl.col("target_start").cast(pl.Int64).alias("tstart"),
            pl.col("target_end").cast(pl.Int64).alias("tend"),
            pl.col("region_poisson_score").cast(pl.Float64).alias("score"),
        )

    if direct:
        cols = ["query", "pfam_id", "qstart", "qend", "score", "evalue"]
        lf = pl.scan_csv(path, separator="\t", has_header=False, new_columns=cols)
        return lf.select(
            extract_accession(pl.col("query")).alias("query_acc"),
            # hmmscan reports versioned Pfam accessions (PF00001.24); the tables key on
            # the unversioned id.
            pl.col("pfam_id").str.split(".").list.get(0).alias("pfam_id"),
            pl.col("qstart").cast(pl.Int64),
            pl.col("qend").cast(pl.Int64),
            pl.col("score").cast(pl.Float64),
        )

    # 8 columns for the aligners; motif tools (Folddisco) append a 9th holding how many
    # residues actually matched, so the envelope's density survives into the metrics.
    # Widths are read off the file rather than declared, so one loader serves both.
    cols = ["query", "target", "qstart", "qend", "tstart", "tend", "score", "evalue",
            "n_matched_residues"]
    probe = pl.scan_csv(path, separator="\t", has_header=False)
    width = len(probe.collect_schema().names())
    lf = pl.scan_csv(path, separator="\t", has_header=False,
                     new_columns=cols[:width])
    selection = [
        extract_accession(pl.col("query")).alias("query_acc"),
        extract_accession(pl.col("target")).alias("target_acc"),
        pl.col("qstart").cast(pl.Int64),
        pl.col("qend").cast(pl.Int64),
        pl.col("tstart").cast(pl.Int64),
        pl.col("tend").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
    ]
    if width >= 9:
        selection.append(pl.col("n_matched_residues").cast(pl.Int64))
    return lf.select(selection)


def overlap_expr(a_start: str, a_end: str, b_start: str, b_end: str) -> pl.Expr:
    lo = pl.max_horizontal(pl.col(a_start), pl.col(b_start))
    hi = pl.min_horizontal(pl.col(a_end), pl.col(b_end))
    return (hi - lo).clip(lower_bound=0)


def transfer_domains(regions: pl.LazyFrame, domain_map: pl.LazyFrame, min_overlap: float) -> pl.LazyFrame:
    """Label each region with every target-side Pfam domain it covers."""
    joined = regions.join(
        domain_map.select(
            pl.col("accession").alias("target_acc"),
            "pfam_id",
            pl.col("domain_start").alias("t_dom_start"),
            pl.col("domain_end").alias("t_dom_end"),
        ),
        on="target_acc",
        how="inner",
    )
    return (
        joined.with_columns(
            overlap_expr("tstart", "tend", "t_dom_start", "t_dom_end").alias("t_overlap"),
            (pl.col("t_dom_end") - pl.col("t_dom_start")).alias("t_dom_len"),
        )
        .filter(pl.col("t_overlap") >= min_overlap * pl.col("t_dom_len"))
        .select("query_acc", "pfam_id", "qstart", "qend", "score")
    )


def score_calls(calls: pl.LazyFrame, truth: pl.LazyFrame, min_overlap: float,
                semantics: str = "alignment") -> pl.DataFrame:
    """Match each call to the best true instance of the same family on the same protein.

    `semantics` picks what counts as correctly placed:

      alignment  IoU >= min_overlap. The call must coincide with the true domain. This is
                 the right test for a tool that reports an alignment, where a predicted
                 interval claims every residue inside it.
      motif      coverage of the true domain >= min_overlap. The right test for Folddisco,
                 whose interval is the envelope of a discontinuous residue set rather than
                 a claim on the residues between them. Judging that envelope by IoU would
                 score the envelope reduction, not the prediction.

    IoU is recorded either way, so the two are always inspectable side by side -- but
    is_tp, and therefore precision and recall, follow the tool's own semantics.
    """
    matched = (
        calls.join(
            truth.select(
                pl.col("accession").alias("query_acc"),
                "pfam_id",
                pl.col("domain_start").alias("true_start"),
                pl.col("domain_end").alias("true_end"),
            ),
            on=["query_acc", "pfam_id"],
            how="left",
        )
        .with_columns(overlap_expr("qstart", "qend", "true_start", "true_end").alias("ov"))
        .with_columns(
            # The null guard is load-bearing. A left join leaves true_start/true_end null
            # when the query protein has no instance of the transferred family -- the
            # definitive false positive. polars' max_horizontal/min_horizontal SKIP nulls
            # rather than propagating them, so without this branch the union collapses to
            # the call's own span, IoU comes out 1.0, and every call for a family the
            # protein does not have scores as a true positive. That inverts the metric.
            pl.when(pl.col("true_start").is_null() | pl.col("true_end").is_null())
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("ov")
                / (
                    pl.max_horizontal("qend", "true_end")
                    - pl.min_horizontal("qstart", "true_start")
                )
            )
            .fill_null(0.0)
            .alias("iou")
        )
        .with_columns(
            # Fraction of the true domain the call covers. Same null guard as above:
            # max/min_horizontal skip nulls, so an absent truth row must be branched on
            # explicitly rather than divided through.
            pl.when(pl.col("true_start").is_null() | pl.col("true_end").is_null())
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("ov") / (pl.col("true_end") - pl.col("true_start"))
            )
            .fill_null(0.0)
            .alias("cover")
        )
    )

    # One row per call: its best-matching true instance. A call whose family is absent from
    # the protein has iou 0 and stays a false positive.
    #
    # sort_by inside the aggregation, not a sort before group_by: polars does not promise
    # that a group_by preserves the input order within a group, so picking .first() off a
    # pre-sorted frame can silently return a non-best match.
    best = (
        matched.group_by(["query_acc", "pfam_id", "qstart", "qend"])
        .agg(
            # A call can appear once per target that transferred it; they are the same
            # call, so keep its strongest evidence rather than an arbitrary row.
            pl.col("score").max(),
            pl.col("iou").max(),
            pl.col("cover").max(),
            pl.col("true_start").sort_by("iou", descending=True).first(),
            pl.col("true_end").sort_by("iou", descending=True).first(),
        )
        .with_columns(
            (pl.col("cover") if semantics == "motif" else pl.col("iou"))
            .ge(min_overlap)
            .alias("is_tp")
        )
    )
    return best.collect(engine="streaming")


def rank_roc_auc(calls: pl.DataFrame) -> float | None:
    """Call-level ROC-AUC: P(a correctly placed call outranks an incorrectly placed one).

    Computed by the Mann-Whitney rank identity rather than by integrating the curve, so
    tied scores are handled exactly. Ties matter here: HP alphabets at low ksize produce
    large blocks of identical region scores, and trapezoid integration over a coarse
    curve would quietly round them in the tool's favour.

    Returns None, not 0.0, when one class is absent. A tool whose every call is correct
    has no ROC-AUC to report, and writing 0.0 there would rank it below a coin flip.
    """
    n = calls.height
    if n == 0:
        return None
    n_pos = int(calls["is_tp"].sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranked = calls.with_columns(
        pl.col("score").fill_null(float("-inf")).rank(method="average").alias("rank")
    )
    sum_pos_ranks = float(ranked.filter("is_tp")["rank"].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def operating_points(calls: pl.DataFrame, n_reachable: int) -> pl.DataFrame:
    """Every score threshold's full operating point, in one descending-score pass.

    Each row is "keep all calls scoring at least this much". Cumulative counts are taken
    at the last row of each distinct score so a threshold never splits a block of ties.

    Two different denominators sit side by side on purpose:
      precision, tpr, fpr    call-level -- of what was reported, how much was right
      recall_reachable       instance-level -- of the domains that could be found, how
                             many were, counting each true instance once no matter how
                             many regions hit it
    """
    if calls.height == 0:
        return pl.DataFrame(
            schema={
                "score_threshold": pl.Float64, "n_calls": pl.Int64, "tp_calls": pl.Int64,
                "fp_calls": pl.Int64, "instances_found": pl.Int64, "precision": pl.Float64,
                "recall_reachable": pl.Float64, "f1": pl.Float64, "tpr": pl.Float64,
                "fpr": pl.Float64,
            }
        )

    ranked = (
        calls.sort("score", descending=True, nulls_last=True)
        .with_columns(
            pl.when("is_tp")
            .then(pl.struct("query_acc", "pfam_id", "true_start", "true_end"))
            .otherwise(None)
            .alias("tp_key")
        )
        .with_columns(
            (pl.col("tp_key").is_not_null() & pl.col("tp_key").is_first_distinct())
            .alias("novel_tp")
        )
        .with_columns(
            pl.col("is_tp").cum_sum().alias("tp_calls"),
            (~pl.col("is_tp")).cum_sum().alias("fp_calls"),
            pl.col("novel_tp").cum_sum().alias("instances_found"),
        )
    )

    total_tp = int(ranked["is_tp"].sum())
    total_fp = ranked.height - total_tp

    pts = (
        ranked.group_by("score", maintain_order=True)
        .agg(
            pl.col("tp_calls").last(),
            pl.col("fp_calls").last(),
            pl.col("instances_found").last(),
        )
        .rename({"score": "score_threshold"})
        .with_columns((pl.col("tp_calls") + pl.col("fp_calls")).alias("n_calls"))
    )

    return pts.with_columns(
        (pl.col("tp_calls") / pl.col("n_calls")).alias("precision"),
        (pl.col("instances_found") / n_reachable if n_reachable else pl.lit(0.0)).alias(
            "recall_reachable"
        ),
        (pl.col("tp_calls") / total_tp if total_tp else pl.lit(0.0)).alias("tpr"),
        (pl.col("fp_calls") / total_fp if total_fp else pl.lit(0.0)).alias("fpr"),
    ).with_columns(
        pl.when(pl.col("precision") + pl.col("recall_reachable") > 0)
        .then(
            2
            * pl.col("precision")
            * pl.col("recall_reachable")
            / (pl.col("precision") + pl.col("recall_reachable"))
        )
        .otherwise(0.0)
        .alias("f1")
    )


def average_precision(points: pl.DataFrame) -> float:
    """Average precision: sum of precision weighted by the recall gained at each step.

    The step-wise sum, not a trapezoid, which is the standard AP definition and does not
    interpolate credit across a gap the tool never actually covered.
    """
    if points.height == 0:
        return 0.0
    recall = points["recall_reachable"]
    delta = recall - recall.shift(1, fill_value=0.0)
    return float((points["precision"] * delta).sum())


def downsample(points: pl.DataFrame, max_points: int) -> pl.DataFrame:
    """Thin the curve for storage, always keeping both ends.

    A 1017-combo sweep writing one row per distinct score would dwarf the metrics it
    supports. Every scalar metric is computed on the FULL curve before this runs, so
    thinning changes the plot's resolution and nothing else.
    """
    n = points.height
    if n <= max_points:
        return points
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    return points[sorted(set(idx))]


def compute_metrics(calls: pl.DataFrame, points: pl.DataFrame, truth: pl.DataFrame,
                    reachable: pl.DataFrame, min_overlap: float) -> dict:
    n_calls = calls.height
    n_tp_calls = int(calls["is_tp"].sum()) if n_calls else 0

    # Counts distinct true instances found, not calls: several regions hitting one domain
    # is one recovery, not many.
    found = (
        calls.filter("is_tp").select("query_acc", "pfam_id", "true_start", "true_end")
        .unique().height
        if n_calls
        else 0
    )
    n_truth = truth.height
    n_reachable = reachable.height

    precision = n_tp_calls / n_calls if n_calls else 0.0
    recall = found / n_truth if n_truth else 0.0
    recall_reachable = found / n_reachable if n_reachable else 0.0

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    metrics = {
        "n_calls": n_calls,
        "n_tp_calls": n_tp_calls,
        "n_fp_calls": n_calls - n_tp_calls,
        "n_truth_instances": n_truth,
        "n_reachable_instances": n_reachable,
        "n_instances_found": found,
        # --- operating point the tool actually reported at ---
        "precision": precision,
        "recall": recall,
        # Recall against what was transferable at all. A human family absent from this
        # target proteome cannot be recovered by any search, so raw recall above
        # understates every tool by the same species-specific amount. Compare tools on
        # this one.
        "recall_reachable": recall_reachable,
        "f1": f1(precision, recall),
        "f1_reachable": f1(precision, recall_reachable),
        # --- threshold-free ---
        "roc_auc": rank_roc_auc(calls),
        "auprc": average_precision(points),
        "min_overlap": min_overlap,
        "median_iou_tp": float(calls.filter("is_tp")["iou"].median()) if n_tp_calls else 0.0,
    }

    # --- best achievable operating point, and where it sits ---
    # The reported point above depends on each tool's own default cutoff, which differs
    # between tools and is not a property of the method. This is the comparable one.
    if points.height:
        best = points.sort("f1", descending=True).head(1).to_dicts()[0]
        metrics.update({
            "best_f1": best["f1"],
            "best_f1_threshold": best["score_threshold"],
            "best_f1_precision": best["precision"],
            "best_f1_recall_reachable": best["recall_reachable"],
        })
    else:
        metrics.update({
            "best_f1": 0.0, "best_f1_threshold": None,
            "best_f1_precision": 0.0, "best_f1_recall_reachable": 0.0,
        })
    return metrics


def attach_strata(truth: pl.DataFrame, covariates: pl.DataFrame | None) -> pl.DataFrame:
    """Add one column per covariate axis, holding that protein's stratum label."""
    if covariates is None:
        return truth.with_columns(pl.lit("all").alias("stratum_hgnc"))

    cov = covariates
    exprs = []
    for axis, (col, edges) in STRATA.items():
        name = f"stratum_{axis}"
        if col not in cov.columns:
            exprs.append(pl.lit(None, dtype=pl.String).alias(name))
            continue
        expr = pl.when(pl.col(col).is_null()).then(pl.lit(None, dtype=pl.String))
        for lo, hi in zip(edges[:-1], edges[1:]):
            expr = expr.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(
                pl.lit(f"{lo}-{hi}")
            )
        exprs.append(expr.otherwise(None).alias(name))

    hgnc = "hgnc_gene_group"
    exprs.append(
        pl.col(hgnc).alias("stratum_hgnc") if hgnc in cov.columns
        else pl.lit(None, dtype=pl.String).alias("stratum_hgnc")
    )
    cov = cov.with_columns(exprs)

    keep = ["accession"] + [c for c in cov.columns if c.startswith("stratum_")]
    return truth.join(cov.select(keep), on="accession", how="left")


def strata_of(truth: pl.DataFrame) -> list[tuple[str, str]]:
    """Enumerate (axis, value) cuts worth reporting, always including the ungrouped one."""
    out = [("all", "all")]
    for col in (c for c in truth.columns if c.startswith("stratum_")):
        axis = col.removeprefix("stratum_")
        counts = (
            truth.filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.col("accession").n_unique().alias("n"))
            .filter(pl.col("n") >= MIN_STRATUM_PROTEINS)
            .sort("n", descending=True)
        )
        out.extend((axis, v) for v in counts[col].to_list())
    return out


def subset(truth: pl.DataFrame, calls: pl.DataFrame, split: str,
           axis: str, value: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restrict both the answer key and the calls to one split x stratum cell.

    Truth and calls must be cut the same way or the metrics are incoherent: keeping a
    call whose protein is outside the stratum would count against a denominator that
    never included it.
    """
    t = truth
    if split != "all":
        t = t.filter(pl.col("split") == split)
    if axis != "all":
        t = t.filter(pl.col(f"stratum_{axis}") == value)

    if t.height == 0:
        return t, calls.head(0)

    proteins = t.select("accession").unique().rename({"accession": "query_acc"})
    c = calls.join(proteins, on="query_acc", how="inner")
    if split != "all":
        # Splits are grouped by Pfam family, so a call is in the split iff its claimed
        # family is. Filtering calls by protein alone would leak selection-half families
        # into the held-out numbers.
        fams = t.select("pfam_id").unique()
        c = c.join(fams, on="pfam_id", how="inner")
    return t, c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", required=True, type=Path)
    p.add_argument("--tool", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--domain-map", type=Path)
    p.add_argument("--covariates", type=Path,
                   help="per-protein HGNC group / omega / pLDDT / disorder table")
    p.add_argument("--direct-annotation", action="store_true")
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--strict-iou", type=float, default=0.8)
    p.add_argument("--interval-semantics", choices=["alignment", "motif"],
                   default="alignment",
                   help="motif for tools reporting discontinuous residue sets (Folddisco)")
    p.add_argument("--calls-out", required=True, type=Path)
    p.add_argument("--metrics-out", required=True, type=Path)
    p.add_argument("--curve-out", type=Path)
    p.add_argument("--max-curve-points", type=int, default=2000)
    args = p.parse_args()

    if not args.direct_annotation and args.domain_map is None:
        raise SystemExit("--domain-map is required unless --direct-annotation is set")

    truth_lf = pl.scan_parquet(args.truth)
    truth = truth_lf.collect()
    covariates = pl.read_parquet(args.covariates) if args.covariates else None
    truth = attach_strata(truth, covariates)

    regions = load_regions(args.regions, args.direct_annotation)

    if args.direct_annotation:
        target_families = None
        calls_lf = regions
    else:
        map_lf = pl.scan_parquet(args.domain_map)
        target_families = map_lf.select("pfam_id").unique().collect()
        calls_lf = (
            transfer_domains(regions, map_lf, args.min_overlap) if regions is not None else None
        )

    if calls_lf is None:
        scored = pl.DataFrame(
            schema={
                "query_acc": pl.String, "pfam_id": pl.String, "qstart": pl.Int64,
                "qend": pl.Int64, "score": pl.Float64, "iou": pl.Float64,
                "cover": pl.Float64, "true_start": pl.Int64, "true_end": pl.Int64,
                "is_tp": pl.Boolean,
            }
        )
    else:
        scored = score_calls(calls_lf, truth_lf, args.min_overlap, args.interval_semantics)

    scored.write_parquet(args.calls_out, compression="zstd")

    # IC is estimated once on the whole answer key, not per stratum. A family's rarity is
    # a property of the proteome; re-estimating it inside each cut would make the same
    # family worth different amounts in different strata and break comparability.
    ic = cm.information_content(truth)

    ident = {"tool": args.tool, "variant": args.variant, "species": args.species,
             # Stamped on every row so an alignment tool and a motif tool are never
             # silently compared on boundary metrics that mean different things.
             "interval_semantics": args.interval_semantics}
    rows, curves = [], []

    for split in ("all", "selection", "heldout"):
        if split != "all" and "split" not in truth.columns:
            continue
        for axis, value in strata_of(truth):
            t_sub, c_sub = subset(truth, scored, split, axis, value)
            if t_sub.height == 0:
                continue

            reachable = (
                t_sub if target_families is None
                else t_sub.join(target_families, on="pfam_id", how="inner")
            )
            points = operating_points(c_sub, reachable.height)
            m = compute_metrics(c_sub, points, t_sub, reachable, args.min_overlap)

            pc = cm.protein_centric_curve(c_sub, t_sub, ic)
            m.update(cm.cafa_scalars(pc))
            m.update(cm.boundary_metrics(c_sub, t_sub, args.strict_iou))
            m.update(cm.domain_count_metrics(c_sub, t_sub))
            m.update(ident)
            m.update({"split": split, "stratum_axis": axis, "stratum": value})
            rows.append(m)

            # Curves only for the ungrouped cut. One per split x stratum x combo would
            # dwarf the metrics they support across a 1017-combo sweep.
            if axis == "all" and args.curve_out is not None:
                curves.append(
                    downsample(points, args.max_curve_points).with_columns(
                        pl.lit(split).alias("split"),
                        **{k: pl.lit(v) for k, v in ident.items()},
                    )
                )

    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        args.metrics_out, compression="zstd"
    )
    if args.curve_out is not None:
        (pl.concat(curves, how="diagonal_relaxed") if curves
         else pl.DataFrame(schema={"split": pl.String})).write_parquet(
            args.curve_out, compression="zstd"
        )

    headline = next(
        (r for r in rows if r["split"] == "all" and r["stratum_axis"] == "all"), {}
    )
    print(json.dumps(
        {k: v for k, v in headline.items() if not k.startswith("stratum")}, indent=2
    ))
    print(f"\nemitted {len(rows)} metric rows across splits x strata")


if __name__ == "__main__":
    main()
