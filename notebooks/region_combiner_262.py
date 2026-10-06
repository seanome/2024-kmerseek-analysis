"""Combiner B, best of n: every merged region any alphabet-ksize pair called is ranked by
the best score any pair gave it. Built for notebook 262 on the files notebook 260 writes.

Three ranking scores, kept separate:

* ``evalue_min``: the lowest ``region_evalue`` any pair gave the region (lower is better).
* ``evalue_best_of_n_corrected`` = ``n_arms_tried * evalue_min`` (lower is better).
  ``n_arms_tried`` is the number of pairs whose search was run on that query and target
  species, whether or not they called anything there. Taking the lowest of n E-values is
  n chances to get a low one by luck; multiplying by n undoes that when the n searches are
  independent, and over-corrects when they are not.
* ``mean_idf_max``: the highest ``region_mean_idf`` any pair gave the region (higher is
  better). region_mean_idf is the mean rarity of one shared k-mer in the region.

Scores are taken over every call in the region, not only the one row per pair that the
region table keeps, so the calls are re-read from notebook 260's ``reduced/`` files and
re-merged with notebook 260's own merge. `merged_calls` checks the merge reproduces the
table's merged regions.

Truth follows notebook 260 and 261: a merged region is a true call in a truth set when at
least half of it lies inside one feature of that set. Precision counts calls on queries
that have at least one feature in that set; recall counts features found, once per target
species searched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import region_combiner_261 as rc
import region_table_260 as rt

KEY = rc.KEY
TRUTH_SETS = rc.TRUTH_SETS
FEATURE_KEY = rc.FEATURE_KEY

#: Ranking scores. True = lower is better.
SCORES = {
    "evalue_min": True,
    "evalue_best_of_n_corrected": True,
    "mean_idf_max": False,
}
SCORE_LABELS = {
    "evalue_min": "lowest E-value over pairs (raw)",
    "evalue_best_of_n_corrected": "lowest E-value x pairs tried (corrected)",
    "mean_idf_max": "highest mean IDF over pairs",
}
#: The precision the threshold is frozen at, on tune.
TARGET_PRECISION = 0.5
#: Score for the threshold rule and every reported number.
RANK_SCORE = "evalue_best_of_n_corrected"


def arms_tried(summary: pl.DataFrame) -> pl.DataFrame:
    """Pairs searched per target species: one regions file is one pair's search of every
    query in the run against that species, so the count of files is n_arms_tried for each
    query. Files with no E-value fit count too: they were run."""
    return summary.group_by("species").agg(
        n_arms_tried=pl.col("arm").n_unique(),
        arms_tried=pl.col("arm").unique().sort(),
    )


def merged_calls(reduced_dir: Path, table: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every call that passed notebook 260's E-value cut, with its merged region id.

    With `table`, check that the re-merge gives the same merged regions (ids and extents)
    as notebook 260's table, and raise if not.
    """
    files = sorted(reduced_dir.glob("*.kept.parquet"))
    calls = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    m = rt.merge_calls(calls)
    if table is not None:
        a = m.select(KEY + ["merged_start", "merged_end"]).unique().sort(KEY)
        b = table.select(KEY + ["merged_start", "merged_end"]).unique().sort(KEY)
        if not a.equals(b):
            raise ValueError(f"re-merge differs from the table: {a.height} vs {b.height} regions")
    return m


def region_scores(calls: pl.DataFrame, tried: pl.DataFrame) -> pl.DataFrame:
    """One row per merged region: the three ranking scores, the pairs that gave the best
    E-value and the best IDF (all of them when tied), and how many pairs called it."""
    per_arm = calls.group_by(KEY + ["arm"]).agg(
        e=pl.col("region_evalue").min(), idf=pl.col("region_mean_idf").max())
    best = per_arm.group_by(KEY).agg(
        evalue_min=pl.col("e").min(),
        mean_idf_max=pl.col("idf").max(),
        n_arms_fired=pl.len(),
        evalue_min_arms=pl.col("arm").filter(pl.col("e") == pl.col("e").min()).sort(),
        mean_idf_max_arms=pl.col("arm").filter(pl.col("idf") == pl.col("idf").max()).sort(),
    )
    return (best.join(tried.select("species", "n_arms_tried"), on="species", how="left")
            .with_columns(evalue_best_of_n_corrected=pl.col("n_arms_tried") * pl.col("evalue_min"))
            .sort(KEY))


def attach_truth(scores: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Add split, length, truth flags and the best Swiss-Prot feature from the region table."""
    cols = KEY + ["split", "merged_start", "merged_end", "merged_length",
                  "merged_swissprot_feature", "merged_pfam_feature"]
    cols += [f"merged_{ts}_is_true" for ts in TRUTH_SETS] + [f"query_has_{ts}" for ts in TRUTH_SETS]
    cols = [c for c in cols if c in table.columns]
    return scores.join(table.select(cols).unique(subset=KEY), on=KEY, how="left")


@dataclass
class Calls:
    """Scored calls on one split, ready for precision-recall.

    `regions`: one row per call with ``rid``, ``accession``, ``species``, the score
    columns, ``<ts>_is_true`` and ``query_has_<ts>``. `landed`: (rid, species, feature)
    rows where the call puts at least half of itself inside the feature.
    """

    regions: pl.DataFrame
    landed: pl.DataFrame
    n_features: dict


def kmerseek_calls(regions: pl.DataFrame, truth_pairs: pl.DataFrame,
                   n_features: dict) -> Calls:
    r = regions.sort(KEY).with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int64))
    r = r.rename({f"merged_{ts}_is_true": f"{ts}_is_true" for ts in TRUTH_SETS})
    landed = (truth_pairs.filter(pl.col("landed") >= rt.LANDED_MIN)
              .join(r.select(KEY + ["rid"]), on=KEY, how="inner")
              .select(["rid", "species"] + FEATURE_KEY))
    return Calls(r, landed, n_features)


def n_features(truth: pl.DataFrame, split: str, n_species: int) -> dict:
    feats = (truth.unique(subset=FEATURE_KEY)
             .with_columns(split=rt.query_split(pl.col("accession")))
             .filter(pl.col("split") == split))
    return {ts: feats.filter(pl.col("truth_set") == ts).height * n_species for ts in TRUTH_SETS}


def kd_calls(seqs: dict[str, str], truth: pl.DataFrame, species: list[str], split: str,
             n_feat: dict) -> Calls:
    """Notebook 231's Kyte-Doolittle scan (19-residue window, mean hydropathy above 1.6)
    as calls on the split's queries. It reads only the query, so the same calls stand for
    every target species, which keeps its recall on the same denominator as kmerseek's.
    Score: the segment's best window mean (higher is better), as ``mean_idf_max`` is."""
    import swissprot_control_utils as sc

    kd = (sc.kd_transmem_calls(seqs)
          .select(accession="query_acc", start="qstart", end="qend", kd_score="score")
          .with_columns(split=rt.query_split(pl.col("accession")))
          .filter(pl.col("split") == split))
    kd = kd.join(pl.DataFrame({"species": species}), how="cross").sort(
        "accession", "species", "start").with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int64))
    pairs = rt.region_truth_pairs(kd.select("rid", "accession", "species", "start", "end"),
                                  truth, "start", "end")
    hit = pairs.filter(pl.col("landed") >= rt.LANDED_MIN)
    has = truth.select("accession", "truth_set").unique()
    for ts in TRUTH_SETS:
        true_rid = hit.filter(pl.col("truth_set") == ts)["rid"].unique().implode()
        acc = has.filter(pl.col("truth_set") == ts)["accession"].implode()
        kd = kd.with_columns(pl.col("rid").is_in(true_rid).alias(f"{ts}_is_true"),
                             pl.col("accession").is_in(acc).alias(f"query_has_{ts}"))
    return Calls(kd, hit.select(["rid", "species"] + FEATURE_KEY), n_feat)


def pr_curve(c: Calls, score: str, lower_is_better: bool, ts: str) -> pl.DataFrame:
    """Precision and recall at every distinct score, calling every region at least as
    good as it. One row per threshold, loosest last."""
    r = c.regions.filter(pl.col(score).is_not_null() & pl.col(score).is_not_nan())
    s = r[score].to_numpy().astype(float)
    key = s if lower_is_better else -s
    order = np.argsort(key, kind="stable")
    k = key[order]
    on = r[f"query_has_{ts}"].to_numpy()[order]
    tru = r[f"{ts}_is_true"].to_numpy()[order] & on
    last = np.r_[np.flatnonzero(np.diff(k) != 0), len(k) - 1]  # end of each tie group
    n_called = last + 1
    n_on = np.cumsum(on)[last]
    n_true = np.cumsum(tru)[last]
    # A feature is found at threshold t when its best-scoring landed call passes t.
    rid_key = dict(zip(r["rid"].to_list(), key.tolist()))
    lf = c.landed.filter(pl.col("truth_set") == ts).with_columns(
        k=pl.col("rid").replace_strict(rid_key, default=None, return_dtype=pl.Float64))
    best = (lf.filter(pl.col("k").is_not_null())
            .group_by(["species"] + FEATURE_KEY).agg(pl.col("k").min())["k"].to_numpy().copy())
    best.sort()
    n_found = np.searchsorted(best, k[last], side="right")
    thr = k[last] if lower_is_better else -k[last]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(n_on > 0, n_true / np.maximum(n_on, 1), np.nan)
    return pl.DataFrame({
        "threshold": thr, "n_called": n_called, "n_called_on_truth": n_on, "n_true": n_true,
        "precision": prec, "n_found": n_found,
        "recall": n_found / c.n_features[ts] if c.n_features[ts] else np.full(len(last), np.nan),
    })


def average_precision(curve: pl.DataFrame) -> float:
    """Sum over thresholds of precision x the recall gained there (step rule, no
    interpolation); higher is better."""
    rec = np.r_[0.0, curve["recall"].to_numpy()]
    prec = np.nan_to_num(curve["precision"].to_numpy())
    return float(np.sum(np.diff(rec) * prec))


def loosest_threshold(curve: pl.DataFrame, target: float = TARGET_PRECISION) -> dict | None:
    """The loosest threshold whose precision is still at least `target`, or None when no
    threshold reaches it. Rows are ordered strictest first, so the loosest is the last."""
    ok = curve.filter(pl.col("precision") >= target)
    return ok.row(-1, named=True) if ok.height else None


def at_threshold(c: Calls, score: str, lower_is_better: bool, thr: float) -> dict:
    """Calls, precision and recall in both truth sets for one fixed threshold."""
    r = c.regions
    sel = r.filter(pl.col(score) <= thr if lower_is_better else pl.col(score) >= thr)
    out = {"threshold": thr, "n_called": sel.height}
    rid = sel["rid"].implode()
    for ts in TRUTH_SETS:
        on = sel.filter(pl.col(f"query_has_{ts}"))
        out[f"n_called_on_{ts}_queries"] = on.height
        out[f"precision_{ts}"] = (on[f"{ts}_is_true"].sum() / on.height) if on.height else float("nan")
        found = (c.landed.filter((pl.col("truth_set") == ts) & pl.col("rid").is_in(rid))
                 .unique(subset=["species"] + FEATURE_KEY).height)
        out[f"n_found_{ts}"] = found
        out[f"recall_{ts}"] = found / c.n_features[ts] if c.n_features[ts] else float("nan")
    return out


def winner_shares(regions: pl.DataFrame, arms_col: str, by: str, arms: list[str]) -> pl.DataFrame:
    """Share of regions in each `by` group where each pair gave the best score. A region
    where k pairs tie for the best gives each 1/k, so every column sums to 1."""
    long = (regions.select(KEY + [by, arms_col])
            .with_columns(w=1.0 / pl.col(arms_col).list.len())
            .explode(arms_col).rename({arms_col: "arm"}))
    n = regions.group_by(by).agg(n_regions=pl.len())
    shares = (long.group_by(by, "arm").agg(pl.col("w").sum())
              .join(n, on=by).with_columns(share=pl.col("w") / pl.col("n_regions")))
    grid = pl.DataFrame({"arm": arms}).join(n, how="cross")
    return (grid.join(shares.select(by, "arm", "share"), on=[by, "arm"], how="left")
            .with_columns(pl.col("share").fill_null(0.0)))


def load_query_seqs(accessions: list[str]) -> dict[str, str]:
    """Human query sequences from the QfO 2020_04 human proteome, as notebook 231 read them."""
    import swissprot_control_utils as sc

    return sc.read_fasta(sc.HUMAN_FASTA, keep=set(accessions))
