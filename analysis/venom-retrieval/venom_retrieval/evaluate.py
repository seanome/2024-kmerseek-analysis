"""What counts as a true positive, and the numbers each method is scored on.

Four endpoints, kept separate because they are different claims:

  framework          the reference peptide has the same cysteine framework as the query.
                     This is a pattern of cysteine positions, nothing more.
  fold_connectivity  both peptides have a complete disulfide connectivity measured
                     experimentally, and the two patterns are the same. This is the only
                     endpoint here backed by a measurement of the fold.
  fold_knottin       both peptides carry UniProt's Knottin keyword. A curator's fold
                     assignment, not a measurement, and reported separately for that
                     reason.
  target             the two peptides share at least one activity label, within one
                     database's vocabulary. Multilabel: a peptide with three activities
                     counts as a hit for any of them, and no peptide is forced to one.

An absent label is unknown. A reference peptide with no target label is neither a true
positive nor a hard negative for the target endpoint; it is left out of both. A hard
negative has to be a peptide whose labels are known and do not overlap the query's.

Six numbers per query per method, so that a win can be attributed:

  candidate_recall              the share of this query's true positives the method could
                                compare at all. A method that filters on framework
                                compatibility throws some away before ranking; this is
                                how many.
  recall_at_budget_conditional  recall within the budget, counting only the true positives
                                that survived candidate generation. Reranking quality,
                                with candidate loss taken out.
  recall_at_budget              recall within the budget over every true positive in the
                                reference, including the ones candidate generation lost.
                                This is the end-to-end number, and it is the one that
                                decides anything.
  precision_at_budget           true positives in the budget, over the budget.
  hard_negative_auc             the chance that a true positive outranks a same-framework
                                peptide whose labels are known and do not overlap. 0.5 is
                                no better than a coin toss on the pairs the framework
                                cannot separate.
  seconds, n_candidates         what it cost and how much it had to look at.

A true positive excluded by framework filtering is an end-to-end miss. `recall_at_budget`
is computed against every true positive in the reference, so a method that never sees a
peptide is not excused for not retrieving it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from . import methods

#: The review budget, declared before any result was looked at: a person checking a
#: retrieval list reads about ten hits. Precision and recall are reported at this depth.
DEFAULT_BUDGET = 10

ENDPOINTS = ("framework", "fold_connectivity", "fold_knottin", "target")


@dataclass(frozen=True)
class Labels:
    """One peptide's labels for every endpoint, read off the manifest."""

    framework: str
    n_cysteines: int
    connectivity: str | None
    connectivity_experimental: bool
    knottin: bool
    targets: frozenset[str]

    @classmethod
    def from_row(cls, row: dict) -> Labels:
        return cls(
            framework=row["framework"] or "",
            n_cysteines=row["n_cysteines"] or 0,
            connectivity=row["connectivity"] if row["connectivity_complete"] else None,
            connectivity_experimental=row["connectivity_evidence"] == "experimental",
            knottin=bool(row["knottin"]),
            targets=frozenset(
                part for part in (row["target_labels"] or "").split(";") if part
            ),
        )


def has_label(labels: Labels, endpoint: str, min_cysteines: int = 4) -> bool:
    """Whether this peptide can take part in `endpoint` at all."""
    if endpoint == "framework":
        return bool(labels.framework) and labels.n_cysteines >= min_cysteines
    if endpoint == "fold_connectivity":
        return bool(labels.connectivity) and labels.connectivity_experimental
    if endpoint == "fold_knottin":
        return labels.knottin
    if endpoint == "target":
        return bool(labels.targets)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def is_true_positive(query: Labels, target: Labels, endpoint: str) -> bool:
    """Whether `target` is a true positive for `query` under `endpoint`.

    False whenever either peptide lacks the label, which keeps an unknown out of the
    positives. `truth_is_known` is what separates an unknown from a negative.
    """
    if not (has_label(query, endpoint) and has_label(target, endpoint)):
        return False
    if endpoint == "framework":
        return query.framework == target.framework
    if endpoint == "fold_connectivity":
        return query.connectivity == target.connectivity
    if endpoint == "fold_knottin":
        return True  # both carry the keyword, which is the whole label
    if endpoint == "target":
        return bool(query.targets & target.targets)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def truth_is_known(query: Labels, target: Labels, endpoint: str) -> bool:
    """Whether the pair can be called either way, rather than being unlabelled."""
    return has_label(query, endpoint) and has_label(target, endpoint)


@dataclass
class QueryResult:
    """One query, one method, one endpoint."""

    query_id: str
    method: str
    endpoint: str
    reference_mode: str
    n_reference: int
    n_true_positives: int
    n_candidates: int
    n_true_positives_in_candidates: int
    candidate_recall: float
    tp_at_budget: int
    precision_at_budget: float
    recall_at_budget: float
    recall_at_budget_conditional: float
    n_hard_negatives: int
    n_same_framework_true_positives: int
    hard_negative_auc: float | None
    chance_tp_at_budget: float
    seconds: float

    def as_row(self) -> dict:
        return dict(vars(self))


def _auc(positive_ranks: np.ndarray, negative_ranks: np.ndarray) -> float:
    """The share of (positive, negative) pairs where the positive ranks higher.

    Ranks are positions in the returned list, so smaller is better. A tie counts as half,
    which matters here: `framework_only` gives every same-framework peptide the same
    score, and counting ties as wins would make a method that cannot separate them look
    perfect.
    """
    wins = 0.0
    for rank in positive_ranks:
        wins += float((negative_ranks > rank).sum()) + 0.5 * float(
            (negative_ranks == rank).sum()
        )
    return wins / (len(positive_ranks) * len(negative_ranks))


def truth_arrays(
    query_labels: Labels,
    target_labels: list[Labels],
    query_framework: str,
    target_frameworks: list[str],
    reference_indices: list[int],
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, same-framework hard negatives, and same-framework true positives.

    None of the three depends on the method, so they are computed once per query, endpoint
    and reference set and handed to every method's `evaluate_ranking`.

    A hard negative has the same cysteine framework as the query, has a known label for
    this endpoint, and is not a true positive. These are the peptides the framework cannot
    tell apart from a real hit.

    The third array is the true positives that also share the query's framework, and it
    exists because the hard-negative comparison has to be made inside the same-framework
    block to mean anything. Scored against all true positives, a framework-only method
    came out at an AUC of 0.08: every hard negative shares the framework and so scores
    1.0, while most true positives do not and score 0.0, so the number measured the
    definition of a hard negative rather than any ranking. Comparing same-framework
    positives against same-framework negatives asks the question that was intended: among
    the peptides a framework cannot separate, does the method put the real hits first?

    A peptide with no label for the endpoint is neither positive nor negative: unknown is
    not a measured absence.
    """
    truth = np.fromiter(
        (
            is_true_positive(query_labels, target_labels[i], endpoint)
            for i in reference_indices
        ),
        dtype=bool,
        count=len(reference_indices),
    )
    hard_negative = np.fromiter(
        (
            bool(query_framework)
            and target_frameworks[i] == query_framework
            and truth_is_known(query_labels, target_labels[i], endpoint)
            and not is_true_positive(query_labels, target_labels[i], endpoint)
            for i in reference_indices
        ),
        dtype=bool,
        count=len(reference_indices),
    )
    same_framework = np.fromiter(
        (
            bool(query_framework) and target_frameworks[i] == query_framework
            for i in reference_indices
        ),
        dtype=bool,
        count=len(reference_indices),
    )
    return truth, hard_negative, truth & same_framework


def evaluate_ranking(
    query_id: str,
    truth: np.ndarray,
    hard_negative: np.ndarray,
    same_framework_truth: np.ndarray,
    n_reference: int,
    order: np.ndarray,
    seconds: float,
    method_name: str,
    endpoint: str,
    reference_mode: str,
    budget: int = DEFAULT_BUDGET,
) -> QueryResult | None:
    """Count everything about one ranking, for one endpoint.

    `order` is what `methods.rank` returned: positions in `reference_indices`, best first,
    holding only the targets the method could compare. A target missing from `order` was
    not judged badly, it was not judged at all, and that is what `candidate_recall`
    reports.

    The ranking is computed once per query and method and scored against every endpoint
    the query has a label for, because a method's ranking does not depend on which
    endpoint it is being judged on. `seconds` is the time that one ranking took.

    Returns None when the reference holds no true positive for this query: that is not a
    result either way, and averaging it in as a zero would punish a method for the
    database rather than for its ranking.
    """
    n_true = int(truth.sum())
    if n_true == 0:
        return None

    in_candidates = np.zeros(n_reference, dtype=bool)
    in_candidates[order] = True
    n_true_in_candidates = int((truth & in_candidates).sum())

    top = order[:budget]
    tp_at_budget = int(truth[top].sum()) if len(top) else 0

    rank_of = np.full(n_reference, n_reference + 1, dtype=int)
    rank_of[order] = np.arange(len(order))
    # Inside the same-framework block only: same-framework true positives against
    # same-framework hard negatives. See `truth_arrays` for why.
    positive_ranks = rank_of[same_framework_truth & in_candidates]
    negative_ranks = rank_of[hard_negative & in_candidates]
    auc = (
        _auc(positive_ranks, negative_ranks)
        if len(positive_ranks) and len(negative_ranks)
        else None
    )

    return QueryResult(
        query_id=query_id,
        method=method_name,
        endpoint=endpoint,
        reference_mode=reference_mode,
        n_reference=n_reference,
        n_true_positives=n_true,
        n_candidates=len(order),
        n_true_positives_in_candidates=n_true_in_candidates,
        candidate_recall=n_true_in_candidates / n_true,
        tp_at_budget=tp_at_budget,
        precision_at_budget=tp_at_budget / budget,
        recall_at_budget=tp_at_budget / n_true,
        recall_at_budget_conditional=(
            tp_at_budget / n_true_in_candidates if n_true_in_candidates else float("nan")
        ),
        n_hard_negatives=int(hard_negative.sum()),
        n_same_framework_true_positives=int(same_framework_truth.sum()),
        hard_negative_auc=auc,
        chance_tp_at_budget=budget * n_true / n_reference,
        seconds=seconds,
    )


def summarise(results: pl.DataFrame) -> pl.DataFrame:
    """One row per method, endpoint and reference mode, averaged over queries.

    Means over queries, not over hits: each query counts once however many true positives
    it has, so a single large family cannot carry a method's score.
    """
    return (
        results.group_by("method", "endpoint", "reference_mode")
        .agg(
            n_queries=pl.len(),
            mean_candidate_recall=pl.col("candidate_recall").mean(),
            mean_precision_at_budget=pl.col("precision_at_budget").mean(),
            mean_recall_at_budget=pl.col("recall_at_budget").mean(),
            # The conditional recall is undefined for a query whose true positives were
            # all lost at candidate generation, so those queries are left out of this
            # mean rather than turning the whole column into NaN.
            mean_recall_at_budget_conditional=pl.col("recall_at_budget_conditional")
            .drop_nans()
            .mean(),
            n_queries_with_a_candidate_true_positive=(
                pl.col("n_true_positives_in_candidates") > 0
            ).sum(),
            mean_tp_at_budget=pl.col("tp_at_budget").mean(),
            mean_chance_tp_at_budget=pl.col("chance_tp_at_budget").mean(),
            mean_hard_negative_auc=pl.col("hard_negative_auc").mean(),
            n_queries_with_hard_negatives=(pl.col("n_hard_negatives") > 0).sum(),
            median_candidates=pl.col("n_candidates").median(),
            total_seconds=pl.col("seconds").sum(),
            seconds_per_query=pl.col("seconds").mean(),
        )
        .sort("endpoint", "reference_mode", "mean_recall_at_budget", descending=[False, False, True])
    )
