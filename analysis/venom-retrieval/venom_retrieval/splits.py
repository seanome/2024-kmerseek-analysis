"""Remove duplicates, group related peptides, and fix the query and reference sets.

A retrieval benchmark on venom peptides fails in two ways if the sets are built carelessly.
The same peptide appears in ConoServer and in Swiss-Prot, so a method can retrieve the
query itself and look perfect. And venom-peptide families are large and recent, so a query
usually has close relatives in the database; a method that only finds those has not shown
it can reach a peptide whose residues have diverged, which is the whole claim being tested.

So this module does three things, in order:

  1. Drop peptides whose mature sequence is character-for-character identical to another's,
     keeping one representative and recording which identifiers collapsed onto it.
  2. Work out, for each peptide, which other peptides are too close to it to be a fair
     retrieval target: those similar above a stated threshold, and those stating the same
     gene family. This is a neighbourhood around each peptide, not a transitive group.
  3. Build two reference sets for every query: one that keeps the query's neighbours,
     which measures ordinary retrieval, and one with the whole neighbourhood removed,
     which is the only one that says anything about reaching another family.

Similarity here is a BLOSUM62 local-alignment score divided by the smaller of the two
peptides' self-scores, not percent identity. The score ratio is what the grouping
threshold is set on, and the grouping is deliberately broad: a pair wrongly grouped only
costs the benchmark a query, while a pair wrongly separated leaks a near-duplicate into
the reference and inflates every method's score.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import polars as pl

from .align import local_score, self_scores

#: Two peptides join one group at or above this score ratio. 0.5 of the smaller
#: self-score: a pair this similar would be called the same family by eye, and anything
#: looser starts merging unrelated cysteine-rich peptides through their cysteines alone.
DEFAULT_GROUP_THRESHOLD = 0.5


@dataclass(frozen=True)
class DuplicateReport:
    """What collapsing identical sequences removed."""

    n_before: int
    n_after: int
    n_removed: int
    #: One row per kept representative that stood for more than one identifier.
    collapsed: pl.DataFrame


def remove_exact_duplicates(frame: pl.DataFrame) -> tuple[pl.DataFrame, DuplicateReport]:
    """Keep one peptide per distinct mature sequence.

    The representative is the row with the most labels on it, so a sequence held by both
    ConoServer and Swiss-Prot keeps whichever copy carries the connectivity and target
    annotations. Ties go to the lexicographically first identifier, so the choice does not
    depend on row order.
    """
    scored = frame.with_columns(
        _label_count=(
            (~pl.col("fold_missing")).cast(pl.Int32)
            + (~pl.col("target_missing")).cast(pl.Int32)
            + (~pl.col("framework_missing")).cast(pl.Int32)
            + (pl.col("gene_family").is_not_null()).cast(pl.Int32)
            + (pl.col("connectivity_evidence") == "experimental").cast(pl.Int32)
        )
    ).sort(["mature_peptide", "_label_count", "peptide_id"], descending=[False, True, False])

    kept = scored.unique(subset=["mature_peptide"], keep="first", maintain_order=True)
    collapsed = (
        scored.join(
            kept.select("mature_peptide", representative=pl.col("peptide_id")),
            on="mature_peptide",
            how="left",
        )
        .filter(pl.col("peptide_id") != pl.col("representative"))
        .group_by("representative")
        .agg(
            duplicate_ids=pl.col("peptide_id").sort().str.join(";"),
            n_duplicates=pl.len(),
        )
        .sort("representative")
    )
    return kept.drop("_label_count").sort("peptide_id"), DuplicateReport(
        n_before=frame.height,
        n_after=kept.height,
        n_removed=frame.height - kept.height,
        collapsed=collapsed,
    )


#: Set once per worker process by `_init`. The sequences and their self-scores are the
#: same for every block, so they are sent to each worker once at start-up rather than
#: pickled alongside every block of pairs.
_WORKER: dict = {}


def _init(sequences: list[str], selves: list[float], threshold: float) -> None:
    _WORKER.update(sequences=sequences, selves=selves, threshold=threshold)


def _ratio_block(index_pairs):
    """Score one block of peptide pairs, returning those at or above the threshold."""
    sequences = _WORKER["sequences"]
    selves = _WORKER["selves"]
    threshold = _WORKER["threshold"]
    out = []
    for i, j in index_pairs:
        smaller = min(selves[i], selves[j])
        if smaller <= 0:
            continue
        ratio = local_score(sequences[i], sequences[j]) / smaller
        if ratio >= threshold:
            out.append((i, j, ratio))
    return out


def similar_pairs(
    sequences: list[str],
    threshold: float = DEFAULT_GROUP_THRESHOLD,
    processes: int = 1,
    block: int = 200_000,
) -> list[tuple[int, int, float]]:
    """Every pair of peptides at or above `threshold`, as (i, j, ratio) with i < j.

    Every pair is scored; there is no seeding heuristic and nothing is skipped. A seed
    filter would be much faster and would silently miss pairs, and a missed pair here is a
    near-duplicate left in the reference set, which is the error this whole module exists
    to prevent.
    """
    selves = self_scores(sequences)
    pairs = itertools.combinations(range(len(sequences)), 2)
    blocks = iter(lambda: list(itertools.islice(pairs, block)), [])
    found: list[tuple[int, int, float]] = []
    if processes <= 1:
        _init(sequences, selves, threshold)
        for chunk in blocks:
            found.extend(_ratio_block(chunk))
        return found
    with ProcessPoolExecutor(
        max_workers=processes, initializer=_init, initargs=(sequences, selves, threshold)
    ) as pool:
        for result in pool.map(_ratio_block, blocks):
            found.extend(result)
    return found


def neighbourhoods(
    frame: pl.DataFrame,
    pairs: list[tuple[int, int, float]],
    use_gene_family: bool = True,
) -> list[set[int]]:
    """For each peptide, the peptides too close to it to be a fair retrieval target.

    A neighbour is a peptide similar above the threshold, or one stating the same gene
    family in the same database's vocabulary. A peptide is its own neighbour.

    Neighbourhoods, not transitive groups. Single-linkage grouping was tried first and
    chained 88% of these 9_571 peptides into one group: with short peptides at a 0.5 score
    ratio there is almost always a chain A to B to C whose ends are unrelated, and a
    reference set with that group removed would have had an eighth of the database left in
    it. A neighbourhood asks only about the query's own distance to each peptide, so no
    chain can grow.

    The cost of dropping transitivity is that two peptides in one reference set may be
    near-duplicates of each other. That inflates the number of true positives a query has
    but cannot leak the query itself, which is the leak that matters.
    """
    sets: list[set[int]] = [{i} for i in range(frame.height)]
    for i, j, _ in pairs:
        sets[i].add(j)
        sets[j].add(i)
    if use_gene_family:
        by_family: dict[tuple[str, str], list[int]] = {}
        for i, (family, source) in enumerate(
            zip(frame["gene_family"].to_list(), frame["gene_family_source"].to_list())
        ):
            if not family:
                # An unknown family links nothing: "unknown" is not a family, and treating
                # it as one would merge every unannotated peptide into a single block.
                continue
            # A family string is only comparable inside the database that wrote it:
            # ConoServer's "O1 superfamily" and a UniProt family name are different
            # vocabularies, so the source is part of the key.
            by_family.setdefault((source, family), []).append(i)
        for members in by_family.values():
            members_set = set(members)
            for i in members:
                sets[i] |= members_set
    return sets


def neighbourhood_sizes(sets: list[set[int]]) -> pl.DataFrame:
    """One row per peptide with the size of its neighbourhood, for the report."""
    return pl.DataFrame({"n_neighbours": [len(s) for s in sets]})


def choose_queries(
    frame: pl.DataFrame,
    sets: list[set[int]],
    endpoint: str,
    n_queries: int,
    seed: int,
) -> pl.DataFrame:
    """The query peptides for one endpoint: labelled, mutually distant, a fixed sample.

    A query must carry the endpoint's label, and no two queries may be neighbours, so the
    per-query averages are not dominated by whichever family happens to be largest in the
    database. The order candidates are considered in is shuffled with `seed`, so the same
    seed picks the same queries, and a different seed is a different sample of the same
    construction rather than a different rule.
    """
    from . import evaluate

    if endpoint not in evaluate.ENDPOINTS:
        raise ValueError(
            f"endpoint must be one of {list(evaluate.ENDPOINTS)}, not {endpoint!r}"
        )
    rows = frame.to_dicts()
    eligible = [
        i
        for i, row in enumerate(rows)
        if evaluate.has_label(evaluate.Labels.from_row(row), endpoint)
    ]
    # Shuffle, then take greedily: the first peptide that neighbours nothing already
    # taken. Strongest evidence first within the shuffle would bias the sample towards
    # the best-studied families, so the shuffle is the only ordering.
    order = pl.Series("i", eligible).sample(
        fraction=1.0, shuffle=True, seed=seed
    ).to_list()
    taken: list[int] = []
    blocked: set[int] = set()
    for i in order:
        if i in blocked:
            continue
        taken.append(i)
        blocked |= sets[i]
        if len(taken) >= n_queries:
            break
    return frame[sorted(taken)].sort("peptide_id")
