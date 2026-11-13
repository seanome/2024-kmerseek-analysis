"""Loading and labelling for notebook 271: which kmerseek region score to filter on.

Inputs are two runs of `make run-mini-scaled-270` (branch olgabot/scaled-sweep-270), both with
extension on and scaled 1, pulled from Sherlock to DATA:

  extend/kmerseek  the 200 mini-set human queries against yeast and E. coli
  decoy/kmerseek   the same 200 queries, each shuffled keeping its amino-acid pairs
                   (dipeptide shuffle), against the same targets
  annotations/     Pfam domains for human, yeast and E. coli (1-based, inclusive)

kmerseek writes 0-based starts and exclusive ends; they are converted to 1-based inclusive on
load so they compare directly with the Pfam coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/Users/olga/data/qfo-pfam-region-benchmark-mini-scaled-270")
SPECIES = ["yeast", "ecoli"]
N_QUERIES = 200
#: Queries x target species: every per-query rate below divides by this.
N_QUERY_SEARCHES = N_QUERIES * len(SPECIES)

#: The alphabet and k of each pair in the scaled-270 run (one k per alphabet).
ALPHABET_K = {
    "hp_lehninger_c_nonpolar2": 18,
    "hp_thomas_dill2": 19,
    "hp_thomas_dill_no_c2": 19,
    "hp_kyte_doolittle2": 19,
    "gbmr4": 16,
    "mmseqs12": 7,
    "sdm12": 6,
    "protein20": 5,
}
HP_ALPHABETS = [a for a in ALPHABET_K if a.startswith("hp_")]
AMINO_ACID_LIKE = ["protein20", "sdm12", "mmseqs12"]

#: A region sits on a Pfam domain when at least this fraction of it lies inside the domain.
MIN_FRACTION_IN_DOMAIN = 0.5

REGION_COLUMNS = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_length",
    "region_evalue",
    "region_poisson_score",
    "region_tail_probability",
    "region_search_space",
    "db_n_targets",
    "region_tfidf",
    "region_mean_idf",
    "region_ka_bits",
    "query_tfidf",
]

#: Each score as "higher is better", with the name used in figures. E-values are -log10'd.
SCORES = {
    "E-value": -pl.col("region_evalue").log10(),
    "Poisson score": pl.col("region_poisson_score"),
    "TF-IDF": pl.col("region_tfidf"),
    "mean IDF": pl.col("region_mean_idf"),
    "region length": pl.col("region_length").cast(pl.Float64),
}


def accession(column: str) -> pl.Expr:
    """UniProt accession from a `db|ACC|NAME description` header; drops the DECOY_ prefix."""
    return pl.col(column).str.split("|").list.get(1).str.replace("^DECOY_", "")


def pfam_domains(species: str) -> pl.DataFrame:
    """Pfam domains with positions for one species: accession, pfam_id, domain_start, domain_end."""
    return (
        pl.read_parquet(DATA / "annotations" / f"{species}_pfam_domains.parquet")
        .filter(pl.col("has_position"))
        .select(
            "accession",
            "pfam_id",
            pl.col("domain_start").cast(pl.Int64),
            pl.col("domain_end").cast(pl.Int64),
        )
    )


def families_under_region(
    regions: pl.DataFrame, domains: pl.DataFrame, side: str
) -> pl.DataFrame:
    """(row_id, pfam_id) for every Pfam family whose domain holds at least half of one side.

    `side` is "query" or "target"; it picks the protein and interval the region is checked on.
    """
    acc, start, end = f"{side}_acc", f"{side}_from", f"{side}_to"
    overlap = (
        pl.min_horizontal(end, "domain_end")
        - pl.max_horizontal(start, "domain_start")
        + 1
    )
    return (
        regions.select("row_id", acc, start, end)
        .join(domains, left_on=acc, right_on="accession")
        .filter(overlap >= MIN_FRACTION_IN_DOMAIN * (pl.col(end) - pl.col(start) + 1))
        .select("row_id", "pfam_id")
        .unique()
    )


def load_labelled_regions(
    run: str,
    species: str,
    alphabet: str,
    human_domains: pl.DataFrame,
    target_domains: pl.DataFrame,
) -> pl.DataFrame:
    """One alphabet's regions for one species, with a `label` column.

    run "decoy": every region is "shuffled" (a false hit by construction).
    run "extend": "same family" when at least half of the region lies in a domain of one Pfam
    family on the query and in a domain of that same family on the target; "no shared family"
    when the query and target proteins share no Pfam family at all; "other" for the rest.
    """
    k = ALPHABET_K[alphabet]
    path = (
        DATA
        / run
        / "kmerseek"
        / f"human_vs_{species}.{alphabet}.k{k}.lctrue.regions.parquet"
    )
    regions = (
        pl.read_parquet(path, columns=REGION_COLUMNS)
        .with_row_index("row_id")
        .with_columns(
            query_acc=accession("query_name"),
            target_acc=accession("target_name"),
            query_from=pl.col("region_start") + 1,
            query_to=pl.col("region_end"),
            target_from=pl.col("target_start") + 1,
            target_to=pl.col("target_end"),
            # kmerseek's own suggestion for turning the Poisson score into an E-value
            # (doc comment on MatchedRegion::poisson_score, kmerseek 0.4.0)
            poisson_evalue=pl.col("region_tail_probability")
            * pl.col("region_search_space")
            * pl.col("db_n_targets"),
            species=pl.lit(species),
            alphabet=pl.lit(alphabet),
            k=pl.lit(k),
        )
    )
    if run == "decoy":
        return regions.with_columns(label=pl.lit("shuffled"))
    query_families = human_domains.group_by("accession").agg(
        query_families=pl.col("pfam_id").unique()
    )
    target_families = target_domains.group_by("accession").agg(
        target_families=pl.col("pfam_id").unique()
    )
    pairs = (
        regions.select("query_acc", "target_acc")
        .unique()
        .join(query_families, left_on="query_acc", right_on="accession", how="left")
        .join(target_families, left_on="target_acc", right_on="accession", how="left")
        .with_columns(
            shares_family=pl.col("query_families")
            .list.set_intersection("target_families")
            .list.len()
            .fill_null(0)
            > 0
        )
        .select("query_acc", "target_acc", "shares_family")
    )
    same_family = (
        families_under_region(regions, human_domains, "query")
        .join(
            families_under_region(regions, target_domains, "target"),
            on=["row_id", "pfam_id"],
        )
        .select("row_id")
        .unique()
        .with_columns(same_family=pl.lit(True))
    )
    return (
        regions.join(pairs, on=["query_acc", "target_acc"], how="left")
        .join(same_family, on="row_id", how="left")
        .with_columns(
            label=pl.when(pl.col("same_family"))
            .then(pl.lit("same family"))
            .when(~pl.col("shares_family"))
            .then(pl.lit("no shared family"))
            .otherwise(pl.lit("other"))
        )
    )


def load_all() -> pl.DataFrame:
    """Every alphabet, species and run, labelled, with the score columns of SCORES added."""
    human = pfam_domains("human")
    frames = []
    for species in SPECIES:
        targets = pfam_domains(species)
        for alphabet in ALPHABET_K:
            for run in ["extend", "decoy"]:
                frames.append(
                    load_labelled_regions(run, species, alphabet, human, targets).drop(
                        "same_family", "shares_family", strict=False
                    )
                )
    # E-values of 0 or infinity become +-inf after -log10; clip so they still sort first/last.
    return pl.concat(frames, how="diagonal").with_columns(
        **{name: expr.clip(-1e6, 1e6) for name, expr in SCORES.items()}
    )


def false_hits_per_query(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Shuffled-query regions with an E-value at or below each cut-off, per query search."""
    v = np.sort(values)
    return np.searchsorted(v, cutoffs, side="right") / N_QUERY_SEARCHES


def kept_at_false_rate(
    true_scores: np.ndarray, shuffled_scores: np.ndarray, false_per_query: float
) -> tuple[float, float]:
    """(cut-off, fraction of true regions kept) when shuffled queries pass `false_per_query`.

    Scores are higher-is-better. The cut-off is the score of the (n+1)th best shuffled region,
    n = false_per_query x query searches; a region must beat it strictly, so ties at the
    cut-off (for example the Poisson score's ceiling) are not kept.
    """
    n_allowed = int(false_per_query * N_QUERY_SEARCHES)
    ranked = np.sort(shuffled_scores)[::-1]
    cutoff = ranked[n_allowed] if len(ranked) > n_allowed else -np.inf
    return float(cutoff), float((true_scores > cutoff).mean())
