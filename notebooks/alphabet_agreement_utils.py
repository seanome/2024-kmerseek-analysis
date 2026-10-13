"""Count how many kmerseek alphabets report a region at the same spot of a query-target pair.

Shared by scripts/284_alphabet_agreement.py (notebook 241's real queries) and
notebooks/284_shuffle_agreement.py (the same queries, dipeptide-shuffled), so the real and
shuffled numbers come from one function.

A spot joins regions from different alphabets that sit on the same diagonal (target start
minus query start is equal, so the two stretches are in register) and whose query stretches
overlap. Coordinates are kmerseek's: 0-based, end-exclusive.

Three counts per spot:
  n_alphabets                 alphabets with a region there (of 19);
  n_families                  the same, with the seven two- and three-letter H/P alphabets
                              counted as one;
  n_families_identity_as_one  n_families, except that a spot protein20 also calls counts
                              as 1: protein20 only matches identical residues, and an
                              identical stretch matches in every alphabet, so the other
                              alphabets add no independent support there.
"""

from __future__ import annotations

import polars as pl

HP = [
    "hp_kyte_doolittle2",
    "hp_lehninger2",
    "hp_lehninger_c_nonpolar2",
    "hp_lehninger_hpc3",
    "hp_pbotc_1st_ed2",
    "hp_thomas_dill2",
    "hp_thomas_dill_no_c2",
]
COUNTS = ["n_alphabets", "n_families", "n_families_identity_as_one"]
# E-value cuts applied to each alphabet's own calls before counting. None keeps every region,
# including those with no E-value fit (E = inf).
E_CUTS = [None, 1e5, 1e4, 1e3, 1e2, 10, 1]


def prepare(regions: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Add query start/end, diagonal and alphabet family to a kmerseek region table."""
    return regions.lazy().with_columns(
        qs=pl.col("region_start").cast(pl.Int64),
        qe=pl.col("region_end").cast(pl.Int64),
        diag=(pl.col("target_start") - pl.col("region_start")).cast(pl.Int64),
        family=pl.when(pl.col("alphabet").is_in(HP))
        .then(pl.lit("H/P"))
        .otherwise(pl.col("alphabet")),
    )


SPOT_KEYS = ["query_name", "gene", "target_name", "diag", "spot"]


def assign_spots(d: pl.LazyFrame, e_cut: float | None = None) -> pl.LazyFrame:
    """Each region with the number of its spot on its (query, target, diagonal). `d` comes
    from prepare(). Regions with E >= e_cut are dropped first, so a cut can split a spot.
    """
    if e_cut is not None:
        d = d.filter(pl.col("region_evalue") < e_cut)
    keys = ["query_name", "target_name", "diag"]
    d = d.sort(*keys, "qs")
    d = d.with_columns(prev_end=pl.col("qe").cum_max().shift(1).over(keys))
    return d.with_columns(
        spot=(pl.col("prev_end").is_null() | (pl.col("qs") >= pl.col("prev_end")))
        .cast(pl.Int32)
        .cum_sum()
        .over(keys)
    )


def spots(d: pl.LazyFrame, e_cut: float | None = None) -> pl.DataFrame:
    """One row per spot with its three counts. `d` comes from prepare()."""
    d = assign_spots(d, e_cut)
    s = d.group_by(*SPOT_KEYS).agg(
        n_alphabets=pl.col("alphabet").n_unique(),
        n_families=pl.col("family").n_unique(),
        has_protein20=(pl.col("alphabet") == "protein20").any(),
        qs=pl.col("qs").min(),
        qe=pl.col("qe").max(),
    )
    s = s.with_columns(
        n_families_identity_as_one=pl.when(pl.col("has_protein20"))
        .then(1)
        .otherwise(pl.col("n_families"))
        .cast(pl.UInt32)
    )
    return s.collect()


def best_per_target(s: pl.DataFrame, count: str) -> pl.DataFrame:
    """Each target protein's highest `count` over its spots, one row per (query, target)."""
    return (
        s.sort(count, "n_alphabets", descending=True)
        .group_by("query_name", "gene", "target_name", maintain_order=True)
        .first()
    )


def partner_summary(s: pl.DataFrame, query: str, partner: str) -> dict:
    """For each count: the partner's best value, how many proteins beat it, and the proteome's top."""
    out = {}
    for c in COUNTS:
        b = best_per_target(s.filter(pl.col("query_name") == query), c)
        p = b.filter(pl.col("gene") == partner)
        v = int(p[c].max()) if p.height else 0
        out[c] = {
            "partner": v,
            "n_more": int(b.filter(pl.col(c) > v).height),
            "max": int(b[c].max()) if b.height else 0,
            "n_targets": b.height,
        }
    return out
