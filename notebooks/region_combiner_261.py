"""Combiner A, the intersection: a merged region is called when every alphabet-ksize pair
in a fixed panel calls it at ``region_evalue < E_max``. Built for notebook 261 on the
table notebook 260 writes (``region_table.parquet``, one row per query, target species,
merged region and alphabet-ksize pair; ``arm`` is the pair's label).

Each pair's set of passing merged regions is held as a sorted integer array, so a panel is
scored by intersecting arrays rather than by re-filtering the table. The greedy search
over panels calls `score` a few hundred times per E_max.

Truth follows notebook 260: a merged region is a true call in a truth set when at least
half of it lies inside one feature of that set (``merged_<set>_is_true``). A truth feature
is found when at least one called merged region on its query and target species puts at
least half of itself inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import region_table_260 as rt

KEY = ["accession", "species", "merged_region_id"]
TRUTH_SETS = ("swissprot", "pfam")
FEATURE_KEY = ["accession", "truth_set", "feature", "feature_start", "feature_end"]
EMAX_GRID = (0.1, 1.0, 10.0)
MAX_PANEL = 5
#: The number the greedy search raises: precision on Swiss-Prot features, landed >= 0.5.
OBJECTIVE = "precision_swissprot"


def _region_index(table: pl.DataFrame) -> pl.DataFrame:
    """One row per merged region with an integer id, in a fixed order."""
    return (table.select(KEY + ["split"]).unique(subset=KEY).sort(KEY)
            .with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int64)))


def _arm_sets(table: pl.DataFrame, regions: pl.DataFrame, emax: float) -> dict[str, np.ndarray]:
    """For each pair, the sorted region ids where it has a call with region_evalue < emax."""
    passing = (table.filter(pl.col("region_evalue") < emax)
               .join(regions.select(KEY + ["rid"]), on=KEY, how="inner")
               .group_by("arm").agg(pl.col("rid").unique().sort()))
    return {a: np.asarray(r, dtype=np.int64) for a, r in passing.iter_rows()}


def _intersect(sets: dict[str, np.ndarray], panel: list[str]) -> np.ndarray:
    out = sets.get(panel[0], np.empty(0, np.int64))
    for a in panel[1:]:
        out = np.intersect1d(out, sets.get(a, np.empty(0, np.int64)), assume_unique=True)
    return out


@dataclass
class Scorer:
    """Panel scoring on one split, real queries and (if given) shuffled queries."""

    table: pl.DataFrame
    truth_pairs: pl.DataFrame
    truth: pl.DataFrame
    split: str
    decoy_table: pl.DataFrame | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        t = self.table.filter(pl.col("split") == self.split)
        self.regions = _region_index(t)
        self.t = t
        # Per region: is it a true call, and is it on a query that has any feature of that set.
        reg = t.unique(subset=KEY).join(self.regions.select(KEY + ["rid"]), on=KEY).sort("rid")
        self.is_true = {ts: reg[f"merged_{ts}_is_true"].to_numpy() for ts in TRUTH_SETS}
        self.has_truth = {ts: reg[f"query_has_{ts}"].to_numpy() for ts in TRUTH_SETS}
        # Region -> truth feature it lands >= 0.5 in, for recall.
        self.landed = (self.truth_pairs.filter(pl.col("landed") >= rt.LANDED_MIN)
                       .join(self.regions.select(KEY + ["rid"]), on=KEY, how="inner")
                       .select(["rid", "species"] + FEATURE_KEY))
        # Recall denominator: every truth feature on a query of this split, once per target
        # species searched.
        species = t["species"].unique().sort().to_list()
        feats = (self.truth.unique(subset=FEATURE_KEY)
                 .with_columns(split=rt.query_split(pl.col("accession")))
                 .filter(pl.col("split") == self.split))
        self.n_features = {ts: feats.filter(pl.col("truth_set") == ts).height * len(species)
                           for ts in TRUTH_SETS}
        self.species = species
        if self.decoy_table is not None:
            self.decoy_t = self.decoy_table.filter(pl.col("split") == self.split)
            self.decoy_regions = _region_index(self.decoy_t)
        self.arms = sorted(t["arm"].unique().to_list())

    def sets(self, emax: float):
        if emax not in self._cache:
            real = _arm_sets(self.t, self.regions, emax)
            decoy = (_arm_sets(self.decoy_t, self.decoy_regions, emax)
                     if self.decoy_table is not None else None)
            self._cache[emax] = (real, decoy)
        return self._cache[emax]

    def called(self, panel: list[str], emax: float) -> np.ndarray:
        return _intersect(self.sets(emax)[0], panel)

    def score(self, panel: list[str], emax: float) -> dict:
        real, decoy = self.sets(emax)
        rid = _intersect(real, panel)
        row = {"split": self.split, "emax": emax, "n_arms": len(panel),
               "panel": " + ".join(panel), "n_called": int(len(rid))}
        hit = self.landed.filter(pl.col("rid").is_in(rid))
        for ts in TRUTH_SETS:
            on = self.has_truth[ts][rid]
            n_on = int(on.sum())
            n_true = int((self.is_true[ts][rid] & on).sum())
            row[f"n_called_on_{ts}_queries"] = n_on
            row[f"precision_{ts}"] = n_true / n_on if n_on else float("nan")
            n_found = hit.filter(pl.col("truth_set") == ts).unique(
                subset=["species"] + FEATURE_KEY).height
            row[f"n_found_{ts}"] = n_found
            row[f"recall_{ts}"] = n_found / self.n_features[ts] if self.n_features[ts] else float("nan")
        row["n_decoy_called"] = (int(len(_intersect(decoy, panel)))
                                 if decoy is not None else None)
        return row


def _rank(row: dict) -> tuple:
    """Higher objective first, then more regions called; NaN objective ranks last."""
    v = row[OBJECTIVE]
    return (-(v if v == v else -1.0), -row["n_called"], row["panel"])


def greedy(scorer: Scorer, emax: float, max_n: int = MAX_PANEL) -> pl.DataFrame:
    """Greedy forward selection from the best single pair.

    Every step adds the pair that gives the highest objective (ties to more regions
    called, then the name), up to `max_n` pairs, and records whether the step raised the
    objective. The search's own stop is the first step that does not; the path past it is
    kept so the figure can show what adding more pairs would have done.
    """
    panel: list[str] = []
    rows = []
    for n in range(1, min(max_n, len(scorer.arms)) + 1):
        cands = [scorer.score(panel + [a], emax) for a in scorer.arms if a not in panel]
        best = min(cands, key=_rank)
        best["added_arm"] = best["panel"].split(" + ")[-1]
        prev = rows[-1][OBJECTIVE] if rows else None
        best["improved"] = prev is None or (best[OBJECTIVE] == best[OBJECTIVE]
                                            and best[OBJECTIVE] > prev)
        rows.append(best)
        panel = best["panel"].split(" + ")
    path = pl.DataFrame(rows, infer_schema_length=None)
    # The panel the search stops at: the last step before the first non-improving one. At
    # least two pairs, since one pair alone is not an intersection.
    stop = next((i for i, r in enumerate(rows) if not r["improved"]), len(rows))
    stop_n = max(stop, 2) if len(rows) >= 2 else len(rows)
    return path.with_columns(is_stop=pl.col("n_arms") == stop_n)


def freeze(paths: pl.DataFrame) -> dict:
    """The (panel, E_max) to carry to test: among each E_max's stop panel, the highest
    objective, ties to more regions called, then the smaller E_max."""
    stops = paths.filter(pl.col("is_stop")).to_dicts()
    best = min(stops, key=lambda r: _rank(r) + (r["emax"],))
    return best


def load_phmmer(path: Path, species: str) -> pl.DataFrame:
    """phmmer's per-domain hits from the region benchmark: query span 1-based inclusive
    (domtblout hmm from/to, the query being the profile), i-Evalue per domain."""
    cols = ["query", "target", "qstart", "qend", "tstart", "tend", "score", "evalue"]
    return (pl.read_csv(path, separator="\t", has_header=False, new_columns=cols)
            .with_columns(accession=pl.col("query").str.split("|").list.get(1),
                          species=pl.lit(species)))


def domains_landed(spans: pl.DataFrame, domains: pl.DataFrame,
                   start: str, end: str) -> pl.DataFrame:
    """Per (accession, species): how many of the query's domains at least one span puts at
    least half of itself inside. `domains` has accession, feature, feature_start,
    feature_end; `spans` has accession, species and the two named columns."""
    pairs = rt.region_truth_pairs(spans.select("accession", "species", start, end),
                                  domains.select("accession", "feature", "feature_start",
                                                 "feature_end"),
                                  start, end)
    return (pairs.filter(pl.col("landed") >= rt.LANDED_MIN)
            .unique(subset=["accession", "species", "feature", "feature_start", "feature_end"])
            .group_by("accession", "species").agg(n_landed=pl.len()))
