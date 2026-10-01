"""Loading and scoring for notebook 270: kmerseek --scaled on the mini set.

Inputs are three runs of `make run-mini-scaled-270` (branch olgabot/scaled-sweep-270),
pulled to DATA:

  exact/   extension off: each region is an exact run of agreeing residues that holds at
           least one kept k-mer (a seed)
  extend/  extension on: each seed's run grown through mismatches (the calls)
  decoy/   extension on, the 200 queries replaced by dipeptide-shuffled copies
  trace/   <run>.<date>.trace.txt, each run's Nextflow trace

Coordinates. kmerseek writes 0-based starts and exclusive ends; Swiss-Prot features are
1-based and inclusive. Everything here is converted to 1-based inclusive on load.

Matching a region to a truth instance. An instance is one human Swiss-Prot range feature
(not a 1-2 residue point feature) on a query protein, scored separately in each target
species. A region matches an instance when (a) it is on the same query protein and its
query interval overlaps the instance by at least one residue, and (b) its target interval
overlaps at least one residue of a Swiss-Prot feature of the same type on that target
protein. Rule (b) is looser than the pipeline's transfer rule (the region covers at least
half of the target feature): an exact seed of 6-19 residues rarely covers half of a target
feature, and the same rule has to serve the exact and the extended runs, so that the gap
between them is extension and not a change of rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/Users/olga/data/qfo-pfam-region-benchmark-mini-scaled-270")
SPECIES = ["yeast", "ecoli"]
SCALED = [1, 2, 5, 10]
N_QUERIES = 200

#: Notebook 244's arm per feature type (mask on), one k per alphabet. protein20 k5 is
#: also the control at protein20's smallest k.
ARMS = [
    ("hp_lehninger_c_nonpolar2", 18, "TRANSMEM"),
    ("hp_thomas_dill2", 19, "REGION"),
    ("hp_thomas_dill_no_c2", 19, "ZN_FING, DOMAIN"),
    ("hp_kyte_doolittle2", 19, "REPEAT, DNA_BIND"),
    ("gbmr4", 16, "COILED"),
    ("mmseqs12", 7, "INTRAMEM"),
    ("sdm12", 6, "BINDING"),
    ("protein20", 5, "MOTIF, SITE; control"),
]
ARM_K = {a: k for a, k, _ in ARMS}
ARM_ORDER = [a for a, _, _ in ARMS]

LENGTH_BINS = ["< 30", "30-59", "60-119", ">= 120"]
SHORT_BINS = ["< 30", "30-59"]

LANDED_MIN = 0.5  # overlap / call length

_REGION_RE = re.compile(
    r"^human_vs_(?P<sp>[a-z]+)\.(?P<alpha>.+)\.k(?P<k>\d+)(?:\.s(?P<s>\d+))?"
    r"\.lc(?P<lc>true|false)\.regions\.parquet$"
)
_TAG_RE = re.compile(
    r"^(?P<sp>[a-z]+)_(?P<alpha>.+)_k(?P<k>\d+)(?:\.s(?P<s>\d+))?_lc(?P<lc>true|false)$"
)


def length_bin(length: pl.Expr) -> pl.Expr:
    return (
        pl.when(length < 30)
        .then(pl.lit("< 30"))
        .when(length < 60)
        .then(pl.lit("30-59"))
        .when(length < 120)
        .then(pl.lit("60-119"))
        .otherwise(pl.lit(">= 120"))
    )


def predicted_reach(n_kmers, scaled):
    """Pr(at least one of n k-mers is kept) when each is kept with probability 1/scaled."""
    return 1.0 - (1.0 - 1.0 / np.asarray(scaled, float)) ** np.asarray(n_kmers, float)


def accession(col: str) -> pl.Expr:
    return pl.col(col).str.split("|").list.get(1)


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
def region_files(run: str) -> list[Path]:
    return sorted((DATA / run / "kmerseek").glob("human_vs_*.regions.parquet"))


def load_regions(run: str) -> pl.DataFrame:
    """Every region of one run, with species, alphabet, k, scaled, and 1-based coordinates."""
    parts = []
    for f in region_files(run):
        m = _REGION_RE.match(f.name)
        if not m or m["sp"] not in SPECIES or m["alpha"] not in ARM_K:
            continue
        d = pl.read_parquet(
            f,
            columns=[
                "query_name", "target_name", "region_start", "region_end",
                "target_start", "target_end", "region_n_shared_kmers",
                "region_n_mismatches", "region_evalue", "region_mean_idf", "scaled",
            ],
        )
        parts.append(
            d.with_columns(
                species=pl.lit(m["sp"]),
                alphabet=pl.lit(m["alpha"]),
                k=pl.lit(int(m["k"])),
                file_scaled=pl.lit(int(m["s"] or 1)),
            )
        )
    d = pl.concat(parts)
    # The scaled column kmerseek writes and the one the file name says must agree.
    bad = d.filter(pl.col("scaled") != pl.col("file_scaled")).height
    assert bad == 0, f"{bad} rows whose scaled column disagrees with the file name"
    return d.drop("file_scaled").with_columns(
        query_acc=accession("query_name"),
        target_acc=accession("target_name"),
        qs=pl.col("region_start") + 1,
        qe=pl.col("region_end"),
        ts=pl.col("target_start") + 1,
        te=pl.col("target_end"),
    ).with_columns(call_length=pl.col("qe") - pl.col("qs") + 1)


def load_instances(run: str = "extend") -> pl.DataFrame:
    """Human Swiss-Prot range instances on the 200 queries, one row per target species."""
    t = pl.read_parquet(DATA / run / "truth_swissprot" / "human_swissprot_truth.parquet")
    t = t.filter(~pl.col("is_point")).select(
        query_acc="accession",
        ftype="pfam_id",
        ds="domain_start",
        de="domain_end",
    ).unique()
    t = t.with_columns(feature_length=pl.col("de") - pl.col("ds") + 1)
    t = t.with_columns(length_bin=length_bin(pl.col("feature_length")))
    return t.join(pl.DataFrame({"species": SPECIES}), how="cross")


def load_target_features(run: str = "extend") -> pl.DataFrame:
    parts = []
    for sp in SPECIES:
        m = pl.read_parquet(DATA / run / "truth_swissprot" / f"{sp}_domain_map.parquet")
        parts.append(
            m.select(
                target_acc="accession", ftype="pfam_id", fs="domain_start", fe="domain_end"
            ).with_columns(species=pl.lit(sp))
        )
    return pl.concat(parts).unique()


def extension_status(regions: pl.DataFrame) -> pl.DataFrame:
    """Whether each (alphabet, scaled, species) table was extended.

    When an index has no Karlin-Altschul fit, the pipeline's search step logs "searching
    again without extension" and writes exact regions under the extended run's name. Such a
    table has no finite region_evalue and no region with a mismatch, which is how it is
    found here.
    """
    return regions.group_by(["alphabet", "scaled", "species"]).agg(
        n_regions=pl.len(),
        n_finite_evalue=pl.col("region_evalue").is_finite().sum(),
        n_with_mismatch=(pl.col("region_n_mismatches") > 0).sum(),
    ).with_columns(
        extended=(pl.col("n_finite_evalue") > 0) | (pl.col("n_with_mismatch") > 0)
    )


def extended_at_every_scaled(*statuses: pl.DataFrame) -> pl.DataFrame:
    """(alphabet, species) pairs extended at all four scaled values in every given run."""
    st = pl.concat(statuses)
    return (
        st.group_by(["alphabet", "species"])
        .agg(n_ok=pl.col("extended").sum(), n=pl.len())
        .filter(pl.col("n_ok") == pl.col("n"))
        .filter(pl.col("n") == len(SCALED) * len(statuses))
        .select("alphabet", "species")
        .sort("alphabet", "species")
    )


# ---------------------------------------------------------------------------
# Matching.
# ---------------------------------------------------------------------------
def typed_regions(regions: pl.DataFrame, target_features: pl.DataFrame) -> pl.DataFrame:
    """Each region once per feature type it touches on the target side (rule b)."""
    keys = ["species", "alphabet", "k", "scaled", "query_acc", "target_acc", "qs", "qe", "ts", "te"]
    touched = (
        regions.select(keys)
        .unique()
        .join(target_features, on=["species", "target_acc"], how="inner")
        .filter((pl.col("ts") <= pl.col("fe")) & (pl.col("te") >= pl.col("fs")))
        .select(keys + ["ftype"])
        .unique()
    )
    return touched.join(regions, on=keys, how="inner")


def match_instances(typed: pl.DataFrame, instances: pl.DataFrame) -> pl.DataFrame:
    """One row per (region, instance) pair that satisfies rules (a) and (b)."""
    m = typed.join(instances, on=["species", "query_acc", "ftype"], how="inner").filter(
        (pl.col("qs") <= pl.col("de")) & (pl.col("qe") >= pl.col("ds"))
    )
    overlap = pl.min_horizontal("qe", "de") - pl.max_horizontal("qs", "ds") + 1
    union = pl.max_horizontal("qe", "de") - pl.min_horizontal("qs", "ds") + 1
    return m.with_columns(
        overlap=overlap,
        inside=overlap / pl.col("call_length"),
        iou=overlap / union,
        start_err=(pl.col("qs") - pl.col("ds")).abs(),
        end_err=(pl.col("qe") - pl.col("de")).abs(),
    )


INSTANCE_KEY = ["species", "query_acc", "ftype", "ds", "de"]
ARM_KEY = ["alphabet", "k", "scaled"]


def reached(matches: pl.DataFrame, instances: pl.DataFrame, landed: bool) -> pl.DataFrame:
    """Every (arm, scaled, instance), with reached True/False.

    landed=False: any matching region. landed=True: a matching region with at least half
    of its length inside the instance. The best call per instance is the landed one with
    the highest IoU, ties to the smaller start error, then end error, then target, then
    start, so the choice does not depend on row order.
    """
    m = matches.filter(pl.col("inside") >= LANDED_MIN) if landed else matches
    best = (
        m.sort(
            ["iou", "start_err", "end_err", "target_acc", "qs"],
            descending=[True, False, False, False, False],
        )
        .group_by(ARM_KEY + INSTANCE_KEY, maintain_order=True)
        .first()
        .select(ARM_KEY + INSTANCE_KEY + ["iou", "start_err", "end_err", "target_acc"])
    )
    grid = pl.DataFrame(
        [(a, ARM_K[a], s) for a in ARM_ORDER for s in SCALED],
        schema=["alphabet", "k", "scaled"],
        orient="row",
    ).with_columns(pl.col("k").cast(pl.Int32), pl.col("scaled").cast(pl.Int64))
    full = grid.join(instances, how="cross")
    best = best.with_columns(pl.col("k").cast(pl.Int32), pl.col("scaled").cast(pl.Int64))
    return full.join(best, on=ARM_KEY + INSTANCE_KEY, how="left").with_columns(
        reached=pl.col("iou").is_not_null()
    )


def seed_kmers_at_scaled1(exact_matches: pl.DataFrame) -> pl.DataFrame:
    """Per (arm, instance): the k-mers in every exact run that matches it at scaled 1.

    A run of L residues holds L - k + 1 k-mers. At scaled s the run is found again if any
    one of them is kept, wherever it sits in the run, so the prediction for the instance
    is 1 - (1 - 1/s)^n with n summed over its runs.
    """
    s1 = exact_matches.filter(pl.col("scaled") == 1)
    runs = s1.select(
        ["alphabet", "k"] + INSTANCE_KEY + ["target_acc", "qs", "qe", "ts", "te"]
    ).unique()
    return runs.group_by(["alphabet", "k"] + INSTANCE_KEY).agg(
        n_kmers=(pl.col("qe") - pl.col("qs") + 1 - pl.col("k") + 1).sum(),
        n_runs=pl.len(),
    )


# ---------------------------------------------------------------------------
# Cost.
# ---------------------------------------------------------------------------
_SIZE = {"B": 1 / 1024 ** 3, "KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}
_TIME = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _gb(text):
    if text in (None, "-", ""):
        return None
    num, unit = text.split()
    return float(num) * _SIZE[unit]


def _seconds(text):
    if text in (None, "-", ""):
        return None
    total = 0.0
    for part in text.split():
        m = re.fullmatch(r"([0-9.]+)(ms|s|m|h|d)", part)
        total += float(m[1]) * _TIME[m[2]]
    return total


def load_cost(run: str) -> pl.DataFrame:
    """kmerseekIndex and kmerseekSearch rows of one run's Nextflow trace, keyed by arm.

    On the mini runs every task ran inside the head's one Slurm allocation, so sacct has
    no per-task numbers; peak_rss and realtime are Nextflow's, read from /proc on Linux.
    Index tasks served from the store run nothing and leave no row.
    """
    traces = sorted((DATA / "trace").glob(f"{run}.*.trace.txt"))
    assert len(traces) == 1, f"expected one trace for {run}, found {traces}"
    d = pl.read_csv(traces[0], separator="\t", infer_schema=False)
    d = d.filter(
        pl.col("process").is_in(["kmerseekIndex", "kmerseekSearch"])
        & pl.col("status").is_in(["COMPLETED", "CACHED"])
    )
    rows = []
    for r in d.iter_rows(named=True):
        m = _TAG_RE.match(r["tag"])
        if not m:
            raise ValueError(f"cannot parse tag {r['tag']}")
        rows.append(
            dict(
                run=run,
                process=r["process"],
                status=r["status"],
                species=m["sp"],
                alphabet=m["alpha"],
                k=int(m["k"]),
                scaled=int(m["s"] or 1),
                peak_rss_gb=_gb(r["peak_rss"]),
                realtime_s=_seconds(r["realtime"]),
            )
        )
    return pl.DataFrame(rows)


def pfam_homolog_pairs() -> pl.DataFrame:
    """(species, query_acc, target_acc) where the human query and the target protein share
    at least one Pfam family, from the mini set's own Pfam tables (DATA / annotations)."""
    h = pl.read_parquet(DATA / "annotations" / "human_pfam_domains.parquet").select(
        query_acc="accession", pfam_id="pfam_id"
    ).unique()
    parts = []
    for sp in SPECIES:
        t = pl.read_parquet(DATA / "annotations" / f"{sp}_pfam_domains.parquet").select(
            target_acc="accession", pfam_id="pfam_id"
        ).unique()
        parts.append(
            h.join(t, on="pfam_id").select("query_acc", "target_acc").unique()
            .with_columns(species=pl.lit(sp))
        )
    return pl.concat(parts).select("species", "query_acc", "target_acc")
