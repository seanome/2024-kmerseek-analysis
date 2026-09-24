"""Loaders, splits and controls for notebook 251 (DisProt functional-region transfer).

The tables come from three scripts, named at each loader:

* ``scripts/fetch_disprot.py``: the DisProt release, filtered to functional regions with
  experimental evidence that sit on experimentally shown disorder.
* ``scripts/prep_251_region_covariates.py``: pLDDT, metapredict disorder and the
  Kyte-Doolittle scan per region.
* ``scripts/reduce_disprot_region_landing.py``: one row per (arm, target, human region,
  label rule), plus the ranks and the length check.

Coordinates in the landing tables are 0-based with the end excluded (kmerseek's
convention). DisProt's own are 1-based inclusive; the covariate tables keep those, and the
join below converts.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import polars as pl

import hero_example_utils as he

DISPROT_DIR = Path.home() / "data" / "disprot-region-transfer"
EXTRACT = DISPROT_DIR / "extract"
#: Where the reduction wrote its tables. The dry run by default; the full run sets
#: DISPROT_LANDING_DIR.
LANDING_DIR = Path(
    os.environ.get(
        "DISPROT_LANDING_DIR", DISPROT_DIR / "dryrun" / "results" / "landing_disprot"
    )
)
#: The HGNC table the midi-plus run used, copied next to its data.
HGNC_FILE = he.MIDI / "hgnc_complete_set.txt"
QFO_DIR = he.QFO_DIR
HUMAN_TAXON = 9606

TARGET_ORDER = [
    "mouse",
    "chicken",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
    "disprot",
]
TARGET_LABEL = {t: t for t in TARGET_ORDER} | {"disprot": "all DisProt, pooled"}

#: Comparison tools, by the arm name the reduction gives them.
STRUCTURE_ARMS = {"foldseek": "Foldseek", "prostt5": "ProstT5", "reseek": "Reseek"}
SEQUENCE_ARMS = {
    "hmmer3_phmmer": "phmmer",
    "hmmer3_jackhmmer": "jackhmmer",
    "mmseqs2_seqseq": "MMseqs2",
    "mmseqs2_iterative": "MMseqs2 iterative",
}
COMPARISON_ARMS = {**STRUCTURE_ARMS, **SEQUENCE_ARMS}
LABEL_RULES = {
    "same_term": "same function term",
    "any_term": "any function term",
}
#: A function term needs this many regions in the choose half to get its own arm;
#: fewer, and it takes the arm chosen over all terms pooled.
MIN_REGIONS_PER_TERM = 5
REGION_KEY = ["query_acc", "term_id", "true_start", "true_end"]


# ---------------------------------------------------------------------------
# DisProt tables.
# ---------------------------------------------------------------------------
def proteins() -> pl.DataFrame:
    return pl.read_parquet(DISPROT_DIR / "disprot_proteins.parquet")


def fetch_summary() -> dict:
    import json

    return json.loads((DISPROT_DIR / "fetch_summary.json").read_text())


def human_regions() -> pl.DataFrame:
    """Every scored human region with its term name, length and covariates.

    pLDDT and disorder are over the region's own residues. kd_landed says whether the
    Kyte-Doolittle scan landed on it (a scan that knows nothing about homology).
    """
    reg = pl.read_parquet(DISPROT_DIR / "disprot_function_regions.parquet").filter(
        (pl.col("taxon") == HUMAN_TAXON) & pl.col("qfo_same_sequence")
    )
    plddt = pl.read_parquet(EXTRACT / "251_region_plddt.parquet")
    dis = pl.read_parquet(EXTRACT / "251_region_disorder.parquet")
    kd = pl.read_parquet(EXTRACT / "251_kd_scan_regions.parquet")
    base = (
        reg.select(
            pl.col("accession").alias("query_acc"),
            "term_id",
            "term_name",
            (pl.col("start") - 1).alias("true_start"),
            pl.col("end").alias("true_end"),
            pl.col("start").alias("domain_start"),
            pl.col("end").alias("domain_end"),
            "length",
        )
        .unique(REGION_KEY)
        .join(
            plddt.rename({"accession": "query_acc"}),
            on=["query_acc", "domain_start", "domain_end"],
            how="left",
        )
        .join(
            dis.rename({"accession": "query_acc"}).drop("pfam_id", strict=False),
            on=["query_acc", "domain_start", "domain_end"],
            how="left",
        )
        .join(kd, on=REGION_KEY, how="left")
        .with_columns(pl.col("kd_landed").fill_null(False))
    )
    prot = proteins().select(
        pl.col("accession").alias("query_acc"),
        pl.col("gene").alias("gene"),
        pl.col("length").alias("protein_length"),
    )
    return (
        base.join(prot, on="query_acc", how="left")
        .join(query_halves(), on="query_acc", how="left")
        .join(term_halves(base["term_id"].unique()), on="term_id", how="left")
    )


# ---------------------------------------------------------------------------
# Splits.
# ---------------------------------------------------------------------------
def sha1_half(name: str) -> str:
    """The rule notebook 244 uses: parity of the first byte of the SHA-1 of the name."""
    return "choose" if hashlib.sha1(name.encode()).digest()[0] % 2 == 0 else "report"


def query_halves() -> pl.DataFrame:
    """Each human query protein's half, by HGNC gene group (notebook 244's rule).

    The unit is the HGNC gene group, so paralogs in one group never sit on both sides; a
    protein with no group is its own unit, named by its symbol or, with no HGNC entry,
    its accession.
    """
    hgnc = pl.read_csv(
        HGNC_FILE,
        separator="\t",
        columns=["symbol", "gene_group", "uniprot_ids"],
        infer_schema_length=0,
    ).filter(pl.col("uniprot_ids").is_not_null())
    by_acc = (
        hgnc.with_columns(pl.col("uniprot_ids").str.split("|"))
        .explode("uniprot_ids")
        .rename({"uniprot_ids": "query_acc"})
        .unique("query_acc")
    )
    q = (
        proteins()
        .filter(pl.col("taxon") == HUMAN_TAXON)
        .select(pl.col("accession").alias("query_acc"))
    )
    unit = q.join(by_acc, on="query_acc", how="left").with_columns(
        split_unit=pl.coalesce("gene_group", "symbol", "query_acc")
    )
    return unit.select(
        "query_acc",
        pl.col("symbol").alias("hgnc_symbol"),
        pl.col("gene_group").alias("hgnc_gene_group"),
        "split_unit",
        pl.col("split_unit")
        .map_elements(sha1_half, return_dtype=pl.String)
        .alias("query_half"),
    )


def term_halves(terms) -> pl.DataFrame:
    """Each DisProt function term's half, for the held-out-function split."""
    terms = sorted(set(terms))
    return pl.DataFrame({"term_id": terms, "term_half": [sha1_half(t) for t in terms]})


# ---------------------------------------------------------------------------
# Landing tables.
# ---------------------------------------------------------------------------
def _concat(pattern: str, landing_dir: Path) -> pl.DataFrame:
    files = sorted(landing_dir.glob(pattern))
    frames = [pl.read_parquet(f) for f in files if f.stat().st_size]
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_landing(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    return _concat("*.regions.parquet", landing_dir)


def load_hits(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    return _concat("*.hits.parquet", landing_dir)


def load_ranks(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    return _concat("*.ranks.parquet", landing_dir)


def load_length_rho(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    return _concat("*.length_rho.parquet", landing_dir)


def arm_label(arm: str) -> str:
    return COMPARISON_ARMS.get(arm) or he.arm_short(arm)


def arm_targets(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    """Every (arm, tool, target) the reduction processed, from its file names.

    Read from the names, not the rows: an arm that labelled nothing writes an empty
    table, and taking the arm list from the rows would drop it instead of counting it
    as zero.
    """
    rows = []
    for f in sorted(landing_dir.glob("*.regions.parquet")):
        stem = f.name[: -len(".regions.parquet")]
        arm, _, target = stem.rpartition(".")
        rows.append(dict(arm=arm, tool=arm.split(".")[0], target=target))
    return pl.DataFrame(rows, schema=dict(arm=pl.String, tool=pl.String, target=pl.String))


def full_grid(
    regions: pl.DataFrame, landing: pl.DataFrame, landing_dir: Path = LANDING_DIR
) -> pl.DataFrame:
    """Every (arm, target, rule) x every human region, a region with no call included.

    The landing table only has rows for regions some labelled call overlapped, so a
    fraction taken over it alone would drop every region a tool missed.
    """
    combos = arm_targets(landing_dir).join(
        pl.DataFrame({"label_rule": list(LABEL_RULES)}), how="cross"
    )
    grid = combos.join(regions.select(REGION_KEY), how="cross")
    if landing.is_empty():
        # Nothing landed anywhere: the columns the reduction would have written, empty.
        ints = ["qstart", "qend", "tstart", "tend", "t_feat_start", "t_feat_end"]
        floats = ["rank_score", "iou", "inside", "cover"]
        cols = {
            f"{p}_{c}": pl.lit(None, dtype=pl.Int64) for p in ("best", "land") for c in ints
        }
        cols |= {
            f"{p}_{c}": pl.lit(None, dtype=pl.Float64)
            for p in ("best", "land")
            for c in floats
        }
        cols |= {
            f"{p}_target_acc": pl.lit(None, dtype=pl.String) for p in ("best", "land")
        }
        return grid.with_columns(
            landed=pl.lit(False),
            n_overlapping_calls=pl.lit(0, dtype=pl.UInt32),
            n_overlapping_targets=pl.lit(0, dtype=pl.UInt32),
            n_landed_targets=pl.lit(0, dtype=pl.UInt32),
            any_inside_half=pl.lit(None, dtype=pl.Boolean),
            rank_by=pl.lit(None, dtype=pl.String),
            **cols,
        )
    return grid.join(
        landing.drop("tool"), on=["arm", "target", "label_rule", *REGION_KEY], how="left"
    ).with_columns(
        pl.col("landed").fill_null(False),
        pl.col("n_overlapping_calls").fill_null(0),
    )


def landing_summary(grid: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Fraction of regions landed on, and the median IoU over the regions landed on."""
    return (
        grid.group_by(by)
        .agg(
            n_regions=pl.len(),
            n_landed=pl.col("landed").sum(),
            median_iou_landed=pl.col("land_iou").filter(pl.col("landed")).median(),
        )
        .with_columns(frac_landed=pl.col("n_landed") / pl.col("n_regions"))
        .sort(by)
    )


def choose_arms(
    grid: pl.DataFrame,
    regions: pl.DataFrame,
    target: str,
    rule: str,
    half_col: str = "query_half",
    per_term: bool = True,
) -> pl.DataFrame:
    """The kmerseek arm per function term, picked on the choose half.

    Best landing fraction, then higher median IoU over the landed regions, then the arm
    name, so the pick is the same every run. A term with fewer than
    MIN_REGIONS_PER_TERM choose-half regions takes the arm picked over all terms pooled.

    half_col="term_half" is the held-out-function split. A term in the report half has no
    choose-half regions by construction, so there every term takes the pooled arm
    (per_term=False).
    """
    km = grid.filter(
        (pl.col("tool") == "kmerseek")
        & (pl.col("target") == target)
        & (pl.col("label_rule") == rule)
    ).join(regions.select(*REGION_KEY, half_col), on=REGION_KEY)
    choose = km.filter(pl.col(half_col) == "choose")
    if choose.is_empty():
        return pl.DataFrame(
            schema=dict(
                term_id=pl.String,
                chosen_arm=pl.String,
                own_arm=pl.Boolean,
                n_choose_regions=pl.UInt32,
            )
        )

    def best(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        s = landing_summary(df, [*by, "arm"]).with_columns(
            pl.col("median_iou_landed").fill_null(0.0)
        )
        return (
            s.sort(
                [*by, "frac_landed", "median_iou_landed", "arm"],
                descending=[False] * len(by) + [True, True, False],
            )
            .group_by(by, maintain_order=True)
            .first()
        )

    pooled = best(choose.with_columns(_all=pl.lit(1)), ["_all"]).row(0, named=True)
    per = best(choose, ["term_id"])
    if not per_term:
        per = per.with_columns(n_regions=pl.lit(0, dtype=pl.UInt32))
    terms = regions.select("term_id").unique()
    return (
        terms.join(per, on="term_id", how="left")
        .with_columns(
            own_arm=pl.col("n_regions").fill_null(0) >= MIN_REGIONS_PER_TERM,
        )
        .with_columns(
            chosen_arm=pl.when("own_arm")
            .then(pl.col("arm"))
            .otherwise(pl.lit(pooled["arm"])),
            n_choose_regions=pl.col("n_regions").fill_null(0),
        )
        .select("term_id", "chosen_arm", "own_arm", "n_choose_regions")
    )


def has_model(accessions) -> dict[str, bool]:
    """Whether the human protein has an AlphaFold model, the input Foldseek and Reseek
    need (ProstT5 predicts its structure letters from the sequence and needs none)."""
    d = DISPROT_DIR / "structures" / "human"
    have = {f.name.split("-")[1] for f in d.glob("AF-*-F1-model_v*.cif")}
    return {a: a in have for a in accessions}


# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------
def placement_p(
    protein_length: int, call_start: int, call_end: int, true_start: int, true_end: int
) -> float:
    """Chance that a window of the call's length, dropped at a uniformly random position
    on the human protein, lands on the region by the same rule (>= 80% inside, >= 30% of
    the region covered). A landed call only counts when this is small."""
    w = call_end - call_start
    n = protein_length - w + 1
    if w <= 0 or n <= 0:
        return float("nan")
    s = np.arange(n)
    ov = np.clip(np.minimum(s + w, true_end) - np.maximum(s, true_start), 0, None)
    ok = (ov / w >= he.INSIDE_MIN) & (
        ov / max(true_end - true_start, 1) >= he.COVER_MIN
    )
    return float(ok.mean())


def add_placement_p(df: pl.DataFrame) -> pl.DataFrame:
    """placement_p for every landed row (land_qstart/land_qend)."""
    ps = [
        (
            placement_p(
                r["protein_length"],
                r["land_qstart"],
                r["land_qend"],
                r["true_start"],
                r["true_end"],
            )
            if r["landed"]
            else None
        )
        for r in df.select(
            "protein_length",
            "land_qstart",
            "land_qend",
            "true_start",
            "true_end",
            "landed",
        ).iter_rows(named=True)
    ]
    return df.with_columns(placement_p=pl.Series(ps, dtype=pl.Float64))


def random_query_p(
    cases: pl.DataFrame,
    ranks: pl.DataFrame,
    regions: pl.DataFrame,
    n_draw: int = 200,
    tol: float = 0.1,
) -> pl.DataFrame:
    """Length-matched random-query control for landed cases.

    For a case (human protein q, target protein T that passed the label, at rank r in q's
    list under the arm's ranking score), draw up to n_draw other human query proteins
    whose length is within tol of q's and whose HGNC unit is different, and count how
    many put T at rank <= r. A query that does not list T at all did not reach it.
    p = (1 + reached) / (1 + drawn). The draw is the first n_draw by SHA-1 of
    accession + case key, so it is the same every run.
    """
    plen = regions.select("query_acc", "protein_length", "split_unit").unique(
        "query_acc"
    )
    lens = dict(zip(plen["query_acc"], plen["protein_length"]))
    units = dict(zip(plen["query_acc"], plen["split_unit"]))
    rk = {
        (a, rb, q, t): r
        for a, rb, q, t, r in ranks.select(
            "arm", "rank_by", "query_acc", "target_acc", "rank"
        ).iter_rows()
    }
    rows = []
    for c in cases.iter_rows(named=True):
        q, t, arm, rb = c["query_acc"], c["land_target_acc"], c["arm"], c["rank_by"]
        r = rk.get((arm, rb, q, t))
        L = lens.get(q)
        if r is None or L is None:
            rows.append(
                dict(
                    **{k: c[k] for k in (*REGION_KEY, "arm", "target", "label_rule")},
                    case_rank=r,
                    n_drawn=0,
                    n_reached=0,
                    random_query_p=None,
                )
            )
            continue
        pool = [
            a
            for a, la in lens.items()
            if a != q and units.get(a) != units.get(q) and abs(la - L) <= tol * L
        ]
        key = f"{q}|{c['term_id']}|{arm}|{t}"
        pool.sort(key=lambda a: hashlib.sha1(f"{a}|{key}".encode()).hexdigest())
        drawn = pool[:n_draw]
        reached = sum(1 for a in drawn if (rk.get((arm, rb, a, t)) or 10**9) <= r)
        rows.append(
            dict(
                **{k: c[k] for k in (*REGION_KEY, "arm", "target", "label_rule")},
                case_rank=r,
                n_drawn=len(drawn),
                n_reached=reached,
                random_query_p=(1 + reached) / (1 + len(drawn)),
            )
        )
    schema = {
        **{k: pl.String for k in ("query_acc", "term_id", "arm", "target", "label_rule")},
        "true_start": pl.Int64,
        "true_end": pl.Int64,
        "case_rank": pl.Int64,
        "n_drawn": pl.Int64,
        "n_reached": pl.Int64,
        "random_query_p": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Sequences, for the alignment printout.
# ---------------------------------------------------------------------------
def target_sequences(target: str, accessions: set[str]) -> dict[str, str]:
    if target == "disprot":
        p = proteins().filter(pl.col("accession").is_in(list(accessions)))
        return dict(zip(p["accession"], p["sequence"]))
    return he.sequences(target, accessions)


def human_sequences(accessions: set[str]) -> dict[str, str]:
    p = proteins().filter(pl.col("accession").is_in(list(accessions)))
    return dict(zip(p["accession"], p["sequence"]))
