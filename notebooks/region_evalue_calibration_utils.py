"""Counting chance regions on decoy queries against what kmerseek's E-values predict.

Notebook 344 reads the outputs of nextflow-runs/region-evalue-calibration and asks: when
kmerseek reports a region at E <= E*, do unrelated queries produce about that many such
regions by chance? An E-value E for a region means "about E regions this good turn up by
chance for one query searched against the whole database". So Q decoy queries should
produce about Q x E* regions at E <= E*.

The strata, cutoffs and pass rule are fixed here, in constants, so the notebook and the
tests use one copy. They were committed before any decoy was searched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# Length bins, in residues: the leaderboard's RULES.md edges, [0,100), [100,300),
# [300,1000], (1000, inf). The third bin includes 1000.
LENGTH_BINS = (
    ("<100 aa", 0, 100),
    ("100-299 aa", 100, 300),
    ("300-1000 aa", 300, 1001),
    (">1000 aa", 1001, math.inf),
)
# Share of the real protein's residues that SEG masks (segmasker defaults: window 12, low
# cut 2.2, high cut 2.5; Wootton and Federhen 1993): [0, 0.05), [0.05, 0.15), [0.15, 1].
# Chosen from the human proteome's SEG shares alone (median 0.069), before any search.
SEG_BINS = (
    ("SEG <5%", 0.0, 0.05),
    ("SEG 5-15%", 0.05, 0.15),
    ("SEG >=15%", 0.15, math.inf),
)
STRATA = tuple(f"{lb[0]}, {sb[0]}" for lb in LENGTH_BINS for sb in SEG_BINS)

# Expected chance regions per stratum at the two cutoffs: E* = TARGET / Q_s, where Q_s is
# the number of decoy queries in the stratum, so Q_s x E* = TARGET.
TARGETS = (10, 100)
FACTOR = 2.0
# The two shipped columns the release gate judges, and the one users filter on (reported,
# not judged).
GATE_COLUMNS = ("region_run_evalue", "region_ka_evalue")
REPORT_COLUMNS = GATE_COLUMNS + ("region_evalue",)
GATE_KIND = "dipeptide"
KINDS = ("composition", "dipeptide", "window10")
# The pipeline keeps a region row only when one of its E-values is <= this
# (params.max_evalue in nextflow-runs/region-evalue-calibration/main.nf).
KEPT_UP_TO = 1.0

# hp_lehninger2's classes, from kmerseek src/rust/alphabets.rs (LEHNINGER_HP, line 298 at
# 0a22970): h = AFGILMPVWY, p = CDEHKNQRST.
HP_LEHNINGER2 = {**{a: "h" for a in "AFGILMPVWY"}, **{a: "p" for a in "CDEHKNQRST"}}


def length_bin(length: int) -> str:
    for name, lo, hi in LENGTH_BINS:
        if lo <= length < hi:
            return name
    raise ValueError(f"length {length} is in no bin")


def seg_bin(fraction: float) -> str:
    for name, lo, hi in SEG_BINS:
        if lo <= fraction < hi:
            return name
    raise ValueError(f"SEG share {fraction} is in no bin")


def stratum(length: int, seg_fraction: float) -> str:
    return f"{length_bin(length)}, {seg_bin(seg_fraction)}"


def poisson_cdf(k: int, mean: float) -> float:
    """Pr(X <= k) for X ~ Poisson(mean), summed term by term (k is small here)."""
    if k < 0:
        return 0.0
    term = total = math.exp(-mean)
    for i in range(1, k + 1):
        term *= mean / i
        total += term
    return min(total, 1.0)


def chance_inside_factor(expected: float, factor: float = FACTOR) -> float:
    """Pr(expected / factor <= X <= expected x factor) for X ~ Poisson(expected): the
    chance that a calibrated E-value passes one stratum, if regions arrive independently.
    """
    lo, hi = math.ceil(expected / factor), math.floor(expected * factor)
    return poisson_cdf(hi, expected) - poisson_cdf(lo - 1, expected)


def passes(observed: int, expected: float, factor: float = FACTOR) -> bool:
    return expected / factor <= observed <= expected * factor


def load_decoys(results: Path) -> pl.DataFrame:
    """One row per decoy: its id, kind, length, its source protein's SEG share, stratum."""
    decoys = pl.read_csv(results / "decoys" / "decoys.tsv", separator="\t")
    seg = pl.read_csv(results / "seg" / "human.seg.tsv", separator="\t").select(
        pl.col("id").alias("source_id"), "seg_fraction"
    )
    out = decoys.join(seg, on="source_id", how="left", validate="m:1")
    if out["seg_fraction"].null_count():
        raise ValueError("some decoys have no SEG share for their source protein")
    return out.with_columns(
        pl.struct("length", "seg_fraction")
        .map_elements(
            lambda r: stratum(r["length"], r["seg_fraction"]), return_dtype=pl.String
        )
        .alias("stratum")
    )


def load_regions(results: Path, kind: str) -> pl.DataFrame:
    """Every kept region row of one decoy kind (E <= 1 in at least one E-value column)."""
    return pl.read_csv(
        results / "regions" / f"{kind}.regions.tsv.gz",
        separator="\t",
        infer_schema_length=0,
    ).with_columns(
        pl.col(list(REPORT_COLUMNS) + ["region_pr_same", "region_ka_bits"]).cast(
            pl.Float64, strict=False
        ),
        pl.col(
            [
                "region_start",
                "region_end",
                "target_start",
                "target_end",
                "region_length",
                "region_n_mismatches",
                "region_run_length",
            ]
        ).cast(pl.Int64),
    )


# A decoy the shuffle left identical to its source, or whose hp_lehninger2 string equals
# its source's (searched in that alphabet it is the protein itself), is not a decoy.
NOT_A_DECOY = (pl.col("unchanged") == 1) | (pl.col("same_hp_lehninger2") == 1)


def accession_of(name: pl.Expr) -> pl.Expr:
    """P10415 from sp|P10415|BCL2_HUMAN (and from a name with no | fields, the name)."""
    return (
        pl.when(name.str.count_matches(r"\|") >= 2)
        .then(name.str.split("|").list.get(1, null_on_oob=True))
        .otherwise(name)
    )


def tag_regions(decoys: pl.DataFrame, regions: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Each region row with its decoy's stratum and two flags: `own_source` (the target is
    the protein the decoy was shuffled from) and `not_a_decoy` (see NOT_A_DECOY)."""
    d = decoys.filter(pl.col("kind") == kind).select(
        pl.col("decoy_id").alias("query_name"),
        "stratum",
        "accession",
        NOT_A_DECOY.alias("not_a_decoy"),
    )
    r = regions.join(d, on="query_name", how="left")
    if r["stratum"].null_count():
        raise ValueError(
            f"{r['stratum'].null_count()} region rows have a query that is not a {kind} decoy"
        )
    return r.with_columns(
        (accession_of(pl.col("target_name")) == pl.col("accession")).alias("own_source")
    )


def calibration_table(
    decoys: pl.DataFrame, regions: pl.DataFrame, kind: str
) -> pl.DataFrame:
    """Observed against expected chance regions, per stratum, column and target.

    Q_s counts every decoy of this kind in the stratum, including those with no region,
    except those that are not decoys (NOT_A_DECOY). A region on the decoy's own source protein is not counted: that one database
    protein has the decoy's composition and pair counts, so it is not unrelated to it.
    A row whose E-value is empty is not counted for that column.
    """
    d = decoys.filter((pl.col("kind") == kind) & (~NOT_A_DECOY))
    q = d.group_by("stratum").len("n_decoys")
    r = tag_regions(decoys, regions, kind).filter(
        ~pl.col("not_a_decoy") & ~pl.col("own_source")
    )
    rows = []
    for s in STRATA:
        q_s = q.filter(pl.col("stratum") == s)["n_decoys"].sum()
        in_s = r.filter(pl.col("stratum") == s)
        for column in REPORT_COLUMNS:
            for target in TARGETS:
                cutoff = target / q_s if q_s else math.nan
                if cutoff > KEPT_UP_TO:
                    raise ValueError(
                        f"{s}: cutoff {cutoff:.3g} is above the {KEPT_UP_TO} the pipeline kept"
                    )
                observed = int((in_s[column] <= cutoff).sum()) if q_s else 0
                rows.append(
                    {
                        "kind": kind,
                        "stratum": s,
                        "column": column,
                        "expected": float(target),
                        "n_decoys": int(q_s),
                        "cutoff": cutoff,
                        "observed": observed,
                        "ratio": observed / target,
                        "inside_factor_two": bool(q_s) and passes(observed, target),
                    }
                )
    return pl.DataFrame(rows)


def verdict(table: pl.DataFrame) -> dict[str, bool]:
    """The pre-registered rule: a column passes only if every stratum is inside a factor of
    two at both targets, on the dipeptide decoys."""
    g = table.filter(
        (pl.col("kind") == GATE_KIND) & pl.col("column").is_in(GATE_COLUMNS)
    )
    out = {}
    for column in GATE_COLUMNS:
        c = g.filter(pl.col("column") == column)
        if c.height != len(STRATA) * len(TARGETS):
            raise ValueError(
                f"{column}: {c.height} rows, expected {len(STRATA) * len(TARGETS)}"
            )
        out[column] = bool(c["inside_factor_two"].all())
    return out


def observed_curve(
    decoys: pl.DataFrame,
    regions: pl.DataFrame,
    kind: str,
    column: str,
    cutoffs: list[float],
) -> pl.DataFrame:
    """All strata together: regions at E <= cutoff against Q x cutoff, for each cutoff,
    with the same exclusions as `calibration_table`."""
    n = decoys.filter((pl.col("kind") == kind) & (~NOT_A_DECOY)).height
    r = tag_regions(decoys, regions, kind).filter(
        ~pl.col("not_a_decoy") & ~pl.col("own_source")
    )
    values = r[column].drop_nulls()
    return pl.DataFrame(
        {
            "cutoff": cutoffs,
            "expected": [n * c for c in cutoffs],
            "observed": [int((values <= c).sum()) for c in cutoffs],
        }
    ).with_columns(pl.lit(kind).alias("kind"), pl.lit(column).alias("column"))


@dataclass
class AlignedRegion:
    query: str
    target: str
    query_start: int  # 1-based, inclusive
    target_start: int

    def lines(
        self, classes: dict[str, str] = HP_LEHNINGER2, width: int = 60
    ) -> list[str]:
        """The residues (decoy, then target) with a match line (| identical), then the
        hp_lehninger2 classes (h hydrophobic, p polar; decoy, then target) with their
        match line (| same class), in blocks of `width`, with 1-based coordinates."""
        if len(self.query) != len(self.target):
            raise ValueError("a gapless region has the same length on both sides")
        qc = "".join(classes.get(a, "?") for a in self.query)
        tc = "".join(classes.get(a, "?") for a in self.target)
        res_match = "".join(
            "|" if a == b else " " for a, b in zip(self.query, self.target)
        )
        cls_match = "".join("|" if a == b != "?" else " " for a, b in zip(qc, tc))
        out = []
        for i in range(0, len(self.query), width):
            j = min(i + width, len(self.query))
            qs, ts = self.query_start + i, self.target_start + i
            out += [
                f"decoy   {qs:>5} {self.query[i:j]} {qs + j - i - 1}",
                f"              {res_match[i:j]}",
                f"target  {ts:>5} {self.target[i:j]} {ts + j - i - 1}",
                f"hp      {qs:>5} {qc[i:j]}",
                f"              {cls_match[i:j]}",
                f"hp      {ts:>5} {tc[i:j]}",
                "",
            ]
        return out

    def counts(self, classes: dict[str, str] = HP_LEHNINGER2) -> dict[str, int]:
        """Identical residues, same-class positions, and the longest same-class run."""
        same = [
            classes.get(a, "?") == classes.get(b, "!")
            for a, b in zip(self.query, self.target)
        ]
        longest = run = 0
        for s in same:
            run = run + 1 if s else 0
            longest = max(longest, run)
        return {
            "length": len(self.query),
            "identical": sum(a == b for a, b in zip(self.query, self.target)),
            "same_class": sum(same),
            "longest_same_class_run": longest,
        }
