"""One table of kmerseek region calls: one row per (query, species, merged region, arm).

Built for notebook 260 from the midi-plus kmerseek 0.4 run, and from its dipeptide-shuffled
twin. Three steps, each a function here and a subcommand of the CLI at the bottom:

1. ``reduce``: read one ``human_vs_<species>.<alphabet>.k<k>...regions.parquet``, keep the
   rows with ``region_evalue < EVALUE_MAX``, keep every column kmerseek wrote under its own
   name, and add ``species``, ``alphabet``, ``k``, ``arm``. One summary row per file says
   how many rows and queries the cut kept, and whether the file has any E-value at all.
2. ``merge``: on one query and one target species, two calls belong to one merged region
   when their overlap is at least half the shorter call. Taken pairwise and chained (call A
   joins B, B joins C, so A, B and C are one region even if A and C barely touch).
3. ``truth``: every merged region against every truth feature on its query, Pfam and
   Swiss-Prot. ``landed`` = overlap / region length, ``coverage`` = overlap / feature
   length. A region is a true call when ``landed >= LANDED_MIN``. No IoU gate.

Coordinates. kmerseek writes ``region_start``/``region_end`` 0-based and end-exclusive;
both truth tables are 1-based and inclusive. Every overlap here is computed on 1-based
inclusive intervals: ``call_start = region_start + 1``, ``call_end = region_end``, length
``end - start + 1``. The original kmerseek columns are kept unchanged beside them.

polars and numpy only, so the same code runs on the Mac and inside the kmerseek 0.4 image
on Sherlock (polars 1.43, numpy 2.5, no scipy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

EVALUE_MAX = 10.0
OVERLAP_OF_SHORTER = 0.5
LANDED_MIN = 0.5

# human_vs_zebrafish.encoded_hp_pbotc_1st_ed2.k19.lcfalse.extend-c1.63.regions.parquet
# The extend-c<C> part exists only in the random-alphabet control's files; the pipeline's
# own extended searches do not record the penalty in the name (one C per alphabet).
FILE_RE = re.compile(
    r"^human_vs_(?P<species>[a-z]+)\.(?P<alphabet>.+?)\.k(?P<k>\d+)"
    r"(?:\.s(?P<scaled>\d+))?\.lc(?P<lc>true|false)"
    r"(?:\.extend-c(?P<c>[0-9.]+))?\.regions\.parquet$"
)

DECOY_PREFIX = "DECOY_"


def parse_name(path: Path) -> dict:
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not a kmerseek regions file name: {path.name}")
    d = m.groupdict()
    k = int(d["k"])
    arm = f"{d['alphabet']}_k{k}"
    # Two searches of one alphabet and k at different extension penalties would otherwise
    # share an arm name; the penalty goes in only when the file name carries it.
    if d["c"]:
        arm += f"_ext{d['c']}"
    return {
        "species": d["species"],
        "alphabet": d["alphabet"],
        "k": k,
        "scaled": int(d["scaled"] or 1),
        "lowcomp": d["lc"] == "true",
        "extend_penalty": float(d["c"]) if d["c"] else None,
        "arm": arm,
        "source_file": path.name,
    }


def fit_refused(path: Path) -> bool | None:
    """The search's own record of a refused E-value fit, from its timings.jsonl, or None
    when the file is missing or does not say."""
    t = path.with_name(path.name.replace(".regions.parquet", ".timings.jsonl"))
    if not t.exists():
        return None
    for line in t.read_text().splitlines():
        rec = json.loads(line)
        if "extension_fit_refused" in rec:
            return bool(rec["extension_fit_refused"])
    return None


def accession_expr() -> pl.Expr:
    # query_name is the whole FASTA header: sp|P12345|NAME_HUMAN ... (or sp|DECOY_P12345|...)
    return pl.col("query_name").str.split("|").list.get(1)


def reduce_file(path: Path, evalue_max: float = EVALUE_MAX) -> tuple[pl.DataFrame, dict]:
    meta = parse_name(path)
    lf = pl.scan_parquet(path)
    summary = (
        lf.select(
            n_regions=pl.len(),
            n_queries=pl.col("query_name").n_unique(),
            n_evalue_finite=pl.col("region_evalue").is_finite().sum(),
            n_kept=(pl.col("region_evalue") < evalue_max).sum(),
            n_queries_kept=pl.col("query_name")
            .filter(pl.col("region_evalue") < evalue_max)
            .n_unique(),
            evalue_min=pl.col("region_evalue").min(),
        )
        .collect()
        .to_dicts()[0]
    )
    summary.update(meta)
    summary["evalue_max"] = evalue_max
    summary["has_evalue"] = summary["n_evalue_finite"] > 0
    summary["extension_fit_refused"] = fit_refused(path)
    kept = (
        lf.filter(pl.col("region_evalue") < evalue_max)
        .with_columns(**{k: pl.lit(v) for k, v in meta.items()})
        .with_columns(accession=accession_expr())
        .collect()
    )
    return kept, summary


# ---------------------------------------------------------------------------
# Merging.
# ---------------------------------------------------------------------------


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component labels of n nodes joined by the edges (a[i], b[i]).

    Label propagation with the smallest index as the label: each pass pulls every edge's
    two ends down to their smaller label, then jumps every node to its label's label.
    No scipy in the Sherlock image, so no csgraph.
    """
    lab = np.arange(n)
    if len(a) == 0:
        return lab
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def merge_group(start: np.ndarray, end: np.ndarray, frac: float = OVERLAP_OF_SHORTER) -> np.ndarray:
    """Merged-region labels for calls on one (query, species), 1-based inclusive.

    `start` must be sorted ascending. For call i, every later call j with start_j <= end_i
    overlaps it, and those j are one contiguous run after i (sorted starts), so the pairs
    are generated directly rather than by an all-against-all matrix.
    """
    n = len(start)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    length = end - start + 1
    # r[i]: first index whose start is past end_i; candidates for i are i+1 .. r[i]-1.
    r = np.searchsorted(start, end, side="right")
    cnt = np.maximum(r - np.arange(n) - 1, 0)
    if cnt.sum() == 0:
        return np.arange(n)
    i = np.repeat(np.arange(n), cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    j = i + 1 + offs
    ov = np.minimum(end[i], end[j]) - start[j] + 1
    keep = ov >= frac * np.minimum(length[i], length[j])
    lab = _components(n, i[keep], j[keep])
    # Relabel 0..m-1 in order of first appearance (that is, by leftmost call).
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(first)
    remap = np.empty(lab.max() + 1, dtype=np.int64)
    remap[np.unique(lab)[order]] = np.arange(len(order))
    return remap[lab]


def merge_calls(calls: pl.DataFrame, frac: float = OVERLAP_OF_SHORTER) -> pl.DataFrame:
    """Add call_start/call_end (1-based inclusive) and merged_region_id per (accession,
    species), then merged_start/merged_end and n_arms for each merged region."""
    calls = calls.with_columns(
        call_start=pl.col("region_start") + 1,
        call_end=pl.col("region_end"),
    ).sort(["accession", "species", "call_start", "call_end", "arm", "target_name"])
    parts = []
    for (acc, sp), g in calls.group_by(["accession", "species"], maintain_order=True):
        lab = merge_group(g["call_start"].to_numpy(), g["call_end"].to_numpy(), frac)
        parts.append(g.with_columns(merged_region_id=pl.Series(lab, dtype=pl.Int64)))
    merged = pl.concat(parts) if parts else calls.with_columns(merged_region_id=pl.lit(0, pl.Int64))
    key = ["accession", "species", "merged_region_id"]
    return merged.with_columns(
        merged_start=pl.col("call_start").min().over(key),
        merged_end=pl.col("call_end").max().over(key),
        n_calls_merged=pl.len().over(key),
        n_arms=pl.col("arm").n_unique().over(key),
    )


def one_row_per_arm(merged: pl.DataFrame) -> pl.DataFrame:
    """Keep one call per (accession, species, merged region, arm): the lowest
    region_evalue, ties to the highest region_tfidf, then the leftmost call, then the
    target name, so the choice never depends on row order. n_calls_arm says how many
    calls of that arm the region absorbed."""
    key = ["accession", "species", "merged_region_id", "arm"]
    return (
        merged.with_columns(n_calls_arm=pl.len().over(key))
        .sort(key + ["region_evalue", "region_tfidf", "call_start", "target_name"],
              descending=[False] * len(key) + [False, True, False, False])
        .unique(subset=key, keep="first", maintain_order=True)
        .sort(["accession", "species", "merged_region_id", "arm"])
    )


# ---------------------------------------------------------------------------
# Truth.
# ---------------------------------------------------------------------------


def load_truth(pfam_path: Path, swissprot_path: Path) -> pl.DataFrame:
    """Both truth sets as (truth_set, accession, feature, feature_start, feature_end,
    pfam_split), 1-based inclusive. feature is the Pfam family or the Swiss-Prot feature
    type (DOMAIN, REGION, TRANSMEM, ...), the key notebook 231 scored on."""
    pf = pl.read_parquet(pfam_path).select(
        truth_set=pl.lit("pfam"),
        accession="accession",
        feature="pfam_id",
        feature_start=pl.col("domain_start").cast(pl.Int64),
        feature_end=pl.col("domain_end").cast(pl.Int64),
        pfam_split="split",
    )
    sp = pl.read_parquet(swissprot_path).select(
        truth_set=pl.lit("swissprot"),
        accession="accession",
        feature="pfam_id",
        feature_start=pl.col("domain_start").cast(pl.Int64),
        feature_end=pl.col("domain_end").cast(pl.Int64),
        pfam_split=pl.lit(None, pl.String),
    )
    return pl.concat([pf, sp])


def region_truth_pairs(regions: pl.DataFrame, truth: pl.DataFrame,
                       start: str, end: str) -> pl.DataFrame:
    """Every (region, truth feature) on the same query that overlap by >= 1 residue,
    with landed and coverage. `regions` must carry accession and the two named columns."""
    j = regions.join(truth, on="accession", how="inner")
    ov = (pl.min_horizontal(pl.col(end), pl.col("feature_end"))
          - pl.max_horizontal(pl.col(start), pl.col("feature_start")) + 1)
    return (
        j.with_columns(overlap=ov)
        .filter(pl.col("overlap") > 0)
        .with_columns(
            landed=pl.col("overlap") / (pl.col(end) - pl.col(start) + 1),
            coverage=pl.col("overlap") / (pl.col("feature_end") - pl.col("feature_start") + 1),
        )
    )


def best_truth(pairs: pl.DataFrame, key: list[str], prefix: str) -> pl.DataFrame:
    """Per key and truth set: the feature with the largest landed fraction (ties to the
    larger coverage, then the feature name), its coverage, and is_true = landed >= 0.5."""
    best = (
        pairs.sort(key + ["truth_set", "landed", "coverage", "feature"],
                   descending=[False] * len(key) + [False, True, True, False])
        .unique(subset=key + ["truth_set"], keep="first", maintain_order=True)
    )
    out = None
    for ts in ("pfam", "swissprot"):
        b = best.filter(pl.col("truth_set") == ts).select(
            key
            + [
                pl.col("feature").alias(f"{prefix}{ts}_feature"),
                pl.col("feature_start").alias(f"{prefix}{ts}_feature_start"),
                pl.col("feature_end").alias(f"{prefix}{ts}_feature_end"),
                pl.col("landed").alias(f"{prefix}{ts}_landed"),
                pl.col("coverage").alias(f"{prefix}{ts}_coverage"),
            ]
            + ([pl.col("pfam_split").alias(f"{prefix}pfam_split")] if ts == "pfam" else [])
        )
        out = b if out is None else out.join(b, on=key, how="full", coalesce=True)
    return out


def query_split(accession: pl.Expr) -> pl.Expr:
    """tune / test by the parity of the first byte of SHA-1(source accession). A decoy gets
    its source protein's half, so a shuffled copy is never tuned on while its source is
    reported on."""
    src = accession.str.strip_prefix(DECOY_PREFIX)
    return src.map_elements(
        lambda a: "tune" if hashlib.sha1(a.encode()).digest()[0] % 2 == 0 else "test",
        return_dtype=pl.String,
    )


def build_table(calls: pl.DataFrame, truth: pl.DataFrame | None,
                species_mya: dict[str, int], is_decoy: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The per-arm table and the long (merged region x truth feature) table."""
    merged = one_row_per_arm(merge_calls(calls))
    key = ["accession", "species", "merged_region_id"]
    regions = merged.select(key + ["merged_start", "merged_end"]).unique(subset=key)
    if truth is not None and not is_decoy:
        reg_pairs = region_truth_pairs(regions, truth, "merged_start", "merged_end")
        merged = merged.join(best_truth(reg_pairs, key, "merged_"), on=key, how="left")
        call_key = key + ["arm"]
        call_pairs = region_truth_pairs(
            merged.select(call_key + ["call_start", "call_end"]), truth, "call_start", "call_end"
        )
        merged = merged.join(best_truth(call_pairs, call_key, "call_").drop("call_pfam_split"),
                             on=call_key, how="left")
        long = reg_pairs.select(key + ["merged_start", "merged_end", "truth_set", "feature",
                                       "feature_start", "feature_end", "pfam_split",
                                       "overlap", "landed", "coverage"])
    else:
        long = pl.DataFrame()
    for ts in ("pfam", "swissprot"):
        for lvl in ("merged", "call"):
            c = f"{lvl}_{ts}_landed"
            if c not in merged.columns:
                merged = merged.with_columns(pl.lit(None, pl.Float64).alias(c))
            merged = merged.with_columns(
                (pl.col(c).fill_null(0.0) >= LANDED_MIN).alias(f"{lvl}_{ts}_is_true")
            )
    has_truth = (
        truth.select("accession", "truth_set").unique()
        .group_by("accession").agg(
            query_has_pfam=(pl.col("truth_set") == "pfam").any(),
            query_has_swissprot=(pl.col("truth_set") == "swissprot").any(),
        )
        if truth is not None else None
    )
    if has_truth is not None and not is_decoy:
        merged = merged.join(has_truth, on="accession", how="left").with_columns(
            pl.col("query_has_pfam").fill_null(False),
            pl.col("query_has_swissprot").fill_null(False),
        )
    else:
        merged = merged.with_columns(query_has_pfam=pl.lit(False), query_has_swissprot=pl.lit(False))
    merged = merged.with_columns(
        is_decoy=pl.lit(is_decoy),
        source_accession=pl.col("accession").str.strip_prefix(DECOY_PREFIX),
        split=query_split(pl.col("accession")),
        species_mya=pl.col("species").replace_strict(species_mya, default=None,
                                                    return_dtype=pl.Int64),
        merged_length=pl.col("merged_end") - pl.col("merged_start") + 1,
        call_length=pl.col("call_end") - pl.col("call_start") + 1,
    )
    return merged, long


# ---------------------------------------------------------------------------
# CLI, for the full run on Sherlock.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce", help="E-value cut of every regions file in a directory")
    r.add_argument("--kmerseek-dir", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--evalue-max", type=float, default=EVALUE_MAX)
    b = sub.add_parser("build", help="merge + truth on the reduced files")
    b.add_argument("--reduced-dir", type=Path, required=True)
    b.add_argument("--pfam-truth", type=Path, required=True)
    b.add_argument("--swissprot-truth", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--decoy", action="store_true")
    b.add_argument("--species-mya", type=str, required=True,
                   help="JSON object, e.g. '{\"mouse\": 100}'")
    a = p.parse_args(argv)

    if a.cmd == "reduce":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in sorted(a.kmerseek_dir.glob("human_vs_*.regions.parquet")):
            kept, summ = reduce_file(f, a.evalue_max)
            kept.write_parquet(a.out_dir / f.name.replace(".regions.parquet", ".kept.parquet"))
            rows.append(summ)
            print(f"{f.name}: {summ['n_kept']:_} of {summ['n_regions']:_} kept", flush=True)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(a.out_dir / "reduce_summary.parquet")
        return 0

    files = sorted(a.reduced_dir.glob("*.kept.parquet"))
    calls = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    truth = load_truth(a.pfam_truth, a.swissprot_truth)
    table, long = build_table(calls, truth, json.loads(a.species_mya), a.decoy)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "region_table_decoy" if a.decoy else "region_table"
    table.write_parquet(a.out_dir / f"{stem}.parquet")
    if long.height:
        long.write_parquet(a.out_dir / f"{stem}_truth_pairs.parquet")
    print(f"{stem}: {table.height:_} rows, {table['accession'].n_unique():_} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
