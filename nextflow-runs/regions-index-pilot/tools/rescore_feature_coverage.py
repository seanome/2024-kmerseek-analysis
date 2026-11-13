#!/usr/bin/env python3
"""A second ranking: calls ranked by how much of the target feature they cover.

Added 2026-10-13 at Olga's request, next to the E-value ranking, never instead of it. On
the regions index every entry is a known feature, so a natural score for a call is the
share of that feature's own residues (its flanks left out) that the call covers on the
target side. The same score is computed on the whole-protein index, once per target
feature the call overlaps, so the two indexes are ranked alike.

Per set (tool x index x decoy window x setting):
  1. Every call, target or decoy, is paired with each target feature whose residues it
     covers: on the regions index the one feature its entry was cut around (position from
     the entry name), on the whole-protein index every feature of the target protein. A
     decoy is paired with its source's features (a window-shuffled decoy keeps its
     source's coordinates).
  2. feature coverage = target-feature residues inside the call's target span / feature
     length.
  3. Pairs are ranked by feature coverage, highest first, then by E-value, lowest first.
     The threshold is the longest top of that list in which decoy pairs / target pairs
     <= 5%.
  4. A target pair above the threshold finds a query feature by the rules score_calls.py
     uses (same feature type; landing >= 0.5 for the whole call and for the aligned piece).

The composition classifier is left out: its "target span" is the whole entry, so every call
would cover its feature fully.

This script runs after the pipeline, outside it: the pipeline's bin/ is part of every
task's cache key, so changing bin/score_calls.py would rerun every search. It imports
functions from bin/score_calls.py without changing them, and finds each set's call files
through the run's trace.

    python3 tools/rescore_feature_coverage.py --launch-dir runs/full --results <outdir> \\
        --out-dir <outdir>/scored_feature_coverage
    python3 bin/compare_indexes.py --scored <outdir>/scored_feature_coverage/* \\
        --truth <outdir>/queries/truth.parquet \\
        --feature-identity <outdir>/comparison/feature_identity.tsv
"""

import argparse
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from score_calls import CALL_COLUMNS, CALL_TYPES, aligned_piece, clean  # noqa: E402

TARGET_RE = re.compile(r"--target (\S+)\.kmerseek\.rocksdb")
MMSEQS_LOG_RE = re.compile(r"mmseqs\.(\S+)\.w(\d+)\.log$")


def sets_from_trace(launch_dir: Path) -> dict[str, list[Path]]:
    """{label: [call files]} for every finished kmerseek and MMseqs2 search."""
    out: dict[str, list[Path]] = defaultdict(list)
    seen = set()
    for trace in sorted(launch_dir.glob("*.trace.txt")):
        t = pl.read_csv(trace, separator="\t", infer_schema_length=0)
        t = t.filter(
            pl.col("status").is_in(["COMPLETED", "CACHED"])
            & pl.col("process").is_in(["kmerseekSearch", "mmseqsSearch"])
        )
        for proc, h in t.select("process", "hash").iter_rows():
            if h in seen:
                continue
            seen.add(h)
            hits = glob.glob(
                str(launch_dir / "work" / h.split("/")[0] / (h.split("/")[1] + "*"))
            )
            if len(hits) != 1:
                print(
                    f"[coverage] {proc} {h}: {len(hits)} work dirs, skipped",
                    file=sys.stderr,
                )
                continue
            wd = Path(hits[0])
            if proc == "kmerseekSearch":
                script = (wd / ".command.sh").read_text()
                m = TARGET_RE.search(script)
                name, window, slug = m.group(1).split(".", 2)
                kind = (
                    "whole"
                    if name.startswith("whole")
                    else ("pfam" if name.startswith("pfam") else "regions")
                )
                if kind != "whole" and "--ka-r-database" in script:
                    kind += "shared"
                label = f"kmerseek.{kind}.{window}.{slug}"
                out[label].extend(sorted(wd.glob("*.calls.tsv.gz")))
            else:
                logs = [
                    p for p in wd.glob("mmseqs.*.log") if MMSEQS_LOG_RE.search(p.name)
                ]
                name, window = MMSEQS_LOG_RE.search(logs[0].name).groups()
                kind = (
                    "whole"
                    if name.startswith("whole")
                    else ("pfam" if name.startswith("pfam") else "regions")
                )
                out[f"mmseqs2.{kind}.w{window}.mmseqs2"].append(wd / "calls.tsv.gz")
    return out


def read_calls(files: list[Path]) -> tuple[pl.DataFrame, bool]:
    nofit = False
    frames = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            nofit |= fh.readline().startswith("#nofit")
        frames.append(
            pl.read_csv(
                f,
                separator="\t",
                comment_prefix="#",
                quote_char=None,
                schema_overrides=CALL_TYPES,
            ).select(CALL_COLUMNS)
        )
    calls = pl.concat(frames).with_row_index("call_id")
    return (
        calls.with_columns(
            pl.col("target").str.starts_with("DECOY_").alias("is_decoy")
        ),
        nofit,
    )


def pair_with_target_features(
    calls: pl.DataFrame, kind: str, tfeat: pl.DataFrame, flank: int
) -> pl.DataFrame:
    """One row per (call, target feature it covers), with feature coverage."""
    if kind == "whole":
        j = calls.with_columns(
            pl.col("target").str.replace(r"^DECOY_", "").alias("_src")
        ).join(
            tfeat.select(
                pl.col("accession").alias("_src"),
                pl.col("feature_type").alias("hit_type"),
                pl.col("description").alias("_desc"),
                pl.col("start").cast(pl.Int64).alias("feat_tstart"),
                pl.col("end").cast(pl.Int64).alias("feat_tend"),
            ),
            on="_src",
        )
        j = j.with_columns(
            pl.col("_desc").map_elements(clean, return_dtype=pl.Utf8).alias("hit_desc")
        ).drop("_src", "_desc")
    else:
        parts = pl.col("target").str.replace(r"^DECOY_", "").str.split("|")
        se = parts.list.get(3).str.split("-")
        start = se.list.get(0).cast(pl.Int64)
        end = se.list.get(1).cast(pl.Int64)
        cut_start = pl.max_horizontal(pl.lit(1), start - flank)
        j = calls.with_columns(
            parts.list.get(1).alias("hit_type"),
            parts.list.get(2).alias("hit_desc"),
            (start - cut_start + 1).alias("feat_tstart"),
            (end - cut_start + 1).alias("feat_tend"),
        )
    ov = (
        pl.min_horizontal("tend", "feat_tend")
        - pl.max_horizontal("tstart", "feat_tstart")
        + 1
    ).clip(lower_bound=0)
    return j.with_columns(
        (ov / (pl.col("feat_tend") - pl.col("feat_tstart") + 1)).alias(
            "feature_coverage"
        )
    ).filter(pl.col("feature_coverage") > 0)


def coverage_threshold(pairs: pl.DataFrame, max_rate: float) -> tuple[int, dict]:
    """Rank pairs by (coverage desc, E-value asc); the longest top with decoys/targets <= rate.

    Returns the number of ranked groups kept and the summary. Pairs tied on both keys are
    kept or dropped together.
    """
    g = (
        pairs.group_by("feature_coverage", "evalue")
        .agg(
            pl.col("is_decoy").sum().alias("d"),
            (~pl.col("is_decoy")).sum().alias("t"),
        )
        .sort(["feature_coverage", "evalue"], descending=[True, False])
        .with_columns(
            pl.col("d").cum_sum().alias("cd"), pl.col("t").cum_sum().alias("ct")
        )
        .with_row_index("rank")
    )
    ok = g.filter((pl.col("ct") > 0) & (pl.col("cd") <= max_rate * pl.col("ct")))
    if ok.height == 0:
        return -1, {"threshold": None, "n_target": 0, "n_decoy": 0}
    last = ok.sort("rank").tail(1).row(0, named=True)
    return int(last["rank"]), {
        "threshold": f"coverage >= {last['feature_coverage']:.3f}, E <= {last['evalue']:.3g}",
        "n_target": int(last["ct"]),
        "n_decoy": int(last["cd"]),
    }


def score_set(label, files, truth, tfeat, flank, max_rate, min_landing):
    kind = label.split(".")[1]
    kind = "whole" if kind == "whole" else "regions"
    calls, nofit = read_calls(files)
    pairs = pair_with_target_features(calls, kind, tfeat, flank)
    keep_rank, thr = (-1, {"threshold": None, "n_target": 0, "n_decoy": 0})
    if not nofit:
        keep_rank, thr = coverage_threshold(pairs, max_rate)
    thr.update(
        {
            "label": label,
            "nofit": nofit,
            "ranking": "feature coverage, then E-value",
            "pairs_total": pairs.height,
            "decoy_pairs_total": int(pairs["is_decoy"].sum()),
            "max_decoy_rate": max_rate,
        }
    )
    ranked = pairs.join(
        pairs.group_by("feature_coverage", "evalue")
        .agg()
        .sort(["feature_coverage", "evalue"], descending=[True, False])
        .with_row_index("rank"),
        on=["feature_coverage", "evalue"],
    ).filter(~pl.col("is_decoy") & (pl.col("rank") <= keep_rank))
    q = ranked.join(
        truth.select(
            "truth_id",
            pl.col("accession").alias("query"),
            pl.col("feature_type").alias("hit_type"),
            pl.col("start").alias("fstart"),
            pl.col("end").alias("fend"),
            "desc_clean",
        ),
        on=["query", "hit_type"],
    )
    pieces = [
        aligned_piece(qs, ts, qa, ta, fs, fe)
        for qs, ts, qa, ta, fs, fe in q.select(
            "qstart", "tstart", "qseq", "tseq", "feat_tstart", "feat_tend"
        ).iter_rows()
    ]
    q = q.with_columns(
        pl.Series("aq_start", [p[0] for p in pieces], dtype=pl.Int64),
        pl.Series("aq_end", [p[1] for p in pieces], dtype=pl.Int64),
    )
    ov = (
        pl.min_horizontal("qend", "fend") - pl.max_horizontal("qstart", "fstart") + 1
    ).clip(lower_bound=0)
    ova = (
        pl.min_horizontal("aq_end", "fend")
        - pl.max_horizontal("aq_start", "fstart")
        + 1
    ).clip(lower_bound=0)
    q = q.with_columns(
        (ov / (pl.col("qend") - pl.col("qstart") + 1) >= min_landing).alias("ok_call"),
        (
            (ova / (pl.col("aq_end") - pl.col("aq_start") + 1)).fill_null(0.0)
            >= min_landing
        ).alias("ok_aligned"),
        (pl.col("hit_desc") == pl.col("desc_clean")).alias("name_match"),
    )
    found = q.group_by("truth_id").agg(
        pl.col("ok_call").any().alias("found"),
        pl.col("ok_aligned").any().alias("found_aligned"),
        (pl.col("ok_call") & pl.col("name_match")).any().alias("found_name"),
        (pl.col("ok_aligned") & pl.col("name_match")).any().alias("found_name_aligned"),
    )
    out = (
        truth.select("truth_id")
        .join(found, on="truth_id", how="left")
        .with_columns(
            pl.col(
                "found", "found_aligned", "found_name", "found_name_aligned"
            ).fill_null(False),
            pl.lit(label).alias("label"),
        )
        .sort("truth_id")
    )
    thr["features_found"] = int(out["found"].sum())
    thr["features_found_aligned"] = int(out["found_aligned"].sum())
    thr["features_total"] = out.height
    return out, thr


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--launch-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True, help="the run's --outdir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--searched", type=Path, help="default: <results>/queries/queries.fasta"
    )
    ap.add_argument("--flank", type=int, default=21, help="k_max - 1 of the run")
    ap.add_argument("--max-decoy-rate", type=float, default=0.05)
    ap.add_argument("--min-landing", type=float, default=0.5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    searched_fa = args.searched or args.results / "queries" / "queries.fasta"
    with open(searched_fa) as fh:
        searched = [ln[1:].split()[0] for ln in fh if ln.startswith(">")]
    truth = (
        pl.read_parquet(args.results / "queries" / "truth.parquet")
        .filter(pl.col("accession").is_in(searched))
        .with_columns(
            pl.col("description")
            .map_elements(clean, return_dtype=pl.Utf8)
            .alias("desc_clean")
        )
    )
    tfeat = pl.read_parquet(
        args.results / "swissprot" / "sprot_features.parquet"
    ).filter(~pl.col("is_mammal"))
    sets = sets_from_trace(args.launch_dir)
    print(f"[coverage] {len(sets)} sets from {args.launch_dir}", file=sys.stderr)
    for label, files in sorted(sets.items()):
        out, thr = score_set(
            label,
            files,
            truth,
            tfeat,
            args.flank,
            args.max_decoy_rate,
            args.min_landing,
        )
        out.write_parquet(args.out_dir / f"{label}.features.parquet")
        (args.out_dir / f"{label}.threshold.json").write_text(json.dumps(thr, indent=2))
        print(
            f"[coverage] {label}: {len(files)} files, {thr['n_target']} target pairs at "
            f"{thr['threshold']}, found {thr['features_found_aligned']} (aligned piece)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
