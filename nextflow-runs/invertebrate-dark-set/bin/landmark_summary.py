#!/usr/bin/env python3
"""One species' landmark pairs across every arm.

A landmark pair is a query in this species and a target in the reference that the run
follows by name (params.landmarks; human BCL2 P10415 and C. elegans CED-9 P41958 are the
default, each way round). For each pair this reports whether the query is in the dark set,
what each sequence arm's best hit on the target was, and for every kmerseek arm whether any
region joined the two, with the best region's E-value, region score and coordinates.

A target of `*` is every target: the query is followed on its own (the Botryllus
histocompatibility factor has no named partner), and each arm reports how many targets
and regions it put on the query, and its best target.

The kmerseek side reads the per-search landmark CSVs the search task writes: every row of
a landmark query, all columns. The sequence side reads the arms' hit tables.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl

HIT_COLS = ["query", "target", "tstart", "tend", "bits", "evalue"]
# <chunk>.<alphabet>.k<ksize>.s<scaled>.lc<true|false>.<exact|extend-c<C>>.landmarks.csv
NAME = re.compile(r"^(?P<chunk>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.s(?P<scaled>\d+)"
                  r"\.lc(?P<lc>true|false)\.(?P<ext>exact|extend-c[0-9.]+)\.landmarks\.csv$")


def arm_label(alphabet: str, ksize: int, scaled: int, ext: str) -> str:
    parts = [f"{alphabet} k{ksize}"]
    if scaled != 1:
        parts.append(f"scaled {scaled}")
    if ext != "exact":
        parts.append(f"extend C={ext[len('extend-c'):]}")
    return " ".join(parts)


def arm_of_hits(path: Path) -> str:
    return path.name.rsplit(".", 3)[-3]  # <species>.<chunk>.<arm>.tsv.gz


def read_hits(path: Path) -> pl.DataFrame:
    """An arm's hit table, tagged with the arm; an arm that found nothing (an empty file,
    or a gzip of nothing) is an empty frame, not an error."""
    empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in HIT_COLS})
    if path.stat().st_size == 0:
        return empty.with_columns(pl.lit(arm_of_hits(path)).alias("arm"))
    try:
        df = pl.read_csv(path, separator="\t", has_header=False, infer_schema_length=0,
                         truncate_ragged_lines=True, new_columns=HIT_COLS)
    except pl.exceptions.NoDataError:
        df = empty
    return df.with_columns(pl.lit(arm_of_hits(path)).alias("arm"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--pairs", default="", help="query:target, comma separated; may be empty")
    ap.add_argument("--dark", type=Path, required=True)
    ap.add_argument("--hits", type=Path, nargs="*", default=[])
    ap.add_argument("--landmark-csvs", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    args = ap.parse_args()

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p.strip()]
    dark = set(pl.read_parquet(args.dark)["accession"].to_list()) if pairs else set()

    hits = (pl.concat([read_hits(p) for p in args.hits]) if args.hits and pairs
            else pl.DataFrame({c: [] for c in HIT_COLS + ["arm"]}))
    hits = hits.with_columns(pl.col("evalue").cast(pl.Float64, strict=False),
                             pl.col("bits").cast(pl.Float64, strict=False))

    # kmerseek: one frame of every landmark row, tagged with its arm.
    frames = []
    for f in args.landmark_csvs:
        m = NAME.match(f.name)
        if not m:
            raise SystemExit(f"could not parse the arm from {f.name}")
        df = pl.read_csv(f, infer_schema_length=0)
        if df.height == 0:
            continue
        frames.append(df.with_columns(
            pl.lit(m["alphabet"]).alias("alphabet"), pl.lit(int(m["ksize"])).alias("ksize"),
            pl.lit(int(m["scaled"])).alias("scaled"), pl.lit(m["ext"]).alias("extension"),
            pl.lit(m["lc"] == "true").alias("low_complexity_mask")))
    rows = pl.concat(frames, how="diagonal_relaxed") if frames else None
    arms = sorted({(m["alphabet"], int(m["ksize"]), int(m["scaled"]), m["ext"])
                   for m in (NAME.match(f.name) for f in args.landmark_csvs) if m})

    out_pairs = []
    arm_rows = []
    for query, target in pairs:
        any_target = target == "*"
        seq = {}
        for arm in ["phmmer", "jackhmmer", "mmseqs2"]:
            h = hits.filter((pl.col("arm") == arm) & (pl.col("query") == query))
            if not any_target:
                h = h.filter(pl.col("target") == target)
            h = h.sort("evalue")
            seq[arm] = ({"found": True, "best_evalue": h["evalue"][0], "bits": h["bits"][0],
                         "tstart": h["tstart"][0], "tend": h["tend"][0],
                         "best_target": h["target"][0], "n_targets": h["target"].n_unique()}
                        if h.height else {"found": False, "n_targets": 0})
        km = []
        for alphabet, ksize, scaled, ext in arms:
            found = False
            best = None
            n = 0
            n_targets = 0
            if rows is not None:
                r = rows.filter((pl.col("query_name") == query)
                                & (pl.col("alphabet") == alphabet) & (pl.col("ksize") == ksize)
                                & (pl.col("scaled") == scaled) & (pl.col("extension") == ext))
                if not any_target:
                    r = r.filter(pl.col("target_name") == target)
                n = r.height
                n_targets = r["target_name"].n_unique() if n else 0
                if n:
                    found = True
                    r = r.with_columns(pl.col("region_evalue").cast(pl.Float64, strict=False),
                                       pl.col("region_poisson_score").cast(pl.Float64, strict=False))
                    # Best region: smallest finite E-value if the arm has one, else the
                    # highest region score.
                    finite = r.filter(pl.col("region_evalue").is_finite())
                    b = (finite.sort("region_evalue").row(0, named=True) if finite.height
                         else r.sort("region_poisson_score", descending=True).row(0, named=True))
                    best = {k: b.get(k) for k in ["region_evalue", "region_poisson_score",
                                                  "region_ka_bits", "region_start", "region_end",
                                                  "target_start", "target_end", "region_length",
                                                  "region_n_mismatches"]}
                    best["best_target"] = b.get("target_name")
            row = {"species": args.species, "query": query, "target": target,
                   "alphabet": alphabet, "ksize": ksize, "scaled": scaled, "extension": ext,
                   "arm": arm_label(alphabet, ksize, scaled, ext), "found": found,
                   "n_regions": n, "n_targets": n_targets,
                   "best_target": (best or {}).get("best_target"),
                   **{k: (best or {}).get(k) for k in [
                       "region_evalue", "region_poisson_score", "region_ka_bits",
                       "region_start", "region_end", "target_start", "target_end",
                       "region_length", "region_n_mismatches"]}}
            km.append(row)
            arm_rows.append(row)
        out_pairs.append({"query": query, "target": target, "query_dark": query in dark,
                          "sequence_arms": seq, "kmerseek_arms": km,
                          "n_kmerseek_arms_found": sum(1 for r in km if r["found"]),
                          "n_kmerseek_arms": len(km)})

    schema = {"species": pl.Utf8, "query": pl.Utf8, "target": pl.Utf8, "alphabet": pl.Utf8,
              "ksize": pl.Int32, "scaled": pl.Int32, "extension": pl.Utf8, "arm": pl.Utf8,
              "found": pl.Boolean, "n_regions": pl.Int64, "n_targets": pl.Int64,
              "best_target": pl.Utf8, "region_evalue": pl.Float64,
              "region_poisson_score": pl.Float64, "region_ka_bits": pl.Float64,
              "region_start": pl.Utf8, "region_end": pl.Utf8, "target_start": pl.Utf8,
              "target_end": pl.Utf8, "region_length": pl.Utf8, "region_n_mismatches": pl.Utf8}
    pl.DataFrame(arm_rows, schema=schema).write_parquet(args.out, compression="zstd")
    summary = {"species": args.species, "pairs": out_pairs}
    args.summary_out.write_text(json.dumps(summary, indent=2, default=str))
    for p in out_pairs:
        print(f"{p['query']} -> {p['target']}: query dark={p['query_dark']}; "
              f"sequence arms found it: {[a for a, v in p['sequence_arms'].items() if v['found']]}; "
              f"kmerseek arms: {p['n_kmerseek_arms_found']} of {p['n_kmerseek_arms']}")


if __name__ == "__main__":
    main()
