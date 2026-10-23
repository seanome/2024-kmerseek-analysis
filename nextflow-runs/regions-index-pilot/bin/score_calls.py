#!/usr/bin/env python3
"""Score one set of calls: the 5% decoy threshold, and which query features it finds.

A set is one tool x index x decoy window x kmerseek setting. Calls arrive in the shared
format every search step writes (calls.tsv.gz):

  query, qstart, qend, target, tstart, tend, evalue, qseq, tseq

with coordinates 1-based and inclusive on both proteins.

Rules (PREDICTIONS.md):
- A call to a target whose name starts with DECOY_ is a decoy call.
- Threshold: calls are ranked by E-value; the threshold is the largest E-value at which
  decoy calls / target calls <= --max-decoy-rate among calls at or below it.
- The hit entry's feature types:
    regions index         the type in the entry's name (accession|type|description|start-end)
    whole-protein index   the types of the target protein's features whose interval,
                          widened by --flank residues each side, overlaps the target side
                          of the call. The widening mirrors the regions index, whose
                          entries carry the same flank.
- A call is correct for a query feature when the feature's type is one of the hit's types
  and landing = (call residues inside the feature) / (call length) >= --min-landing.
  Coverage = (feature residues inside the call) / (feature length) is reported.
- A second landing rule, added 2026-10-07 after the 20-query test and reported next to the
  first (found_aligned): the call is cut down to the query residues aligned to the target
  feature's own residues (no flank), and landing is judged on that piece. Without it a long
  alignment on the whole-protein index spans several domains, lands under half in any one
  of them, and counts as wrong, while the same alignment on the regions index is confined to
  one feature by the entry's length. The alignment strings (qseq, tseq; equal length with
  gaps for MMseqs2, ungapped for kmerseek) give the mapping. The composition classifier has
  no alignment, so its piece is the whole window.
- --annotated-targets-only (whole-protein index; added 2026-10-13, Olga's "reading 2"):
  before the threshold, keep only calls whose target side lies inside one target feature
  widened by --flank, the span a regions-index entry covers. A decoy call is judged by its
  source protein's features (a window-shuffled decoy keeps its source's coordinates). The
  index and its E-values are unchanged, so this isolates the effect of searching only
  annotated targets from the effect of a smaller index.
- --all-calls (the composition classifier): no threshold; every target call is scored and
  written with its rank to <prefix>.call_hits.parquet, so the comparison can give the
  classifier as many calls as kmerseek makes at its 5% threshold.
- Secondary: a name match, when the hit feature's description (cleaned the way
  extract_regions.py cleans it) equals the query feature's.

Outputs:
  <prefix>.features.parquet  one row per query feature: found (at the threshold),
                             found_name, best_landing, best_coverage, best_evalue,
                             best_target, best_qstart/qend/tstart/tend, best_qseq, best_tseq
  <prefix>.threshold.json    the threshold and call counts
  <prefix>.call_hits.parquet with --all-calls: call_id, rank (by score, target calls only),
                             truth_id found by that call under each landing rule
"""

import argparse
import gzip
import io
import json
import re
import sys
from pathlib import Path

import polars as pl

WS_RE = re.compile(r"\s+")
CALL_COLUMNS = [
    "query",
    "qstart",
    "qend",
    "target",
    "tstart",
    "tend",
    "evalue",
    "qseq",
    "tseq",
]


CALL_TYPES = {
    "query": pl.Utf8,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "target": pl.Utf8,
    "tstart": pl.Int64,
    "tend": pl.Int64,
    "evalue": pl.Float64,
    "qseq": pl.Utf8,
    "tseq": pl.Utf8,
}


def clean(desc: str | None) -> str:
    d = WS_RE.sub("_", (desc or "").strip()).replace("|", "/").replace(",", ";")
    return d or "none"


def decoy_threshold(target_ev: pl.Series, decoy_ev: pl.Series, max_rate: float) -> dict:
    """Largest E-value E with (decoys <= E) / (targets <= E) <= max_rate.

    Takes only the E-values of the target calls and of the decoy calls. Both are sorted,
    and the counts at or below each E-value are found by binary search.
    """
    for values in (target_ev, decoy_ev):
        if values.null_count() or not values.is_finite().all() or (values < 0).any():
            raise ValueError("E-values must be finite, nonnegative numbers")
    if target_ev.len() + decoy_ev.len() == 0:
        return {"threshold": None, "n_target": 0, "n_decoy": 0}
    t, d = target_ev.sort(), decoy_ev.sort()
    best = None
    # Every E-value a call has is a candidate threshold, decoy or target.
    for e in (t, d):
        g = pl.DataFrame(
            {
                "evalue": e,
                "ct": t.search_sorted(e, side="right"),
                "cd": d.search_sorted(e, side="right"),
            }
        )
        ok = g.filter((pl.col("ct") > 0) & (pl.col("cd") <= max_rate * pl.col("ct")))
        if ok.height:
            row = ok.tail(1).row(0, named=True)
            if best is None or row["evalue"] > best["evalue"]:
                best = row
        del g, ok
    if best is None:
        return {"threshold": None, "n_target": 0, "n_decoy": 0}
    return {
        "threshold": float(best["evalue"]),
        "n_target": int(best["ct"]),
        "n_decoy": int(best["cd"]),
    }


def hit_types_regions(calls: pl.DataFrame, flank: int) -> pl.DataFrame:
    """The type in the entry name, and the feature's own residues in entry coordinates.

    An entry holds protein residues cut_start..cut_end with cut_start = max(1, start -
    flank), so the feature sits at (start - cut_start + 1)..(end - cut_start + 1).
    """
    parts = pl.col("target").str.split("|")
    se = parts.list.get(3).str.split("-")
    start = se.list.get(0).cast(pl.Int64)
    end = se.list.get(1).cast(pl.Int64)
    cut_start = pl.max_horizontal(pl.lit(1), start - flank)
    return calls.with_columns(
        parts.list.get(1).alias("hit_type"),
        parts.list.get(2).alias("hit_desc"),
        (start - cut_start + 1).alias("feat_tstart"),
        (end - cut_start + 1).alias("feat_tend"),
    )


def aligned_piece(
    qstart: int, tstart: int, qseq: str, tseq: str, fts: int, fte: int
) -> tuple[int | None, int | None]:
    """First and last query residue aligned to target residues fts..fte, or (None, None)."""
    if len(qseq) != len(tseq):
        return None, None
    q, t = qstart - 1, tstart - 1
    lo = hi = None
    for a, b in zip(qseq, tseq):
        if a != "-":
            q += 1
        if b != "-":
            t += 1
        if a != "-" and b != "-" and fts <= t <= fte:
            lo = q if lo is None else lo
            hi = q
    return lo, hi


def hit_types_whole(
    calls: pl.DataFrame, tfeat: pl.DataFrame, flank: int
) -> pl.DataFrame:
    """One row per (call, target feature) whose widened interval overlaps the call."""
    j = calls.join(
        tfeat.select(
            pl.col("accession").alias("target"),
            pl.col("feature_type").alias("hit_type"),
            pl.col("description").alias("hit_desc_raw"),
            (pl.col("start") - flank).alias("_fs"),
            (pl.col("end") + flank).alias("_fe"),
            pl.col("start").cast(pl.Int64).alias("feat_tstart"),
            pl.col("end").cast(pl.Int64).alias("feat_tend"),
        ),
        on="target",
        how="inner",
    ).filter((pl.col("_fs") <= pl.col("tend")) & (pl.col("_fe") >= pl.col("tstart")))
    return j.with_columns(
        pl.col("hit_desc_raw")
        .map_elements(clean, return_dtype=pl.Utf8)
        .alias("hit_desc")
    ).drop("_fs", "_fe", "hit_desc_raw")


def annotated_targets_only(
    calls: pl.DataFrame, tfeat: pl.DataFrame, flank: int
) -> pl.DataFrame:
    """Calls whose target side lies inside one target feature widened by `flank`."""
    spans = tfeat.select(
        pl.col("accession").alias("_src"),
        (pl.col("start") - flank).alias("_fs"),
        (pl.col("end") + flank).alias("_fe"),
    )
    keep = (
        calls.with_row_index("_row")
        .with_columns(pl.col("target").str.replace(r"^DECOY_", "").alias("_src"))
        .join(spans, on="_src", how="inner")
        .filter((pl.col("_fs") <= pl.col("tstart")) & (pl.col("tend") <= pl.col("_fe")))
        .select("_row")
        .unique()
    )
    return calls.with_row_index("_row").join(keep, on="_row", how="semi").drop("_row")


# Uncompressed bytes of call text parsed at a time. A label's calls run to 27 GB gzipped,
# about 100 GB of text, and all but a few thousand of them are dropped after the threshold
# is counted, so the files are read in pieces rather than whole (2026-10-16: reading them
# whole needed 18-32 GB of memory per GB of gzipped calls).
CHUNK_BYTES = 1 << 28


def read_calls(path: Path):
    """Yield the calls of one file as DataFrames of about CHUNK_BYTES of text each.

    Each piece ends at a line break and is parsed exactly as the whole file used to be,
    with the file's header line put in front of every piece after the first.
    """

    def parse(text: bytes) -> pl.DataFrame:
        return (
            pl.read_csv(
                io.BytesIO(text),
                separator="\t",
                comment_prefix="#",
                # Fixed types: a set with no fit writes only a header, and polars would
                # read every column of a header-only file as text (2026-10-15: hp_kyte k22
                # on the whole-protein index stopped the full run this way).
                schema_overrides=CALL_TYPES,
                quote_char=None,
            )
            .select(CALL_COLUMNS)
            .with_columns(pl.col("target").str.starts_with("DECOY_").alias("is_decoy"))
        )

    header = None
    rest = b""
    with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK_BYTES)
            text = rest + block
            if block:
                end = text.rfind(b"\n") + 1
                if end == 0:
                    rest = text
                    continue
                text, rest = text[:end], text[end:]
            if header is None:
                start = 0  # the header is the first line that is not a comment
                while text.startswith(b"#", start):
                    start = text.find(b"\n", start) + 1 or len(text)
                end = text.find(b"\n", start) + 1 or len(text)
                header = text[start:end]
                yield parse(text)
            elif text:
                yield parse(header + text)
            if not block:
                return


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--calls",
        type=Path,
        nargs="+",
        required=True,
        help="one or more call files (one per query chunk), gzipped or not",
    )
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--index-kind", choices=["regions", "whole"], required=True)
    ap.add_argument(
        "--target-features",
        type=Path,
        help="sprot_features.parquet; needed for the whole-protein index",
    )
    ap.add_argument("--flank", type=int, required=True)
    ap.add_argument("--max-decoy-rate", type=float, default=0.05)
    ap.add_argument("--min-landing", type=float, default=0.5)
    ap.add_argument(
        "--searched",
        type=Path,
        required=True,
        help="the FASTA of the queries searched; truth is limited to these proteins",
    )
    ap.add_argument("--annotated-targets-only", action="store_true")
    ap.add_argument("--all-calls", action="store_true")
    ap.add_argument("--label", required=True, help="tool.index.window.setting")
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    # A search whose index holds no Karlin-Altschul fit for its setting writes a #nofit
    # line and no rows; one such chunk marks the whole set as having no E-values.
    nofit = False
    for f in args.calls:
        with gzip.open(f, "rt") if f.suffix == ".gz" else open(f) as fh:
            nofit |= fh.readline().startswith("#nofit")

    tfeat = pl.read_parquet(args.target_features) if args.target_features else None

    # First pass: check each call and keep only its E-value, apart for target and decoy
    # calls: 8 bytes a call, where holding whole calls took 18-32 GB per GB of call files.
    target_ev, decoy_ev, n_bad, first_bad, n_before = [], [], 0, None, 0
    for f in args.calls:
        for piece in read_calls(f):
            bad = piece.filter(
                (pl.col("qstart") < 1)
                | (pl.col("qend") < pl.col("qstart"))
                | pl.col("evalue").is_null()
                | ~pl.col("evalue").is_finite()
                | (pl.col("evalue") < 0)
            )
            if bad.height and first_bad is None:
                first_bad = bad.row(0)
            n_bad += bad.height
            n_before += piece.height
            if args.annotated_targets_only:
                piece = annotated_targets_only(piece, tfeat, args.flank)
            target_ev.append(piece.filter(~pl.col("is_decoy"))["evalue"])
            decoy_ev.append(piece.filter(pl.col("is_decoy"))["evalue"])
    if n_bad:
        sys.exit(f"{n_bad} calls with bad coordinates or no E-value, e.g. {first_bad}")
    target_ev = pl.concat([pl.Series("evalue", [], pl.Float64)] + target_ev)
    decoy_ev = pl.concat([pl.Series("evalue", [], pl.Float64)] + decoy_ev)
    n_calls, n_decoy = target_ev.len() + decoy_ev.len(), decoy_ev.len()
    if args.annotated_targets_only:
        print(
            f"[annotated targets only] kept {n_calls} of {n_before} calls",
            file=sys.stderr,
        )

    thr = (
        {"threshold": None, "n_target": 0, "n_decoy": 0}
        if nofit
        else decoy_threshold(target_ev, decoy_ev, args.max_decoy_rate)
    )
    thr["nofit"] = nofit
    thr.update(
        {
            "label": args.label,
            "calls_total": n_calls,
            "decoy_calls_total": n_decoy,
            "max_decoy_rate": args.max_decoy_rate,
        }
    )
    del target_ev, decoy_ev

    with open(args.searched) as fh:
        searched = [ln[1:].split()[0] for ln in fh if ln.startswith(">")]
    truth = pl.read_parquet(args.truth).filter(pl.col("accession").is_in(searched))
    truth = truth.with_columns(
        pl.col("description")
        .map_elements(clean, return_dtype=pl.Utf8)
        .alias("desc_clean")
    )
    # Only target calls at or below the threshold can find a feature.
    cut = thr["threshold"] if thr["threshold"] is not None else -1.0
    if args.all_calls:
        cut = float("inf")
    # Second pass: keep only those calls, in file order as before. The annotated-targets
    # rule is judged per call, so applying it after the E-value cut keeps the same calls.
    keep = ~pl.col("is_decoy") & (pl.col("evalue") <= cut)

    def kept_calls(piece: pl.DataFrame) -> pl.DataFrame:
        if args.annotated_targets_only:
            return annotated_targets_only(piece, tfeat, args.flank)
        return piece

    real = (
        pl.concat(
            [pl.DataFrame(schema={**CALL_TYPES, "is_decoy": pl.Boolean})]
            + [kept_calls(p.filter(keep)) for f in args.calls for p in read_calls(f)]
        )
        .sort("evalue", "query", "qstart", "target")
        .with_row_index("call_id")
    )
    if args.index_kind == "regions":
        typed = hit_types_regions(real, args.flank)
    else:
        typed = hit_types_whole(real, tfeat, args.flank)

    # Pair each typed call with every query feature of the same type on the same protein.
    pairs = typed.join(
        truth.select(
            "truth_id",
            pl.col("accession").alias("query"),
            pl.col("feature_type").alias("hit_type"),
            pl.col("start").alias("fstart"),
            pl.col("end").alias("fend"),
            "desc_clean",
        ),
        on=["query", "hit_type"],
        how="inner",
    )
    no_alignment = args.label.split(".")[0] == "composition"
    pieces = [
        (None, None) if no_alignment else aligned_piece(qs, ts, qa, ta, fs, fe)
        for qs, ts, qa, ta, fs, fe in pairs.select(
            "qstart", "tstart", "qseq", "tseq", "feat_tstart", "feat_tend"
        ).iter_rows()
    ]
    pairs = pairs.with_columns(
        pl.Series("aq_start", [p[0] for p in pieces], dtype=pl.Int64),
        pl.Series("aq_end", [p[1] for p in pieces], dtype=pl.Int64),
    )
    if no_alignment:
        pairs = pairs.with_columns(
            pl.col("qstart").cast(pl.Int64).alias("aq_start"),
            pl.col("qend").cast(pl.Int64).alias("aq_end"),
        )
    ov = (
        pl.min_horizontal("qend", "fend") - pl.max_horizontal("qstart", "fstart") + 1
    ).clip(lower_bound=0)
    ova = (
        pl.min_horizontal("aq_end", "fend")
        - pl.max_horizontal("aq_start", "fstart")
        + 1
    ).clip(lower_bound=0)
    pairs = pairs.with_columns(
        (ov / (pl.col("qend") - pl.col("qstart") + 1)).alias("landing"),
        (ov / (pl.col("fend") - pl.col("fstart") + 1)).alias("coverage"),
        (ova / (pl.col("aq_end") - pl.col("aq_start") + 1))
        .fill_null(0.0)
        .alias("landing_aligned"),
        (pl.col("hit_desc") == pl.col("desc_clean")).alias("name_match"),
    )
    if args.all_calls:
        # call_id is the rank: target calls sorted by score, best first.
        hits = pl.concat(
            [
                pairs.filter(pl.col("landing") >= args.min_landing)
                .select("call_id", "truth_id")
                .with_columns(pl.lit("whole call").alias("landing_rule")),
                pairs.filter(pl.col("landing_aligned") >= args.min_landing)
                .select("call_id", "truth_id")
                .with_columns(pl.lit("aligned piece").alias("landing_rule")),
            ]
        ).unique()
        hits.rename({"call_id": "rank"}).write_parquet(
            f"{args.prefix}.call_hits.parquet"
        )
    aligned_found = (
        pairs.filter(pl.col("landing_aligned") >= args.min_landing)
        .group_by("truth_id")
        .agg(
            pl.col("landing_aligned").max().alias("best_landing_aligned"),
            pl.col("name_match").any().alias("found_name_aligned"),
        )
    )
    pairs = pairs.filter(pl.col("landing") >= args.min_landing).with_columns(
        pl.lit(True).alias("passes")
    )

    best = (
        pairs.filter("passes")
        .sort(
            ["truth_id", "evalue", "landing", "call_id"],
            descending=[False, False, True, False],
        )
        .group_by("truth_id", maintain_order=True)
        .agg(
            pl.col("landing").first().alias("best_landing"),
            pl.col("coverage").first().alias("best_coverage"),
            pl.col("evalue").first().alias("best_evalue"),
            pl.col("target").first().alias("best_target"),
            pl.col("qstart").first().alias("best_qstart"),
            pl.col("qend").first().alias("best_qend"),
            pl.col("tstart").first().alias("best_tstart"),
            pl.col("tend").first().alias("best_tend"),
            pl.col("qseq").first().alias("best_qseq"),
            pl.col("tseq").first().alias("best_tseq"),
            pl.col("name_match").any().alias("found_name"),
        )
    )
    out = (
        truth.select("truth_id")
        .join(best, on="truth_id", how="left")
        .join(aligned_found, on="truth_id", how="left")
        .with_columns(
            pl.col("best_landing").is_not_null().alias("found"),
            pl.col("best_landing_aligned").is_not_null().alias("found_aligned"),
            pl.col("found_name").fill_null(False),
            pl.col("found_name_aligned").fill_null(False),
            pl.lit(args.label).alias("label"),
        )
        .sort("truth_id")
    )
    out.write_parquet(f"{args.prefix}.features.parquet")
    thr["features_found"] = int(out["found"].sum())
    thr["features_found_aligned"] = int(out["found_aligned"].sum())
    thr["features_total"] = out.height
    Path(f"{args.prefix}.threshold.json").write_text(json.dumps(thr, indent=2))
    print(json.dumps(thr), file=sys.stderr)


if __name__ == "__main__":
    main()
