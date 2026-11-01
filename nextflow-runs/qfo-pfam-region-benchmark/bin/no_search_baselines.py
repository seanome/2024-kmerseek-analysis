#!/usr/bin/env python3
"""Calls from the query sequence alone, with no database: the leaderboard's baseline rows.

Gillis and Pavlidis (2013, BMC Bioinformatics 14 suppl. 3 S15) asked every function
benchmark for a row that uses no search at all, so a reader can see what score comes for
free. Four kinds are made here. Each writes calls in the layout evaluate_domain_calls.py
reads with --direct-annotation (query, label, qstart, qend, score, evalue; tab-separated,
no header), so they are scored by the same code and against the same truth tables as every
search tool.

    kd          Kyte-Doolittle hydropathy scan, copied from notebook 231
                (notebooks/swissprot_control_utils.py, kd_transmem_calls; used in cell
                code-05 of notebooks/231_swissprot_composition_controls.ipynb): 19-residue
                window, mean above 1.6. Labelled TRANSMEM or INTRAMEM (one file each).
    runs        Runs of residues whose per-residue disorder score is above a cutoff, at
                least --min-length residues long. Used for IUPred long and metapredict.
    seg         segmasker (NCBI BLAST+), the NCBI copy of SEG, at its default settings
                (window 12, locut 2.2, hicut 2.5). Masked intervals become calls.
    frequency   Every query gets, over its whole length, the Pfam family held by the most
                proteins of its length bin in a reference set that excludes the queries.
                The "term frequency" baseline of CAFA, at the region level.

Coordinates: calls are 1-based with an inclusive end, the same convention notebook 231's
scan used and the truth tables carry.

Which truth set and Swiss-Prot feature type each baseline can address is listed in
assets/no_search_baselines.tsv. A baseline row is reported only there; every other cell
is "not applicable", never zero.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Kyte & Doolittle 1982, J Mol Biol 157:105, Table 2. Copied from
# notebooks/swissprot_control_utils.py (notebook 231) without change.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
KD_WINDOW = 19
KD_THRESHOLD = 1.6

# Length bins for the frequency baseline, in residues, written before any result was read.
# Lower edge inclusive, upper edge exclusive; the last bin is open.
FREQUENCY_LENGTH_EDGES = [0, 100, 200, 300, 400, 600, 800, 1200]

CALL_COLUMNS = ["query", "label", "qstart", "qend", "score", "evalue"]


def accession_of(name: str) -> str:
    """`sp|P04637|P53_HUMAN` -> `P04637`; a bare name passes through."""
    first = name.split()[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 2 and parts[1] else first


def read_fasta(path: Path) -> dict[str, str]:
    """{accession: sequence}, accession taken as evaluate_domain_calls does."""
    seqs: dict[str, list[str]] = {}
    acc = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                acc = accession_of(line[1:])
                seqs[acc] = []
            elif acc is not None:
                seqs[acc].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) 0-based half-open runs of True."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def empty_calls() -> pl.DataFrame:
    return pl.DataFrame(schema={"query": pl.String, "label": pl.String,
                                "qstart": pl.Int64, "qend": pl.Int64,
                                "score": pl.Float64, "evalue": pl.String})


def to_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return empty_calls()
    return pl.DataFrame(rows).with_columns(pl.lit("NA").alias("evalue")).select(
        pl.col("query").cast(pl.String), pl.col("label").cast(pl.String),
        pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64),
        pl.col("score").cast(pl.Float64), pl.col("evalue"),
    )


def write_calls(calls: pl.DataFrame, out: Path) -> None:
    calls.select(CALL_COLUMNS).write_csv(out, separator="\t", include_header=False)
    print(f"{out}: {calls.height} calls on {calls['query'].n_unique()} proteins",
          file=sys.stderr)


def kd_calls(seqs: dict[str, str], label: str, window: int = KD_WINDOW,
             threshold: float = KD_THRESHOLD) -> pl.DataFrame:
    """Notebook 231's scan, with the label a parameter and the window test made exact.

    Windows whose mean hydropathy exceeds `threshold`, merged into segments spanning the
    full windows. Score is the segment's best window mean.

    The test is done in whole tenths: every Kyte-Doolittle value has one decimal, so ten
    times a window's sum is an integer and "mean above 1.6" is "sum of tenths above
    16 x 19". Notebook 231 compared floating-point means, and a window whose mean is
    exactly 1.6 then fell either side by rounding. The 2026-10-19 Sherlock run (numpy
    2.5.2) and the laptop (numpy 2.3.5) disagreed on 16 such calls out of about 1580 and
    gave TRANSMEM Fmax 0.7775 against 0.7781; the exact test agrees with neither on 3
    calls each. Scores are still the float mean, which only orders calls.
    """
    rows = []
    for acc, s in seqs.items():
        h = np.array([KYTE_DOOLITTLE.get(c, 0.0) for c in s.upper()])
        if h.size < window:
            continue
        tenths = np.rint(h * 10).astype(np.int64)
        # Window starting at residue i (0-based) covers residues i..i+window-1.
        sums = np.convolve(tenths, np.ones(window, dtype=np.int64), mode="valid")
        means = sums / (10 * window)
        for a, b in segments(sums > round(threshold * 10) * window):
            rows.append({"query": acc, "label": label, "qstart": a + 1,
                         "qend": b - 1 + window, "score": float(means[a:b].max())})
    return to_frame(rows)


def read_residue_scores(path: Path) -> dict[str, np.ndarray]:
    """accession, position (1-based), score TSV -> {accession: scores in residue order}."""
    t = pl.read_csv(path, separator="\t", has_header=False,
                    new_columns=["accession", "position", "score"],
                    schema_overrides={"accession": pl.String, "position": pl.Int64,
                                      "score": pl.Float64})
    out = {}
    for (acc,), g in t.sort("accession", "position").group_by("accession",
                                                               maintain_order=True):
        pos = g["position"].to_numpy()
        if not np.array_equal(pos, np.arange(1, len(pos) + 1)):
            raise SystemExit(f"{path}: positions for {acc} are not 1..{len(pos)}")
        out[acc] = g["score"].to_numpy()
    return out


def run_calls(scores: dict[str, np.ndarray], labels: list[str], cutoff: float,
              min_length: int) -> pl.DataFrame:
    """Runs of residues scoring strictly above `cutoff`, at least `min_length` long.

    Score is the run's mean residue score. One copy of each run per label.
    """
    rows = []
    for acc, s in scores.items():
        for a, b in segments(s > cutoff):
            if b - a < min_length:
                continue
            for label in labels:
                rows.append({"query": acc, "label": label, "qstart": a + 1, "qend": b,
                             "score": float(s[a:b].mean())})
    return to_frame(rows)


def read_segmasker_intervals(path: Path) -> list[tuple[str, int, int]]:
    """segmasker -outfmt interval: '>name' then '<start> - <end>' lines, 0-based inclusive."""
    out, acc = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                acc = accession_of(line[1:])
                continue
            a, _, b = line.partition(" - ")
            out.append((acc, int(a), int(b)))
    return out


def seg_calls(intervals: list[tuple[str, int, int]], labels: list[str]) -> pl.DataFrame:
    """Masked intervals as calls (1-based inclusive). SEG gives no score, so the score is
    the interval's length: a longer masked stretch is ranked as the more confident call."""
    rows = [{"query": acc, "label": label, "qstart": a + 1, "qend": b + 1,
             "score": float(b - a + 1)}
            for acc, a, b in intervals for label in labels]
    return to_frame(rows)


def length_bin(n: int) -> str:
    edges = FREQUENCY_LENGTH_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def frequency_table(reference_annotation: pl.DataFrame, reference_lengths: dict[str, int],
                    exclude: set[str]) -> pl.DataFrame:
    """Per length bin: the Pfam family held by the most reference proteins.

    Proteins are counted, not domain instances, so a 30-repeat protein counts once.
    Ties go to the alphabetically first accession, so the choice is reproducible.
    `share` is the fraction of the bin's proteins that hold the family.
    """
    lengths = pl.DataFrame(
        {"accession": list(reference_lengths),
         "length": list(reference_lengths.values())},
        schema={"accession": pl.String, "length": pl.Int64},
    ).filter(~pl.col("accession").is_in(list(exclude)))
    lengths = lengths.with_columns(
        pl.col("length").map_elements(length_bin, return_dtype=pl.String).alias("bin"))
    per_bin = lengths.group_by("bin").agg(pl.len().alias("n_proteins_bin"))
    held = (
        reference_annotation.select("accession", "pfam_id").unique()
        .join(lengths.select("accession", "bin"), on="accession", how="inner")
        .group_by("bin", "pfam_id").agg(pl.len().alias("n_proteins_family"))
        .sort(["bin", "n_proteins_family", "pfam_id"], descending=[False, True, False])
        .group_by("bin", maintain_order=True).first()
    )
    return (
        held.join(per_bin, on="bin")
        .with_columns((pl.col("n_proteins_family") / pl.col("n_proteins_bin"))
                      .alias("share"))
        .sort("bin")
    )


def frequency_calls(queries: dict[str, str], table: pl.DataFrame) -> pl.DataFrame:
    best = {r["bin"]: r for r in table.iter_rows(named=True)}
    rows = []
    for acc, s in queries.items():
        r = best.get(length_bin(len(s)))
        if r is None:
            continue
        rows.append({"query": acc, "label": r["pfam_id"], "qstart": 1, "qend": len(s),
                     "score": float(r["share"])})
    return to_frame(rows)


def read_applies(path: Path) -> pl.DataFrame:
    """assets/no_search_baselines.tsv: which (truth set, stratum) each baseline addresses."""
    return pl.read_csv(path, separator="\t", infer_schema=False)


def mark_applicable(metrics: pl.DataFrame, applies: pl.DataFrame) -> pl.DataFrame:
    """Add `applicable`: True on a row the baseline can address, else False.

    A table row with stratum_axis "*" covers every stratum of its truth set; otherwise
    only the one (stratum_axis, stratum) cell it names. A cell marked False is reported as
    "not applicable", never as a score of zero.
    """
    keys = ["tool", "variant", "truth_set"]
    wild = (applies.filter(pl.col("stratum_axis") == "*").select(keys).unique()
            .with_columns(pl.lit(True).alias("_wild")))
    cell = (applies.filter(pl.col("stratum_axis") != "*")
            .select(*keys, "stratum_axis", "stratum").unique()
            .with_columns(pl.lit(True).alias("_cell")))
    out = (metrics.join(wild, on=keys, how="left")
           .join(cell, on=[*keys, "stratum_axis", "stratum"], how="left"))
    return out.with_columns(
        (pl.col("_wild").fill_null(False) | pl.col("_cell").fill_null(False))
        .alias("applicable"),
        pl.lit("no search").alias("tool_kind"),
    ).drop("_wild", "_cell")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("kd", help="Kyte-Doolittle scan (notebook 231)")
    k.add_argument("--fasta", required=True, type=Path)
    k.add_argument("--label", action="append", required=True,
                   help="one output per label, e.g. TRANSMEM, INTRAMEM")
    k.add_argument("--out", action="append", required=True, type=Path)

    r = sub.add_parser("runs", help="runs above a per-residue cutoff")
    r.add_argument("--scores", required=True, type=Path)
    r.add_argument("--cutoff", required=True, type=float)
    r.add_argument("--min-length", type=int, default=20)
    r.add_argument("--label", action="append", required=True)
    r.add_argument("--out", action="append", required=True, type=Path)

    s = sub.add_parser("seg", help="segmasker intervals to calls")
    s.add_argument("--intervals", required=True, type=Path)
    s.add_argument("--label", action="append", required=True)
    s.add_argument("--out", action="append", required=True, type=Path)

    f = sub.add_parser("frequency", help="most common Pfam family per length bin")
    f.add_argument("--fasta", required=True, type=Path, help="query FASTA")
    f.add_argument("--reference-fasta", required=True, type=Path)
    f.add_argument("--reference-annotation", required=True, type=Path,
                   help="Pfam domains parquet (accession, pfam_id, ...)")
    f.add_argument("--table-out", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)

    r2 = sub.add_parser("restrict", help="keep the truth rows of the query proteins only")
    r2.add_argument("--truth", required=True, type=Path)
    r2.add_argument("--fasta", required=True, type=Path, help="query FASTA")
    r2.add_argument("--out", required=True, type=Path)

    c = sub.add_parser("concat", help="join the per-baseline metrics into one table")
    c.add_argument("--metrics-dir", required=True, type=Path)
    c.add_argument("--applies", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)

    a = p.parse_args()

    if a.cmd in ("kd", "runs", "seg") and len(a.label) != len(a.out):
        raise SystemExit("--label and --out must be given the same number of times")

    if a.cmd == "kd":
        seqs = read_fasta(a.fasta)
        for label, out in zip(a.label, a.out):
            write_calls(kd_calls(seqs, label), out)
    elif a.cmd == "runs":
        scores = read_residue_scores(a.scores)
        for label, out in zip(a.label, a.out):
            write_calls(run_calls(scores, [label], a.cutoff, a.min_length), out)
    elif a.cmd == "seg":
        intervals = read_segmasker_intervals(a.intervals)
        for label, out in zip(a.label, a.out):
            write_calls(seg_calls(intervals, [label]), out)
    elif a.cmd == "frequency":
        queries = read_fasta(a.fasta)
        ref = read_fasta(a.reference_fasta)
        ann = pl.read_parquet(a.reference_annotation, columns=["accession", "pfam_id"])
        table = frequency_table(ann, {k: len(v) for k, v in ref.items()}, set(queries))
        table.write_csv(a.table_out, separator="\t")
        print(table, file=sys.stderr)
        write_calls(frequency_calls(queries, table), a.out)
    elif a.cmd == "restrict":
        # The Pfam-N truth table covers the whole human proteome (44_315 regions on 18_724
        # proteins). Scored as it stands, recall is divided by regions of proteins nobody
        # searched; the midi-plus run's queries hold 2_218 of them on 966 proteins. Same
        # fix as restrictTruthToQueries in PR 165.
        queries = set(read_fasta(a.fasta))
        t = pl.read_parquet(a.truth)
        out = t.filter(pl.col("accession").is_in(list(queries)))
        out.write_parquet(a.out)
        print(f"{a.truth.name}: kept {out.height} of {t.height} rows, "
              f"{out['accession'].n_unique()} of {t['accession'].n_unique()} proteins",
              file=sys.stderr)
    elif a.cmd == "concat":
        files = sorted(a.metrics_dir.glob("*.metrics.parquet"))
        if not files:
            raise SystemExit(f"no *.metrics.parquet under {a.metrics_dir}")
        m = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        m = mark_applicable(m, read_applies(a.applies))
        m.write_parquet(a.out)
        print(m.filter(pl.col("applicable") & (pl.col("split") == "all"))
              .select("truth_set", "tool", "variant", "stratum_axis", "stratum",
                      "n_truth_instances", "fmax")
              .sort("truth_set", "tool", "variant", "stratum_axis", "stratum"),
              file=sys.stderr)


if __name__ == "__main__":
    main()
