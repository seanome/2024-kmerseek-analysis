#!/usr/bin/env python3
"""Tables for the leaderboard's region pile-up figure.

    results/leaderboard/pileup/regions.parquet  one row per true Pfam region on a human query:
        row_id, accession, symbol, pfam_id, true_start, true_end (1-based, inclusive),
        protein_length, region_length, best_pident (closest same-family domain in the
        relative, NaN when it has none), region_plddt (mean AlphaFold pLDDT over the
        region's residues), and window_seq: the protein's residues from true_start - 100 to
        true_end + 100, clipped at the protein's ends, with window_offset = the 1-based
        residue number of its first letter.
    results/leaderboard/pileup/calls.parquet    one row per call a tool made on the same
        protein and the same Pfam family as a region, overlapping that region's window:
        tool, setting, row_id, call_start, call_end (1-based, inclusive), score, is_tp.
    results/leaderboard/pileup/provenance.json  inputs with sha256, and the coordinate rules.

Coordinates. Pfam truth and the aligners' calls are 1-based and inclusive. kmerseek's
calls carry region_start/region_end as written by kmerseek, 0-based and end-exclusive, so
1 is added to their start and the end is kept (checked on 300 mouse regions on
2026-09-24: read 0-based, the query and target class strings agree at every position).
provenance.json records the median call start minus true start over correct calls for each
tool: hmmscan, whose models define the Pfam truth, sits at 0, which confirms the 1-based
reading of the truth and the aligners.

The same interval reported many times (one human region aligned to many of the relative's
proteins) is kept once, with its highest score.

Only calls naming the region's own family count: a call of another family over the same
residues is not finding this region.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import polars as pl

FLANK = 100
TOOLS = {
    # file stem in calls/ -> (tool, setting)
    "hmmer3_phmmer.default": ("hmmer3_phmmer", "default"),
    "hmmer3_jackhmmer.n3": ("hmmer3_jackhmmer", "n3"),
    "mmseqs2_seqseq.s7": ("mmseqs2_seqseq", "s7"),
    "mmseqs2_iterative.s7": ("mmseqs2_iterative", "s7"),
    "hhblits.single_seq": ("hhblits", "single_seq"),
    "foldseek.3di_aa": ("foldseek", "3di_aa"),
    "prostt5.3di_from_seq": ("prostt5", "3di_from_seq"),
}
KMERSEEK = [
    "gbmr4_k15_lcFalse",
    "gbmr4_k17_lcFalse",
    "gbmr4_k19_lcFalse",
    "gbmr4_k21_lcFalse",
    "hp_thomas_dill2_k24_lcFalse",
    "hp_thomas_dill2_k27_lcFalse",
    "hp_thomas_dill2_k30_lcFalse",
    "hp_thomas_dill2_k33_lcFalse",
    "protein20_k6_lcFalse",
    "protein20_k7_lcFalse",
    "protein20_k8_lcFalse",
    "wwmj5_k11_lcFalse",
    "wwmj5_k12_lcFalse",
    "wwmj5_k14_lcFalse",
    "wwmj5_k15_lcFalse",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    """UniProt FASTA: the accession is the second |-separated field of the header."""
    seqs, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            head = line[1:].split()[0]
            acc = head.split("|")[1] if "|" in head else head
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def region_plddt(regions: pl.DataFrame, residue_plddt: Path) -> pl.DataFrame:
    """Mean pLDDT over each region's residues, from the F1 model of each protein."""
    with gzip.open(residue_plddt, "rt") as fh:
        pr = pl.read_csv(fh, separator="\t", schema_overrides={"plddt": pl.Float64})
    # A model of a different sequence version must not lend its pLDDT to this one. Proteins
    # up to 2,700 residues have one AlphaFold model covering the whole sequence, so its
    # length must equal protein_length; longer proteins only have the first 1,400-residue
    # piece (F1) here, which must not run past the protein's end. Anything else: no pLDDT.
    model_len = (
        pr.filter(pl.col("fragment") == 1)
        .group_by("accession")
        .agg(pl.col("residue").max().alias("model_len"))
    )
    plen = regions.select("accession", "protein_length").unique()
    ok = (
        plen.join(model_len, on="accession", how="inner")
        .filter(
            pl.when(pl.col("protein_length") <= 2700)
            .then(pl.col("model_len") == pl.col("protein_length"))
            .otherwise(pl.col("model_len") <= pl.col("protein_length"))
        )
        .select("accession")
    )
    pr = (
        pr.filter(pl.col("fragment") == 1)
        .join(ok, on="accession", how="semi")
        .select("accession", "residue", "plddt")
    )
    joined = (
        regions.select("row_id", "accession", "true_start", "true_end")
        .join(pr, on="accession", how="inner")
        .filter(pl.col("residue").is_between(pl.col("true_start"), pl.col("true_end")))
        .group_by("row_id")
        .agg(
            pl.col("plddt").mean().alias("region_plddt"),
            pl.len().alias("n_plddt_residues"),
        )
    )
    return regions.join(joined, on="row_id", how="left")


def load_calls(path: Path, tool: str, setting: str) -> pl.DataFrame:
    c = pl.read_parquet(path).select(
        pl.col("query_acc").alias("accession"),
        "pfam_id",
        "qstart",
        "qend",
        "score",
        "is_tp",
    )
    start = pl.col("qstart") + 1 if tool == "kmerseek" else pl.col("qstart")
    return c.select(
        pl.lit(tool).alias("tool"),
        pl.lit(setting).alias("setting"),
        "accession",
        "pfam_id",
        start.cast(pl.Int64).alias("call_start"),
        pl.col("qend").cast(pl.Int64).alias("call_end"),
        pl.col("score").cast(pl.Float64),
        "is_tp",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pileup-dir",
        type=Path,
        required=True,
        help="folder holding calls/, truth/ and fasta/ copied from the run",
    )
    ap.add_argument("--species", default="zebrafish")
    ap.add_argument("--outdir", type=Path, default=Path("results/leaderboard/pileup"))
    a = ap.parse_args()
    d = a.pileup_dir

    truth = pl.read_parquet(d / "truth/human_domain_truth.parquet").sort(
        "accession", "domain_start", "domain_end", "pfam_id"
    )
    cov = pl.read_parquet(d / "truth/human_query_covariates.parquet").select(
        "accession", pl.col("hgnc_symbol").alias("symbol")
    )
    ident = pl.read_parquet(d / f"truth/{a.species}.domain_identity.parquet").select(
        "accession", "pfam_id", "domain_start", "domain_end", "best_pident"
    )
    regions = (
        truth.with_row_index("row_id")
        .join(
            ident, on=["accession", "pfam_id", "domain_start", "domain_end"], how="left"
        )
        .join(cov, on="accession", how="left")
        .rename({"domain_start": "true_start", "domain_end": "true_end"})
        .with_columns(
            (pl.col("true_end") - pl.col("true_start") + 1).alias("region_length")
        )
    )
    if regions.height != truth.height:
        raise SystemExit("a join duplicated truth rows")
    regions = region_plddt(regions, d / "truth/residue_plddt.tsv.gz")

    seqs = read_fasta(next((d / "fasta").glob("UP000005640_*.fasta")))
    win = []
    for r in regions.iter_rows(named=True):
        s = seqs.get(r["accession"])
        if s is None:
            win.append((None, None))
            continue
        if len(s) != r["protein_length"]:
            raise SystemExit(
                f"{r['accession']}: FASTA length {len(s)} != truth {r['protein_length']}"
            )
        lo = max(1, r["true_start"] - FLANK)
        hi = min(len(s), r["true_end"] + FLANK)
        win.append((s[lo - 1 : hi], lo))
    regions = regions.with_columns(
        pl.Series("window_seq", [w[0] for w in win], dtype=pl.String),
        pl.Series("window_offset", [w[1] for w in win], dtype=pl.Int64),
    ).select(
        "row_id",
        "accession",
        "symbol",
        "pfam_id",
        "true_start",
        "true_end",
        "protein_length",
        "region_length",
        "best_pident",
        "region_plddt",
        "n_plddt_residues",
        "window_seq",
        "window_offset",
        "split",
    )

    frames = []
    inputs = {}
    for stem, (tool, setting) in TOOLS.items():
        p = d / f"calls/pfam.{stem}.{a.species}.calls.parquet"
        frames.append(load_calls(p, tool, setting))
        inputs[p.name] = sha256(p)
    for setting in KMERSEEK:
        p = d / f"calls/pfam.kmerseek.{setting}.{a.species}.calls.parquet"
        frames.append(load_calls(p, "kmerseek", setting))
        inputs[p.name] = sha256(p)
    p = d / "calls/hmmscan.direct.all.calls.parquet"
    frames.append(load_calls(p, "hmmscan", "direct"))
    inputs[p.name] = sha256(p)
    calls = pl.concat(frames)

    # Keep calls of the region's own family that overlap its window.
    calls = (
        calls.join(
            regions.select("row_id", "accession", "pfam_id", "true_start", "true_end"),
            on=["accession", "pfam_id"],
            how="inner",
        )
        .filter(
            (pl.col("call_end") >= pl.col("true_start") - FLANK)
            & (pl.col("call_start") <= pl.col("true_end") + FLANK)
        )
        .group_by("tool", "setting", "row_id", "pfam_id", "call_start", "call_end")
        .agg(pl.col("score").max(), pl.col("is_tp").any(), pl.len().alias("n_copies"))
        .sort("tool", "setting", "row_id", "call_start", "call_end")
    )

    # Coordinate check on correct calls: after the shift, starts should centre on 0.
    chk = (
        calls.filter("is_tp")
        .join(regions.select("row_id", "true_start"), on="row_id")
        .group_by("tool")
        .agg(
            (pl.col("call_start") - pl.col("true_start"))
            .median()
            .alias("median_start_diff")
        )
    )

    a.outdir.mkdir(parents=True, exist_ok=True)
    regions.write_parquet(a.outdir / "regions.parquet")
    calls.write_parquet(a.outdir / "calls.parquet")
    for name in (
        "human_domain_truth.parquet",
        "human_query_covariates.parquet",
        f"{a.species}.domain_identity.parquet",
        "residue_plddt.tsv.gz",
    ):
        inputs[name] = sha256(d / "truth" / name)
    prov = {
        "species": a.species,
        "flank": FLANK,
        "truth_set": "pfam",
        "calls": "raw",
        "coordinates": "1-based inclusive; kmerseek call_start = region_start + 1",
        "median_start_diff_correct_calls": {
            r["tool"]: r["median_start_diff"] for r in chk.iter_rows(named=True)
        },
        "inputs": inputs,
        "outputs": {
            n: sha256(a.outdir / n) for n in ("regions.parquet", "calls.parquet")
        },
        "n_regions": regions.height,
        "n_regions_without_plddt": regions.filter(
            pl.col("region_plddt").is_null()
        ).height,
        "n_calls": calls.height,
    }
    (a.outdir / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: prov[k]
                for k in (
                    "n_regions",
                    "n_regions_without_plddt",
                    "n_calls",
                    "median_start_diff_correct_calls",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
