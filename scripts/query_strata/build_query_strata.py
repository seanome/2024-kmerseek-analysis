#!/usr/bin/env python3
"""Per-query strata of the 998 midi-plus queries, from the outputs of the five jobs.

Reads the job outputs (scripts/query_strata/0*.sbatch) from --data, a mirror of
/scratch/users/olgabot (`make query-strata-fetch` makes it), and writes into --outdir:

    query_strata.tsv          one row per query: the raw value of every protein-level
                              stratum and its bin (RULES.md)
    query_nearest_hits.tsv    one row per (target species, query): the best MMseqs2 hit
                              at E < 1e-3, or none, and its identity bin
    pfam_family_breadth.tsv   one row per Pfam family: superkingdoms and phyla it occurs in
    busco_summary.json        the human proteome's BUSCO C, S, D, F, M percentages
    human_query_covariates.strata.parquet
                              the midi-plus run's own covariates table plus the strata
                              columns; the file main.nf --query_covariates takes

The bins here are computed with the same rules the scorer applies
(evaluate_domain_calls.attach_query_strata), and query_strata.tsv carries them so a reader
can count queries per bin without the scorer. check_query_strata.py recounts them from
the raw files with its own code.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import polars as pl

SCRATCH = "/scratch/users/olgabot"
QS = "leaderboard-query-strata/out"
QFO = "2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark"
SPECIES = [
    "arabidopsis",
    "btaurus",
    "chicken",
    "ciona",
    "clupus",
    "ecoli",
    "fly",
    "mdomestica",
    "mouse",
    "ptroglodytes",
    "rnorvegicus",
    "worm",
    "xtropicalis",
    "yeast",
    "zebrafish",
]
M8 = [
    "query",
    "target",
    "fident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
]
DEEPTMHMM_CLASS = {
    "TM": "alpha-helical TM",
    "SP+TM": "alpha-helical TM",
    "BETA": "beta-barrel",
    "SP": "neither",
    "GLOB": "neither",
}
INPUTS = {
    "best_hits": f"{QS}/identity/best_hits.tsv",
    "mmseqs_version": f"{QS}/identity/mmseqs_version.txt",
    "seg": f"{QS}/seg/queries.seg.intervals",
    "segmasker_version": f"{QS}/seg/segmasker_version.txt",
    "length": f"{QS}/length/queries.length.tsv",
    "deeptmhmm": f"{QS}/deeptmhmm/predicted_topologies.3line",
    "busco_table": f"{QS}/busco/busco_human/run_primates_odb12/full_table.tsv",
    "busco_summary": f"{QS}/busco/busco_human/short_summary.specific.primates_odb12.busco_human.json",
    "busco_version": f"{QS}/busco/busco_version.txt",
    "breadth": f"{QS}/pfam_breadth/pfam_family_breadth.tsv",
    "covariates": f"{QFO}/data/midi-plus/results/truth/human_query_covariates.parquet",
    "pfam_truth": f"{QFO}/data/midi-plus/results/truth/human_domain_truth.parquet",
    "pfamn_truth": f"{QFO}/data/pfamn/human_pfamn_truth.parquet",
}


def accession(name: str) -> str:
    """sp|P12345|NAME_HUMAN -> P12345; a bare accession passes through."""
    parts = name.split("|")
    return parts[1] if len(parts) >= 3 else name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_bin(pct: float | None) -> str:
    if pct is None:
        return "no hit"
    if pct < 30:
        return "<30%"
    if pct < 50:
        return "30-50%"
    if pct < 90:
        return "50-90%"
    return ">=90%"


def seg_bin(f: float) -> str:
    return "<5%" if f < 0.05 else ("5-20%" if f <= 0.20 else ">20%")


def length_bin(n: int) -> str:
    if n < 100:
        return "<100"
    if n < 300:
        return "100-300"
    return "300-1000" if n <= 1000 else ">1000"


def read_seg(path: Path) -> dict[str, int]:
    """segmasker -outfmt interval: '>header' then 'start - end' lines, 0-based inclusive.

    Residues masked, each counted once. segmasker writes overlapping intervals (A1A5C7:
    51-63 and 63-77; 162 of the 998 queries have some), so summing interval lengths
    counts the shared residues twice; the union of the intervals is taken instead.
    """
    masked: dict[str, set[int]] = {}
    cur = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            cur = accession(line[1:].split()[0])
            masked[cur] = set()
        elif line.strip():
            a, b = (int(x) for x in line.split("-"))
            masked[cur].update(range(a, b + 1))
    return {k: len(v) for k, v in masked.items()}


def read_deeptmhmm(path: Path) -> dict[str, str]:
    """predicted_topologies.3line: '>id | TYPE', sequence, labels."""
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name, typ = line[1:].split(" | ")
            out[accession(name.strip())] = typ.strip()
    return out


def read_busco(path: Path) -> dict[str, str]:
    """full_table.tsv -> per protein: complete if any Complete/Duplicated row, else
    fragmented if any Fragmented row (RULES.md)."""
    status: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 3 or f[1] not in ("Complete", "Duplicated", "Fragmented"):
            continue
        acc = accession(f[2].split()[0])
        s = "complete" if f[1] in ("Complete", "Duplicated") else "fragmented"
        if status.get(acc) != "complete":
            status[acc] = s
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--outdir", type=Path, default=Path("results/leaderboard/query_strata")
    )
    args = ap.parse_args()
    p = {k: args.data / v for k, v in INPUTS.items()}

    lengths = pl.read_csv(
        p["length"], separator="\t", has_header=False, new_columns=["name", "length_aa"]
    )
    lengths = lengths.with_columns(
        pl.col("name")
        .map_elements(accession, return_dtype=pl.String)
        .alias("accession")
    )
    accs = lengths["accession"].to_list()
    assert len(accs) == len(set(accs)) == 998, f"{len(accs)} queries, expected 998"

    masked = read_seg(p["seg"])
    dtm = read_deeptmhmm(p["deeptmhmm"])
    busco = read_busco(p["busco_table"])
    missing = {"seg": set(accs) - set(masked), "deeptmhmm": set(accs) - set(dtm)}
    for k, v in missing.items():
        assert not v, f"{k} has no row for {len(v)} queries, e.g. {sorted(v)[:3]}"

    best = pl.read_csv(
        p["best_hits"],
        separator="\t",
        has_header=False,
        new_columns=["species", *M8],
        infer_schema_length=0,
    )
    best = best.with_columns(
        pl.col("query")
        .map_elements(accession, return_dtype=pl.String)
        .alias("accession"),
        pl.col("fident").cast(pl.Float64),
        pl.col("evalue").cast(pl.Float64),
        pl.col("bits").cast(pl.Float64),
    ).filter(pl.col("evalue") < 1e-3)
    # MMseqs2 prints fident as a fraction (0-1); the bins are in percent.
    assert best["fident"].max() <= 1.0, "fident above 1: this MMseqs2 prints percent"
    best = best.with_columns((pl.col("fident") * 100).alias("identity_pct"))
    dup = best.group_by("species", "accession").len().filter(pl.col("len") > 1)
    assert (
        dup.height == 0
    ), "best_hits.tsv holds more than one hit per query and species"

    grid = pl.DataFrame({"species": SPECIES}).join(
        pl.DataFrame({"accession": accs}), how="cross"
    )
    hits = grid.join(
        best.select("species", "accession", "target", "identity_pct", "evalue", "bits"),
        on=["species", "accession"],
        how="left",
    ).with_columns(
        pl.col("identity_pct")
        .map_elements(identity_bin, return_dtype=pl.String, skip_nulls=False)
        .alias("nearest_identity")
    )

    q = (
        lengths.select("accession", "length_aa")
        .with_columns(
            pl.col("accession")
            .replace_strict(masked, return_dtype=pl.Int64)
            .alias("seg_masked"),
            pl.col("accession")
            .replace_strict(dtm, return_dtype=pl.String)
            .alias("deeptmhmm_type"),
            pl.col("accession")
            .replace_strict(busco, default="not a BUSCO gene", return_dtype=pl.String)
            .alias("busco_status"),
        )
        .with_columns(
            (pl.col("seg_masked") / pl.col("length_aa")).alias("seg_fraction")
        )
    )
    q = q.with_columns(
        pl.col("deeptmhmm_type").replace_strict(DEEPTMHMM_CLASS).alias("membrane"),
        pl.col("seg_fraction")
        .map_elements(seg_bin, return_dtype=pl.String)
        .alias("low_complexity"),
        pl.col("length_aa")
        .map_elements(length_bin, return_dtype=pl.String)
        .alias("protein_length"),
        pl.col("busco_status").alias("busco"),
    )
    wide_ident = hits.pivot(on="species", index="accession", values="identity_pct")
    wide_ident = wide_ident.rename({s: f"nearest_identity_{s}" for s in SPECIES})

    # Pfam family breadth, for every family a query carries in either Pfam truth set.
    breadth = pl.read_csv(p["breadth"], separator="\t")
    fams = (
        pl.concat(
            [
                pl.read_parquet(p["pfam_truth"]).select("accession", "pfam_id"),
                pl.read_parquet(p["pfamn_truth"]).select("accession", "pfam_id"),
            ]
        )
        .filter(pl.col("accession").is_in(accs))
        .unique()
    )
    fam_b = fams.join(
        breadth.select("pfam_id", "n_superkingdoms", "n_phyla"),
        on="pfam_id",
        how="left",
    ).sort("accession", "pfam_id")
    fam_list = fam_b.group_by("accession").agg(
        pl.struct("pfam_id", "n_superkingdoms", "n_phyla").alias("pfam_family_breadth")
    )

    cov = pl.read_parquet(p["covariates"])
    assert sorted(cov["accession"].to_list()) == sorted(
        accs
    ), "the run's covariates and the query FASTA name different proteins"
    new_cols = q.select(
        "accession", "length_aa", "seg_fraction", "deeptmhmm_type", "busco_status"
    ).join(wide_ident, on="accession")
    clash = (set(new_cols.columns) - {"accession"}) & set(cov.columns)
    assert not clash, f"covariates already carry {clash}"
    strata_cov = (
        cov.join(new_cols, on="accession", how="left")
        .join(fam_list, on="accession", how="left")
        .with_columns(pl.col("pfam_family_breadth").fill_null([]))
    )
    assert strata_cov.height == cov.height

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    table = q.join(wide_ident, on="accession").sort("accession")
    table.write_csv(out / "query_strata.tsv", separator="\t", float_precision=4)
    hits.sort("species", "accession").write_csv(
        out / "query_nearest_hits.tsv", separator="\t", float_precision=4
    )
    breadth.write_csv(out / "pfam_family_breadth.tsv", separator="\t")
    strata_cov.write_parquet(
        out / "human_query_covariates.strata.parquet", statistics=False
    )

    summ = json.loads(p["busco_summary"].read_text())
    res = summ.get("results", summ)
    keep = {
        k: res[k]
        for k in res
        if re.match(
            r"^(Complete|Single|Multi|Fragmented|Missing|"
            r"n_markers|one_line_summary)",
            k,
        )
    }
    busco_out = {
        "lineage": summ.get("lineage_dataset", {}).get("name"),
        "busco_version": p["busco_version"].read_text().strip(),
        "proteome": f"{SCRATCH}/{QFO}/data/qfo/Eukaryota/UP000005640_9606.fasta",
        "results": keep,
        "queries_by_status": dict(q.group_by("busco").len().sort("busco").iter_rows()),
    }
    (out / "busco_summary.json").write_text(json.dumps(busco_out, indent=1) + "\n")

    prov = {
        "inputs": {f"{SCRATCH}/{v}": sha256(p[k]) for k, v in INPUTS.items()},
        "tools": {
            "mmseqs2": p["mmseqs_version"].read_text().strip(),
            "segmasker": p["segmasker_version"].read_text().strip().splitlines()[0],
            "busco": busco_out["busco_version"],
        },
        "outputs": {
            n: sha256(out / n)
            for n in (
                "query_strata.tsv",
                "query_nearest_hits.tsv",
                "pfam_family_breadth.tsv",
                "busco_summary.json",
                "human_query_covariates.strata.parquet",
            )
        },
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")

    for axis in ("membrane", "low_complexity", "protein_length", "busco"):
        print(q.group_by(axis).len().sort(axis))
    print(hits.group_by("nearest_identity").len().sort("nearest_identity"))
    print(
        f"families with breadth: {fam_b['n_phyla'].is_not_null().sum()} "
        f"of {fam_b.height} query-family pairs"
    )


if __name__ == "__main__":
    main()
