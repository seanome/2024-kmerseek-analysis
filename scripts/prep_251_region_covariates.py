#!/usr/bin/env python3
"""Per-region tables for notebook 251 that need no search result.

Reads the tables scripts/fetch_disprot.py wrote under --disprot-dir and writes three
parquets into <disprot-dir>/extract/:

  251_region_plddt.parquet
      AlphaFold pLDDT over each DisProt functional region's own residues, human and
      target side both: bin/build_query_covariates.py's read_plddt + domain_plddt on the
      AlphaFold models under <disprot-dir>/structures/<human|disprot> and the per-species
      folders of --structure-cache. pLDDT is AlphaFold's per-residue confidence, 0-100;
      below 50 the model is usually no structure at all, and below 70 its backbone
      should not be trusted.

  251_region_disorder.parquet
      metapredict 3 disorder over each region: the pipeline's own
      bin/predict_disorder_metapredict.py with --domains set to the DisProt regions. The
      disordered fraction is the share of residues scoring >= 0.5, metapredict's default.

  251_kd_scan_regions.parquet
      The Kyte-Doolittle window scan of notebook 231 (swissprot_control_utils
      .kd_transmem_calls, window 19, mean hydropathy > 1.6) scored against every human
      DisProt region with the landing rule of notebook 244 (>= 80% of the scan segment
      inside the region, covering >= 30% of it). A scan knows nothing about homology, so
      anything it lands on is landed by composition alone.

Coordinates: DisProt's, 1-based and inclusive, which is what domain_plddt and
predict_disorder_metapredict read. The KD scan is compared after moving both onto the
0-based, end-excluded convention the reduction uses.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(REPO / "notebooks"))
import build_query_covariates as bqc  # noqa: E402
import evaluate_domain_calls as ev  # noqa: E402
import swissprot_control_utils as su  # noqa: E402

HUMAN_TAXON = 9606
INSIDE_MIN = 0.8
COVER_MIN = 0.3


def model_path(acc: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for pat in (f"AF-{acc}-F1-model_v*.cif", f"AF-{acc}-F1.cif"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[-1]
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--disprot-dir", type=Path, default=Path.home() / "data/disprot-region-transfer"
    )
    ap.add_argument(
        "--structure-cache",
        type=Path,
        default=Path.home() / "data/alphafold_structures/structures",
        help="per-species AlphaFold folders (mouse/, yeast/, ...) for the QfO targets",
    )
    args = ap.parse_args()
    D = args.disprot_dir
    out = D / "extract"
    out.mkdir(parents=True, exist_ok=True)

    prot = pl.read_parquet(D / "disprot_proteins.parquet")
    reg = pl.read_parquet(D / "disprot_function_regions.parquet")
    domains = reg.select(
        "accession",
        pl.col("term_id").alias("pfam_id"),
        pl.col("start").alias("domain_start"),
        pl.col("end").alias("domain_end"),
    ).unique()

    # 1. pLDDT per region, every region with a model.
    dirs_for = {
        "human": [D / "structures" / "human"],
    }
    tracks: dict[str, list[float]] = {}
    for r in prot.filter(
        pl.col("accession").is_in(reg["accession"].unique().implode())
    ).iter_rows(named=True):
        if r["taxon"] == HUMAN_TAXON:
            dirs = dirs_for["human"]
        else:
            dirs = [D / "structures" / "disprot"]
            if r["qfo_label"]:
                dirs.append(args.structure_cache / r["qfo_label"])
        path = model_path(r["accession"], dirs)
        if path is None:
            continue
        t = bqc.read_plddt(path)
        # A model of a different sequence is a model of a different protein.
        if t and len(t) == r["length"]:
            tracks[r["accession"]] = t
    plddt = bqc.domain_plddt(domains, tracks)
    plddt.write_parquet(out / "251_region_plddt.parquet")
    print(
        f"pLDDT: {len(tracks)} of {domains['accession'].n_unique()} proteins have a model "
        f"of the same length; {plddt['mean_plddt_region'].is_not_null().sum()} of "
        f"{plddt.height} regions measured"
    )

    # 2. metapredict per region, with the pipeline's own script.
    fasta = out / "251_disprot_proteins.fasta"
    with open(fasta, "w") as fh:
        for r in prot.filter(
            pl.col("accession").is_in(reg["accession"].unique().implode())
        ).iter_rows(named=True):
            fh.write(f">dp|{r['accession']}|\n{r['sequence']}\n")
    dom_path = out / "251_disprot_regions_as_domains.parquet"
    domains.write_parquet(dom_path)
    subprocess.run(
        [
            sys.executable,
            str(BIN / "predict_disorder_metapredict.py"),
            "--fasta",
            str(fasta),
            "--out",
            str(out / "251_disprot_proteins.disorder_metapredict.parquet"),
            "--domains",
            str(dom_path),
            "--domains-out",
            str(out / "251_region_disorder.parquet"),
        ],
        check=True,
    )

    # 3. Kyte-Doolittle scan, scored against every human region by position alone.
    human = prot.filter((pl.col("taxon") == HUMAN_TAXON) & pl.col("qfo_same_sequence"))
    seqs = dict(zip(human["accession"], human["sequence"]))
    kd = su.kd_transmem_calls(seqs, 19, 1.6).with_columns(
        (pl.col("qstart") - 1).alias("qstart")
    )
    inst = (
        reg.filter((pl.col("taxon") == HUMAN_TAXON) & pl.col("qfo_same_sequence"))
        .select(
            pl.col("accession").alias("query_acc"),
            "term_id",
            (pl.col("start") - 1).alias("true_start"),
            pl.col("end").alias("true_end"),
        )
        .unique()
    )
    m = (
        kd.drop("pfam_id")
        .join(inst, on="query_acc", how="inner")
        .with_columns(ov=ev.overlap_expr("qstart", "qend", "true_start", "true_end"))
        .filter(pl.col("ov") > 0)
        .with_columns(
            inside=pl.col("ov")
            / (pl.col("qend") - pl.col("qstart")).clip(lower_bound=1),
            cover=pl.col("ov")
            / (pl.col("true_end") - pl.col("true_start")).clip(lower_bound=1),
            iou=pl.col("ov")
            / (
                pl.max_horizontal("qend", "true_end")
                - pl.min_horizontal("qstart", "true_start")
            ).clip(lower_bound=1),
        )
        .with_columns(
            landed=(pl.col("inside") >= INSIDE_MIN) & (pl.col("cover") >= COVER_MIN)
        )
    )
    key = ["query_acc", "term_id", "true_start", "true_end"]
    kd_inst = (
        m.sort(["iou", "qstart"], descending=[True, False])
        .group_by(key)
        .agg(
            kd_landed=pl.col("landed").any(),
            kd_best_iou=pl.col("iou").first(),
            kd_qstart=pl.col("qstart").first(),
            kd_qend=pl.col("qend").first(),
            kd_inside=pl.col("inside").first(),
        )
    )
    kd_all = inst.join(kd_inst, on=key, how="left").with_columns(
        pl.col("kd_landed").fill_null(False)
    )
    kd_all.write_parquet(out / "251_kd_scan_regions.parquet")
    print(
        f"KD scan: {kd.height} segments on {kd['query_acc'].n_unique()} human proteins; "
        f"{kd_inst.height} of {inst.height} human regions overlapped, "
        f"{int(kd_inst['kd_landed'].sum())} landed"
    )


if __name__ == "__main__":
    main()
