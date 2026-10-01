#!/usr/bin/env python3
"""Per-instance tables for notebook 244 that need no search result.

Writes four parquets into /Users/olga/data/qfo-pfam-region-midi-plus/extract/:

  244_swissprot_feature_notes.parquet
      The Swiss-Prot /note text (e.g. "SH2", "C2H2-type 3", "Disordered") for every range
      feature on the 998 human queries and on every target protein in the nine midi-plus
      domain maps. The truth key (bin/build_swissprot_truth.py) keeps the type only. Parsed
      from the same uniprot_sprot.dat.gz release, with build_swissprot_truth's own FT_RE,
      so (accession, type, start, end) joins back onto the key exactly.

  244_swissprot_region_plddt.parquet
      pLDDT over each human feature's own residues: bin/build_query_covariates.py's
      read_plddt + domain_plddt, run on the pipeline's own human AlphaFold models with the
      Swiss-Prot key in place of the Pfam key.

  244_swissprot_region_disorder.parquet
      metapredict disorder over each human feature's residues: the pipeline's own
      bin/predict_disorder_metapredict.py with --domains set to the Swiss-Prot key.

  244_kd_scan_instances.parquet
      The Kyte-Doolittle window scan of notebook 231 (swissprot_control_utils
      .kd_transmem_calls, window 19, threshold 1.6) scored against every human range
      instance with notebook 244's landing rule (>= 80% of the scan segment inside the
      instance, covering >= 30% of it), whatever the instance's type.

Usage:
    python scripts/prep_244_instance_covariates.py
"""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(REPO / "notebooks"))
import build_query_covariates as bqc  # noqa: E402
import build_swissprot_truth as sprot  # noqa: E402
import evaluate_domain_calls as ev  # noqa: E402
import swissprot_control_utils as su  # noqa: E402

MIDI = Path("/Users/olga/data/qfo-pfam-region-midi-plus")
OUT = MIDI / "extract"
DAT = Path("/Users/olga/data/uniprot/uniprot_sprot.dat.gz")
#: The pipeline's own human AlphaFold models (data/structures/human on Sherlock), copied
#: here for the 933 query proteins with a Swiss-Prot feature.
STRUCTURES = MIDI / "structures_human"
SPECIES = [
    "mouse",
    "chicken",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
]
PYTHON = sys.executable


def parse_notes(accessions: set[str]) -> pl.DataFrame:
    """(accession, feature type, start, end, note) for every ranged feature of `accessions`.

    Includes BINDING and SITE features written as a range ("301..318"): the truth key keeps
    those as intervals, not points.

    Same record logic as build_swissprot_truth.parse (primary accession = first AC token,
    fuzzy endpoints dropped); the /note qualifier is the text on the following FT lines.
    """
    rows = []
    acc = None
    cur = None  # [acc, type, start, end, note]
    in_note = False
    with gzip.open(DAT, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("ID "):
                acc, cur, in_note = None, None, False
            elif line.startswith("AC ") and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif line.startswith("FT "):
                if acc not in accessions:
                    continue
                m = sprot.FT_RE.match(line.rstrip("\n"))
                if m:
                    if cur is not None:
                        rows.append(cur)
                    cur, in_note = None, False
                    ftype, s, e = m.groups()
                    if (
                        ftype in sprot.DEFAULT_FEATURES
                        and e
                        and not any(c in s + e for c in "<>?")
                    ):
                        cur = [acc, ftype, int(s), int(e), None]
                    continue
                if cur is None:
                    continue
                text = line[21:].rstrip("\n")
                # BINDING features name their ligand in /ligand, not /note; take it when
                # there is no note, written "ligand: ATP".
                if text.startswith('/ligand="') and cur[4] is None:
                    cur[4] = "ligand: " + text[len('/ligand="') :].rstrip('"')
                    in_note = False
                    continue
                if text.startswith('/note="'):
                    cur[4] = text[len('/note="') :]
                    in_note = not text.endswith('"')
                    if not in_note:
                        cur[4] = cur[4][:-1]
                elif in_note:
                    piece = text.rstrip('"')
                    cur[4] = f"{cur[4]} {piece}"
                    in_note = not text.endswith('"')
                elif text.startswith("/"):
                    in_note = False
            elif line.startswith("//"):
                if cur is not None:
                    rows.append(cur)
                acc, cur, in_note = None, None, False
    return pl.DataFrame(
        rows,
        schema=["accession", "pfam_id", "domain_start", "domain_end", "note"],
        orient="row",
    ).unique()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = pl.read_parquet(MIDI / "truth_swissprot" / "human_swissprot_truth.parquet")
    ranges = truth.filter(~pl.col("is_point"))
    queries = set(pl.read_parquet(MIDI / "query_gene_map.parquet")["accession"])

    # 1. Feature notes, human queries and every target protein with a Swiss-Prot feature.
    accs = set(queries)
    for sp in SPECIES:
        accs |= set(
            pl.read_parquet(MIDI / "truth_swissprot" / f"{sp}_domain_map.parquet")[
                "accession"
            ]
        )
    notes = parse_notes(accs)
    notes.write_parquet(OUT / "244_swissprot_feature_notes.parquet")
    hit = ranges.join(
        notes, on=["accession", "pfam_id", "domain_start", "domain_end"], how="semi"
    )
    print(
        f"notes: {notes.height} range features on {notes['accession'].n_unique()} proteins; "
        f"{hit.height} of {ranges.height} human range instances matched"
    )

    # 2. pLDDT per human range instance.
    tracks: dict[str, list[float]] = {}
    for acc in sorted(set(ranges["accession"])):
        # F1 only, as the pipeline's load_plddt does: a protein over 2_700 aa is modelled as
        # 1_400-residue fragments, and a feature past residue 1_400 gets no pLDDT.
        path = STRUCTURES / f"AF-{acc}-F1.cif"
        if path.exists():
            t = bqc.read_plddt(path)
            if t:
                tracks[acc] = t
    plddt = bqc.domain_plddt(ranges, tracks)
    plddt.write_parquet(OUT / "244_swissprot_region_plddt.parquet")
    print(
        f"pLDDT: {len(tracks)} of {ranges['accession'].n_unique()} proteins have a model; "
        f"{plddt['mean_plddt_region'].is_not_null().sum()} of {plddt.height} instances measured"
    )

    # 3. metapredict per human range instance, with the pipeline's own script.
    seqs = su.read_fasta(su.HUMAN_FASTA, queries)
    fasta = OUT / "244_human_queries.fasta"
    with open(fasta, "w") as fh:
        for acc, s in seqs.items():
            fh.write(f">sp|{acc}|\n{s}\n")
    domains = OUT / "244_human_swissprot_ranges.parquet"
    ranges.write_parquet(domains)
    subprocess.run(
        [
            PYTHON,
            str(BIN / "predict_disorder_metapredict.py"),
            "--fasta",
            str(fasta),
            "--out",
            str(OUT / "244_human_queries.disorder_metapredict.parquet"),
            "--domains",
            str(domains),
            "--domains-out",
            str(OUT / "244_swissprot_region_disorder.parquet"),
        ],
        check=True,
    )

    # 4. Kyte-Doolittle scan against every range instance, label ignored.
    kd = su.kd_transmem_calls(seqs, 19, 1.6)
    inst = ranges.select(
        pl.col("accession").alias("query_acc"),
        "pfam_id",
        pl.col("domain_start").alias("true_start"),
        pl.col("domain_end").alias("true_end"),
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
        .with_columns(landed=(pl.col("inside") >= 0.8) & (pl.col("cover") >= 0.3))
    )
    key = ["query_acc", "pfam_id", "true_start", "true_end"]
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
    kd_inst.write_parquet(OUT / "244_kd_scan_instances.parquet")
    print(
        f"KD scan: {kd.height} segments; {kd_inst.height} instances overlapped, "
        f"{int(kd_inst['kd_landed'].sum())} landed"
    )


if __name__ == "__main__":
    main()
