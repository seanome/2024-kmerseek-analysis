"""Experiment 7: the alphabet dose-response on Pfam with membrane instances held out.

nb 234 says HP retention sits on the BLOSUM line; the midi-plus report's dose-response says
a 2-letter alphabet recognises the right family (family_fmax) far better than protein20 on
the same engine. If that holds once every Pfam instance that is really a transmembrane
segment is taken out of the truth, the alphabet result stands on its own: the same
conservation the aligners encode, read at 2 bits per residue with an index.

Membrane instances: Pfam instances whose interval is at least half covered by Swiss-Prot
TRANSMEM/INTRAMEM features of the same protein, or that contain two or more of them
(a multi-pass bundle). Both truth tables are in the midi-plus extract.

Re-scores every Pfam call table (dedup) on `all` and `heldout` (non-membrane) instances
with the pipeline's own Fmax and family_fmax. Needs the call tables pulled first:

    rsync -az --include='pfam.*.dedup.calls.parquet' --exclude='*' \
      sherlock:/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results/calls/ \
      /Users/olga/data/qfo-pfam-region-midi-plus/calls_pfam/

Usage:
    python scripts/rescore_pfam_membrane_heldout.py --qfo-bin <bin dir at e136731> \
        --out /Users/olga/data/qfo-pfam-region-midi-plus/240_pfam_membrane_heldout_fmax.parquet
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

D = Path("/Users/olga/data/qfo-pfam-region-midi-plus")
CALLS = D / "calls_pfam"
PFAM_TRUTH = D / "extract" / "human_domain_truth.parquet"
SPROT_TRUTH = D / "truth_swissprot" / "human_swissprot_truth.parquet"
METRICS = D / "extract" / "all_domain_metrics.parquet"

_G: dict = {}


def membrane_instances(pfam: pl.DataFrame, sprot: pl.DataFrame) -> pl.DataFrame:
    tm = sprot.filter(pl.col("pfam_id").is_in(["TRANSMEM", "INTRAMEM"])).select(
        pl.col("accession"), pl.col("domain_start").alias("t_start"), pl.col("domain_end").alias("t_end"))
    j = pfam.join(tm, on="accession", how="left")
    j = j.with_columns(
        pl.when(pl.col("t_start").is_null()).then(0)
          .otherwise((pl.min_horizontal("domain_end", "t_end") - pl.max_horizontal("domain_start", "t_start")).clip(lower_bound=0)).alias("ov"))
    agg = (j.group_by("accession", "pfam_id", "domain_start", "domain_end")
             .agg(pl.col("ov").sum().alias("tm_residues"), (pl.col("ov") > 0).sum().alias("n_tm")))
    return agg.with_columns(
        ((pl.col("tm_residues") / (pl.col("domain_end") - pl.col("domain_start")) >= 0.5) | (pl.col("n_tm") >= 2)).alias("membrane"))


def init(qfo_bin: str):
    sys.path.insert(0, qfo_bin)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import cafa_metrics as cm  # noqa
    import evaluate_domain_calls as ev  # noqa
    import swissprot_control_utils as su  # noqa
    truth = pl.read_parquet(PFAM_TRUTH).select("accession", "pfam_id", "domain_start", "domain_end", "protein_length", "split").unique()
    mem = membrane_instances(truth.select("accession", "pfam_id", "domain_start", "domain_end").unique(), pl.read_parquet(SPROT_TRUTH))
    truth = truth.join(mem.select("accession", "pfam_id", "domain_start", "domain_end", "membrane"), on=["accession", "pfam_id", "domain_start", "domain_end"], how="left").with_columns(pl.col("membrane").fill_null(False))
    _G.update(cm=cm, ev=ev, su=su, truth=truth, ic=cm.information_content(truth))


def score(calls: pl.DataFrame, ident: dict) -> list[dict]:
    cm, ev, truth, ic = _G["cm"], _G["ev"], _G["truth"], _G["ic"]
    rows = []
    for split in ("all", "heldout"):
        for subset, t in (("all", truth), ("non_membrane", truth.filter(~pl.col("membrane"))), ("membrane", truth.filter(pl.col("membrane")))):
            if split != "all":
                t = t.filter(pl.col("split") == split)
            if t.height == 0:
                continue
            proteins = t.select("accession").unique().rename({"accession": "query_acc"})
            c = calls.join(proteins, on="query_acc", how="inner")
            fams = t.select("pfam_id").unique()
            c = c.join(fams, on="pfam_id", how="inner")
            c = ev.restrict_tp_to_cut(t, c)
            sc = cm.cafa_scalars(cm.protein_centric_curve(c, t, ic))
            fam = cm.cafa_scalars(cm.protein_centric_curve(c, t, ic, level="family"))
            rows.append({**ident, "split": split, "subset": subset, "n_truth": t.height, "n_calls": c.height,
                         "fmax": sc["fmax"], "fmax_precision": sc["fmax_precision"], "fmax_recall": sc["fmax_recall"], "family_fmax": fam["fmax"]})
    return rows


def one(path: Path) -> list[dict]:
    ident = _G["su"].parse_arm(path.name.replace("pfam.", "swissprot.", 1))
    ident["truth_set"] = "pfam"
    return score(pl.read_parquet(path), ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qfo-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    files = sorted(CALLS.glob("pfam.*.dedup.calls.parquet"))
    if not files:
        raise SystemExit(f"no call tables under {CALLS}; see the rsync in the module docstring")
    init(args.qfo_bin)
    print(_G["truth"].group_by("membrane").len())
    rows = []
    with Pool(args.procs, initializer=init, initargs=(args.qfo_bin,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, files, chunksize=4)):
            rows += r
            if i % 500 == 0:
                print(f"  {i} tables", flush=True)
    df = pl.DataFrame(rows).join(pl.read_parquet(METRICS).select("species", "species_mya").unique(), on="species", how="left")
    df.write_parquet(args.out)
    print(df.shape)


if __name__ == "__main__":
    main()
