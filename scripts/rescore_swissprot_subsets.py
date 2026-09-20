"""Re-score every Swiss-Prot call table of the midi-plus run on feature-type subsets.

For each (arm, species) call table the pipeline wrote, compute interval-level Fmax with the
pipeline's own curve (cafa_metrics.protein_centric_curve) on several truth subsets:

  all          every feature type, the report's headline
  composition  TRANSMEM, INTRAMEM, COILED, REGION, REPEAT: features a hydrophobicity or
               periodicity scan could reproduce without homology
  heldout      everything else (DOMAIN, ZN_FING, DNA_BIND, MOTIF, BINDING, ACT_SITE, SITE,
               CA_BIND, NP_BIND, METAL): the family- and site-bearing features
  transmem     TRANSMEM alone
  domain_only  DOMAIN, ZN_FING, DNA_BIND, MOTIF: range features that name a family, no points

Held-out semantics: the truth is cut to the subset's types, calls are kept only on proteins
that still have truth and only when their label is one of the subset's types, and a true
positive whose instance left the cut is re-judged (evaluate_domain_calls.restrict_tp_to_cut).
That is "what the benchmark would have said had those feature types never been annotated".

Also emits the same for two no-search control arms (Kyte-Doolittle scan, HP-class runs).

Usage:
    python scripts/rescore_swissprot_subsets.py --qfo-bin <bin dir from the pipeline branch> \
        --out /Users/olga/data/qfo-pfam-region-midi-plus/231_swissprot_subset_fmax.parquet
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

CALLS = Path("/Users/olga/data/qfo-pfam-region-midi-plus/calls_swissprot")
TRUTH = Path("/Users/olga/data/qfo-pfam-region-midi-plus/truth_swissprot/human_swissprot_truth.parquet")
QUERIES = Path("/Users/olga/data/qfo-pfam-region-midi-plus/query_gene_map.parquet")
METRICS = Path("/Users/olga/data/qfo-pfam-region-midi-plus/extract/all_domain_metrics.parquet")

COMPOSITION = ["TRANSMEM", "INTRAMEM", "COILED", "REGION", "REPEAT"]
DOMAIN_ONLY = ["DOMAIN", "ZN_FING", "DNA_BIND", "MOTIF"]

_G: dict = {}


def init(qfo_bin: str):
    sys.path.insert(0, qfo_bin)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import cafa_metrics as cm  # noqa
    import evaluate_domain_calls as ev  # noqa
    import swissprot_control_utils as su  # noqa
    truth = pl.read_parquet(TRUTH)
    all_types = sorted(truth["pfam_id"].unique().to_list())
    subsets = {
        "all": all_types,
        "composition": [t for t in all_types if t in COMPOSITION],
        "heldout": [t for t in all_types if t not in COMPOSITION],
        "transmem": ["TRANSMEM"],
        "domain_only": [t for t in all_types if t in DOMAIN_ONLY],
    }
    _G.update(cm=cm, ev=ev, su=su, truth=truth, subsets=subsets, ic=cm.information_content(truth))


def score_subsets(calls: pl.DataFrame, ident: dict) -> list[dict]:
    cm, ev, truth, ic = _G["cm"], _G["ev"], _G["truth"], _G["ic"]
    rows = []
    for name, types in _G["subsets"].items():
        t = truth.filter(pl.col("pfam_id").is_in(types))
        proteins = t.select("accession").unique().rename({"accession": "query_acc"})
        c = calls.join(proteins, on="query_acc", how="inner").filter(pl.col("pfam_id").is_in(types))
        c = ev.restrict_tp_to_cut(t, c)
        curve = cm.protein_centric_curve(c, t, ic)
        sc = cm.cafa_scalars(curve)
        rows.append({**ident, "subset": name, "n_truth": t.height, "n_truth_proteins": t["accession"].n_unique(),
                     "n_calls": c.height, "n_tp_calls": int(c["is_tp"].sum()) if c.height else 0,
                     "fmax": sc["fmax"], "fmax_precision": sc["fmax_precision"], "fmax_recall": sc["fmax_recall"],
                     "family_fmax": cm.cafa_scalars(cm.protein_centric_curve(c, t, ic, level="family"))["fmax"]})
    return rows


def one_file(path: Path) -> list[dict]:
    ident = _G["su"].parse_arm(path.name)
    calls = pl.read_parquet(path)
    return score_subsets(calls, ident)


def control_arms() -> list[dict]:
    su, ev, truth = _G["su"], _G["ev"], _G["truth"]
    q = set(pl.read_parquet(QUERIES)["accession"])
    seqs = su.read_fasta(su.HUMAN_FASTA, q)
    arms = {
        "control.kyte_doolittle_w19_1.6": su.kd_transmem_calls(seqs, 19, 1.6),
        "control.hp_thomas_dill_run15_p2": su.hp_run_calls(seqs, "ACFILMVWY", 15, 2),
        "control.hp_thomas_dill_run12_p0": su.hp_run_calls(seqs, "ACFILMVWY", 12, 0),
    }
    rows = []
    for arm, calls in arms.items():
        scored = ev.score_calls(calls.lazy(), truth.lazy(), 0.5, "alignment", "cover")
        scored = ev.classify_scoreable(scored, truth, 0.5)
        tool, variant = arm.split(".", 1)
        ident = {"tool": tool, "variant": variant, "species": "none", "dedup": True, "arm": arm}
        rows += score_subsets(scored, ident)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qfo-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    files = sorted(CALLS.glob("swissprot.*.dedup.calls.parquet"))
    print(f"{len(files)} dedup call tables", flush=True)
    rows = []
    with Pool(args.procs, initializer=init, initargs=(args.qfo_bin,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one_file, files, chunksize=4)):
            rows += r
            if i % 500 == 0:
                print(f"  {i} tables scored", flush=True)
    init(args.qfo_bin)
    rows += control_arms()
    df = pl.DataFrame(rows)
    mya = (pl.read_parquet(METRICS).select("species", "species_mya").unique())
    df = df.join(mya, on="species", how="left")
    df.write_parquet(args.out)
    print(df.shape)
    print(df.filter(pl.col("tool") == "control").select("arm", "subset", "n_calls", "n_tp_calls", "fmax", "fmax_precision", "fmax_recall"))


if __name__ == "__main__":
    main()
