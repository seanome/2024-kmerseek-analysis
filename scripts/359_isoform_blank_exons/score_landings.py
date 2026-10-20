#!/usr/bin/env python3
"""Notebook 359, steps 6-8 and 10: thresholds at a 1% decoy rate, the three rules, the
hydropathy scan control, and the surviving landings with their residues.

    score_landings.py --build-dir <build_exons.py out> --best-dir <reduce_calls.py out>
                      --regions <regions.parquet> --settings <settings.tsv>
                      --example-seed <int> --out-dir <dir>

<best-dir> holds <setting>.real.best.tsv and <setting>.decoy.best.tsv for each setting in
settings.tsv, or <setting>.nofit when the index refused the E-value fit. Writes
summary.csv (one row per setting, the union and the scan, overall and by stratum),
landings.parquet, scan.parquet, survivors.parquet, examples.json and verdict.json.
The rule is in notebooks/359_data/preregistered.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import isoform_blank_exons_359 as ib  # noqa: E402

MAX_DECOY_RATE = 0.01
MIN_NET = 0.10
MIN_NAMED = 0.5
MAX_SCAN_OVERLAP = 0.5


def read_best(path: Path) -> dict[str, dict]:
    with open(path) as fh:
        return {r["exon_id"]: {**r, "evalue": float(r["evalue"])} for r in csv.DictReader(fh, delimiter="\t")}


def named_call(call: dict, regions: dict) -> tuple[bool, str, str, float]:
    """Is the best landing call on a named Swiss-Prot feature, with at least half of its
    target interval inside the feature's own residues (not only the 21-residue flanks)?"""
    reg = regions[call["target"]]
    feat = (reg["start"] - reg["cut_start"], reg["end"] - reg["cut_start"] + 1)
    t = (int(call["tstart"]), int(call["tend"]))
    inside = ib.overlap(t, feat) / max(1, t[1] - t[0])
    named = ib.is_named(reg["feature_type"], reg["description"]) and inside >= 0.5
    return named, reg["feature_type"], reg["description"], inside


def rules(row: dict) -> dict:
    n = row["n_blank"]
    row["net_fraction"] = row["net_landings"] / n if n else 0.0
    row["named_fraction"] = row["named"] / row["real_landings"] if row["real_landings"] else 0.0
    row["scan_overlap_fraction"] = row["scan_overlap"] / row["real_landings"] if row["real_landings"] else math.nan
    row["rule1_net_ge_10pct"] = row["net_fraction"] >= MIN_NET
    row["rule2_named_ge_half"] = row["named_fraction"] >= MIN_NAMED
    row["rule3_scan_overlap_lt_half"] = bool(row["real_landings"]) and row["scan_overlap_fraction"] < MAX_SCAN_OVERLAP
    row["passes"] = row["rule1_net_ge_10pct"] and row["rule2_named_ge_half"] and row["rule3_scan_overlap_lt_half"]
    return row


def main(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bl = pl.read_parquet(Path(a.build_dir) / "blank_exons.parquet")
    n_blank = bl.height
    strata = {"all": set(bl["exon_id"])}
    strata["basic_ccds"] = set(bl.filter(pl.col("in_basic_ccds_isoform"))["exon_id"])
    strata["not_basic_ccds"] = strata["all"] - strata["basic_ccds"]
    regions = {
        r["name"]: r
        for r in pl.read_parquet(a.regions, columns=["name", "feature_type", "description", "start", "end", "cut_start"])
        .with_columns(pl.col("description").fill_null(""))
        .iter_rows(named=True)
    }
    settings = [r["setting"] for r in csv.DictReader(open(a.settings), delimiter="\t")]

    # ---- hydropathy scan on the same real and decoy proteins ---------------------------
    scan_rows = []
    for r in bl.iter_rows(named=True):
        seg = (r["p_start"], r["p_end"])
        for side, prot in (("real", r["protein"]), ("decoy", r["decoy_protein"])):
            scan_rows.append({"exon_id": r["exon_id"], "side": side, "score": ib.scan_best_landing(ib.window_means(prot), seg)})
    scan = pl.DataFrame(scan_rows)
    scan.write_parquet(out / "scan.parquet")
    scan_real = {r["exon_id"]: r["score"] for r in scan.filter(pl.col("side") == "real").iter_rows(named=True) if r["score"] < math.inf}
    scan_decoy = {r["exon_id"]: r["score"] for r in scan.filter(pl.col("side") == "decoy").iter_rows(named=True) if r["score"] < math.inf}
    scan_t = ib.threshold_at_decoy_rate(scan_decoy, n_blank, MAX_DECOY_RATE)
    S = ib.landed_at(scan_real, scan_t)
    S_decoy = ib.landed_at(scan_decoy, scan_t)

    # ---- kmerseek, per setting --------------------------------------------------------
    real, decoy, nofit = {}, {}, []
    for s in settings:
        if (Path(a.best_dir) / f"{s}.nofit").exists():
            nofit.append(s)
            continue
        real[s] = read_best(Path(a.best_dir) / f"{s}.real.best.tsv")
        decoy[s] = read_best(Path(a.best_dir) / f"{s}.decoy.best.tsv")
    fitted = [s for s in settings if s not in nofit]
    thresholds = {
        s: ib.threshold_at_decoy_rate({k: v["evalue"] for k, v in decoy[s].items()}, n_blank, MAX_DECOY_RATE)
        for s in fitted
    }
    union_t, union_q = ib.union_thresholds(
        {s: {k: v["evalue"] for k, v in decoy[s].items()} for s in fitted}, n_blank, MAX_DECOY_RATE
    )

    land_rows = []
    for s in fitted:
        for side, d in (("real", real[s]), ("decoy", decoy[s])):
            for k, v in d.items():
                land_rows.append({"setting": s, "side": side, **{c: v[c] for c in ("exon_id", "evalue", "target", "qstart", "qend", "tstart", "tend", "qseq", "tseq")}})
    landings = pl.DataFrame(land_rows) if land_rows else pl.DataFrame()
    landings.write_parquet(out / "landings.parquet")

    def evaluate(label, landed_real, landed_decoy, best_call, threshold):
        rows = []
        for stratum, ids in strata.items():
            K = landed_real & ids
            D = landed_decoy & ids
            named = 0
            for k in K:
                named += named_call(best_call[k], regions)[0]
            rows.append(rules({
                "setting": label, "stratum": stratum, "threshold": threshold,
                "n_blank": len(ids), "real_landings": len(K), "decoy_landings": len(D),
                "net_landings": len(K) - len(D), "named": named,
                "scan_landings": len(S & ids), "scan_overlap": len(K & S),
            }))  # fmt: skip
        return rows

    rows = []
    for s in fitted:
        t = thresholds[s]
        K = ib.landed_at({k: v["evalue"] for k, v in real[s].items()}, t)
        D = ib.landed_at({k: v["evalue"] for k, v in decoy[s].items()}, t)
        rows += evaluate(s, K, D, real[s], t)
    # union: each setting at its union threshold; the exon's call is the smallest E among
    # the settings that land on it there
    K_u, D_u, best_u = set(), set(), {}
    for s in fitted:
        Ks = ib.landed_at({k: v["evalue"] for k, v in real[s].items()}, union_t[s])
        K_u |= Ks
        D_u |= ib.landed_at({k: v["evalue"] for k, v in decoy[s].items()}, union_t[s])
        for k in Ks:
            if k not in best_u or real[s][k]["evalue"] < best_u[k]["evalue"]:
                best_u[k] = {**real[s][k], "setting": s}
    rows += evaluate("union_of_fitted_settings", K_u, D_u, best_u, f"per-setting cap q={union_q:.5f}")
    for stratum, ids in strata.items():
        rows.append({"setting": "kyte_doolittle_scan", "stratum": stratum, "threshold": -scan_t if scan_t not in (math.inf, -math.inf) else scan_t,
                     "n_blank": len(ids), "real_landings": len(S & ids), "decoy_landings": len(S_decoy & ids),
                     "net_landings": len(S & ids) - len(S_decoy & ids)})  # fmt: skip
    summary = pl.DataFrame(rows, infer_schema_length=None)
    summary.write_csv(out / "summary.csv")

    # ---- surviving landings (union) with residues; five at random for the figure --------
    surv = []
    for k in sorted(K_u):
        c = best_u[k]
        named, ftype, desc, inside = named_call(c, regions)
        b = bl.filter(pl.col("exon_id") == k).row(0, named=True)
        q, t = c["qseq"], c["tseq"]
        n_id = sum(x == y for x, y in zip(q, t))
        surv.append({
            "exon_id": k, "gene": b["gene"], "rep_isoform": b["rep_isoform"], "setting": c["setting"],
            "evalue": c["evalue"], "exon_p_start": b["p_start"], "exon_p_end": b["p_end"],
            "qstart": int(c["qstart"]), "qend": int(c["qend"]), "target": c["target"],
            "tstart": int(c["tstart"]), "tend": int(c["tend"]), "feature_type": ftype, "description": desc,
            "fraction_of_call_in_feature": inside, "named": named, "qseq": q, "tseq": t,
            "match": "".join("|" if x == y else " " for x, y in zip(q, t)),
            "n_identical": n_id, "n_aligned": min(len(q), len(t)), "in_basic_ccds_isoform": b["in_basic_ccds_isoform"],
            "scan_also_lands": k in S,
        })  # fmt: skip
    survivors = pl.DataFrame(surv, infer_schema_length=None) if surv else pl.DataFrame()
    survivors.write_parquet(out / "survivors.parquet")
    picks = sorted(random.Random(a.example_seed).sample(sorted(K_u), min(5, len(K_u))))
    json.dump({"seed": a.example_seed, "exon_ids": picks}, open(out / "examples.json", "w"), indent=2)

    overall = summary.filter((pl.col("stratum") == "all") & (pl.col("setting") != "kyte_doolittle_scan"))
    passing = overall.filter(pl.col("passes"))["setting"].to_list()
    verdict = {
        "verdict": "PASS" if passing else "KILL",
        "passing_rows": passing,
        "n_blank": n_blank,
        "settings_without_fit": nofit,
        "thresholds": {s: thresholds[s] for s in fitted},
        "union_thresholds": union_t,
        "union_per_setting_cap": union_q,
        "scan_threshold": scan_t,
        "rule": {"max_decoy_rate": MAX_DECOY_RATE, "min_net": MIN_NET, "min_named": MIN_NAMED, "max_scan_overlap": MAX_SCAN_OVERLAP},
        "rows": overall.to_dicts(),
    }  # fmt: skip
    json.dump(verdict, open(out / "verdict.json", "w"), indent=2, default=str)
    print(summary)
    print(json.dumps({k: verdict[k] for k in ("verdict", "passing_rows", "n_blank", "settings_without_fit")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--build-dir", required=True)
    p.add_argument("--best-dir", required=True)
    p.add_argument("--regions", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--example-seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
