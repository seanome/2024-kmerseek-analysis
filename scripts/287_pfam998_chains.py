"""Notebook 287, part 3: on the Pfam 998 all-against-all, does ranking protein pairs by
their best chain E-value find more correct Pfam placements than ranking by the lowest
single-call E-value?

Data: notebook 243's all-against-all search of the 998 Pfam-annotated human proteins
(PR 48), notebook 241's 152 settings, every region kept
(/Users/olga/data/alphabet-logreg-pfam998/regions/). Self matches dropped. 70 settings have
a fitted E-value. N per setting (k-mers in that setting's 998-protein index) is read from
the index log's "total k-mers found" line, the number kmerseek's E-value uses (checked on
notebook 241's indexes in scripts/287_chain_scores.py).

Chains follow notebook 287's rules (sum_statistics_utils): across settings, every call's
E-value times 70, gap at most 200 residues, no overlap, same order on both proteins.

Labels, two ways:
  placement  notebook 284's rule (PR 44's): one Pfam family covers at least 20% of the
             reported query stretch and at least 20% of the reported target stretch. A
             chain reports the stretch from its first call's start to its last call's end.
  shared family  the two proteins share any Pfam family (the same label for both scores).
Average precision of each ranking of all pairs with at least one eligible call.

Outputs: notebooks/287_pfam998_average_precision.csv, notebooks/287_pfam998_precision_curve.csv,
/Users/olga/data/alphabet-logreg-pfam998/287_pair_scores.parquet
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import sum_statistics_utils as ss  # noqa: E402

D = Path("/Users/olga/data/alphabet-logreg-pfam998")
TRUTH = Path("/Users/olga/data/qfo-pfam-region-benchmark/midi-plus/truth/human_domain_truth.parquet")
MAX_GAP = 200
N_WORKERS = 8

_spec = importlib.util.spec_from_file_location("nb284", ROOT / "scripts/284_pfam998_agreement.py")
nb284 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb284)


def lengths() -> dict[str, int]:
    out, name = {}, None
    for line in open(D / "human_pfam_truth.fa"):
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = 0
        else:
            out[name] += len(line.strip())
    return out


def n_per_setting() -> pl.DataFrame:
    rows = []
    for p in sorted((D / "logs").glob("*.index.log")):
        m = re.search(r"(\d+) total k-mers found", p.read_text())
        a, k = p.name.removesuffix(".index.log").rsplit(".k", 1)
        rows.append(dict(alphabet=a, ksize=int(k), n_db=float(m.group(1))))
    return pl.DataFrame(rows)


def load(queries: list[str] | None) -> pl.DataFrame:
    lens = lengths()
    r = pl.scan_parquet(sorted((D / "regions").glob("*.parquet"))).filter(
        pl.col("region_evalue").is_finite() & (pl.col("query_name") != pl.col("target_name"))
    )
    if queries is not None:
        r = r.filter(pl.col("query_name").is_in(queries))
    r = (
        r.select("query_name", "target_name", "region_start", "region_end", "target_start",
                 "region_evalue", "alphabet", "ksize")
        .join(n_per_setting().lazy(), on=["alphabet", "ksize"])
        .with_columns(
            n_t=pl.col("target_name").replace_strict(lens, return_dtype=pl.Float64),
            setting=pl.col("alphabet") + ".k" + pl.col("ksize").cast(pl.String),
        )
    )
    return ss.add_normalised_score(r).collect()


def score_chunk(args) -> pl.DataFrame:
    queries, rules = args
    c = load(queries)
    s = ss.score_pairs(c, rules)
    if s.height == 0:
        return s
    qs_, qe_, ts_, te_ = (c[k].to_numpy() for k in ("qs", "qe", "ts", "te"))
    setting = c["setting"].to_numpy()
    first = np.array([m[0] for m in s["chain_rows"].to_list()])
    last = np.array([m[-1] for m in s["chain_rows"].to_list()])
    single = s["single_row"].to_numpy()
    # reported stretches: a chain runs from its first call's start to its last call's end
    s = s.with_columns(
        cqs=pl.Series(qs_[first]), cqe=pl.Series(qe_[last]),
        cts=pl.Series(ts_[first]), cte=pl.Series(te_[last]),
        sqs=pl.Series(qs_[single]), sqe=pl.Series(qe_[single]),
        sts=pl.Series(ts_[single]), ste=pl.Series(te_[single]),
        chain_settings=pl.Series([len(set(setting[m])) for m in s["chain_rows"].to_list()]),
    )
    return s.drop("chain_rows", "single_row")


def precision_curve(score: np.ndarray, y: np.ndarray, n_points: int = 400) -> pl.DataFrame:
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    n = np.arange(1, len(y) + 1)
    pick = np.unique(np.geomspace(1, len(y), n_points).astype(int)) - 1
    return pl.DataFrame(dict(n_pairs=n[pick], precision=hits[pick] / n[pick], recall=hits[pick] / max(y.sum(), 1)))


def main() -> None:
    t0 = time.time()
    qs = sorted(lengths())
    n_settings = n_per_setting().height
    found = (pl.scan_parquet(sorted((D / "regions").glob("*.parquet")))
             .filter(pl.col("region_evalue").is_finite())
             .select((pl.col("alphabet") + pl.col("ksize").cast(pl.String)).n_unique()).collect().item())
    pen = math.log(found)
    print(f"{len(qs)} proteins, {n_settings} indexes, {found} settings with an E-value", flush=True)
    rules = ss.ChainRules(MAX_GAP, True, pen)
    chunks = [qs[i::N_WORKERS * 4] for i in range(N_WORKERS * 4)]
    with cf.ProcessPoolExecutor(N_WORKERS) as ex:
        parts = list(ex.map(score_chunk, [(c, rules) for c in chunks]))
    s = pl.concat([p for p in parts if p.height])
    print(f"{s.height} pairs scored in {time.time() - t0:.0f} s", flush=True)

    truth = pl.read_parquet(TRUTH)
    fams = truth.group_by("accession").agg(fams=pl.col("pfam_id").unique())
    lab = []
    for pre in ("c", "s"):
        x = s.select("query_name", "target_name", pl.col(f"{pre}qs").alias("qs"), pl.col(f"{pre}qe").alias("qe"),
                     pl.col(f"{pre}ts").alias("ts"), pl.col(f"{pre}te").alias("te"))
        x = nb284.label(x, truth).select("query_name", "target_name", pl.col("correct").alias(f"correct_{pre}"))
        lab.append(x)
    s = s.join(lab[0], on=["query_name", "target_name"]).join(lab[1], on=["query_name", "target_name"])
    s = (s.join(fams.rename({"accession": "query_name", "fams": "qf"}), on="query_name", how="left")
         .join(fams.rename({"accession": "target_name", "fams": "tf"}), on="target_name", how="left")
         .with_columns(shared_family=pl.col("qf").list.set_intersection(pl.col("tf")).list.len().fill_null(0) > 0)
         .drop("qf", "tf"))
    s.write_parquet(D / "287_pair_scores.parquet")

    s = s.sort("query_name", "target_name")
    chain_score = -s["chain_log_e"].to_numpy()
    single_score = -s["single_log_e"].to_numpy()
    rows, curves = [], []
    for label, yc, ys in (("placement (284 rule)", s["correct_c"].to_numpy(), s["correct_s"].to_numpy()),
                          ("shared Pfam family", s["shared_family"].to_numpy(), s["shared_family"].to_numpy())):
        rows.append(dict(label=label, n_pairs=s.height, n_settings_factor=found, max_gap=MAX_GAP,
                         n_correct_chain=int(yc.sum()), n_correct_single=int(ys.sum()),
                         ap_chain=nb284.average_precision(chain_score, yc),
                         ap_single=nb284.average_precision(single_score, ys),
                         n_pairs_chain_e_below_1=int((s["chain_e"] < 1).sum()),
                         n_pairs_single_e_below_1=int((s["single_e"] < 1).sum()),
                         n_pairs_chain_of_2_or_more=int((s["chain_r"] > 1).sum())))
        curves.append(precision_curve(chain_score, yc).with_columns(label=pl.lit(label), score=pl.lit("chain E-value")))
        curves.append(precision_curve(single_score, ys).with_columns(label=pl.lit(label), score=pl.lit("lowest single E-value")))
    out = pl.DataFrame(rows).with_columns(ap_difference=pl.col("ap_chain") - pl.col("ap_single"))
    out.write_csv(ROOT / "notebooks/287_pfam998_average_precision.csv")
    pl.concat(curves).write_csv(ROOT / "notebooks/287_pfam998_precision_curve.csv")
    print(out.to_pandas().to_string(), flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
