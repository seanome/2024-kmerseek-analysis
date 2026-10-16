#!/usr/bin/env python3
"""Score every method's ranked lists and write the tables the notebook reads.

Every method ranked all of Swiss-Prot for every query. This step turns those lists into
small tables, so the notebook draws figures instead of re-reading 1.2 GB of hits.

A ranked list belongs to one method and one setting: a kmerseek setting label, a scan's
list name, or a tool's one setting. Scores are always "larger is better": a kmerseek
setting ranks by minus its smallest region E-value, a sequence tool and Foldseek by bit
score, and a scan by minus its distance.

For one query and one ranked list:

  threshold       the score of the first false positive: the best-scoring Swiss-Prot entry
                  that is not in the truth set at all. A list with no false positive in it
                  has no threshold, so its recall is left empty and `no_false_positive` is
                  true. A scan's rows are the top ones plus every truth entry, so an empty
                  threshold there means the scan reported too few entries to see one, not
                  that the method made no mistake.
  recall          cross-lineage truth entries scoring strictly better than the threshold,
                  over every cross-lineage truth entry there is. A hit on the query's own
                  lineage is left out of both, as is the query's own entry.
  n_found         how many cross-lineage truth entries that is.
  n_hits          entries scoring strictly better than the threshold. Only truth entries
                  can be there, since the threshold is the first entry that is not one.
  auprc           average precision over the whole list, cross-lineage truth as the
                  positives. A positive the method never reported counts as not retrieved.

A null draw is scored twice: at its own first false positive, and at the score its real
query's first false positive sat at (`n_found_at_real_threshold`). The second is what the
null comparison reads. Judged at its own threshold a scrambled query almost always has a
non-truth entry first, so it finds nothing and would pass any null rule; holding the score
cutoff fixed asks the question the control is for, which is how many ice-binding proteins
a scrambled query reaches when it has to clear the same bar.

Written to --outdir:
  metrics.tsv       one row per method, setting, query and overlap cutoff
  grid.tsv          counts per method, setting, query lineage and target lineage
  verdict.json      the decision rule's three numbers, in sample and held out
  hits_residues.md  the residues of every cross-lineage kmerseek hit above threshold
  afdb_coverage.tsv truth entries per lineage with and without an AlphaFold model, which
                    is what Foldseek on structures can and cannot reach
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HP_ALPHABET = "hp_pbotc_1st_ed2"
# The window scan applies its own minimum-overlap cutoffs in bin/scans.py, before it keeps
# the top rows, and emits one list per cutoff. Filtering here instead left a list of
# nothing but truth entries and so no false positive to set the threshold.
OVERLAP_CUTOFFS = [0]
# Lineage 6, plants, has no reviewed entry, so the odd half is 1, 3, 5, 7 and the even
# half is 2 and 4. The brief fixes this split before any run.
IN_SAMPLE = ["1 fish, type I AFP", "3 fish, type II AFP", "5 insects", "7 bacteria"]
HELD_OUT = ["2 fish, AFGP", "4 fish, type III AFP"]
MARGIN = 0.10
NULL_B_FRACTION = 0.10


def average_precision(is_positive, n_positive):
    """Average precision over a ranked list, with unreported positives never retrieved."""
    if n_positive == 0:
        return float("nan")
    hits = np.cumsum(is_positive)
    ranks = np.arange(1, len(is_positive) + 1)
    precision = hits / ranks
    return float(precision[is_positive].sum() / n_positive)


def score_one(
    targets, scores, truth_lineage, query_lineage, n_cross, fixed_threshold=None
):
    """Metrics for one query against one ranked list.

    targets and scores are that list's rows, in any order; both are read as arrays.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    targets = np.asarray(targets)[order]
    scores = np.asarray(scores, dtype=float)[order]
    lineage = np.array([truth_lineage.get(t, "") for t in targets])
    is_truth = lineage != ""
    is_own = is_truth & (lineage == query_lineage)
    is_cross = is_truth & ~is_own

    fp = np.flatnonzero(~is_truth)
    if len(fp) == 0:
        # No false positive in the list, so nothing bounds the hits: recall is left empty
        # rather than counting every entry as found.
        threshold = None
        above = np.ones(len(scores), dtype=bool)
    else:
        threshold = scores[fp[0]]
        above = scores > threshold
    n_found = int((is_cross & above).sum())
    # Own-lineage rows are dropped before average precision, so they neither help nor
    # hurt; the positives are the cross-lineage truth entries.
    keep = ~is_own
    return {
        "threshold": None if threshold is None else float(threshold),
        "no_false_positive": threshold is None,
        "n_hits": int(above.sum()),
        "n_found": n_found,
        "recall": (None if threshold is None or not n_cross else n_found / n_cross),
        "auprc": average_precision(is_cross[keep], n_cross),
        "n_ranked_kept": int(len(scores)),
        "n_found_at_real_threshold": (
            int((is_cross & (scores > fixed_threshold)).sum())
            if fixed_threshold is not None
            else n_found
        ),
        # A dict, not a joined string: a lineage name holds a comma ("2 fish, AFGP"),
        # so splitting a joined string cut every name in two.
        "found_by_lineage": dict(
            zip(*np.unique(lineage[is_cross & above], return_counts=True))
        ),
    }


def lists_from_scans(path, cutoffs):
    """(method, setting, overlap cutoff) -> frame of query_id, target, score."""
    t = pl.read_parquet(path)
    if "overlap" not in t.columns:
        # A scans.parquet written before the window scan existed has no overlap column.
        t = t.with_columns(overlap=pl.lit(None, dtype=pl.Int64))
    for (method, list_name), g in t.group_by(["method", "list_name"]):
        has_overlap = g["overlap"].null_count() < g.height
        for cut in cutoffs if has_overlap else [0]:
            sub = g.filter(pl.col("overlap") >= cut) if cut else g
            yield method, list_name, cut, sub.select("query_id", "target", "score")


def lists_from_tools(hits_dir):
    """One list per sequence tool, and one per Foldseek mode.

    phmmer and jackhmmer searched the queries in chunks, so their files are concatenated.
    Foldseek's two files are two different queries of the same database: AFDB models for
    the real queries, and 3Di letters predicted from sequence for every query.
    """
    for tool in ["phmmer", "jackhmmer", "mmseqs2"]:
        files = sorted(glob.glob(str(hits_dir / tool / "*.tsv.gz")))
        if not files:
            continue
        t = pl.concat([pl.read_csv(f, separator="\t") for f in files])
        yield tool, "default", 0, t.select("query_id", "target", "score")
    for f in sorted(glob.glob(str(hits_dir / "foldseek" / "*.tsv.gz"))):
        mode = Path(f).name.replace("foldseek_", "").replace(".tsv.gz", "")
        t = pl.read_csv(f, separator="\t")
        yield "foldseek", mode, 0, t.select("query_id", "target", "score")


def lists_from_kmerseek(hits_dir):
    for f in sorted(glob.glob(str(hits_dir / "kmerseek" / "*.best.parquet"))):
        label = Path(f).name.replace(".best.parquet", "")
        t = pl.read_parquet(
            f, columns=["query_name", "target_name", "min_region_evalue"]
        )
        if not t.height:
            continue
        yield "kmerseek", label, 0, t.rename(
            {"query_name": "query_id", "target_name": "target"}
        ).select("query_id", "target", (-pl.col("min_region_evalue")).alias("score"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hits", type=Path, required=True, help="results/hits")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--draws", type=Path, required=True)
    ap.add_argument("--alphabet-classes", type=Path, required=True)
    ap.add_argument(
        "--afdb-accessions",
        type=Path,
        help="accessions.txt from the Foldseek AFDB database: one accession per model",
    )
    ap.add_argument("--max-hit-rows", type=int, default=400)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth = pl.read_csv(args.truth, separator="\t")
    truth_lineage = dict(zip(truth["Entry"], truth["lineage"]))
    draws = pl.read_csv(args.draws, separator="\t")
    q_lineage = dict(zip(draws["query_id"], draws["lineage"]))
    q_kind = dict(zip(draws["query_id"], draws["kind"]))
    q_acc = dict(zip(draws["query_id"], draws["accession"]))
    per_lineage = truth["lineage"].value_counts()
    n_truth_in = dict(zip(per_lineage["lineage"], per_lineage["count"]))
    n_cross_of = {
        lin: truth.height - n_truth_in.get(lin, 0) for lin in set(q_lineage.values())
    }

    if args.afdb_accessions:
        modelled = set(args.afdb_accessions.read_text().split())
        cov = (
            truth.with_columns(has_afdb_model=pl.col("Entry").is_in(list(modelled)))
            .group_by("lineage")
            .agg(
                pl.len().alias("n_truth"),
                pl.col("has_afdb_model").sum().alias("n_with_model"),
            )
            .with_columns(n_without_model=pl.col("n_truth") - pl.col("n_with_model"))
            .sort("lineage")
        )
        cov.write_csv(args.outdir / "afdb_coverage.tsv", separator="\t")
        print(
            f"AFDB models: {len(modelled)} accessions; "
            f"{int(cov['n_without_model'].sum())} of {truth.height} truth entries have none",
            file=sys.stderr,
        )

    sources = list(
        lists_from_scans(args.hits / "scans" / "scans.parquet", OVERLAP_CUTOFFS)
    )
    sources += list(lists_from_tools(args.hits))
    sources += list(lists_from_kmerseek(args.hits))

    rows, grid = [], []
    for method, setting, cut, frame in sources:
        groups = {
            (k[0] if isinstance(k, tuple) else k): g
            for k, g in frame.group_by("query_id")
        }
        unknown = sorted(set(groups) - set(q_lineage))
        if unknown:
            sys.exit(f"{method}/{setting} ranked queries not in the draws: {unknown}")
        # The real queries first: a null draw is also judged at its real query's cutoff.
        real_threshold = {}
        for qid, g in groups.items():
            if q_kind[qid] != "real":
                continue
            lin = q_lineage[qid]
            real_threshold[q_acc[qid]] = score_one(
                g["target"].to_list(),
                g["score"].to_list(),
                truth_lineage,
                lin,
                n_cross_of[lin],
            )["threshold"]
        for qid, g in groups.items():
            lin = q_lineage[qid]
            m = score_one(
                g["target"].to_list(),
                g["score"].to_list(),
                truth_lineage,
                lin,
                n_cross_of[lin],
                real_threshold.get(q_acc[qid]),
            )
            rows.append(
                {
                    "method": method,
                    "setting_label": setting,
                    "overlap_min": cut,
                    "query_id": qid,
                    "accession": q_acc[qid],
                    "query_lineage": lin,
                    "kind": q_kind[qid],
                    "n_cross_truth": n_cross_of[lin],
                    **{k: v for k, v in m.items() if k != "found_by_lineage"},
                }
            )
            if q_kind[qid] == "real":
                for target_lineage, n in m["found_by_lineage"].items():
                    grid.append(
                        {
                            "method": method,
                            "setting_label": setting,
                            "overlap_min": cut,
                            "query_lineage": lin,
                            "target_lineage": target_lineage,
                            "n_found": int(n),
                        }
                    )
        print(f"scored {method}/{setting} overlap>={cut}", file=sys.stderr)

    metrics = pl.DataFrame(rows)
    metrics.write_csv(args.outdir / "metrics.tsv", separator="\t")
    counted = (
        pl.DataFrame(grid).sort(
            ["method", "setting_label", "query_lineage", "target_lineage"]
        )
        if grid
        else pl.DataFrame(
            schema={
                "method": pl.Utf8,
                "setting_label": pl.Utf8,
                "overlap_min": pl.Int64,
                "query_lineage": pl.Utf8,
                "target_lineage": pl.Utf8,
                "n_found": pl.Int64,
            }
        )
    )
    counted.write_csv(args.outdir / "grid.tsv", separator="\t")

    verdict = decide(metrics, args.outdir)
    (args.outdir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    write_hits_residues(
        args.hits,
        metrics,
        truth_lineage,
        q_lineage,
        args.alphabet_classes,
        args.outdir / "hits_residues.md",
        args.max_hit_rows,
        verdict,
    )
    return 0


def decide(metrics, outdir):
    """The decision rule, fixed in the brief before any run."""
    real = metrics.filter(pl.col("kind") == "real")
    ks = real.filter(pl.col("method") == "kmerseek")
    if not ks.height:
        return {
            "verdict": "FAIL",
            "reason": "no kmerseek setting produced a ranked list",
        }

    # 1. Choose the setting with the highest mean recall over the in-sample lineages.
    in_s = (
        ks.filter(
            pl.col("query_lineage").is_in(IN_SAMPLE) & pl.col("recall").is_not_null()
        )
        .group_by("setting_label")
        .agg(
            pl.col("recall").mean().alias("mean_recall"),
            pl.len().alias("n_queries"),
        )
        .sort(["mean_recall", "setting_label"], descending=[True, False])
    )
    chosen = in_s["setting_label"][0]
    chosen_in_sample = float(in_s["mean_recall"][0])

    def best_other(method_names, lineage):
        t = real.filter(
            pl.col("method").is_in(method_names) & (pl.col("query_lineage") == lineage)
        )
        return (
            (
                float(t["recall"].max()),
                t.sort("recall", descending=True)["setting_label"][0],
            )
            if t.height
            else (float("nan"), "")
        )

    per_lineage, passed = {}, []
    for lineage in HELD_OUT:
        k = ks.filter(
            (pl.col("setting_label") == chosen) & (pl.col("query_lineage") == lineage)
        )
        k_recall = (
            float(k["recall"][0])
            if k.height and k["recall"][0] is not None
            else float("nan")
        )
        brief_best, brief_which = best_other(["composition", "period"], lineage)
        all_best, all_which = best_other(
            ["composition", "period", "window", "motif"], lineage
        )
        nulls = metrics.filter(
            (pl.col("method") == "kmerseek")
            & (pl.col("setting_label") == chosen)
            & (pl.col("query_lineage") == lineage)
        )
        real_hits = float(nulls.filter(pl.col("kind") == "real")["n_found"].sum())
        b = nulls.filter(pl.col("kind") == "nullB")["n_found_at_real_threshold"]
        null_b_median = float(b.median()) if b.len() else float("nan")
        ok_margin = (k_recall - brief_best) >= MARGIN
        ok_margin_with_window = (k_recall - all_best) >= MARGIN
        ok_null = (
            null_b_median <= NULL_B_FRACTION * real_hits if real_hits > 0 else False
        )
        per_lineage[lineage] = {
            "kmerseek_recall": k_recall,
            "best_composition_or_period_recall": brief_best,
            "best_composition_or_period_list": brief_which,
            "best_no_search_recall": all_best,
            "best_no_search_list": all_which,
            "margin_over_composition_and_period": k_recall - brief_best,
            "margin_over_every_no_search_method": k_recall - all_best,
            "real_n_found": real_hits,
            "null_b_median_n_found_at_real_threshold": null_b_median,
            "passes_margin_rule": bool(ok_margin),
            "passes_margin_including_window_scan": bool(ok_margin_with_window),
            "passes_null_b_rule": bool(ok_null),
        }
        if ok_margin and ok_null:
            passed.append(lineage)

    return {
        "verdict": "PASS" if len(passed) >= 2 else "FAIL",
        "chosen_setting": chosen,
        "chosen_on": IN_SAMPLE,
        "chosen_mean_recall_in_sample": chosen_in_sample,
        "held_out": HELD_OUT,
        "held_out_lineages_passing": passed,
        "per_held_out_lineage": per_lineage,
        "rule": {
            "margin": MARGIN,
            "null_b_fraction": NULL_B_FRACTION,
            "needs_held_out_lineages": 2,
        },
    }


def hp_string(seq, hydrophobic):
    return "".join("h" if a in hydrophobic else "p" for a in seq)


def write_hits_residues(
    hits, metrics, truth_lineage, q_lineage, classes_path, out, max_rows, verdict
):
    """Residues of every cross-lineage kmerseek hit that beats its list's threshold."""
    cls = pl.read_csv(classes_path, separator="\t")
    hydrophobic = set(
        cls.filter((pl.col("alphabet") == HP_ALPHABET) & (pl.col("class_index") == 0))[
            "residue"
        ]
    )
    thresholds = {
        (r["setting_label"], r["query_id"]): r["threshold"]
        for r in metrics.filter(
            (pl.col("method") == "kmerseek") & (pl.col("kind") == "real")
        ).iter_rows(named=True)
    }
    lines = [
        "# Cross-lineage kmerseek hits, residue by residue",
        "",
        "Every hit a kmerseek setting makes on an ice-binding protein of another lineage,",
        "scoring better than that setting's first false positive. Coordinates are",
        "kmerseek's own: 0-based and end-exclusive. kmerseek alignments have no gaps.",
        "The h/p line is the two-letter hydrophobic-polar alphabet "
        f"({HP_ALPHABET}): h where the two agree.",
        "",
    ]
    n = 0
    for f in sorted(glob.glob(str(hits / "kmerseek" / "*.truth.parquet"))):
        label = Path(f).name.replace(".truth.parquet", "")
        t = pl.read_parquet(f)
        if not t.height:
            continue
        for r in t.iter_rows(named=True):
            qid, target = r["query_name"], r["target_name"]
            if qid not in q_lineage or truth_lineage.get(target, "") in (
                "",
                q_lineage.get(qid, ""),
            ):
                continue
            thr = thresholds.get((label, qid))
            if thr is None or not (-r["region_evalue"] > thr):
                continue
            n += 1
            if n > max_rows:
                continue
            qs, ts = r.get("region_subseq", ""), r.get("target_subseq", "")
            if not qs or not ts or len(qs) != len(ts):
                lines += [
                    f"## {label}: {qid} vs {target}",
                    "",
                    f"region_evalue {r['region_evalue']:.3g}, "
                    f"query {r['region_start']}-{r['region_end']}, "
                    f"target {r['target_start']}-{r['target_end']}; "
                    "residues not written: the two spans differ in length "
                    f"({len(qs)} and {len(ts)}), which is a chained region.",
                    "",
                ]
                continue
            match = "".join("|" if a == b else " " for a, b in zip(qs, ts))
            hq, ht = hp_string(qs, hydrophobic), hp_string(ts, hydrophobic)
            hp_match = "".join("|" if a == b else " " for a, b in zip(hq, ht))
            runs = [len(x) for x in "".join(hp_match).split(" ") if x]
            lines += [
                f"## {label}: {qid} vs {target} ({truth_lineage[target]})",
                "",
                f"region_evalue {r['region_evalue']:.3g}, "
                f"region_ka_bits {r.get('region_ka_bits', float('nan')):.2f}, "
                f"{sum(1 for a, b in zip(qs, ts) if a == b)} of {len(qs)} identical, "
                f"longest matching h/p run {max(runs) if runs else 0}",
                "",
                "```",
                f"query  {r['region_start']:>6} {qs} {r['region_end']}",
                f"              {match}",
                f"target {r['target_start']:>6} {ts} {r['target_end']}",
                "",
                f"query  h/p    {hq}",
                f"              {hp_match}",
                f"target h/p    {ht}",
                "```",
                "",
            ]
    header = f"{n} cross-lineage hits above threshold"
    if n > max_rows:
        header += f"; the first {max_rows} are written out"
    lines.insert(6, header + ".")
    lines.insert(7, "")
    out.write_text("\n".join(lines) + "\n")
    print(f"hits_residues.md: {n} cross-lineage hits above threshold", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
