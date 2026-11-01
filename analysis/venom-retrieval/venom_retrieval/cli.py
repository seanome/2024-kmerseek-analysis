"""The command line for the venom-peptide retrieval benchmark.

Five subcommands, each writing its outputs to the output directory and reading the ones
before it, so any step can be rerun without repeating the others:

  manifest    build the peptide table from the downloaded sources
  splits      drop duplicate sequences, group related peptides, pick the query sets
  collisions  count the peptides each alphabet cannot tell apart
  run         rank the reference for every query with every method
  evaluate    turn the rankings into the per-method numbers
  all         the five in order

Every run writes `config.used.json` next to its results, holding the config as it was
read, so a results file can always be traced to the settings that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from . import encodings, evaluate, manifest, methods, splits
from .align import GAP_EXTEND, GAP_OPEN, MATRIX

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pilot.json"


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    config["kmer_settings"] = [tuple(pair) for pair in config["kmer_settings"]]
    config["collision_report_settings"] = [
        tuple(pair) for pair in config["collision_report_settings"]
    ]
    return config


def cmd_manifest(config: dict, data_dir: Path, out_dir: Path) -> None:
    frame, dropped = manifest.build(data_dir)
    frame.write_parquet(out_dir / "manifest.parquet")
    dropped.write_csv(out_dir / "manifest_dropped.csv")
    print(f"manifest: {frame.height} peptides, {frame.width} columns")
    print(
        frame.group_by("control_class", "source")
        .len()
        .sort("len", descending=True)
    )
    print("\nlabel coverage:")
    for column, endpoint in (
        ("framework_missing", "framework"),
        ("connectivity_missing", "complete connectivity"),
        ("fold_missing", "a fold label"),
        ("target_missing", "a target or activity label"),
    ):
        have = frame.height - int(frame[column].sum())
        print(f"  {have:6d} of {frame.height} peptides have {endpoint}")
    inferred = int(frame["mature_boundary_inferred"].sum())
    print(
        f"  {inferred} of {frame.height} mature-peptide boundaries were derived rather "
        f"than read from an annotation; every one is written out in "
        f"mature_boundary_inference"
    )


def cmd_splits(config: dict, data_dir: Path, out_dir: Path) -> None:
    frame = pl.read_parquet(out_dir / "manifest.parquet")
    kept, duplicates = splits.remove_exact_duplicates(frame)
    duplicates.collapsed.write_csv(out_dir / "duplicate_sequences.csv")
    print(
        f"exact duplicates: {duplicates.n_before} peptides -> {duplicates.n_after}, "
        f"{duplicates.n_removed} identical sequences collapsed"
    )

    started = time.perf_counter()
    pairs = splits.similar_pairs(
        kept["mature_peptide"].to_list(),
        threshold=config["group_threshold"],
        processes=config["processes"],
    )
    print(
        f"similarity: {len(pairs)} pairs at or above a {config['group_threshold']} "
        f"score ratio, in {time.perf_counter() - started:.0f} s"
    )
    kept.write_parquet(out_dir / "peptides.parquet")
    identifiers = kept["peptide_id"].to_list()
    pl.DataFrame(
        {
            "peptide_a": [identifiers[i] for i, _, _ in pairs],
            "peptide_b": [identifiers[j] for _, j, _ in pairs],
            "score_ratio": [ratio for _, _, ratio in pairs],
        }
    ).sort("peptide_a", "peptide_b").write_parquet(out_dir / "similar_pairs.parquet")

    sets = splits.neighbourhoods(kept, pairs, config["use_gene_family"])
    sizes = splits.neighbourhood_sizes(sets)
    sizes.write_csv(out_dir / "neighbourhood_sizes.csv")
    print(
        f"neighbourhoods: median {int(sizes['n_neighbours'].median())} peptides, "
        f"largest {int(sizes['n_neighbours'].max())}, "
        f"{int((sizes['n_neighbours'] == 1).sum())} peptides with no neighbour"
    )

    queries = []
    for endpoint in config["endpoints"]:
        chosen = splits.choose_queries(
            kept, sets, endpoint, config["n_queries_per_endpoint"], config["seed"]
        )
        print(f"  {endpoint}: {chosen.height} queries")
        queries.append(chosen.select("peptide_id").with_columns(endpoint=pl.lit(endpoint)))
    pl.concat(queries).sort("endpoint", "peptide_id").write_csv(
        out_dir / "queries.csv"
    )


def _load_neighbourhoods(config: dict, out_dir: Path, frame: pl.DataFrame) -> list[set[int]]:
    """Rebuild the neighbourhoods from the pair table the splits step wrote."""
    position = {name: i for i, name in enumerate(frame["peptide_id"].to_list())}
    pairs_frame = pl.read_parquet(out_dir / "similar_pairs.parquet")
    pairs = [
        (position[a], position[b], ratio)
        for a, b, ratio in zip(
            pairs_frame["peptide_a"], pairs_frame["peptide_b"], pairs_frame["score_ratio"]
        )
    ]
    return splits.neighbourhoods(frame, pairs, config["use_gene_family"])


def cmd_collisions(config: dict, data_dir: Path, out_dir: Path) -> None:
    grouped = pl.read_parquet(out_dir / "peptides.parquet")
    rows = []
    for subset_name, subset in (
        ("all_peptides", grouped),
        ("toxins", grouped.filter(pl.col("is_toxin"))),
        (
            "cysteine_rich_toxins",
            grouped.filter(pl.col("is_toxin") & (pl.col("n_cysteines") >= 4)),
        ),
    ):
        peptides = dict(zip(subset["peptide_id"], subset["mature_peptide"]))
        for alphabet, ksize in config["collision_report_settings"]:
            row = encodings.collision_report(peptides, alphabet, ksize).as_row()
            row["subset"] = subset_name
            rows.append(row)
    report = pl.DataFrame(rows).select(["subset"] + [c for c in rows[0] if c != "subset"])
    report.write_csv(out_dir / "collision_report.csv")
    print(report)

    # The groups themselves, for the alphabet that collapses the most, so a reader can see
    # which peptides an encoding cannot tell apart rather than only how many.
    cysrich = grouped.filter(pl.col("is_toxin") & (pl.col("n_cysteines") >= 4))
    peptides = dict(zip(cysrich["peptide_id"], cysrich["mature_peptide"]))
    worst = max(
        {alphabet for alphabet, _ in config["collision_report_settings"]},
        key=lambda a: len(encodings.collision_groups(peptides, a)),
    )
    groups = encodings.collision_groups(peptides, worst)
    pl.DataFrame(
        {
            "alphabet": [worst] * len(groups),
            "n_peptides": [len(g) for g in groups],
            "peptide_ids": [";".join(g) for g in groups],
            "encoding": [encodings.encode(peptides[g[0]], worst) for g in groups],
        }
    ).write_csv(out_dir / "collision_groups.csv")
    print(
        f"\n{worst} writes {len(groups)} groups of cysteine-rich toxins as one string "
        f"each; see collision_groups.csv"
    )


def _method_table(config: dict) -> list[tuple[str, object, str | None]]:
    """Every method to run, as (name, scoring function, which candidate filter it uses).

    The candidate filter is None for a method that scores the whole reference, `"slack"`
    for the framework filter with the config's loop slack, and `"any_count"` for the
    framework filter with the cysteine count alone and no loop-length condition.

    Two variants of the framework-guided method, because they fail differently and the
    difference is the point: with a loop slack the filter is strict and throws away most
    true positives before anything is ranked, while the cysteine count alone keeps far
    more. Running only the strict one would blame the idea of framework-guided retrieval
    for a choice of slack.

    The first five methods are the ablation ladder, in order: each knows one more kind of
    thing about the peptide than the one before it.
    """
    table: list[tuple[str, object, str | None]] = [
        ("framework_only", methods.framework_only, None),
        ("loop_lengths_only", methods.loop_lengths_only, None),
        ("loop_composition_only", methods.loop_composition_only, None),
        ("loop_anchored", methods.loop_anchored, None),
        ("full_sequence", methods.full_sequence, None),
        ("framework_then_rerank", methods.full_sequence, "slack"),
        ("cysteine_count_then_rerank", methods.full_sequence, "any_count"),
    ]
    for alphabet, ksize in config["kmer_settings"]:
        table.append(
            (f"kmer_{alphabet}_k{ksize}", methods.kmer_containment(alphabet, ksize), None)
        )
    return table


def cmd_run(config: dict, data_dir: Path, out_dir: Path) -> None:
    grouped = pl.read_parquet(out_dir / "peptides.parquet")
    queries = pl.read_csv(out_dir / "queries.csv")

    rows = grouped.to_dicts()
    labels = [evaluate.Labels.from_row(row) for row in rows]
    features = methods.build_features(
        grouped["peptide_id"].to_list(),
        grouped["mature_peptide"].to_list(),
        config["kmer_settings"],
        include_termini=config["include_termini"],
    )
    position = {row["peptide_id"]: i for i, row in enumerate(rows)}
    neighbours = _load_neighbourhoods(config, out_dir, grouped)
    frameworks_of = [f.framework for f in features]

    # Which endpoints each query identifier was chosen for. A query chosen for two
    # endpoints is ranked once and scored for both.
    endpoints_of: dict[str, list[str]] = {}
    for row in queries.iter_rows(named=True):
        endpoints_of.setdefault(row["peptide_id"], []).append(row["endpoint"])

    method_table = _method_table(config)
    results = []
    feature_rows = []
    started_all = time.perf_counter()
    for number, (query_id, endpoints) in enumerate(sorted(endpoints_of.items()), 1):
        q = position[query_id]
        for reference_mode in config["reference_modes"]:
            if reference_mode == "with_family":
                reference = [i for i in range(len(rows)) if i != q]
            elif reference_mode == "held_out_family":
                # Every peptide similar to the query above the threshold, and every
                # peptide stating the query's gene family, is taken out. A hit therefore
                # has to come from somewhere else, which is the only setting that says
                # anything about generalising past the query's own family.
                reference = [
                    i for i in range(len(rows)) if i not in neighbours[q]
                ]
            else:
                raise ValueError(f"unknown reference mode {reference_mode!r}")
            pool = [features[i] for i in reference]
            candidate_pools = {
                "slack": methods.framework_candidates(
                    features[q], pool, config["framework_slack"]
                ),
                "any_count": methods.framework_candidates(features[q], pool, None),
            }
            # The truth and the hard negatives do not depend on the method, so they are
            # computed once per endpoint here and handed to every method below.
            truth_by_endpoint = {
                endpoint: evaluate.truth_arrays(
                    query_labels=labels[q],
                    target_labels=labels,
                    query_framework=features[q].framework,
                    target_frameworks=frameworks_of,
                    reference_indices=reference,
                    endpoint=endpoint,
                )
                for endpoint in endpoints
            }
            for name, score, candidate_filter in method_table:
                started = time.perf_counter()
                order, _ = methods.rank(
                    features[q],
                    pool,
                    score,
                    candidates=candidate_pools.get(candidate_filter),
                )
                seconds = time.perf_counter() - started
                for endpoint in endpoints:
                    truth, hard_negative, same_framework_truth = truth_by_endpoint[
                        endpoint
                    ]
                    result = evaluate.evaluate_ranking(
                        query_id=query_id,
                        truth=truth,
                        hard_negative=hard_negative,
                        same_framework_truth=same_framework_truth,
                        n_reference=len(reference),
                        order=order,
                        seconds=seconds,
                        method_name=name,
                        endpoint=endpoint,
                        reference_mode=reference_mode,
                        budget=config["budget"],
                    )
                    if result is not None:
                        results.append(result.as_row())
        if number % 25 == 0:
            print(
                f"  {number}/{len(endpoints_of)} queries, "
                f"{time.perf_counter() - started_all:.0f} s",
                flush=True,
            )

    for alphabet, ksize in config["kmer_settings"]:
        counts = [len(f.kmers[(alphabet, ksize)]) for f in features]
        feature_rows.append(
            {
                "alphabet": alphabet,
                "ksize": ksize,
                "n_peptides": len(features),
                "n_peptides_with_no_kmers": sum(1 for c in counts if c == 0),
                "total_distinct_kmers_summed": sum(counts),
                "mean_kmers_per_peptide": sum(counts) / len(counts),
                "subsampling": "none (scaled=1): every k-mer kept",
            }
        )
    pl.DataFrame(feature_rows).write_csv(out_dir / "feature_counts.csv")
    # infer_schema_length=None reads every row before fixing a column's type:
    # hard_negative_auc is null for any query with no same-framework hard negative, and
    # the first hundred rows can all be null.
    scored = pl.DataFrame(results, infer_schema_length=None)
    scored.write_parquet(out_dir / "ranked_results.parquet")
    print(
        f"ran {len(method_table)} methods over {len(endpoints_of)} queries in "
        f"{time.perf_counter() - started_all:.0f} s, {len(results)} scored rows"
    )


def cmd_evaluate(config: dict, data_dir: Path, out_dir: Path) -> None:
    results = pl.read_parquet(out_dir / "ranked_results.parquet")
    summary = evaluate.summarise(results)
    summary.write_csv(out_dir / "summary.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(12)
    for endpoint in config["endpoints"]:
        for mode in config["reference_modes"]:
            block = summary.filter(
                (pl.col("endpoint") == endpoint) & (pl.col("reference_mode") == mode)
            )
            if not block.height:
                continue
            print(f"\n=== {endpoint}, reference {mode} ===")
            print(
                block.select(
                    "method",
                    "n_queries",
                    "mean_recall_at_budget",
                    "mean_precision_at_budget",
                    "mean_candidate_recall",
                    "mean_recall_at_budget_conditional",
                    "mean_hard_negative_auc",
                    "median_candidates",
                    "seconds_per_query",
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="venom_retrieval", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "command",
        choices=["manifest", "splits", "collisions", "run", "evaluate", "all"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data", type=Path, default=Path.home() / "data" / "venom-retrieval"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = read_config(args.config)
    # tables/ rather than a results/ folder beside the code: the repository gitignores
    # every results/ directory, and these outputs are the deliverable.
    out_dir = args.out or (
        Path(__file__).resolve().parents[3] / "tables" / "venom_retrieval" / config["name"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.json").write_text(
        json.dumps(
            {
                **config,
                "alignment": {
                    "matrix": MATRIX,
                    "gap_open": GAP_OPEN,
                    "gap_extend": GAP_EXTEND,
                },
                "data_dir": str(args.data),
            },
            indent=2,
            default=list,
        )
        + "\n"
    )

    steps = {
        "manifest": cmd_manifest,
        "splits": cmd_splits,
        "collisions": cmd_collisions,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
    }
    to_run = list(steps) if args.command == "all" else [args.command]
    for name in to_run:
        print(f"\n########## {name}", flush=True)
        steps[name](config, args.data, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
