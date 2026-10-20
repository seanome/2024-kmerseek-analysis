"""Command line for the construct insert check. CPU only, bounded concurrency.

    python src/cli.py validate  --input <dir>
    python src/cli.py run       --input <dir> --output <dir> [--development <dir>] [--workers N]
    python src/cli.py controls  --input <dir> --output <dir>

``run`` writes the per-construct report, every flag, the paired table, the controls, the
pass-rule verdict and a resources file. It never edits a sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import controls as controls_module
import evaluate as evaluate_module
import records as records_module
from dictionary import build_for, leakage_report, own_inserts_already_known
from workflows import COMPARISON, WORKFLOWS, Flag

FLAG_COLUMNS = (
    "workflow",
    "construct_id",
    "version",
    "start",
    "end",
    "residues",
    "insert_name",
    "support",
    "permitted",
    "outcome",
    "region_evalue",
    "setting",
    "notes",
)


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            if columns:
                csv.writer(handle).writerow(columns)
        return
    fieldnames = list(columns) if columns else list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flags_for_construct(payload: tuple) -> list[Flag]:
    construct, dictionary, l_distinguish = payload
    flags: list[Flag] = []
    for workflow in WORKFLOWS.values():
        flags.extend(workflow(construct, dictionary, l_distinguish))
    return flags


def validate(input_dir: Path) -> records_module.Collection:
    collection = records_module.load(input_dir)
    problems = leakage_report(collection)
    if problems:
        raise SystemExit(
            "dictionary leakage, refusing to run:\n  " + "\n  ".join(problems)
        )
    counts = collection.by_status()
    print(f"input: {input_dir}")
    for name, digest in sorted(collection.checksums.items()):
        print(f"  sha256 {digest}  {name}")
    print(
        f"  {len(collection.constructs)} construct versions: "
        f"{counts['clean']} clean, {counts['error']} with errors, "
        f"{counts['unresolved']} unresolved"
    )
    print(
        f"  {collection.documented_event_count} documented unexpected-insert events, "
        f"{collection.with_dna_count} constructs with DNA, "
        f"{len(collection.dictionary)} development dictionary inserts, "
        f"{len(collection.review_sessions)} recorded review sessions"
    )
    print("  group check and leakage check passed")
    return collection


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    collection = validate(input_dir)

    l_distinguish = None
    control_summary: dict[str, object] = {}
    if args.development:
        development = records_module.load(Path(args.development))
        l_distinguish, fit_summary = controls_module.fit_l_distinguish(development)
        control_summary["l_distinguish"] = l_distinguish
        control_summary["l_distinguish_fit"] = fit_summary
        control_summary["l_distinguish_source"] = str(args.development)
        print(f"  L_distinguish fitted on development data: {l_distinguish}")
    else:
        control_summary["l_distinguish"] = None
        control_summary["l_distinguish_source"] = (
            "not fitted; no development collection given, so the short-run ambiguity "
            "test is inactive"
        )

    payloads = [
        (c, build_for(c, collection), l_distinguish) for c in collection.constructs
    ]
    already_known = sum(
        own_inserts_already_known(c, d) for c, d, _ in payloads
    )
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    if workers == 1:
        results = [_flags_for_construct(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_flags_for_construct, payloads))

    flags = [flag for group in results for flag in group]
    by_workflow: dict[str, list[Flag]] = {name: [] for name in WORKFLOWS}
    for flag in flags:
        by_workflow[flag.workflow].append(flag)
    by_workflow["comparison"] = [
        f for f in flags if f.workflow in COMPARISON
    ]

    scores = evaluate_module.score(collection, by_workflow)
    rule = evaluate_module.apply_pass_rule(collection, scores)
    sufficient, shortfalls = evaluate_module.enough_data(collection)
    verdict = rule.verdict if sufficient else "INSUFFICIENT_EVIDENCE"

    control_summary["shuffled_inserts"] = controls_module.shuffled_insert_runs(collection)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "flags.csv", [f.as_row() for f in flags], FLAG_COLUMNS)

    per_construct = []
    for construct in collection.constructs:
        here = [
            f
            for f in by_workflow["reduced"]
            if f.construct_id == construct.construct_id and f.version == construct.version
        ]
        per_construct.append(evaluate_module.construct_report_row(construct, here))
    _write_csv(output_dir / "per_construct_report.csv", per_construct)

    _write_csv(output_dir / "paired_events.csv", evaluate_module.paired_table(collection, scores))

    false_flags = [
        f.as_row()
        for f in by_workflow["reduced"]
        if f.outcome in ("unexpected", "ambiguous")
        and any(
            c.is_clean
            for c in collection.constructs
            if (c.construct_id, c.version) == (f.construct_id, f.version)
        )
    ]
    _write_csv(output_dir / "false_flag_examples.csv", false_flags, FLAG_COLUMNS)

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_bytes = max(rusage.ru_maxrss, children.ru_maxrss)
    if sys.platform != "darwin":
        peak_bytes *= 1024  # Linux reports kilobytes, macOS reports bytes
    elapsed = time.monotonic() - started
    cpu_seconds = (
        rusage.ru_utime + rusage.ru_stime + children.ru_utime + children.ru_stime
    )
    disk_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())

    summary = {
        "input": str(input_dir),
        "input_checksums": collection.checksums,
        "plan_checksum": records_module.sha256(
            Path(__file__).resolve().parents[1] / "PLAN.md"
        ),
        "verdict": verdict,
        "pass_rule": {
            "criterion_1_event_recall": rule.event_recall_ok,
            "criterion_2_clean_flag_rate": rule.clean_rate_ok,
            "criterion_3_beats_comparison": rule.beats_comparison_ok,
            "criterion_4_review_time": rule.review_time_ok,
            "events_the_comparison_missed": rule.events_beating_comparison,
            "reasons": rule.reasons,
        },
        "data_sufficiency": {"enough": sufficient, "shortfalls": shortfalls},
        "counts": {
            "construct_versions": len(collection.constructs),
            "by_status": collection.by_status(),
            "clean_constructs": collection.clean_count,
            "documented_events": collection.documented_event_count,
            "constructs_with_dna": collection.with_dna_count,
            "review_sessions": len(collection.review_sessions),
            "events_whose_insert_was_already_in_the_dictionary": already_known,
        },
        "scores": {
            name: {
                "events_total": s.events_total,
                "events_recovered": s.events_recovered,
                "event_recall": s.event_recall,
                "clean_constructs": s.clean_constructs,
                "clean_constructs_flagged": s.clean_constructs_flagged,
                "clean_flag_rate": s.clean_flag_rate,
                "flags_total": s.flags_total,
                "flags_unexpected": s.flags_unexpected,
                "flags_ambiguous": s.flags_ambiguous,
                "flags_permitted": s.flags_permitted,
                "events_by_insert_class": s.by_class,
            }
            for name, s in scores.items()
        },
        "review_time_minutes_median": evaluate_module.review_time(collection),
        "controls": control_summary,
        "resources": {
            "cpu_seconds": round(cpu_seconds, 2),
            "cpu_hours": round(cpu_seconds / 3600, 5),
            "elapsed_seconds": round(elapsed, 2),
            "peak_rss_mb": round(peak_bytes / 1e6, 1),
            "output_disk_bytes": disk_bytes,
            "workers": workers,
            "gpu": "none",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nverdict: {verdict}")
    for reason in rule.reasons:
        print(f"  {reason}")
    for shortfall in shortfalls:
        print(f"  not enough data: {shortfall}")
    print(f"  wrote {output_dir}/summary.json and four CSV reports")
    return 0


def controls(args: argparse.Namespace) -> int:
    collection = records_module.load(Path(args.input))
    l_distinguish, fit_summary = controls_module.fit_l_distinguish(collection)
    out = {
        "l_distinguish": l_distinguish,
        "fit": fit_summary,
        "shuffled_inserts": controls_module.shuffled_insert_runs(collection),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "controls.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="check the input files and stop")
    p_validate.add_argument("--input", required=True)

    p_run = sub.add_parser("run", help="run every workflow and apply the pass rule")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output", required=True)
    p_run.add_argument(
        "--development",
        help="development collection used to fit L_distinguish; never the evaluation set",
    )
    p_run.add_argument("--workers", type=int, default=4)

    p_controls = sub.add_parser("controls", help="fit L_distinguish and shuffle controls")
    p_controls.add_argument("--input", required=True)
    p_controls.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        validate(Path(args.input))
        return 0
    if args.command == "run":
        return run(args)
    return controls(args)


if __name__ == "__main__":
    raise SystemExit(main())
