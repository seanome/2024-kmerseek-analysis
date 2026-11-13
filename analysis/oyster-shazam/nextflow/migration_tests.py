"""Run first/resume/query-change/failure gates inside one Slurm allocation."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from artifacts import sha, write_new


def test(root, audited, nextflow, python, sourmash):
    root.mkdir()
    workflow = Path(__file__).parent / "main.nf"
    base = [
        str(nextflow),
        "run",
        str(workflow),
        "-profile",
        "allocation_test",
        "--access_confirmed",
        "true",
        "--resources_reviewed",
        "true",
        "--ownership_reviewed",
        "true",
        "--python",
        str(python),
        "--sourmash",
        str(sourmash),
        "--task_cpus",
        "4",
        "--task_memory",
        "8 GB",
        "--task_time",
        "30 min",
        "--partition",
        "normal,hns",
        "--reference_manifest",
        str(audited / "reference.json"),
    ]

    def run(label, query, expected_ok=True, extra=()):
        command = (
            base
            + [
                "--query_manifest",
                str(query),
                "-with-trace",
                f"{label}.trace.tsv",
                "-with-report",
                f"{label}.report.html",
                "-with-timeline",
                f"{label}.timeline.html",
                "-with-dag",
                f"{label}.dag.html",
            ]
            + list(extra)
        )
        (root / f"{label}.command.json").write_text(json.dumps(command))
        with (root / f"{label}.log").open("w") as log:
            status = subprocess.run(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        assert (status == 0) == expected_ok, (
            label,
            status,
            (root / f"{label}.log").read_text()[-5000:],
        )
        with (root / f"{label}.trace.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        print(label, status, [(r["name"], r["status"]) for r in rows], flush=True)
        return rows

    def selected(rows, name):
        return [r for r in rows if r["name"].split(" (")[0] == name]

    def cached(rows, names):
        for name in names:
            assert (
                len(selected(rows, name)) == 1
                and selected(rows, name)[0]["status"] == "CACHED"
            ), (name, rows)

    first = run("first", audited / "query.json")
    assert len(first) == 5 and all(r["status"] == "COMPLETED" for r in first)
    baseline = {
        str(p.relative_to(root)): sha(p)
        for p in (root / "work").glob("*/*/validated.json")
    }
    resumed = run("resume", audited / "query.json", extra=["-resume"])
    cached(resumed, ["ADOPT_REFERENCE", "ADOPT_QUERY", "INDEX", "SEARCH", "AUDIT"])
    assert baseline == {
        str(p.relative_to(root)): sha(p)
        for p in (root / "work").glob("*/*/validated.json")
    }
    # Compare the real one-run migration result with the independently audited old shard.
    result = json.loads(next((root / "work").glob("*/*/validated.json")).read_text())
    expected = json.loads((audited / "expected-one-run.json").read_text())
    assert result["match_pairs"] == len(expected)
    changed = run("query-change", audited / "changed-query.json", extra=["-resume"])
    cached(changed, ["ADOPT_REFERENCE", "INDEX"])
    assert selected(changed, "SEARCH")[0]["status"] == "COMPLETED"
    # Controlled application stop after completed upstream work, then resume.
    stop_manifest = json.loads((audited / "changed-query.json").read_text())
    stop_manifest["artifact_id"] = "controlled-stop-test"
    stop_query = root / "stop-query.json"
    stop_query.write_text(json.dumps(stop_manifest))
    failed = run(
        "stopped",
        stop_query,
        False,
        ["-resume", "--test_stop_before_search", "true"],
    )
    cached(failed, ["ADOPT_REFERENCE", "INDEX"])
    restored = run(
        "after-stop",
        stop_query,
        extra=["-resume", "--test_stop_before_search", "false"],
    )
    cached(restored, ["ADOPT_REFERENCE", "ADOPT_QUERY", "INDEX"])
    assert selected(restored, "SEARCH")[0]["status"] == "COMPLETED"
    for label in ("corrupt", "wrong-protocol"):
        meta = json.loads((audited / "query.json").read_text())
        if label == "corrupt":
            path = root / "truncated.sig.gz"
            path.write_bytes(Path(meta["artifact_path"]).read_bytes()[:20])
            meta["artifact_path"] = str(path)
        else:
            meta["settings"]["stop_policy"] = "split-X"
        manifest = root / f"{label}.json"
        manifest.write_text(json.dumps(meta))
        rows = run(label, manifest, False, ["-resume"])
        assert not selected(rows, "SEARCH"), rows
    write_new(
        root / "migration-audit.json",
        dict(
            first_run=True,
            unchanged_resume=True,
            unchanged_output_hashes=baseline,
            query_change_reference_cached=True,
            controlled_stop_resume=True,
            external_scheduler_interruption="not tested",
            corruption_rejected_before_search=True,
            protocol_mismatch_rejected_before_search=True,
            production_rollout=False,
        ),
    )


if __name__ == "__main__":
    test(*(Path(a).resolve() for a in sys.argv[1:]))
