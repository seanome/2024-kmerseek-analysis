"""Validate immutable DNA100 adoption inputs and independently audit one-run matches."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

SETTINGS = dict(
    moltype="DNA",
    k=31,
    scaled=100,
    seed=42,
    abundance=False,
    stop_policy="not-applicable",
    minimum_aa=None,
)
PREPROCESSING = dict(
    tool="trim_galore", quality=20, minimum_nt=20, paired_mode="independent-files"
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, data):
    # The completion record is the last file written. Never overwrite a prior record.
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x") as handle:
        try:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            # Hard link gives an atomic, no-clobber publication on the same filesystem.
            import os

            os.link(partial, path)
        finally:
            # Remove our own partial file so a retry sees the original error.
            partial.unlink(missing_ok=True)


def metadata(manifest, role):
    if manifest["role"] != role or manifest["settings"] != SETTINGS:
        raise ValueError("Wrong role or scientific protocol")
    if manifest["ownership"] != "completed-read-only":
        raise ValueError("Only completed, read-only artifacts can be adopted")
    if manifest["preprocessing"] != PREPROCESSING:
        raise ValueError("Preprocessing mismatch")
    for field in ("artifact_sha256", "source_sha256"):
        if not re.fullmatch("[a-f0-9]{64}", manifest[field]):
            raise ValueError(f"Missing or invalid {field}")
    code = manifest["source_code_sha256"]
    if code is None:
        if (
            manifest.get("provenance_scope")
            != "historical-build-code-unrecorded-test-only"
        ):
            raise ValueError(
                "Unrecorded historical code is only allowed for bounded tests"
            )
    elif not re.fullmatch("[a-f0-9]{64}", code):
        raise ValueError("Invalid source code checksum")
    if not manifest["artifact_id"] or not manifest["source_tool_versions"]:
        raise ValueError("Missing source provenance")
    names = manifest["expected_ids"]
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("Empty or duplicate expected IDs")
    if role == "reference" and len(names) != 1:
        raise ValueError("First migration milestone requires exactly one public run")
    return manifest


def read_sketches(path):
    import sourmash

    found = {}
    for sig in sourmash.load_file_as_signatures(str(path)):
        mh = sig.minhash
        actual = dict(
            moltype=mh.moltype,
            k=mh.ksize,
            scaled=mh.scaled,
            seed=mh.seed,
            abundance=mh.track_abundance,
            stop_policy="not-applicable",
            minimum_aa=None,
        )
        if actual != SETTINGS or sig.name in found:
            raise ValueError("Wrong sketch settings or duplicate ID")
        found[sig.name] = set(mh.hashes)
    return found


def adopt(manifest_path, sketch, role, output):
    manifest = metadata(json.loads(manifest_path.read_text()), role)
    if sha(sketch) != manifest["artifact_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    found = read_sketches(sketch)
    if set(found) != set(manifest["expected_ids"]):
        raise ValueError("Missing or unexpected sketch IDs")
    if role == "reference" and not any(found.values()):
        raise ValueError("Reference sketch is empty")
    write_new(
        output,
        dict(
            manifest=manifest,
            manifest_sha256=sha(manifest_path),
            empty_ids=sorted(k for k, hashes in found.items() if not hashes),
        ),
    )


def audit(query, reference, query_receipt, reference_receipt, matches, log, output):
    receipts = []
    for sketch, receipt, role in (
        (query, query_receipt, "query"),
        (reference, reference_receipt, "reference"),
    ):
        data = json.loads(receipt.read_text())
        manifest = metadata(data["manifest"], role)
        if sha(sketch) != manifest["artifact_sha256"]:
            raise ValueError("Input changed after adoption")
        receipts.append(data)
    queries, refs = read_sketches(query), read_sketches(reference)
    if set(queries) != set(receipts[0]["manifest"]["expected_ids"]) or set(refs) != set(
        receipts[1]["manifest"]["expected_ids"]
    ):
        raise ValueError("Membership changed after adoption")
    processed = re.findall(r"DONE\. Processed ([\d,]+) search sigs", log.read_text())
    if len(processed) != 1 or int(processed[0].replace(",", "")) != len(queries):
        raise ValueError("Search did not process every query")
    expected = {
        (q, r): len(qh & rh)
        for q, qh in queries.items()
        for r, rh in refs.items()
        if qh & rh
    }
    observed = {}
    with matches.open() as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is not None and not {
            "query_name",
            "match_name",
            "intersect_hashes",
        } <= set(rows.fieldnames):
            raise ValueError("Unexpected result columns")
        for row in rows:
            pair = row["query_name"], row["match_name"]
            if pair in observed:
                raise ValueError("Duplicate match pair")
            observed[pair] = int(row["intersect_hashes"])
    if observed != expected:
        raise ValueError("Search differs from independent exact hash intersection")
    matched = {q for q, _ in observed}
    write_new(
        output,
        dict(
            settings=SETTINGS,
            query_manifest_sha256=receipts[0]["manifest_sha256"],
            reference_manifest_sha256=receipts[1]["manifest_sha256"],
            matches_sha256=sha(matches),
            search_log_sha256=sha(log),
            query_ids=sorted(queries),
            unmatched_ids=sorted(set(queries) - matched),
            empty_ids=sorted(q for q, h in queries.items() if not h),
            references=sorted(refs),
            match_pairs=len(observed),
            validation="Independent exact sampled-hash intersections; candidate sequence matches",
        ),
    )


def versions(executable):
    import sys

    if executable.resolve() != (Path(sys.executable).parent / "sourmash").resolve():
        raise ValueError("sourmash must belong to this Python environment")

    observed = {
        name: importlib.metadata.version(name)
        for name in ("sourmash", "sourmash_plugin_branchwater")
    }
    if observed["sourmash_plugin_branchwater"] != "0.9.15":
        raise ValueError(f"Require Branchwater 0.9.15; found {observed}")
    print(json.dumps(observed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    adoption = commands.add_parser("adopt")
    adoption.add_argument("role", choices=["query", "reference"])
    for name in ("manifest", "sketch", "output"):
        adoption.add_argument(name, type=Path)
    check = commands.add_parser("audit")
    for name in (
        "query",
        "reference",
        "query_receipt",
        "reference_receipt",
        "matches",
        "log",
        "output",
    ):
        check.add_argument(name, type=Path)
    version_check = commands.add_parser("versions")
    version_check.add_argument("executable", type=Path)
    args = parser.parse_args()
    if args.command == "adopt":
        adopt(args.manifest, args.sketch, args.role, args.output)
    elif args.command == "audit":
        audit(
            args.query,
            args.reference,
            args.query_receipt,
            args.reference_receipt,
            args.matches,
            args.log,
            args.output,
        )
    else:
        versions(args.executable)
