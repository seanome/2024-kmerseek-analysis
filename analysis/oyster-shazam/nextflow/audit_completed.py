"""Independently check completed replay/preflight without changing their files."""

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from collections import Counter

import sourmash
from Bio.Seq import Seq
from artifacts import SETTINGS, PREPROCESSING, sha, write_new


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def signatures(path):
    return sourmash.load_file_as_signatures(str(path))


def matches(path):
    found = {}
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            pair = row["query_name"], row["match_name"]
            require(pair not in found, "Duplicate saved pair")
            found[pair] = int(row["intersect_hashes"])
    return found


def done(path, count):
    values = re.findall(r"DONE\. Processed ([\d,]+) search sigs", path.read_text())
    require(
        len(values) == 1 and int(values[0].replace(",", "")) == count,
        "Incomplete search log",
    )


def dna_records(root, manifest):
    full = Path(os.environ["SCRATCH"]) / "oyster-full-extensions-20261023"
    de = Path(os.environ["SCRATCH"]) / "oyster-dekupl-queries-20261024"
    entry = manifest["collections"][0]
    source = full / "splash-trimmed-all/batch-000/compactors.tsv"
    require(sha(source) == entry["source_sha256"], "SPLASH source mismatch")
    with source.open() as handle:
        records = {
            f"splash:full:0:{i}:{r['anchor']}": r["compactor"]
            for i, r in enumerate(csv.DictReader(handle, delimiter="\t"))
            if len(r["compactor"]) > len(r["anchor"])
        }
    selection = load(de / "selection.json")
    require(
        sha(Path(selection["source"])) == selection["source_sha256"],
        "DE source mismatch",
    )
    with gzip.open(de / "all-significant-contigs.tsv.gz", "rt") as handle:
        n = 0
        for row in csv.DictReader(handle, delimiter="\t"):
            require(float(row["pvalue"]) < 0.05, "Unexpected nonsignificant DE record")
            require(row["query_id"] not in records, "Duplicate original ID")
            records[row["query_id"]] = row["contig"]
            n += 1
    require(n == selection["selected_contigs"] == 363635, "DE membership mismatch")
    return records


def manifest_for(path, role, ids, source_sha, output, source_versions):
    manifest = dict(
        role=role,
        artifact_id=path.name,
        artifact_path=str(path),
        artifact_sha256=sha(path),
        source_sha256=source_sha,
        source_code_sha256=None,
        provenance_scope="historical-build-code-unrecorded-test-only",
        source_tool_versions=source_versions,
        settings=SETTINGS,
        preprocessing=PREPROCESSING,
        ownership="completed-read-only",
        expected_ids=ids,
    )
    write_new(output, manifest)


def audit_replay(output):
    group = Path(os.environ["GROUP_SCRATCH"])
    root = group / "oyster-first-public-linear-20261025"
    archive = group / "oyster-bivalve-s10-s100-zero-20261023"
    manifest = load(root / "manifest.json")
    checkpoint = load(root / "dna100/shard-000/validated.json")
    require(
        checkpoint["manifest_sha256"] == sha(root / "manifest.json"),
        "Replay manifest changed",
    )
    records = dna_records(root, manifest)
    refs = {}
    paths = {}
    for row in manifest["reference_runs"][:8]:
        run = row["run"]
        path = archive / "runs" / run[:6] / run / "dna.sig.zip"
        require(sha(path) == row["signatures"]["dna"]["sha256"], "Reference changed")
        sigs = list(signatures(path))
        require(len(sigs) == 1 and sigs[0].name == run, "Reference ID mismatch")
        mh = sigs[0].minhash
        require(
            mh.moltype == "DNA"
            and mh.ksize == 31
            and mh.scaled == 100
            and mh.seed == 42
            and not mh.track_abundance,
            "Wrong reference settings",
        )
        refs[run] = set(mh.hashes)
        paths[run] = path
    require(set(refs) == set(checkpoint["references"]), "Reference membership mismatch")
    summaries = []
    seen = set()
    first_expected = None
    for entry in manifest["collections"]:
        path = root / "queries" / entry["file"]
        require(sha(path) == entry["sha256"], "Query changed")
        expected = {}
        ids = []
        empty = 0
        for sig in signatures(path):
            require(
                sig.name in records and sig.name not in seen,
                "Query membership mismatch",
            )
            seen.add(sig.name)
            ids.append(sig.name)
            independent = sourmash.MinHash(n=0, ksize=31, scaled=100)
            independent.add_sequence(records[sig.name], force=True)
            require(
                independent == sig.minhash, "Query hash mismatch against original DNA"
            )
            hashes = set(sig.minhash.hashes)
            empty += not hashes
            for run, rh in refs.items():
                shared = len(hashes & rh)
                if shared:
                    expected[sig.name, run] = shared
        require(
            len(ids) == entry["queries"] and empty == entry["empty_sketches"],
            "Query count mismatch",
        )
        saved = next(
            x for x in checkpoint["results"] if x["query_set"] == entry["query_set"]
        )
        result = root / "dna100/shard-000" / saved["file"]
        require(sha(result) == saved["sha256"], "Result checksum mismatch")
        require(
            matches(result) == expected and len(expected) == saved["match_pairs"],
            "Replay pairs differ",
        )
        done(result.parent / f"{entry['query_set']}.log", len(ids))
        summaries.append(
            dict(
                query_set=entry["query_set"],
                queries=len(ids),
                empty=empty,
                pairs=len(expected),
                sha256=sha(result),
            )
        )
        if first_expected is None:
            first_expected = expected
            run = max(refs, key=lambda r: sum(pair[1] == r for pair in expected))
            row = next(x for x in manifest["reference_runs"] if x["run"] == run)
            manifest_for(
                path,
                "query",
                ids,
                entry["source_sha256"],
                output / "query.json",
                {"historical_sourmash": "unrecorded"},
            )
            manifest_for(
                paths[run],
                "reference",
                [run],
                sha(paths[run].parent / "validated.json"),
                output / "reference.json",
                {
                    "historical_trim_galore": "0.6.10",
                    "historical_sourmash": "unrecorded",
                },
            )
            write_new(
                output / "expected-one-run.json",
                {q: count for (q, r), count in expected.items() if r == run},
            )
        elif entry["query_set"] == "dekupl-full-000":
            manifest_for(
                path,
                "query",
                ids,
                sha(
                    Path(os.environ["SCRATCH"])
                    / "oyster-dekupl-queries-20261024/selection.json"
                ),
                output / "changed-query.json",
                {"historical_sourmash": "unrecorded"},
            )
        print("AUDITED", entry["query_set"], len(ids), len(expected), flush=True)
    require(seen == set(records), "Original query coverage incomplete")
    write_new(
        output / "replay-audit.json",
        dict(
            queries=len(seen),
            references=list(refs),
            collections=summaries,
            match_pairs=sum(r["pairs"] for r in summaries),
            validation="All original DNA hashes and exact pair/intersection equality; historical build code/version unrecorded",
        ),
    )
    return records


def audit_preflight(output, records):
    root = (
        Path(os.environ["GROUP_SCRATCH"])
        / "oyster-stopfree-public-pilot-linear-20261025"
    )
    meta = load(root / "validated.json")
    summaries = []
    for entry in meta["settings"]:
        mol = entry["moltype"]
        scale_pairs = {}
        for scale in (10, 100):
            refpath = root / f"{mol}.reference{scale}.sig.zip"
            refs = list(signatures(refpath))
            require(
                len(refs) == 1 and refs[0].name == meta["reference"],
                "Preflight reference mismatch",
            )
            ref = refs[0].minhash
            reference_hashes = set(ref.hashes)
            expected, seen, empty = {}, set(), 0
            for sig in signatures(root / f"{mol}.query{scale}.sig.gz"):
                require(
                    sig.name in records and sig.name not in seen,
                    "Preflight query membership mismatch",
                )
                seen.add(sig.name)
                require(
                    sig.minhash.ksize == ref.ksize
                    and sig.minhash.scaled == scale
                    and sig.minhash.moltype == ref.moltype,
                    "Preflight settings mismatch",
                )
                hashes = set(sig.minhash.hashes)
                empty += not hashes
                shared = len(hashes & reference_hashes)
                if shared:
                    expected[sig.name, meta["reference"]] = shared
            require(seen == set(records), "Preflight missing original IDs")
            result = root / f"{mol}-{scale}.matches.csv.gz"
            require(
                sha(result) == entry["result_hashes"][str(scale)],
                "Preflight result changed",
            )
            require(matches(result) == expected, "Preflight exact match mismatch")
            require(
                len(expected) == entry[f"matches{scale}"]
                and empty == entry[f"empty{scale}"],
                "Preflight counts differ",
            )
            done(root / f"{mol}-{scale}-search.log", len(records))
            scale_pairs[scale] = set(expected)
            summaries.append(
                dict(
                    moltype=mol,
                    scaled=scale,
                    queries=len(seen),
                    empty=empty,
                    pairs=len(expected),
                    query_sha256=sha(root / f"{mol}.query{scale}.sig.gz"),
                    reference_sha256=sha(refpath),
                )
            )
            print("AUDITED historical", mol, scale, len(expected), flush=True)
        require(
            not scale_pairs[100] - scale_pairs[10], "Unexpected scale100 gained pairs"
        )
        loss = root / f"{mol}.lost.tsv.gz"
        require(sha(loss) == entry["loss_sha256"], "Loss file changed")
        with gzip.open(loss, "rt") as handle:
            lost = [
                (r["query_id"], r["run"])
                for r in csv.DictReader(handle, delimiter="\t")
            ]
        require(
            len(lost) == len(set(lost))
            and set(lost) == scale_pairs[10] - scale_pairs[100],
            "Loss pair mismatch",
        )
    write_new(
        output / "historical-preflight-audit.json",
        dict(
            protocol="HISTORICAL split stop/X; not corrected >=30aa",
            scope="Saved sketches, identities, match tables and scale losses; translation not independently rebuilt",
            settings=summaries,
        ),
    )


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    destination.mkdir()
    records = audit_replay(destination)
    # Keep a systematic real-input corrected pilot, with original DNA and IDs.
    groups = (
        [k for k in records if k.startswith("splash:")],
        [k for k in records if not k.startswith("splash:")],
    )
    with (destination / "corrected-pilot.fa").open("w") as handle:
        for keys in groups:
            for i in range(1000):
                key = keys[i * len(keys) // 1000]
                handle.write(f">{key}\n{records[key]}\n")
    audit_preflight(destination, records)
