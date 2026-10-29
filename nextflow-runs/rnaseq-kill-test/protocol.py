"""Check the frozen RNA-seq protocol and enumerate condition assignments.

This module does not download reads, build an index, or launch an analysis.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_rules(root=ROOT):
    """Reject changed rules, including whitespace changes."""
    for name in ("decision_rules", "implementation_choices"):
        expected = (root / f"{name}.sha256").read_text().split()[0]
        if digest(root / f"{name}.md") != expected:
            raise ValueError(f"{name}.md differs from its recorded SHA-256")


def read_table(path, required):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate table column")
        if not required <= set(fields):
            raise ValueError(f"{path}: required columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed table")
    return rows


def unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


def positive_integer(value, name):
    if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} is not a measurement")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def check_samples(path, species):
    """Validate one row per biological sample, after combining sequencing lanes."""
    columns = {
        "sample",
        "condition",
        "biological_replicate",
        "taxid",
        "study",
        "runs",
        "layout",
        "strandedness",
        "read_pairs",
        "library_evidence",
        "published_de",
        "selection_date",
    }
    rows = read_table(path, columns)
    for col in ("sample", "biological_replicate"):
        unique([r[col] for r in rows], col)
    runs = [run.strip() for row in rows for run in row["runs"].split(";")]
    if not all(runs):
        raise ValueError("empty sequencing run")
    unique(runs, "sequencing run")
    if len({r["study"] for r in rows}) != 1:
        raise ValueError("samples must come from one experiment")
    counts = Counter(r["condition"] for r in rows)
    if len(counts) != 2 or "" in counts:
        raise ValueError("exactly two nonempty conditions are required")
    expected_taxid = {"mouse": "10090", "zebrafish": "7955"}[species]
    for row in rows:
        if not all(row[c].strip() for c in columns):
            raise ValueError(f"{row['sample']}: missing eligibility evidence")
        if row["taxid"] != expected_taxid or row["layout"] != "PAIRED":
            raise ValueError("wrong species or library layout")
        if row["strandedness"] not in {"forward", "reverse"}:
            raise ValueError("stranded library evidence is required")
        date.fromisoformat(row["selection_date"])
        n = positive_integer(row["read_pairs"], "read_pairs")
        if species == "mouse" and n < 20_000_000:
            raise ValueError("mouse sample has fewer than 20 million read pairs")
    if species == "mouse" and min(counts.values()) < 4:
        raise ValueError(
            "mouse experiment needs four biological replicates per condition"
        )
    if len(set(counts.values())) != 1:
        raise ValueError("balanced label swaps require equal condition sizes")
    return rows


def assignments(samples, labels):
    """Include every labelled balanced reassignment, including the complement."""
    unique(samples, "sample")
    if len(samples) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("one binary label per sample is required")
    n = len(samples)
    if n % 2 or sum(labels) != n // 2:
        raise ValueError("conditions must be balanced")
    true = tuple(labels)
    for chosen in itertools.combinations(range(n), n // 2):
        chosen = set(chosen)
        value = tuple(int(i in chosen) for i in range(n))
        yield {
            "assignment": "".join(map(str, value)),
            "is_true": value == true,
            "is_complement": value == tuple(1 - x for x in true),
            "labels": dict(zip(samples, value)),
        }


def read_taxonomy(path):
    """Read parent links from a versioned NCBI nodes.dmp snapshot."""
    parents = {}
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) < 2:
                raise ValueError(f"malformed NCBI taxonomy row at line {number}")
            node = str(positive_integer(fields[0].strip(), "taxon"))
            parent = str(positive_integer(fields[1].strip(), "parent taxon"))
            if node in parents:
                raise ValueError("duplicate taxonomy node")
            parents[node] = parent
    if parents.get("1") != "1":
        raise ValueError("taxonomy root is missing")
    return parents


def ancestors(taxid, parents):
    lineage = set()
    while taxid not in lineage:
        lineage.add(taxid)
        if taxid == "1":
            return lineage
        if taxid not in parents:
            raise ValueError("species or ancestor missing from taxonomy")
        taxid = parents[taxid]
    raise ValueError("cycle in taxonomy")


def check_index_members(path, heldout, nodes):
    """Derive lineages from the taxonomy snapshot before checking member records."""
    heldout = str(heldout)
    if heldout not in {"7742", "40674"}:
        raise ValueError("heldout must be Vertebrata 7742 or Mammalia 40674")
    parents = read_taxonomy(nodes)
    if heldout not in parents:
        raise ValueError("excluded ancestor is missing from taxonomy")
    rows = read_table(
        path, {"protein", "family", "taxid", "lineage_taxids", "sequence_sha256"}
    )
    seen = set()
    for row in rows:
        pair = (row["protein"], row["family"])
        if pair in seen:
            raise ValueError(f"duplicate protein/family: {pair}")
        seen.add(pair)
        taxid = str(positive_integer(row["taxid"], "taxid"))
        supplied = set(row["lineage_taxids"].split(";"))
        for ancestor in supplied:
            positive_integer(ancestor, "ancestor taxid")
        lineage = ancestors(taxid, parents)
        if heldout in lineage or "10090" in lineage:
            raise ValueError(f"held-out protein in index: {row['protein']}")
        if supplied != lineage:
            raise ValueError(
                f"{row['protein']}: lineage differs from the taxonomy snapshot"
            )
        if not row["protein"] or not row["family"]:
            raise ValueError("missing protein or family")
        sha = row["sequence_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid sequence SHA-256")
    return len(rows)


def check_calibration(path, rules_sha):
    """Require an immutable, development-only threshold record before mouse access."""
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if digest(path) != expected:
        raise ValueError("calibration differs from its recorded SHA-256")
    record = json.loads(path.read_text())
    if record["rules_sha256"] != rules_sha:
        raise ValueError("calibration was made under different rules")
    if record["development_taxid"] != 7955 or record["heldout_taxid"] != 7742:
        raise ValueError("calibration must use zebrafish with vertebrates held out")
    if record["mouse_accessed"] is not False:
        raise ValueError("calibration followed mouse data access")
    if record["source"] != "development_outputs":
        raise ValueError("calibration must come from development outputs")
    date.fromisoformat(record["frozen_date"])
    for name in ("development_manifest_sha256", "development_outputs_sha256"):
        value = record[name]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {name}")
        source = path.parent / record[name.removesuffix("_sha256")]
        if digest(source) != value:
            raise ValueError(f"calibration source has changed: {source}")
    if not record["settings"]:
        raise ValueError("no calibrated settings")
    seen = set()
    for setting in record["settings"]:
        key = (setting["alphabet"], positive_integer(setting["k"], "k"))
        if key in seen or key[0] not in {"protein20", "hp_thomas_dill2"}:
            raise ValueError("duplicate or unknown alphabet/k setting")
        seen.add(key)
        positive_integer(setting["min_unique_kmers"], "min_unique_kmers")
    expected = {
        (a, k)
        for a, ks in (
            ("protein20", (4, 5, 6, 7, 8, 9)),
            ("hp_thomas_dill2", (19, 23, 27, 31, 35)),
        )
        for k in ks
    }
    if seen != expected:
        raise ValueError("calibration does not cover every frozen alphabet/k setting")
    return record


def preflight(root, manifest, species, calibration=None):
    check_rules(root)
    if species == "mouse":
        if calibration is None:
            raise ValueError("mouse access blocked: frozen calibration is missing")
        check_calibration(calibration, digest(root / "decision_rules.md"))
    # Do not open a mouse manifest until the calibration check has passed.
    rows = check_samples(manifest, species)
    return {
        "species": species,
        "biological_samples": len(rows),
        "rules_sha256": digest(root / "decision_rules.md"),
    }


def verdict(parts):
    """A failed component fails a claim; absent components cannot create a PASS."""
    if False in parts:
        return "FAIL"
    return "not measured" if None in parts else "PASS"


def score_claims(measurements):
    """Apply the fixed limits to supplied measurements; never infer missing values."""
    result = {}
    rules = {
        "1": [
            ("wall_ratio_no_reference", 0.1, "lt"),
            ("cpu_ratio_no_reference", 0.1, "lt"),
            ("wall_ratio_with_reference", 0.5, "lt"),
            ("cpu_ratio_with_reference", 0.5, "lt"),
        ],
        "2": [("genes_hit_top300", 70, "ge"), ("go_overlap_top20", 10, "ge")],
        "3": [
            ("in_domain_accuracy", 0.8, "ge"),
            ("outside_protein_false_family_rate", 0.05, "le"),
            ("tag_to_hmmscan_cpu_ratio", 0.1, "lt"),
        ],
        "4": [("recall_gain_at_equal_false_family_rate", 0.05, "ge")],
        "5": [
            ("true_exceeds_every_swap", 1, "eq"),
            ("median_swap_to_true_ratio", 0.05, "lt"),
        ],
    }
    compare = {
        "lt": lambda x, y: x < y,
        "le": lambda x, y: x <= y,
        "ge": lambda x, y: x >= y,
        "eq": lambda x, y: x == y,
    }
    for claim, limits in rules.items():
        components = []
        values = measurements.get(claim, {})
        for name, limit, op in limits:
            raw = values.get(name)
            if raw is None:
                components.append(None)
                continue
            if name == "true_exceeds_every_swap":
                if type(raw) is not bool:
                    raise ValueError("true_exceeds_every_swap must be a boolean")
                value = int(raw)
            elif name == "recall_gain_at_equal_false_family_rate":
                if isinstance(raw, bool):
                    raise ValueError("recall gain is not a boolean")
                value = float(raw)
                if not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError("recall gain must be between -1 and 1")
            else:
                value = finite_number(raw, name)
            if (
                name in {"in_domain_accuracy", "outside_protein_false_family_rate"}
                and value > 1
            ):
                raise ValueError(f"{name} must be a fraction")
            if name in {"genes_hit_top300", "go_overlap_top20"}:
                maximum = 100 if name == "genes_hit_top300" else 20
                if not value.is_integer() or value > maximum:
                    raise ValueError(f"{name} is outside its possible count")
            components.append(compare[op](value, limit))
        result[claim] = {"verdict": verdict(components), "components": components}
    result["alphabet_action"] = (
        "re-report claims 1 to 3 in protein20"
        if result["4"]["verdict"] == "FAIL"
        else (
            "not measured"
            if result["4"]["verdict"] == "not measured"
            else "retain reduced alphabet"
        )
    )
    return result


def score_swaps(rows, expected_assignments):
    """Reject missing or duplicated assignments before evaluating claim 5."""
    expected = {r["assignment"]: r for r in expected_assignments}
    unique([r["assignment"] for r in rows], "assignment result")
    if {r["assignment"] for r in rows} != set(expected):
        raise ValueError("swap results do not cover every balanced assignment")
    counts = {}
    for row in rows:
        raw = row["significant_units"]
        value = finite_number(raw, "significant_units")
        if not value.is_integer():
            raise ValueError("significant_units must be an integer")
        counts[row["assignment"]] = int(value)
    true_key = [key for key, r in expected.items() if r["is_true"]]
    if len(true_key) != 1:
        raise ValueError("exactly one true assignment is required")
    true = counts[true_key[0]]
    swaps = [n for key, n in counts.items() if key != true_key[0]]
    if not swaps:
        raise ValueError("no swaps")
    return {
        "true_exceeds_every_swap": all(true > n for n in swaps),
        "median_swap_to_true_ratio": statistics.median(swaps) / true if true else None,
        "true_significant_units": true,
        "swap_count": len(swaps),
    }


def overlap_curve(ranked_units, top_genes, overlaps):
    """Count distinct top genes with an exon overlap of at least 20 bases."""
    unique(ranked_units, "ranked unit")
    unique(top_genes, "answer-key gene")
    if len(top_genes) > 100:
        raise ValueError("answer key contains more than 100 genes")
    hits = {}
    allowed_units = set(ranked_units)
    allowed_genes = set(top_genes)
    for row in overlaps:
        raw = str(row["overlap_bp"])
        if not raw.isdigit():
            raise ValueError("overlap_bp must be a whole number, 0 or more")
        bases = int(raw)
        if row["unit"] not in allowed_units:
            raise ValueError("overlap contains an unknown unit")
        if row["gene"] in allowed_genes and bases >= 20:
            hits.setdefault(row["unit"], set()).add(row["gene"])
    result = []
    for count in (100, 200, 300, 500, 1000):
        found = set().union(*(hits.get(unit, set()) for unit in ranked_units[:count]))
        result.append(
            {
                "requested_units": count,
                "available_units": min(count, len(ranked_units)),
                "genes_hit": len(found),
                "answer_key_genes": len(top_genes),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-rules")
    p = sub.add_parser("preflight")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--species", choices=["mouse", "zebrafish"], required=True)
    p.add_argument("--calibration", type=Path)
    p = sub.add_parser("assignments")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--species", choices=["mouse", "zebrafish"], required=True)
    p.add_argument("--calibration", type=Path)
    p = sub.add_parser("audit-index")
    p.add_argument("--members", type=Path, required=True)
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--heldout", choices=["7742", "40674"], required=True)
    p = sub.add_parser("check-limits")
    p.add_argument("--measurements", type=Path, required=True)
    args = parser.parse_args()
    try:
        check_rules()
        if args.command == "preflight":
            out = preflight(ROOT, args.manifest, args.species, args.calibration)
        elif args.command == "assignments":
            preflight(ROOT, args.manifest, args.species, args.calibration)
            rows = check_samples(args.manifest, args.species)
            first = sorted({r["condition"] for r in rows})[0]
            out = list(
                assignments(
                    [r["sample"] for r in rows],
                    [int(r["condition"] != first) for r in rows],
                )
            )
        elif args.command == "audit-index":
            out = {
                "member_rows_checked": check_index_members(
                    args.members, args.heldout, args.nodes
                ),
                "taxonomy_sha256": digest(args.nodes),
            }
        elif args.command == "check-limits":
            out = {
                "status": "arithmetic only; not biological evidence",
                "candidate_verdicts": score_claims(
                    json.loads(args.measurements.read_text())
                ),
            }
        else:
            out = {"rules": "unchanged"}
        print(json.dumps(out, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
