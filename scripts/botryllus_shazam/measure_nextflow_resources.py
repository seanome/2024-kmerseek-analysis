"""Record measured process resources before preparing a production launch."""

import argparse
import csv
import json
import math
from pathlib import Path
import re

DEFAULTS = {
    "VALIDATE_PAIR": (8, 4),
    "COUNT_KMERS": (32, 12),
    "JOIN_KMERS": (32, 12),
    "KALLISTO_INDEX": (16, 4),
    "QUANTIFY_GENES": (8, 12),
    "GENE_COUNTS": (16, 4),
    "POOL_COUNTS": (16, 12),
    "NORMALIZE": (8, 4),
    "BLOCKED_MODEL": (32, 24),
    "SELECT_DE_SEEDS": (8, 4),
    "MERGE_TAGS": (8, 4),
    "RECRUIT_READS": (16, 12),
    "SPLASH": (96, 48),
    "SELECT_SPLASH": (8, 4),
    "COMPACT": (96, 48),
    "TRANSLATE": (8, 4),
    "SKETCH": (16, 12),
    "SEARCH": (32, 12),
    "PREPARE_TARGET": (32, 24),
    "JOIN_EVIDENCE": (32, 12),
    "ALIGN_CONTEXT": (32, 12),
    "SPLIT_PFAM": (8, 4),
    "QUERY_PFAM": (4, 2),
}


def memory_gb(value):
    match = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", value.strip(), re.I)
    if not match:
        return None
    number, unit = match.groups()
    return (
        float(number)
        * {
            "B": 1 / 1024**3,
            "KB": 1 / 1024**2,
            "MB": 1 / 1024,
            "GB": 1,
            "TB": 1024,
        }[unit.upper()]
    )


def hours(value):
    parts = re.findall(r"([\d.]+)\s*(ms|[dhms])", value)
    return (
        sum(
            float(n)
            * {"ms": 1 / 3600000, "s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]
            for n, unit in parts
        )
        if parts
        else None
    )


def measure(traces):
    result = {
        name: dict(
            memory_gb=m,
            time_hours=t,
            evidence=[],
            failures=[],
            basis="provisional model; no production measurement",
        )
        for name, (m, t) in DEFAULTS.items()
    }
    for trace in traces:
        with Path(trace).open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                process = row["name"].split(" (")[0].split(":")[-1]
                if process not in result:
                    continue
                item = result[process]
                mem = memory_gb(row.get("peak_rss", ""))
                time = hours(row.get("realtime", ""))
                item["evidence"].append(
                    dict(
                        trace=str(trace),
                        name=row["name"],
                        status=row["status"],
                        exit=row.get("exit"),
                        peak_rss=row.get("peak_rss"),
                        realtime=row.get("realtime"),
                    )
                )
                if row["status"] == "COMPLETED":
                    if mem is not None:
                        item["memory_gb"] = max(item["memory_gb"], math.ceil(mem * 1.5))
                    if time is not None:
                        item["time_hours"] = max(
                            item["time_hours"], math.ceil(time * 1.5)
                        )
                    item["basis"] = (
                        "default or 1.5 times measured peak, whichever is larger; inspect input scale"
                    )
                elif row["status"] == "FAILED":
                    item["failures"].append(row)
                    # Requested resource columns must be in new traces. Unknown failures block launch.
                    requested = memory_gb(row.get("memory", ""))
                    requested_time = hours(row.get("time", ""))
                    if requested is not None:
                        item["memory_gb"] = max(
                            item["memory_gb"], math.ceil(2 * requested)
                        )
                    if requested_time is not None:
                        item["time_hours"] = max(
                            item["time_hours"], math.ceil(2 * requested_time)
                        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--libraries", type=Path)
    p.add_argument("--sizing-library", type=Path)
    p.add_argument("traces", nargs="*", type=Path)
    a = p.parse_args()
    result = measure(a.traces)
    if bool(a.libraries) != bool(a.sizing_library):
        raise ValueError("Both library manifest and sizing library are required")
    if a.libraries:
        libraries = json.loads(a.libraries.read_text())
        measured = json.loads(a.sizing_library.read_text())
        if measured["pairs"] <= 0 or any(
            r["pairs"] > measured["pairs"] for r in libraries
        ):
            raise ValueError("Sizing measurement must cover the largest library")
        if not any(
            e["status"] == "COMPLETED" and measured["run"] in e["name"]
            for e in result["COUNT_KMERS"]["evidence"]
        ):
            raise ValueError(
                "Missing completed count measurement for the sizing library"
            )
        if any(not re.fullmatch(r"SRR[0-9]+", r["run"]) for r in libraries):
            raise ValueError("Unexpected library identifier")
        result["COUNT_KMERS"]["per_library_memory_gb"] = {
            r["run"]: max(
                32,
                math.ceil(
                    result["COUNT_KMERS"]["memory_gb"] * r["pairs"] / measured["pairs"]
                ),
            )
            for r in libraries
        }
        result["COUNT_KMERS"]["library_sizing_basis"] = dict(
            measured_library=measured["run"],
            measured_pairs=measured["pairs"],
            rule="Scale the 1.5x measured largest-library peak by read pairs, with a provisional 32 GB floor for other libraries",
        )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "resources.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.output / "resources.config").open("w") as f:
        for name, row in result.items():
            if row["memory_gb"] > 500 or row["time_hours"] > 168:
                raise ValueError(
                    "Resource prediction exceeds configured ceiling: " + name
                )
            if "per_library_memory_gb" in row:
                mapping = ", ".join(
                    f"'{k}': {v}" for k, v in row["per_library_memory_gb"].items()
                )
                f.write(
                    "process { withName: 'COUNT_KMERS' { memory = { def allocations = ["
                    + mapping
                    + "]; if (!allocations.containsKey(run)) error 'No resource prediction for library'; allocations[run] + ' GB' }; time = '"
                    + str(row["time_hours"])
                    + "h' } }\n"
                )
                continue
            f.write(
                f"process {{ withName: '{name}' {{ memory = '{row['memory_gb']} GB'; time = '{row['time_hours']}h' }} }}\n"
            )
    if any(r["failures"] for r in result.values()):
        raise SystemExit("Failures require diagnosis before launch; see resources.json")
