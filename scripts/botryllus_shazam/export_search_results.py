"""Export completed Swiss-Prot manysearch rows and audit sampled hash intersections."""

import csv
from decimal import Decimal
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
run = root / "manysearch-comparison-20261023-v2"
pilot = root / "primary-search-pilot-20261023"
execution = json.loads((run / "execution.complete.json").read_text())
if not execution["all_settings_agree"]:
    raise ValueError("Engine comparison has disagreements")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "app-export"
out.mkdir(exist_ok=False)
summary = {
    "scope": "Historical pilot queries, complete Swiss-Prot reference; native manysearch completed",
    "query_records": 987,
    "reference_records": 575748,
    "results": [],
    "interpretation": "Candidate sequence matches; no protein function or treatment effect inferred",
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def manifest(z):
    name = next(x for x in z.namelist() if x.lower().endswith("manifest.csv"))
    return csv.DictReader(
        x for x in io.StringIO(z.read(name).decode()) if not x.startswith("#")
    )


def hashes(z, loc):
    data = z.read(loc)
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    obj = json.loads(data)
    if len(obj) != 1 or len(obj[0]["signatures"]) != 1:
        raise ValueError("Expected one stored minhash")
    h = obj[0]["signatures"][0]
    return set(h["mins"]), h["seed"], h["ksize"], h["molecule"]


for entry in execution["settings"]:
    if entry["database"] != "swissprot":
        continue
    mode = entry["mode"]
    source = run / f"swissprot.{mode}.scaled5.csv"
    if sha(source) != entry["csv_sha256"]:
        raise ValueError("Native CSV changed after comparison")
    qzip = zipfile.ZipFile(pilot / f"queries.{mode}.scaled5.sig.zip")
    qmeta = {x["name"]: x for x in manifest(qzip)}
    assert len(qmeta) == 987
    samples = []
    raw = kept = 0
    target = out / f"swissprot.{mode}.scaled5.retained.csv.gz"
    fields = [
        "query_name",
        "match_name",
        "intersect_hashes",
        "query_hashes",
        "containment",
        "ksize",
        "scaled",
        "moltype",
    ]
    with source.open() as inp, gzip.open(target, "wt", newline="") as dst:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(dst, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            raw += 1
            n = Decimal(row["intersect_hashes"])
            q = int(qmeta[row["query_name"]]["n_hashes"])
            if not n.is_finite() or n < 0 or n != n.to_integral_value() or q == 0:
                raise ValueError("Invalid hash count")
            if abs(float(n) / q - float(row["containment"])) > 1e-12:
                raise ValueError("Query containment arithmetic mismatch")
            if len(samples) < 100:
                samples.append(row.copy())
            if n >= 2 and Decimal(row["containment"]) >= Decimal("0.1"):
                row["query_hashes"] = q
                writer.writerow(row)
                kept += 1
    if raw != entry["observed_rows"]:
        raise ValueError("CSV row count differs from completed comparison")
    wanted = {x["match_name"] for x in samples}
    tzip = zipfile.ZipFile(pilot / f"swissprot.{mode}.scaled5.sig.zip")
    tmeta = {x["name"]: x for x in manifest(tzip) if x["name"] in wanted}
    for row in samples:
        q, qs, qk, qm = hashes(qzip, qmeta[row["query_name"]]["internal_location"])
        t, ts, tk, tm = hashes(tzip, tmeta[row["match_name"]]["internal_location"])
        if (qs, qk, qm) != (ts, tk, tm) or len(q & t) != Decimal(
            row["intersect_hashes"]
        ):
            raise ValueError("Sampled shared-hash mismatch")
    summary["results"].append(
        {
            "database": "swissprot",
            "mode": mode,
            "scaled": 5,
            "raw_match_rows": raw,
            "retained_match_rows": kept,
            "query_records": len(qmeta),
            "empty_queries": sum(int(x["n_hashes"]) == 0 for x in qmeta.values()),
            "searchable_queries": sum(int(x["n_hashes"]) > 0 for x in qmeta.values()),
            "eligible_for_two_hash_filter": sum(
                int(x["n_hashes"]) >= 2 for x in qmeta.values()
            ),
            "sampled_pairs_independently_recomputed": len(samples),
            "sample_rule": "First 100 rows of the SHA256-fingerprinted native output",
            "all_row_query_containment_recomputed": True,
            "engine_comparison_agrees": entry["agreement"],
            "raw_csv_sha256": entry["csv_sha256"],
            "retained_csv": target.name,
            "retained_csv_sha256": sha(target),
            "filter": "intersect_hashes >= 2 and query containment >= 0.1",
        }
    )
    qzip.close()
    tzip.close()
(out / "finished-swissprot-manysearch.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2), flush=True)
