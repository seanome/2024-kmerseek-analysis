"""Independent checks against fixture reads and written model tables."""

from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = json.loads((root / "input/inputs.json").read_text())
counts = {}
for record in records:
    tally = Counter()
    for mate in record["files"]:
        with gzip.open(mate["path"], "rt") as f:
            while f.readline():
                sequence = f.readline().strip()
                f.readline()
                f.readline()
                for offset in range(len(sequence) - 30):
                    kmer = sequence[offset : offset + 31]
                    reverse = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                    tally[min(kmer, reverse)] += 1
    counts[record["run"]] = {k: v for k, v in tally.items() if v >= 2}
features = set().union(*(set(c) for c in counts.values()))
eligible = {k for k in features if sum(c.get(k, 0) >= 5 for c in counts.values()) >= 2}
seen = set()
with gzip.open(root / "input/results/counts/raw-counts.tsv.gz", "rt") as f:
    reader = csv.DictReader(f, delimiter="\t")
    assert set(reader.fieldnames[1:]) == set(counts)
    for row in reader:
        key = row["tag"]
        assert key not in seen
        seen.add(key)
        for run, expected in counts.items():
            assert int(row[run]) == expected.get(key, 0), (run, key)
assert seen == eligible
checks = []
for path in sorted(
    (root / "input/results/models").glob("*/*/result/all-features.tsv.gz")
):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows and len({r["tag"] for r in rows}) == len(rows)
    finite = []
    for i, r in enumerate(rows):
        try:
            p = float(r["pvalue"])
        except ValueError:
            continue
        if math.isfinite(p):
            finite.append((p, i))
    finite.sort()
    last = 1.0
    expected = {}
    for rank in range(len(finite), 0, -1):
        p, i = finite[rank - 1]
        last = min(last, p * len(rows) / rank)
        expected[i] = last
    for i, q in expected.items():
        assert abs(float(rows[i]["padj"]) - q) < 1e-10
    checks.append(
        dict(
            path=str(path),
            features=len(rows),
            finite_p=len(finite),
            significant=sum(float(r["padj"]) < 0.05 for r in rows if r["padj"] != "NA"),
        )
    )
assert len(checks) == 6
result = json.loads((root / "input/results/evidence/gene-evidence.json").read_text())
assert result["protocol"] == "six-frame-stop-only-X-min30-v1"
assert any(g["gene_id"] == "G000" and g["paper_membership"] for g in result["genes"])
selection = json.loads(
    (root / "input/results/selected-splash/splash-selected/selection.json").read_text()
)
assert selection["tested_anchors"] > 0 and selection["selected_anchors"] > 0
sources = {}
for source in ("splash", "dekupl"):
    protocol = json.loads(
        (root / f"input/results/queries/{source}/queries/protocol.json").read_text()
    )
    if source == "splash":
        assert protocol["dna"] > 0 and protocol["peptides"] > 0
    sources[source] = protocol
significant_kmers = sum(c["significant"] for c in checks if "/kmers/" in c["path"])
assert (sources["dekupl"]["dna"] > 0) == (significant_kmers > 0)
report = dict(
    splash_selection=selection,
    query_sources=sources,
    software_fixture=True,
    canonical_kmers_checked=len(seen),
    samples=len(records),
    model_checks=checks,
    gene_evidence_rows=len(result["genes"]),
)
(root / "fixture-verified.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
