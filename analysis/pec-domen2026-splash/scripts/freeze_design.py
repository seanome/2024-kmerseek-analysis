#!/usr/bin/env python3
"""Build the frozen sample-to-contrast table from SI Dataset S1 and the ENA report.

Usage: freeze_design.py <Dataset_S1.xlsx> <ena_filereport.tsv> <out_dir>

Writes <out_dir>/samples.tsv (all 64 runs) and <out_dir>/contrast_<name>.cjs (SPLASH
sample sheet, control = -1, PEC = +1). The contrast membership comes ONLY from the SI
sheet "DEseq group annotation"; the alias-based guess in the handoff is reported next
to it for comparison and is not used.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import openpyxl

xlsx, ena, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
print("sheets:", wb.sheetnames)
sheet = next(s for s in wb.sheetnames if "group annotation" in s.lower())
groups, cur = {}, None
for row in wb[sheet].iter_rows(values_only=True):
    cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
    for c in cells:
        if re.match(r"^group\s*\d", c, re.I):
            cur = c
            groups[cur] = []
        elif "PaceMaker" in c and cur:
            groups[cur].append(c)
for g, members in groups.items():
    print(g, len(members))

names = {
    "1": "young_2h",
    "2": "young_24h",
    "3": "old_24h",
    "4": "young_13_49d",
    "5": "young_132d",
    "6": "old_41_93d",
}
ena_rows = list(csv.DictReader(open(ena), delimiter="\t"))


def vial(alias):
    m = re.search(r"_(JA\d+|\d{4})_", alias)
    return m.group(1) if m else None


by_vial = {}
for r in ena_rows:
    v = (
        vial(r["sample_alias"])
        or vial(r["library_name"])
        or vial(r["experiment_alias"])
    )
    if v is None:
        raise SystemExit(f"cannot identify vial in ENA run {r['run_accession']}")
    by_vial.setdefault(v, []).append(r)
dups = {v: len(rs) for v, rs in by_vial.items() if len(rs) != 1}
if dups:
    raise SystemExit(f"vial ids not unique in ENA: {dups}")

assign = {}
for g, members in groups.items():
    num = re.search(r"\d+", g).group(0)
    if num not in names:
        raise SystemExit(f"unknown SI group {g}")
    for m in members:
        v = vial(m)
        if v is None or not ("PaceMakerTre" in m or "PaceMakerCon" in m):
            raise SystemExit(f"unknown SI sample name: {m}")
        if v in assign:
            raise SystemExit(f"{v} in two SI groups")
        assign[v] = (names[num], "PEC" if "PaceMakerTre" in m else "control", m)
        if v not in by_vial:
            raise SystemExit(f"SI sample {m} has no ENA run")

rows = []
for v, (r,) in sorted(by_vial.items()):
    alias = r["sample_alias"]
    parts = alias.split("_")
    contrast, treat, si_name = assign.get(
        v, ("not_in_SI", "PEC" if "PaceMakerTre" in alias else "control", "")
    )
    ena_treat = (
        "PEC"
        if "PaceMakerTre" in alias
        else ("control" if "PaceMakerCon" in alias else "unknown")
    )
    if v in assign and treat != ena_treat:
        raise SystemExit(f"{v}: SI says {treat}, ENA alias says {ena_treat}")
    rows.append(
        dict(
            sample=v,
            run=r["run_accession"],
            treatment=ena_treat,
            genotype=parts[2],
            age_days=next((p[:-1] for p in parts if re.fullmatch(r"\d+d", p)), ""),
            contrast=contrast,
            read_pairs=r["read_count"],
            si_name=si_name,
            ena_alias=alias,
        )
    )
if not rows:
    raise SystemExit("ENA report contains no runs")
for name in names.values():
    treatments = {r["treatment"] for r in rows if r["contrast"] == name}
    if treatments != {"control", "PEC"}:
        raise SystemExit(f"contrast {name} lacks control or PEC samples")
manifest = {
    "sources": {
        str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [xlsx, ena]
    },
    "runs": len(rows),
    "assigned_runs": sum(r["contrast"] != "not_in_SI" for r in rows),
    "pair_ids": "not constructed by this script; must be validated before fitting a paired model",
}
(out / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
cols = list(rows[0])
with open(out / "samples.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
for name in names.values():
    members = [r for r in rows if r["contrast"] == name]
    with open(out / f"contrast_{name}.cjs", "w") as f:
        for r in members:
            f.write(f"{r['sample']},{1 if r['treatment'] == 'PEC' else -1}\n")
    print(
        name,
        len(members),
        "PEC",
        sum(r["treatment"] == "PEC" for r in members),
        "genotypes",
        sorted({r["genotype"] for r in members}),
    )
