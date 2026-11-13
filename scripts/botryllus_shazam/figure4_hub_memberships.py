"""Resolve paper hub labels only within the matching curated DE comparison."""

import argparse, csv, json, re
from pathlib import Path
import openpyxl

p = argparse.ArgumentParser()
p.add_argument("workbook", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
w = openpyxl.load_workbook(a.workbook, read_only=True, data_only=True)
curated = []
for sheet, cohort, groups in [
    ("annotated DE all groups young", "young", [1, 1, 2, 2, 4, 4, 5, 5]),
    ("annotated DE all groups old", "old", [3, 3, 6, 6]),
]:
    for row in w[sheet].iter_rows(min_row=2, values_only=True):
        for j, group in enumerate(groups):
            gid = row[2 * j]
            label = row[2 * j + 1]
            if not gid:
                continue
            if not re.fullmatch("g[0-9]+", str(gid)):
                raise ValueError(gid)
            curated.append(
                dict(
                    gene_id=gid,
                    label=str(label),
                    cohort=cohort,
                    group=group,
                    direction="Up-reg" if j % 2 == 0 else "Down-reg",
                )
            )


def clean(x):
    return re.sub(r"\s+", " ", x.replace("*", "").strip().rstrip(".")).casefold()


hubrows = json.loads((a.output / "published-hub-gene-text.json").read_text())
records = []
for h in hubrows:
    if h["declared_count"] is None:
        continue
    group = (
        1
        if h["time"] == "2 Hours Young"
        else (
            2
            if h["time"] == "24 Hours Young"
            else (
                4
                if h["time"] == "13–49 Days Young"
                else (
                    5
                    if h["time"] == "132 Days Young"
                    else (
                        3
                        if h["time"] == "24 Hours"
                        else 6 if h["time"] == "41–96 Days" else None
                    )
                )
            )
        )
    )
    if group is None:
        raise ValueError(h["time"])
    for token in h["gene_text"].split(","):
        token = token.strip()
        if not token:
            continue
        name = re.sub(r"\s*\(\d+ entries\)", "", token)
        names = {clean(name)} | {clean(x) for x in name.split("/")}
        matches = [
            r
            for r in curated
            if r["group"] == group
            and r["direction"] == h["direction"]
            and clean(r["label"]) in names
        ]
        ids = sorted({r["gene_id"] for r in matches})
        records.append(
            dict(
                cohort=h["cohort"],
                hub=h["hub"],
                direction=h["direction"],
                time=h["time"],
                paper_group=group,
                paper_label=token,
                gene_ids=ids,
                status=(
                    "unique exact label within published comparison"
                    if len(ids) == 1
                    else (
                        "ambiguous label within published comparison"
                        if ids
                        else "unresolved label"
                    )
                ),
            )
        )
(a.output / "hub-gene-memberships.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
with (a.output / "curated-DE-gene-ids.tsv").open("w") as f:
    t = csv.DictWriter(
        f,
        fieldnames=["gene_id", "label", "cohort", "group", "direction"],
        delimiter="\t",
        lineterminator="\n",
    )
    t.writeheader()
    t.writerows(curated)
print(
    "Curated gene/comparison rows",
    len(curated),
    "hub tokens",
    len(records),
    "unique",
    sum(r["status"].startswith("unique") for r in records),
    "ambiguous",
    sum(r["status"].startswith("ambiguous") for r in records),
    "unresolved",
    sum(r["status"] == "unresolved label" for r in records),
)
