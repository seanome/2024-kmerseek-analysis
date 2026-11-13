"""Fetch TimeTree's divergence time from human for every QfO target species.

Writes results/leaderboard/timetree_human_pairwise.tsv, one row per QfO species label, read
by build_leaderboard_long.py as the only source of `target_mya_to_human`.

Why this exists: the QfO region runs carried divergence times typed by hand into the
pipeline at commit a47ea25 (mouse 100, rat 90, cow and dog 95, ...), with no source. They
put mouse after dog and cow, where human splits from rodents before it splits from
Laurasiatheria. The pipeline's own values stay as they are (they reach scoreDomainCalls as
`--species-mya`, so changing them would rehash every scored task); the leaderboard table
replaces them with TimeTree's.

The value used is TimeTree's `adjusted_age` when it is above 0, else `precomputed_age`
(the median of the studies), the same rule as the organism page. A species TimeTree gives
no age for (E. coli: the API returns no row for human against any bacterium) is written
with an empty age, so the leaderboard shows no Mya rather than a guess.

Usage:
    python scripts/fetch_timetree_human_pairwise.py \
        --species nextflow-runs/qfo-pfam-region-benchmark/assets/qfo_species.tsv \
        --out results/leaderboard/timetree_human_pairwise.tsv
"""

import argparse
import csv
import datetime
import sys
import time
import urllib.request
from pathlib import Path

API = "https://timetree.org/api/pairwise"
HUMAN = 9606
# The QfO species that have scored rows in the leaderboard. Others in the registry have no
# divergence axis to fill.
LABELS = [
    "ptroglodytes",
    "rnorvegicus",
    "mouse",
    "btaurus",
    "clupus",
    "mdomestica",
    "chicken",
    "xtropicalis",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
]
COLUMNS = [
    "label",
    "taxon",
    "scientific_name",
    "timetree_taxon_b",
    "timetree_name_b",
    "mya",
    "adjusted_age",
    "precomputed_age",
    "ci_low",
    "ci_high",
    "n_studies",
    "url",
    "fetched",
]


def pairwise(taxon: str) -> dict:
    url = f"{API}/{HUMAN}/{taxon}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read().decode()
            break
        except Exception:  # TimeTree answers HTTP 500 for some pairs it cannot place
            if attempt == 2:
                return {"url": url}
            time.sleep(5)
    lines = body.strip().splitlines()
    if len(lines) < 2 or not lines[0].startswith("taxon_a_id"):
        return {"url": url}
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    adj = float(row.get("adjusted_age") or 0)
    pre = float(row.get("precomputed_age") or 0)
    age = adj if adj > 0 else pre
    return {
        "url": url,
        "timetree_taxon_b": row["taxon_b_id"],
        "timetree_name_b": row["scientific_name_b"],
        "mya": f"{age:.2f}" if age > 0 else "",
        "adjusted_age": row.get("adjusted_age", ""),
        "precomputed_age": row.get("precomputed_age", ""),
        "ci_low": row.get("precomputed_ci_low", ""),
        "ci_high": row.get("precomputed_ci_high", ""),
        "n_studies": row.get("all_total", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.species.open() as f:
        registry = {r["label"]: r for r in csv.DictReader(f, delimiter="\t")}
    today = datetime.date.today().isoformat()
    rows = []
    for label in LABELS:
        sp = registry[label]
        row = {
            "label": label,
            "taxon": sp["taxon"],
            "scientific_name": sp["scientific_name"],
            "fetched": today,
            **pairwise(sp["taxon"]),
        }
        rows.append({c: row.get(c, "") for c in COLUMNS})
        print(f"{label:14s} {row.get('mya') or 'no age'}", file=sys.stderr)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
