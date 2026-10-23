"""Stream public RNA-Seq metadata; never retrieve raw reads here."""

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.ebi.ac.uk/ena/portal/api/"
FIELDS = [
    "run_accession",
    "study_accession",
    "secondary_study_accession",
    "sample_accession",
    "experiment_accession",
    "scientific_name",
    "tax_id",
    "sample_title",
    "study_title",
    "library_layout",
    "library_strategy",
    "library_source",
    "instrument_platform",
    "read_count",
    "base_count",
    "fastq_bytes",
    "fastq_ftp",
    "fastq_md5",
    "first_public",
]


def inventory(taxon, name, output):
    query = f'tax_tree({taxon}) AND library_strategy="RNA-Seq"'
    args = dict(result="read_run", query=query, format="tsv")
    count_url = BASE + "count?" + urllib.parse.urlencode(args)
    count_text = urllib.request.urlopen(count_url, timeout=180).read().decode()
    expected = int(count_text.splitlines()[1])
    url = (
        BASE
        + "search?"
        + urllib.parse.urlencode(dict(**args, fields=",".join(FIELDS), limit=0))
    )
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"{name}-rna-seq.tsv.gz"
    if final.exists():
        raise ValueError("Snapshot already exists; use a new directory")
    partial = final.with_suffix(".gz.partial")
    counts = collections.Counter()
    db = sqlite3.connect(output / f"{name}-identities.sqlite")
    db.execute(
        "CREATE TABLE identities(run TEXT PRIMARY KEY, study TEXT, sample TEXT, taxon TEXT)"
    )
    platforms = collections.Counter()
    layouts = collections.Counter()
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=180) as response, gzip.GzipFile(
        filename=str(partial), mode="wb", mtime=0
    ) as dest:
        # One row at a time keeps metadata inventories bounded in memory.
        lines = io.TextIOWrapper(response, encoding="utf-8", newline="")
        for index, line in enumerate(lines):
            b = line.encode()
            h.update(b)
            dest.write(b)
            if index == 0:
                header = next(
                    csv.reader([line], delimiter="\t", quoting=csv.QUOTE_NONE)
                )
                continue
            # ENA TSV titles can contain literal unmatched quotation marks.
            values = next(csv.reader([line], delimiter="\t", quoting=csv.QUOTE_NONE))
            if len(values) != len(header):
                raise ValueError("Malformed TSV row")
            row = dict(zip(header, values))
            accession = row["run_accession"]
            db.execute(
                "INSERT INTO identities VALUES(?,?,?,?)",
                (
                    accession,
                    row["study_accession"],
                    row["sample_accession"],
                    row["tax_id"],
                ),
            )
            if index % 10000 == 0:
                db.commit()
            counts["runs"] += 1
            counts["bases"] += int(row["base_count"] or 0)
            counts["listed_fastq_bytes"] += sum(
                int(x) for x in row["fastq_bytes"].split(";") if x
            )
            counts["missing_fastq_urls"] += not bool(row["fastq_ftp"])
            counts["missing_fastq_sizes"] += not bool(row["fastq_bytes"])
            platforms[row["instrument_platform"]] += 1
            layouts[row["library_layout"]] += 1
    if counts["runs"] != expected:
        raise ValueError(
            f'Inventory changed or truncated: count endpoint {expected}; received {counts["runs"]}'
        )
    db.commit()
    studies, samples, taxa = db.execute(
        "SELECT count(DISTINCT study),count(DISTINCT sample),count(DISTINCT taxon) FROM identities"
    ).fetchone()
    db.close()
    partial.rename(final)
    report = dict(
        query=query,
        taxon=taxon,
        taxonomy_source=f"https://www.ncbi.nlm.nih.gov/datasets/taxonomy/{taxon}/",
        url=url,
        count_url=count_url,
        expected_runs=expected,
        **counts,
        studies=studies,
        biosamples=samples,
        taxon_ids=taxa,
        platforms=dict(platforms),
        layouts=dict(layouts),
        uncompressed_sha256=h.hexdigest(),
        indexed_runs=0,
        indexing_status="not run",
        snapshot_date="2026-10-23",
    )
    (output / f"{name}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--scope", choices=["tunicate", "chordate", "both"], default="both")
    a = p.parse_args()
    # Names and descendant IDs checked against current NCBI Taxonomy on 2026-10-23.
    if a.scope in {"tunicate", "both"}:
        inventory(7712, "tunicate", a.output)
    if a.scope in {"chordate", "both"}:
        inventory(7711, "chordate", a.output)
