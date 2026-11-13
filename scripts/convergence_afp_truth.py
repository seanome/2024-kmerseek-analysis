#!/usr/bin/env python3
"""Build the ice-binding (antifreeze) protein truth set and pick one query per lineage.

Ice-binding proteins arose separately in fish (type I, type II, type III and the
antifreeze glycoprotein AFGP come from different ancestral genes), insects, plants,
bacteria and fungi. The truth set is every reviewed UniProtKB entry matched by any of
three queries, run against the current UniProt release through the REST API:

  * keyword KW-0047 "Antifreeze protein"
  * GO:0050825 "ice binding"
  * a protein-name search (antifreeze, ice-binding, ice binding, thermal hysteresis,
    ice-structuring), which adds entries that carry neither the keyword nor the GO term,
    such as the spruce budworm antifreeze proteins

Each entry gets a lineage from its UniProt taxonomic lineage and, for fish, the AFP type
read from its protein family or protein name. The query for a lineage is the entry with
the most lines carrying ECO:0000269 (UniProt's evidence code for an experimental result
published in a paper) in its flat-file record.

Writes, under data/convergence_afp/:
  truth_query_log.tsv  one row per query: URL, date, UniProt release, row count
  truth_set.tsv        one row per truth entry, with lineage and evidence count
  queries.tsv          the chosen query per lineage
  raw/                 the downloaded TSV and flat-file text, so a rerun reads them

    python scripts/convergence_afp_truth.py           reuse raw/ if present
    python scripts/convergence_afp_truth.py --refetch download again
"""

import argparse
import datetime as dt
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parent.parent / "data" / "convergence_afp"
RAW = OUT / "raw"
STREAM = "https://rest.uniprot.org/uniprotkb/stream"
RELEASE = "2026_03"

QUERIES = {
    "keyword": "keyword:KW-0047 AND reviewed:true",
    "go": "go:0050825 AND reviewed:true",
    "name": (
        'reviewed:true AND (protein_name:antifreeze OR protein_name:"ice-binding" '
        'OR protein_name:"ice binding" OR protein_name:"thermal hysteresis" '
        'OR protein_name:"ice-structuring")'
    ),
}
FIELDS = (
    "accession,id,protein_name,organism_name,organism_id,lineage,"
    "protein_families,cc_similarity,keyword,go_id,length"
)

# A name-search hit is dropped when its name matches only because it mentions a gene it
# regulates. The one such entry in 2026_03 is a rat DNA helicase whose alternative name
# is "Antifreeze enhancer-binding protein ortholog": it binds DNA near an antifreeze gene
# in fish and does not bind ice.
NAME_ONLY_EXCLUDE = re.compile(r"enhancer-binding", re.IGNORECASE)

LINEAGES = [
    "1 fish, type I AFP",
    "2 fish, AFGP",
    "3 fish, type II AFP",
    "4 fish, type III AFP",
    "5 insects",
    "6 plants",
    "7 bacteria",
    "8 other",
]
FISH_UNKNOWN = "fish, type unknown"


def fetch(url: str, path: Path, refetch: bool) -> tuple[str, str]:
    """Download url to path unless it is already there; return (text, release)."""
    hdr = path.with_suffix(path.suffix + ".release")
    if refetch or not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            release = r.headers.get("X-UniProt-Release", "")
            text = r.read().decode()
        path.write_text(text)
        hdr.write_text(release)
    return path.read_text(), hdr.read_text().strip()


def fish_type(families: str, names: str) -> str:
    """AFP type from the family annotation, else from the protein name.

    Type III is checked before type II and type II before type I, because "Type-III"
    contains "Type-II" and "Type-I" as substrings.
    """
    text = f"{families} {names}".lower()
    if re.search(r"type[- ](iii|3)\b", text):
        return LINEAGES[3]
    if re.search(r"type[- ](ii|2)\b", text):
        return LINEAGES[2]
    if re.search(r"type[- ](i|1)\b", text):
        return LINEAGES[0]
    if "glycoprotein" in text or "glycopeptide" in text:
        return LINEAGES[1]
    return FISH_UNKNOWN


def lineage(row: dict) -> str:
    tax = row["Taxonomic lineage"] or ""
    if "Actinopterygii" in tax:
        return fish_type(row["Protein families"] or "", row["Protein names"] or "")
    if "Insecta" in tax:
        return LINEAGES[4]
    if "Viridiplantae" in tax:
        return LINEAGES[5]
    if "Bacteria (domain)" in tax:
        return LINEAGES[6]
    return LINEAGES[7]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    log, tables = [], []
    for name, q in QUERIES.items():
        url = f"{STREAM}?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS, "format": "tsv"}
        )
        path = RAW / f"query_{name}.tsv"
        text, release = fetch(url, path, args.refetch)
        if release != RELEASE:
            sys.exit(f"{name}: UniProt served release {release!r}, expected {RELEASE}")
        t = pl.read_csv(path, separator="\t", infer_schema_length=0)
        log.append(
            {
                "query_name": name,
                "query": q,
                "url": url,
                "fetched_on": today,
                "uniprot_release": release,
                "n_rows": t.height,
            }
        )
        tables.append(t.with_columns(matched_by=pl.lit(name)))

    allrows = pl.concat(tables)
    truth = allrows.group_by(pl.exclude("matched_by")).agg(
        pl.col("matched_by").unique().sort().str.join("+")
    )
    name_only = truth["matched_by"] == "name"
    excluded = truth.filter(
        name_only
        & truth["Protein names"].str.contains("(?i)" + NAME_ONLY_EXCLUDE.pattern)
    )
    truth = truth.join(excluded.select("Entry"), on="Entry", how="anti")

    # Experimental evidence lines per entry, from the flat-file text.
    acc = truth["Entry"].sort().to_list()
    txt_url = f"{STREAM}?" + urllib.parse.urlencode(
        {"query": " OR ".join(f"accession:{a}" for a in acc), "format": "txt"}
    )
    flat, release = fetch(txt_url, RAW / "truth_entries.txt", args.refetch)
    if release != RELEASE:
        sys.exit(f"flat file: UniProt served release {release!r}, expected {RELEASE}")
    n_exp = {}
    for rec in flat.split("\n//\n"):
        m = re.search(r"^AC   (\w+);", rec, re.MULTILINE)
        if m:
            n_exp[m.group(1)] = sum("ECO:0000269" in ln for ln in rec.splitlines())
    missing = set(acc) - set(n_exp)
    if missing:
        sys.exit(f"no flat-file record for {sorted(missing)}")

    truth = truth.with_columns(
        lineage=pl.Series([lineage(r) for r in truth.iter_rows(named=True)]),
        n_experimental_evidence=pl.col("Entry").replace_strict(n_exp),
        Length=pl.col("Length").cast(pl.Int64),
    ).sort("lineage", "Entry")

    log.append(
        {
            "query_name": "union",
            "query": "keyword OR go OR name, minus excluded",
            "url": "",
            "fetched_on": today,
            "uniprot_release": RELEASE,
            "n_rows": truth.height,
        }
    )
    for r in excluded.iter_rows(named=True):
        log.append(
            {
                "query_name": "excluded",
                "query": f"{r['Entry']}: {r['Protein names']}",
                "url": "",
                "fetched_on": today,
                "uniprot_release": RELEASE,
                "n_rows": 1,
            }
        )
    pl.DataFrame(log).write_csv(OUT / "truth_query_log.tsv", separator="\t")
    truth.write_csv(OUT / "truth_set.tsv", separator="\t")

    # One query per lineage 1-7: most experimental evidence lines; ties go to the
    # longer entry, then to the lower accession, so the choice is reproducible.
    queries = (
        truth.filter(pl.col("lineage").is_in(LINEAGES[:7]))
        .sort(
            ["n_experimental_evidence", "Length", "Entry"],
            descending=[True, True, False],
        )
        .group_by("lineage", maintain_order=True)
        .first()
        .sort("lineage")
    )
    queries.write_csv(OUT / "queries.tsv", separator="\t")

    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_width_chars(220)
    print(pl.DataFrame(log).select("query_name", "uniprot_release", "n_rows", "query"))
    print("\nmatched by:", dict(truth["matched_by"].value_counts().iter_rows()))
    counts = (
        pl.DataFrame({"lineage": LINEAGES + [FISH_UNKNOWN]})
        .join(truth.group_by("lineage").len(), on="lineage", how="left")
        .fill_null(0)
    )
    print(counts)
    print(
        queries.select(
            "lineage",
            "Entry",
            "Protein names",
            "Length",
            "Organism",
            "n_experimental_evidence",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
