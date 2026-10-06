#!/usr/bin/env python3
"""Fetch the two small inputs the pilot commits to assets/, so a later run reads the same ones.

  assets/chr6_queries.<release>.tsv   reviewed human Swiss-Prot entries on chromosome 6:
                                      proteome UP000005640, component "Chromosome 6".
  assets/disprot_human_disorder.tsv   DisProt consensus disorder regions of human proteins
                                      ("Structural state" regions of type D), 1-based,
                                      inclusive.

The UniProt query filters the proteome component client-side, from the "Proteomes" column
("UP000005640: Chromosome 6"), because the REST query field proteomecomponent:"Chromosome 6"
returns 0 hits (checked 2026-10-06) and proteomecomponent:6 matches other components too.

nextflow-runs/disprot-benchmark/bin/parse_disprot.py is not reused, for three reasons.
DisProt pages count from 0 and it asks for page 1 first, so it skips the first page. It
reads the total from `count` or `total`, but the API calls it `size`, so it stops after one
page. On 2026-10-06 those two together kept 1_337 of 3_337 entries. And it reads `regions`,
every annotation term, rather than the consensus disorder.

    python3 tools/fetch_inputs.py --assets assets
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UNIPROT = "https://rest.uniprot.org/uniprotkb/stream"
DISPROT = "https://disprot.org/api/search"
PROTEOME = "UP000005640"
COMPONENT = f"{PROTEOME}: Chromosome 6"


def get(url: str, params: dict) -> tuple[bytes, dict]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=300) as resp:
        return resp.read(), dict(resp.headers)


def fetch_chr6(assets: Path) -> None:
    body, headers = get(UNIPROT, {
        "query": f"proteome:{PROTEOME} AND reviewed:true",
        "fields": "accession,gene_primary,xref_proteomes,length",
        "format": "tsv",
    })
    release = headers.get("X-UniProt-Release") or headers.get("x-uniprot-release")
    if not release:
        sys.exit("UniProt sent no X-UniProt-Release header; cannot name the file")
    lines = body.decode().splitlines()
    rows = [ln.split("\t") for ln in lines[1:]]
    keep = [r for r in rows if COMPONENT in [p.strip() for p in r[2].split(";")]]
    out = assets / f"chr6_queries.{release}.tsv"
    with open(out, "w") as fh:
        fh.write("accession\tgene\tlength\n")
        for acc, gene, _, length in sorted(keep):
            fh.write(f"{acc}\t{gene}\t{length}\n")
    print(f"[chr6] release {release}: {len(rows)} reviewed human entries, "
          f"{len(keep)} on chromosome 6 -> {out}", file=sys.stderr)


def fetch_disprot(assets: Path) -> None:
    entries, page = [], 0   # DisProt pages count from 0
    while True:
        body, _ = get(DISPROT, {"release": "current", "format": "json",
                                "ncbi_taxon_id": "9606", "page_size": 500, "page": page})
        payload = json.loads(body)
        data = payload.get("data", [])
        entries.extend(data)
        total = payload["size"]
        if not data or len(entries) >= total:
            break
        page += 1
    if len(entries) != total:
        sys.exit(f"DisProt returned {len(entries)} of {total} entries")
    rows = []
    for e in entries:
        if str(e.get("ncbi_taxon_id")) != "9606":
            continue
        for r in e.get("disprot_consensus", {}).get("Structural state", []):
            if r.get("type") == "D":
                rows.append((e["acc"], e["disprot_id"], int(r["start"]), int(r["end"])))
    out = assets / "disprot_human_disorder.tsv"
    with open(out, "w") as fh:
        fh.write("accession\tdisprot_id\tstart\tend\n")
        for row in sorted(rows):
            fh.write("\t".join(map(str, row)) + "\n")
    print(f"[disprot] {len(entries)} human entries, {len(rows)} consensus disorder regions "
          f"-> {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, required=True)
    args = ap.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    fetch_chr6(args.assets)
    fetch_disprot(args.assets)


if __name__ == "__main__":
    main()
