"""Read which person gave each sample in a series."""

import gzip
import re
from collections import defaultdict
from pathlib import Path

from geo2kmers.geo import Geo2KmersError

PERSON_FIELD = {"GSE27536": "individual", "GSE83578": "donor", "GSE28998": "subject identifier"}
PERSON_IN_TITLE = {"GSE111555": r"^(\d+)[A-Z]"}


def sample_fields(series, gse):
    """Each sample's accession mapped to its title and its labelled characteristics."""
    columns = defaultdict(list)
    with gzip.open(Path(series) / gse / f"{gse}_series_matrix.txt.gz", "rt", errors="replace") as f:
        for line in f:
            key, _, rest = line.rstrip("\n").partition("\t")
            if key in ("!Sample_geo_accession", "!Sample_title", "!Sample_characteristics_ch1"):
                columns[key].append([cell.strip('"') for cell in rest.split("\t")])
    fields = {}
    for i, gsm in enumerate(columns["!Sample_geo_accession"][0]):
        labelled = dict(row[i].split(": ", 1) for row in columns["!Sample_characteristics_ch1"] if ": " in row[i])
        fields[gsm] = {"title": columns["!Sample_title"][0][i], "gsm": gsm, **labelled}
    return fields


def person_of(gse, fields):
    name = f"{fields.get('gsm', 'a sample')} ({fields.get('title', '')!r})"
    if gse in PERSON_FIELD:
        if PERSON_FIELD[gse] not in fields:
            raise Geo2KmersError(f"{gse} {name} has no '{PERSON_FIELD[gse]}' characteristic")
        return fields[PERSON_FIELD[gse]]
    if gse in PERSON_IN_TITLE:
        match = re.match(PERSON_IN_TITLE[gse], fields["title"])
        if not match:
            raise Geo2KmersError(f"{gse} {name}: the title does not start with a person number")
        return match.group(1)
    return None
