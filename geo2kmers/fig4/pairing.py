"""Check which Fig. 4 array comparisons have the same people before and after.

Usage: pairing.py <groups.tsv> <published_fig4.tsv> <platforms.tsv> <sample_scores.tsv> <series folder> <out.tsv>
Reads who gave each sample from the series matrix: a labelled field, or the title for GSE111555. GSE104079 is
mice assigned to one treatment each, with no animal field, so its groups hold different animals. Also counts
the pre samples whose title says posttraining, and gives the AUC if groups came from the titles instead.
"""

import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.scoring import auc_and_p

groups_file, published_file, platforms_file, scores_file, series, out = sys.argv[1:]
PERSON_FIELD = {"GSE27536": "individual", "GSE83578": "donor", "GSE28998": "subject identifier"}
PERSON_IN_TITLE = {"GSE111555": r"^(\d+)[A-Z]"}
SEPARATE_ANIMALS = {"GSE104079"}


def sample_fields(gse):
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
        fields[gsm] = {"title": columns["!Sample_title"][0][i], **labelled}
    return fields


def person_of(gse, fields):
    if gse in PERSON_FIELD:
        return fields[PERSON_FIELD[gse]]
    if gse in PERSON_IN_TITLE:
        return re.match(PERSON_IN_TITLE[gse], fields["title"]).group(1)
    return None


def design_of(gse, pre, post):
    if gse in SEPARATE_ANIMALS:
        return "unpaired"
    if pre == post:
        return "paired"
    return "partly paired" if pre & post else "unpaired"


def titled_post(fields):
    return "posttraining" in fields["title"]


def auc_from_titles(gse, comparison, samples, fields):
    """The AUC with each sample put in the group its title names, when a title disagrees with its group."""
    scores = {r["sample"]: float(r["score"]) for r in by_series[gse, comparison]}
    post = [titled_post(fields[s]) for s in samples]
    values = np.array([scores[s] for s in samples])
    return auc_and_p(values[~np.array(post)], values[np.array(post)])[0]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


groups = defaultdict(lambda: defaultdict(list))
for r in rows_of(groups_file):
    groups[r["dataset"], r["comparison"]][r["group"]].append(r["sample"])
arrays = {r["gse"] for r in rows_of(platforms_file) if r["type"] == "array"}
by_series = defaultdict(list)
for r in rows_of(scores_file):
    by_series[r["gse"], r["comparison"]].append(r)

columns = (
    "gse comparison group_pair samples_pre samples_post people_pre people_post people_in_both "
    "most_samples_per_person_in_a_group design pre_samples_titled_post auc_with_groups_from_titles"
)
with open(out, "w") as o:
    o.write("\t".join(columns.split()) + "\n")
    for r in rows_of(published_file):
        gse = r["dataset"]
        if r["panel"] not in ("4C", "4D") or gse not in arrays:
            continue
        fields = sample_fields(gse)
        pre, post = (groups[gse, r["comparison"]][g] for g in ("0", "1"))
        people = [[person_of(gse, fields[s]) for s in group] for group in (pre, post)]
        pre_people, post_people = ({p for p in group if p is not None} for group in people)
        per_person = max((group.count(p) for group in people for p in group if p is not None), default="")
        mislabelled = sum(titled_post(fields[s]) for s in pre)
        auc = f"{auc_from_titles(gse, r['comparison'], pre + post, fields):.3f}" if mislabelled else ""
        row = [gse, r["comparison"], f"{r['group0']} vs {r['group1']}", len(pre), len(post), len(pre_people)]
        row += [len(post_people), len(pre_people & post_people), per_person, design_of(gse, pre_people, post_people)]
        o.write("\t".join(map(str, row + [mislabelled, auc])) + "\n")
