"""Correct the six GSE111555 samples whose source field disagrees with their title, and regroup.

Usage: regroup_gse111555.py <GSE111555 series matrix> <groups.tsv> <published_fig4.tsv> <pr212 scores> <out folder>
Titles end a (baseline) or b (after training) after the man's number and tissue, as in 1Pa and 1Pb. A title is
trusted only if the sample's VO2peak equals the value in another sample from that man with the same letter, in
muscle or blood, where title and source agree. Stops if any corrected sample fails that check.
Writes groups.tsv and published_fig4.tsv for fig4.nf's --groups and --published, and gse111555_labels.tsv. The
AUC in published_fig4.tsv is the published score, from PR 212, recomputed on the corrected groups, so both sides
of the comparison use the same samples.
"""

import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo2kmers.scoring import auc_and_p

matrix, groups_file, published_file, scores_file, out = sys.argv[1:]
GSE = "GSE111555"
LETTER = {"a": "pre", "b": "post"}
VO2 = "vo2peak (l/min)"


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def records():
    """Each sample's title, source and labelled characteristics, which sit in different rows per sample."""
    columns = defaultdict(list)
    with gzip.open(matrix, "rt", errors="replace") as f:
        for line in f:
            key, _, rest = line.rstrip("\n").partition("\t")
            if key in (
                "!Sample_geo_accession",
                "!Sample_title",
                "!Sample_source_name_ch1",
                "!Sample_characteristics_ch1",
            ):
                columns[key].append([cell.strip('"') for cell in rest.split("\t")])
    found = {}
    for i, gsm in enumerate(columns["!Sample_geo_accession"][0]):
        labelled = dict(row[i].split(": ", 1) for row in columns["!Sample_characteristics_ch1"] if ": " in row[i])
        title = columns["!Sample_title"][0][i]
        man, tissue, letter = re.match(r"(\d+)([A-Z])([ab]):", title).groups()
        source = columns["!Sample_source_name_ch1"][0][i]
        found[gsm] = dict(
            title=title.split(":")[0],
            man=man,
            tissue=tissue,
            by_title=LETTER[letter],
            by_source="post" if "posttraining" in source else "pre",
            vo2=labelled.get(VO2, ""),
        )
    return found


samples = records()
other_tissue = defaultdict(set)
for s in samples.values():
    if s["tissue"] != "P" and s["vo2"] and s["by_title"] == s["by_source"]:
        other_tissue[s["man"], s["by_title"]].add(s["vo2"])


def vo2_says(sample):
    pre, post = (sample["vo2"] in other_tissue[sample["man"], when] for when in ("pre", "post"))
    return "post" if post and not pre else "pre" if pre and not post else "unclear"


group_rows = rows_of(groups_file)
authors = {r["sample"]: r["group"] for r in group_rows if r["dataset"] == GSE}
score = {r["sample"]: float(r["score"]) for r in rows_of(scores_file)}
labels = []
for gsm, group in authors.items():
    sample = samples[gsm]
    says = vo2_says(sample)
    if sample["by_title"] != sample["by_source"] and says != sample["by_title"]:
        raise SystemExit(f"{gsm} {sample['title']}: VO2peak does not support its title")
    labels.append((gsm, sample, says, group, "1" if sample["by_title"] == "post" else "0"))

Path(out).mkdir(parents=True, exist_ok=True)
with open(f"{out}/gse111555_labels.tsv", "w") as f:
    f.write(
        "sample\ttitle\tby_source\tby_title\tvo2peak_l_min\tvo2peak_says\tgroup_authors\tgroup_corrected\tscore_pr212\n"
    )
    for gsm, sample, says, group, corrected in labels:
        row = [gsm, sample["title"], sample["by_source"], sample["by_title"], sample["vo2"], says, group, corrected]
        f.write("\t".join(row + [str(score[gsm])]) + "\n")

corrected = {gsm: new for gsm, _, _, _, new in labels}
with open(f"{out}/groups.tsv", "w") as f:
    f.write("dataset\tcomparison\tgroup\tsample\n")
    for r in group_rows:
        group = corrected[r["sample"]] if r["dataset"] == GSE else r["group"]
        f.write(f"{r['dataset']}\t{r['comparison']}\t{group}\t{r['sample']}\n")

pre, post = (np.array([score[s] for s, g in corrected.items() if g == want]) for want in "01")
auc, p = auc_and_p(pre, post)
with open(published_file) as source, open(f"{out}/published_fig4.tsv", "w") as f:
    reader = csv.DictReader(source, delimiter="\t")
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for r in reader:
        if r["dataset"] == GSE and r["comparison"] == "main":
            r |= {"n0": len(pre), "n1": len(post), "auc": f"{auc:.3f}", "p": f"{p:.3g}"}
        writer.writerow(r)
