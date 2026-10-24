"""Publish recovered source data without treating queued analyses as results."""

import argparse, csv, gzip, json, math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
a = p.parse_args()
d = a.folder
source = json.loads((d / "source-manifest.json").read_text())
genes = list(csv.DictReader((d / "immune-gene-sets.tsv").open(), delimiter="\t"))
lookup = {g["gene_id"]: g for g in genes}
for g in genes:
    g.update(
        published_tests=[],
        adjusted_tests=[],
        splash_recovery="not measured",
        DEkupl_recovery="not measured",
    )
for i, item in enumerate(source["published_de_tables"], 1):
    with gzip.open(d / (item["sheet"] + ".tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["Gene_ID"] in lookup:
                lookup[r["Gene_ID"]]["published_tests"].append(
                    dict(
                        group=i,
                        log2foldchange=r["log2FoldChange"],
                        pvalue=r["pvalue"],
                        padj=r["padj"],
                    )
                )
# Only import completed adjusted result files; no missing result is a zero.
for contrast in ["young_2h", "young_24h", "young_312h"]:
    file = d / "refit" / f"{contrast}.tsv.gz"
    if not file.exists():
        continue
    with gzip.open(file, "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or {r["scope"] for r in rows} != {"genotype_collection_adjusted"}:
        raise ValueError("Adjusted file scope differs")
    if len({r["gene_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate model gene IDs")
    for r in rows:
        if r["gene_id"] in lookup:
            lookup[r["gene_id"]]["adjusted_tests"].append(
                dict(
                    contrast=contrast,
                    padj=r["padj"],
                    log2foldchange=r["log2FoldChange"],
                )
            )
hubs = json.loads((d / "hub-gene-memberships.json").read_text())
result = dict(
    source=source,
    immune_genes=genes,
    hub_label_mapping=dict(
        unique=sum(r["status"].startswith("unique") for r in hubs),
        ambiguous=sum(r["status"].startswith("ambiguous") for r in hubs),
        unresolved=sum(r["status"] == "unresolved label" for r in hubs),
    ),
    refit=dict(
        first_job="49373699",
        first_job_status="Six condition-only comparisons finished; job then failed on missing jsonlite; outputs not yet downloaded",
        adjusted_status="Corrected TSV-input runner committed; transfer and submission blocked by expired SSH authentication",
        completion=False,
    ),
    recovery=dict(
        job="49373978",
        status="Queued behind full-depth evidence; no recovery output imported",
    ),
    score_reproduction=dict(
        status="Not reproduced",
        reason="146 Botryllus gene IDs recovered. Cited companion mammalian code recovered, but no Botryllus-specific transformed expression matrix or scoring implementation found.",
    ),
    interpretation="Published-count redraw is not an independent reproduction. Original raw-P selection and adjusted-P results remain separate. SPLASH/DE-kupl missing outputs are not negative recoveries.",
)
(d / "dashboard.json").write_text(json.dumps(result, indent=2) + "\n")
