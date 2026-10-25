"""Publish recovered source data without treating queued analyses as results."""

import argparse, csv, gzip, json, math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
a = p.parse_args()
d = a.folder
source = json.loads((d / "source-manifest.json").read_text())
genes = list(csv.DictReader((d / "immune-gene-sets.tsv").open(), delimiter="\t"))
hubs = json.loads((d / "hub-gene-memberships.json").read_text())
labels = {
    r["gene_id"]: r
    for r in csv.DictReader((d / "gene-labels.tsv").open(), delimiter="\t")
}
lookup = {g["gene_id"]: g for g in genes}
for h in hubs:
    for gid in h["gene_ids"]:
        if gid not in lookup:
            lookup[gid] = dict(
                gene_id=gid,
                gene_name=labels.get(gid, {}).get("gene_name", h["paper_label"]),
                role="functional group",
            )
for g in lookup.values():
    g["hub_memberships"] = [h for h in hubs if g["gene_id"] in h["gene_ids"]]
genes = sorted(lookup.values(), key=lambda g: (g["gene_name"].lower(), g["gene_id"]))
for g in genes:
    g.update(
        published_tests=[],
        adjusted_tests=[],
        splash_recovery="not measured",
        DEkupl_recovery="not measured",
        splash_sequences=[],
        DEkupl_sequences=[],
        protein_matches=[],
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
model_summaries = list(
    csv.DictReader((d / "refit/model-summary.tsv").open(), delimiter="\t")
)
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
                    pvalue=r["pvalue"],
                    base_mean=r["baseMean"],
                )
            )
# Import sequence assignments only with their completed recovery provenance.
recovery_folder = d / "recovery"
if (recovery_folder / "complete.json").exists():
    completion = json.loads((recovery_folder / "complete.json").read_text())
    if completion["status"] != "completed":
        raise ValueError("Incomplete recovery cannot be imported")
    sequences = json.loads((recovery_folder / "gene-sequences.json").read_text())
    for gid, g in lookup.items():
        evidence = sequences.get(gid, dict(splash_sequences=[], DEkupl_sequences=[]))
        g.update(evidence)
        g["splash_recovery"] = (
            f"{len(evidence['splash_sequences'])} sequence/comparison assignments; see mapping evidence"
        )
        g["DEkupl_recovery"] = (
            f"{len(evidence['DEkupl_sequences'])} k-mer/comparison assignments; ambiguous mappings shown separately"
        )
hubs = json.loads((d / "hub-gene-memberships.json").read_text())
result = dict(
    source=source,
    immune_genes=[g for g in genes if g["role"] in ("proimmune", "cell_repair")],
    paper_genes=genes,
    unresolved_hub_labels=[h for h in hubs if not h["gene_ids"]],
    model_summaries=model_summaries,
    adjusted_method="Supplied HTSeq counts; collection strata balanced and summed within genotype/condition; DESeq2 ~ genotype + condition; 4/4/3 genotypes; global BH per contrast. Collection is an operational stratum, not a verified calendar date.",
    hub_label_mapping=dict(
        unique=sum(r["status"].startswith("unique") for r in hubs),
        ambiguous=sum(r["status"].startswith("ambiguous") for r in hubs),
        unresolved=sum(r["status"] == "unresolved label" for r in hubs),
    ),
    refit=dict(
        first_job="49373699",
        first_job_status="Six condition-only supplied-count refits completed and retrieved",
        adjusted_job="49387733",
        adjusted_status="Completed and retrieved: young 2 h, 24 h and 312 h; no tested gene passed adjusted P < 0.05",
        completion=True,
    ),
    recovery=dict(
        job="49373978",
        status=(
            "Completed sequence assignments imported"
            if (recovery_folder / "complete.json").exists()
            else "Full-depth SPLASH / DE-kupl pending behind read trimming; no completed mapping output exists"
        ),
    ),
    score_reproduction=dict(
        status="Not reproduced",
        reason="146 Botryllus gene IDs recovered. Cited companion mammalian code recovered, but no Botryllus-specific transformed expression matrix or scoring implementation found.",
    ),
    interpretation="Published-count redraw is not an independent reproduction. Original raw-P selection and adjusted-P results remain separate. SPLASH/DE-kupl missing outputs are not negative recoveries.",
)
(d / "dashboard.json").write_text(json.dumps(result, indent=2) + "\n")

with (d / "paper-gene-evidence.tsv").open("w") as f:
    fields = [
        "gene_id",
        "gene_name",
        "role",
        "hub",
        "paper_time",
        "paper_direction",
        "membership_status",
        "contrast",
        "log2foldchange",
        "padj",
        "expression_status",
        "SPLASH",
        "DEkupl",
        "protein_matches",
    ]
    writer = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for g in genes:
        for h in g["hub_memberships"] or [{}]:
            for contrast in ["young_2h", "young_24h", "young_312h"]:
                test = next(
                    (t for t in g["adjusted_tests"] if t["contrast"] == contrast), {}
                )
                writer.writerow(
                    dict(
                        gene_id=g["gene_id"],
                        gene_name=g["gene_name"],
                        role=g["role"],
                        hub=h.get("hub", "immune model only"),
                        paper_time=h.get("time", ""),
                        paper_direction=h.get("direction", ""),
                        membership_status=h.get(
                            "status", "supplementary immune gene ID"
                        ),
                        contrast=contrast,
                        log2foldchange=test.get("log2foldchange", "NA"),
                        padj=test.get("padj", "NA"),
                        expression_status=(
                            "tested" if test else "below frozen count filter"
                        ),
                        SPLASH=g["splash_recovery"],
                        DEkupl=g["DEkupl_recovery"],
                        protein_matches=(
                            str(len(g["protein_matches"])) + " candidate matches"
                            if (recovery_folder / "complete.json").exists()
                            else "not measured for this paper gene"
                        ),
                    )
                )
