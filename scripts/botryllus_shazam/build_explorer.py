"""Package audited sequence records and a timestamped status snapshot for the Site."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def fasta(path):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append(
                dict(id=line[1:].split()[0], description=line[1:], sequence="")
            )
        elif line.strip():
            records[-1]["sequence"] += line.strip()
    return records


def build(pilot, repo, dest):
    dest.mkdir(parents=True, exist_ok=True)
    data = dest / "data"
    data.mkdir(exist_ok=True)
    for name in ["index.html", "app.js"]:
        shutil.copyfile(repo / "sites/botryllus-shazam" / name, dest / name)
    results = pilot / "results"
    report = repo / "reports/botryllus-shazam"
    audit = json.loads((report / "pilot-audit.json").read_text())
    peptides = fasta(results / "peptides.fasta")
    extensions = fasta(results / "extensions.fasta")
    proteins = fasta(results / "botryllus_candidate_annotated.fasta")
    annotations = rows(results / "botryllus_proteome_annotations.tsv")
    assert len(proteins) == len(annotations) == audit["proteome_records"]
    for record, annotation in zip(proteins, annotations):
        assert record["id"] == annotation["record_id"]
        assert (
            hashlib.sha256(record["sequence"].encode()).hexdigest()
            == annotation["sequence_sha256"]
        )
        record.update(annotation)
    assert len(peptides) == audit["observed"]["unique_peptides"]
    assert len(extensions) == audit["observed"]["extension_records"]
    hits = rows(results / "peptide_search_hits.tsv")
    assert (
        sum(h["query_type"] == "real" and h["database"] == "botryllus" for h in hits)
        == audit["recomputed_real_botryllus_hits"]
    )
    bundle = dict(
        peptides=peptides,
        extensions=extensions,
        hits=hits,
        membership=rows(results / "peptide_membership.tsv"),
        compactors=rows(results / "compactors.tsv"),
    )
    (data / "pilot.json").write_text(json.dumps(bundle, separators=(",", ":")))
    with gzip.open(data / "proteins.json.gz", "wt") as f:
        json.dump(proteins, f, separators=(",", ":"))
    for name in [
        "pilot-audit.json",
        "indexing-pilot-validated.json",
        "primary-search-execution.json",
        "swissprot-size-comparison.json",
        "swissprot-raw-vs-sketch-sizes.png",
        "swissprot-raw-vs-sketch-sizes.tsv",
        "status.json",
        "connection-status.json",
        "finished-search-results.json",
        "external-search-progress.json",
        "manysearch-comparison-complete.json",
        "storage-benchmark-complete.json",
        "native-manysearch-botryllus-protein-scaled5.csv",
        "tunicate-summary.json",
        "chordate-summary.json",
        "pilot-all-anchor-selection.json",
        "samples.tsv",
        "sample-audit.json",
        "PROTOCOL.md",
        "ABUNDANCE-PROTOCOL-20261024.md",
        "abundance-design-20261024.json",
        "PIPELINE.md",
        "pilot-lengths.png",
    ]:
        shutil.copyfile(report / name, data / name)
    for name in [
        "extensions.fasta",
        "peptides.fasta",
        "peptide_membership.tsv",
        "peptide_search_hits.tsv",
        "botryllus_proteome_annotations.tsv",
        "compactors.tsv",
        "peptide_search_summary.json",
        "final_protocol.json",
    ]:
        shutil.copyfile(results / name, data / name)
    with (results / "pilot.after_correction.all_anchors.tsv").open(
        "rb"
    ) as f, gzip.open(data / "pilot.all_anchors.tsv.gz", "wb") as o:
        shutil.copyfileobj(f, o)
    shutil.copyfile(
        pilot / "Botryllus-SPLASH-Report.html", dest / "original-report.html"
    )
    original = (
        repo / "reports/botryllus-shazam/sources/botryllus_new_genome_protein.fasta"
    )
    if (
        hashlib.sha256(original.read_bytes()).hexdigest()
        != "843407c15dacecc88d9c88c982e77c0ca835a6bd1330fd4abf635b5df547d855"
    ):
        raise ValueError("Original uploaded proteome checksum changed")
    shutil.copyfile(original, data / "botryllus_new_genome_protein.fasta")
    for name in ["ena-runs.tsv", "retrieval.json"]:
        shutil.copyfile(report / "sources" / name, data / name)
    shutil.copyfile(
        report / "tunicate-rna-seq.tsv.gz", data / "tunicate-rna-seq.tsv.gz"
    )
    for folder in ["finished-manysearch", "external-results", "match-context"]:
        shutil.copytree(report / folder, data / folder, dirs_exist_ok=True)
    paths = []
    for name in [
        "workflow-overview",
        "search-collections",
        "reads-discovery",
        "protein",
        "context-archive",
        "archive-delivery",
        "metadata-records",
    ]:
        shutil.copyfile(report / "metro" / (name + ".svg"), dest / (name + ".svg"))
        shutil.copyfile(report / "metro" / (name + ".mmd"), data / (name + ".mmd"))
        paths.append(report / "metro" / (name + ".svg"))
    # Keep every component's viewBox and coordinates; stack without changing the SVGs.
    dims = [
        list(map(float, ET.parse(p).getroot().attrib["viewBox"].split())) for p in paths
    ]
    width = max(d[2] for d in dims)
    gap = 50
    height = sum(width * d[3] / d[2] for d in dims) + gap * len(dims) + 130
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>Botryllus full analysis: programs and each output file type</title><rect width="100%" height="100%" fill="white"/><text x="25" y="32" font-family="Arial" font-size="22">Botryllus · Shazam for Biology — analysis plan, not a completion claim</text><text x="25" y="62" font-family="Arial" font-size="15">Each document icon is one named file type. Each panel has its own route key. Read left to right.</text><text x="25" y="87" font-family="Arial" font-size="15">Chordates: metadata inventory only. SPLASH composition does not establish treatment-associated abundance.</text>'
    ]
    y = 125
    for i, (p, d) in enumerate(zip(paths, dims)):
        svg = ET.parse(p).getroot()
        svg.set("x", "0")
        svg.set("y", str(y))
        svg.set("width", str(width))
        svg.set("height", str(width * d[3] / d[2]))
        # SVG IDs must remain unique when several complete documents are nested.
        text = ET.tostring(svg, encoding="unicode")
        import re

        for id_ in re.findall(r'\bid="([^"]+)"', text):
            text = text.replace(
                'id="' + id_ + '"', 'id="p' + str(i) + "-" + id_ + '"'
            ).replace("url(#" + id_ + ")", "url(#p" + str(i) + "-" + id_ + ")")
        parts.append(text)
        y += width * d[3] / d[2] + gap
    parts.append("</svg>")
    (report / "metro-map.svg").write_text("\n".join(parts))
    shutil.copyfile(report / "metro-map.svg", dest / "metro-map.svg")
    manifest = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "asset-manifest.json":
            manifest.append(
                dict(
                    path=str(p.relative_to(dest)),
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    assert max(x["bytes"] for x in manifest) < 25_000_000
    (data / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Packaged {len(peptides)} peptides, {len(extensions)} extensions, {len(proteins)} protein records, {len(hits)} hit rows"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pilot", type=Path)
    p.add_argument("repo", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    build(a.pilot, a.repo, a.dest)
