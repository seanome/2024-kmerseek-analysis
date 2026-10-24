"""Join candidate matches to original protein labels and archive sample attributes."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET


def dump(path, value):
    text = json.dumps(value, separators=(",", ":")) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def overlap(sets, scope):
    result = []
    modes = list(sets)
    for i, left in enumerate(modes):
        for right in modes[i + 1 :]:
            a, b = sets[left], sets[right]
            result.append(
                dict(
                    scope=scope,
                    left=left,
                    right=right,
                    left_count=len(a),
                    right_count=len(b),
                    shared=len(a & b),
                    shared_members=sorted(a & b),
                    left_only=len(a - b),
                    right_only=len(b - a),
                )
            )
    return result


def protein_label(row):
    desc = row["original_description"]
    first, _, rest = desc.partition(" ")
    fields = dict(re.findall(r"(OS|OX|GN|PE|SV)=(.*?)(?= (?:OS|OX|GN|PE|SV)=|$)", rest))
    accession = first.split("|")[1]
    return dict(
        record_id=row["record_id"],
        accession=accession,
        protein=rest.split(" OS=")[0],
        gene=fields.get("GN", "not reported"),
        species=fields.get("OS", "not reported"),
        taxonomy_id=fields.get("OX", "not reported"),
        description=desc,
        source_url="https://www.uniprot.org/uniprotkb/" + accession + "/entry",
    )


def build(report, membership):
    dest = report / "match-context"
    dest.mkdir(exist_ok=True)
    parents = {}
    for row in csv.DictReader(membership.open(), delimiter="\t"):
        parents.setdefault(row["peptide_id"], set()).add(row["extension_id"])
    references = [
        protein_label(r)
        for r in csv.DictReader(
            gzip.open(
                report / "finished-manysearch/matched-swissprot-record-map.tsv.gz", "rt"
            ),
            delimiter="\t",
        )
    ]
    ref = {r["record_id"]: r for r in references}
    matches = []
    pairsets, querysets, targetsets = {}, {}, {}
    enriched = dest / "swissprot-candidates-with-labels.csv.gz"
    fields = [
        "query",
        "alphabet",
        "reference",
        "accession",
        "protein",
        "gene",
        "species",
        "taxonomy_id",
        "shared_hashes",
        "query_hashes",
        "containment",
        "source_extensions",
        "source_url",
    ]
    with gzip.open(enriched, "wt") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        for mode in ("dayhoff", "hp", "protein"):
            pairsets[mode], querysets[mode], targetsets[mode] = set(), set(), set()
            for row in csv.DictReader(
                gzip.open(
                    report
                    / f"finished-manysearch/swissprot.{mode}.scaled5.retained.csv.gz",
                    "rt",
                )
            ):
                q, t = row["query_name"], row["match_name"]
                if q not in parents or t not in ref:
                    raise ValueError("Unmapped query or reference")
                pairsets[mode].add((q, t))
                querysets[mode].add(q)
                targetsets[mode].add(t)
                matches.append(
                    [
                        q,
                        t,
                        mode,
                        int(row["intersect_hashes"]),
                        int(row["query_hashes"]),
                        float(row["containment"]),
                    ]
                )
                meta = ref[t]
                writer.writerow(
                    dict(
                        query=q,
                        alphabet=mode,
                        reference=t,
                        **{
                            k: meta[k]
                            for k in (
                                "accession",
                                "protein",
                                "gene",
                                "species",
                                "taxonomy_id",
                                "source_url",
                            )
                        },
                        shared_hashes=row["intersect_hashes"],
                        query_hashes=row["query_hashes"],
                        containment=row["containment"],
                        source_extensions=";".join(sorted(parents[q])),
                    )
                )
    summary = dict(
        scope="Historical top-300 pilot queries only; full-depth results pending",
        swissprot_records=len(ref),
        swissprot_rows=len(matches),
        swissprot_overlap=overlap(pairsets, "same peptide and protein pair; scaled 5")
        + overlap(querysets, "same peptide query; scaled 5")
        + overlap(targetsets, "same reference protein; scaled 5"),
        swissprot_alphabets=[],
    )
    for mode in pairsets:
        labels = [ref[t] for t in targetsets[mode]]
        summary["swissprot_alphabets"].append(
            dict(
                alphabet=mode,
                queries=len(querysets[mode]),
                proteins=len(labels),
                organism_labels=len(set(x["species"] for x in labels)),
                top_species=Counter(x["species"] for x in labels).most_common(12),
                examples=sorted(
                    labels,
                    key=lambda x: (
                        x["gene"] == "not reported",
                        x["gene"],
                        x["accession"],
                    ),
                )[:12],
            )
        )
    dump(
        dest / "swissprot-context.json.gz",
        dict(
            references=references,
            matches=matches,
            columns=[
                "query",
                "reference",
                "alphabet",
                "shared_hashes",
                "query_hashes",
                "containment",
            ],
            peptide_extensions={k: sorted(v) for k, v in parents.items()},
        ),
    )
    progress = json.loads((report / "external-search-progress.json").read_text())
    manifest = {
        x["run_accession"]: x
        for x in csv.DictReader(
            (report / "external-tunicate-pilot.tsv").open(), delimiter="\t"
        )
    }
    sets = {
        scale: {mode: set() for mode in ("dna", "protein", "dayhoff", "hp")}
        for scale in (10, 100)
    }
    sra_matches = []
    for result in progress["results"]:
        for row in csv.DictReader((report / result["local_retained_csv"]).open()):
            q = row["query_name"]
            run = result["run"]
            mode = result["mode"]
            scale = result["scaled"]
            ext = {q.split()[0]} if mode == "dna" else parents[q]
            if mode == "dna" and q.split()[0] not in set().union(*parents.values()):
                raise ValueError("Unmapped DNA query ID")
            sets[scale][mode].update((x, run) for x in ext)
            sra_matches.append(
                dict(
                    query=q,
                    run=run,
                    alphabet=mode,
                    scaled=scale,
                    source_extensions=sorted(ext),
                    shared_hashes=int(row["intersect_hashes"]),
                    containment=float(row["containment"]),
                )
            )
    summaries = []
    source_dir = dest / "sample-xml"
    source_dir.mkdir(exist_ok=True)
    for run, row in manifest.items():
        sample = row["sample_accession"]
        url = "https://www.ebi.ac.uk/ena/browser/api/xml/" + sample
        path = source_dir / (sample + ".xml")
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            ET.fromstring(data)
            path.write_bytes(data)
        tree = ET.fromstring(path.read_bytes())
        attrs = [
            dict(
                tag=x.findtext("TAG"),
                value=x.findtext("VALUE"),
                units=x.findtext("UNITS"),
            )
            for x in tree.findall(".//SAMPLE_ATTRIBUTE")
        ]
        counts = []
        for result in progress["results"]:
            if result["run"] == run:
                counts.append(
                    dict(
                        alphabet=result["mode"],
                        scaled=result["scaled"],
                        candidate_rows=result["retained_match_rows"],
                    )
                )
        summaries.append(
            dict(
                run=run,
                study=row["study_accession"],
                sample=sample,
                species=row["scientific_name"],
                study_title=row["study_title"],
                sample_title=row["sample_title"],
                sample_attributes=attrs,
                attributes_source=url,
                attributes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                study_url="https://www.ncbi.nlm.nih.gov/bioproject/"
                + row["study_accession"],
                run_url="https://www.ebi.ac.uk/ena/browser/view/" + run,
                counts=counts,
            )
        )
    input_paths = [
        membership,
        report / "finished-manysearch/matched-swissprot-record-map.tsv.gz",
        *[
            report / f"finished-manysearch/swissprot.{mode}.scaled5.retained.csv.gz"
            for mode in ("protein", "dayhoff", "hp")
        ],
        *[report / r["local_retained_csv"] for r in progress["results"]],
    ]
    fingerprints = {}
    for path in input_paths:
        with path.open("rb") as stream:
            fingerprints[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    summary.update(
        input_sha256=fingerprints,
        sra_runs=summaries,
        sra_overlap=sum(
            [
                overlap(sets[s], f"same source DNA extension and SRA run; scaled {s}")
                for s in sets
            ],
            [],
        ),
        recorded_at=datetime.now(timezone.utc).isoformat(),
        interpretation="Candidate sequence evidence only; reference gene labels are not established query functions. Library occurrence is not condition-specific expression. Swiss-Prot is a protein reference, not a condition-labeled RNA-seq experiment.",
    )
    dump(dest / "summary.json", summary)
    dump(
        dest / "sra-context.json.gz",
        dict(runs=summaries, matches=sra_matches, scope=summary["scope"]),
    )
    print(
        json.dumps(
            dict(
                swissprot_rows=len(matches),
                sra_rows=len(sra_matches),
                swissprot_overlap=summary["swissprot_overlap"],
                sra_overlap=summary["sra_overlap"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("membership", type=Path)
    a = p.parse_args()
    build(a.report, a.membership)
