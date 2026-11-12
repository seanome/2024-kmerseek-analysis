"""Build the shared pilot peptide evidence from cached official annotations."""

import csv, gzip, json, hashlib, re, sys
from pathlib import Path
from datetime import datetime, timezone

p = Path(sys.argv[1]) / "match-context"
a = json.load(gzip.open(p / "swissprot-context.json.gz", "rt"))
refs = {r["record_id"]: r for r in a["references"]}
seqfile = Path(sys.argv[2])
seq = seqfile.read_text().split(">PEP000938\n")[1].split(">")[0].replace("\n", "")
mapping = {}
for line in (p / "pfam2go.txt").read_text().splitlines():
    m = re.match(r"Pfam:(PF\d+) (.*?) > GO:(.*?) ; (GO:\d+)", line)
    if m:
        mapping.setdefault(m[1], []).append(dict(id=m[4], term=m[3]))
rows = []
for r in a["matches"]:
    if r[0] != "PEP000938":
        continue
    ref = refs[r[1]]
    raw = json.loads((p / "uniprot" / f"{ref['accession']}.json").read_text())
    pf = []
    go = []
    ipr = []
    for x in raw["uniProtKBCrossReferences"]:
        props = {v["key"]: v["value"] for v in x.get("properties", [])}
        if x["database"] == "Pfam":
            q = p / "pfam" / f"{x['id']}.json"
            name = props.get("EntryName", "not reported")
            if q.exists():
                md = json.loads(q.read_text())["metadata"]
                n = md.get("name", {})
                name = n.get("name", n.get("short", name)) if isinstance(n, dict) else n
            pf.append(
                dict(
                    id=x["id"],
                    name=name,
                    short_name=props.get("EntryName"),
                    pfam2go=mapping.get(x["id"], []),
                )
            )
        if x["database"] == "GO":
            go.append(
                dict(
                    id=x["id"],
                    term=props.get("GoTerm", "not reported"),
                    evidence=props.get("GoEvidenceType", "not reported"),
                )
            )
        if x["database"] == "InterPro":
            ipr.append(dict(id=x["id"], name=props.get("EntryName", "not reported")))
    rows.append(
        dict(
            **ref,
            alphabet=r[2],
            shared_hashes=r[3],
            query_hashes=r[4],
            containment=r[5],
            pfam=pf,
            go=go,
            interpro=ipr,
            annotation_source=f"https://rest.uniprot.org/uniprotkb/{ref['accession']}.json",
        )
    )
out = dict(
    query="PEP000938",
    sequence=seq,
    length=len(seq),
    source_extensions=a["peptide_extensions"]["PEP000938"],
    scope="Historical top-300 pilot; retained candidates at scaled 5",
    matches=rows,
    annotation_scope="Domains and GO annotations belong to the complete reference protein. No peptide-to-domain localization or direct Pfam hit on PEP000938 has been validated.",
    retrieved_at=datetime.now(timezone.utc).isoformat(),
    sources={},
    pfam2go_source="https://current.geneontology.org/ontology/external2go/pfam2go",
    pfam2go_sha256=hashlib.sha256((p / "pfam2go.txt").read_bytes()).hexdigest(),
    peptide_fasta_sha256=hashlib.sha256(seqfile.read_bytes()).hexdigest(),
)
out["sources"] = {
    str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
    for folder in ["uniprot", "pfam"]
    for f in (p / folder).glob("*.json")
}
(p / "PEP000938.json").write_text(json.dumps(out, indent=2) + "\n")
print(seq, len(seq))
print(
    [
        (r["alphabet"], r["accession"], [(x["id"], x["name"]) for x in r["pfam"]])
        for r in rows
    ]
)
