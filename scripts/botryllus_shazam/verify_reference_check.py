"""Verify direct reference searches while retaining annotation-coordinate mismatches."""

import json
from pathlib import Path
import sys
from nextflow_stages import sha, fasta

root = Path(sys.argv[1])
result = {}
for source in ["positive", "empty"]:
    alignment = root / "results/sequence-evidence" / source / "alignment"
    pfam = root / "results/query-pfam" / source / "chunk-00000/pfam"
    a = json.loads((alignment / "alignment.json").read_text())
    p = json.loads((pfam / "pfam.json").read_text())
    expected_a = "completed" if source == "positive" else "no queries"
    expected_p = "completed" if source == "positive" else "no eligible peptides"
    assert a["status"] == expected_a and p["status"] == expected_p
    assert a["source"] == source and p["source"] == source
    assert sha(alignment / "context.paf") == a["output_sha256"]
    mapped = {line.split("\t")[0] for line in (alignment / "context.paf").open()}
    assert mapped <= set(fasta(root / source / "dna.fasta"))
    hits = []
    if source == "positive":
        assert sha(pfam / "domains.domtblout") == p["output_sha256"]
        hits = [
            line.split()
            for line in (pfam / "domains.domtblout").open()
            if not line.startswith("#") and line.strip()
        ]
        assert hits and mapped
        assert {r[3] for r in hits} <= set(fasta(root / source / "peptides.faa"))
    result[source] = dict(
        alignment=a, pfam=p, mapped_queries=len(mapped), pfam_domain_rows=len(hits)
    )
coords = json.loads((root / "coordinates/coordinate-check.json").read_text())
result.update(
    status="direct sequence searches passed",
    coordinate_audit=coords,
    annotation_coordinate_status="Unresolved; do not use this GTF for genome-coordinate claims or exact STAR/HTSeq reproduction",
    safe_scope="The production workflow uses GTF gene/transcript identifiers only; genome alignments remain separate sequence evidence",
)
(root / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k not in ["coordinate_audit", "positive", "empty"]
        }
    )
)
print(
    json.dumps(
        {
            k: {x: result[k][x] for x in ["mapped_queries", "pfam_domain_rows"]}
            for k in ["positive", "empty"]
        }
    )
)
