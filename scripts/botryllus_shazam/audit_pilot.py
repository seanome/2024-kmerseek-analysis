"""Independently check supplied pilot records against original sequences."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import sourmash


def rows(path):
    with path.open() as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def main(pilot, proteome, output):
    result = pilot / "results"
    declared = json.loads((result / "final_summary.json").read_text())
    provenance = json.loads((pilot / "provenance/provenance.json").read_text())
    assert (
        hashlib.file_digest(proteome.open("rb"), "sha256").hexdigest()
        == provenance["uploaded_proteome"]["sha256"]
    )
    refs = {f"BSP{i:06d}": r for i, r in enumerate(SeqIO.parse(proteome, "fasta"), 1)}
    ids = set()
    for r in rows(result / "botryllus_proteome_annotations.tsv"):
        assert r["record_id"] not in ids
        ids.add(r["record_id"])
        original = refs[r["record_id"]]
        assert r["original_id"] == original.id
        assert (
            r["sequence_sha256"]
            == hashlib.sha256(str(original.seq).encode()).hexdigest()
        )
        assert int(r["length_aa"]) == len(original.seq)
    assert ids == set(refs)
    preserved = list(
        SeqIO.parse(result / "botryllus_candidate_annotated.fasta", "fasta")
    )
    assert len(preserved) == len(refs)
    assert len({r.id for r in preserved}) == len(refs)
    assert all(str(r.seq) == str(refs[r.id].seq) for r in preserved)
    selected = list(rows(result / "selected_anchors.tsv"))
    significant = []
    tested = 0
    for row in rows(result / "pilot.after_correction.all_anchors.tsv"):
        tested += 1
    for row in rows(result / "pilot.after_correction.scores.tsv"):
        q = float(row["pval_opt_corrected"])
        assert math.isfinite(q) and 0 <= q <= 0.05
        significant.append(row)
    expected_selection = sorted(
        significant, key=lambda r: (float(r["pval_opt_corrected"]), r["anchor"])
    )[:300]
    assert [r["anchor"] for r in selected] == [r["anchor"] for r in expected_selection]
    extensions = list(rows(result / "compactors.tsv"))
    selected_ids = {r["anchor"] for r in selected}
    by_id = {r["id"]: r for r in extensions}
    assert len(by_id) == len(extensions)
    for r in extensions:
        assert r["anchor"] in selected_ids and r["anchor"] in r["compactor"]
        assert int(r["total_length"]) == len(r["compactor"])
        if r["parent_id"] != "-1":
            assert r["parent_id"] in by_id
            assert by_id[r["parent_id"]]["anchor"] == r["anchor"]
    parents = {r["parent_id"] for r in extensions}
    terminal = [r for r in extensions if r["id"] not in parents]
    peptides = {
        r.id: str(r.seq) for r in SeqIO.parse(result / "peptides.fasta", "fasta")
    }
    dna = {r.id: str(r.seq) for r in SeqIO.parse(result / "extensions.fasta", "fasta")}
    occurrences = 0
    for r in rows(result / "peptide_membership.tsv"):
        segment = Seq(dna[r["extension_id"]][int(r["nt_start"]) : int(r["nt_end"])])
        if r["strand"] == "-":
            segment = segment.reverse_complement()
        assert str(segment.translate()) == peptides[r["peptide_id"]]
        occurrences += 1
    checked = 0
    for r in rows(result / "peptide_search_hits.tsv"):
        if r["query_type"] != "real" or r["database"] != "botryllus":
            continue
        opts = {"is_protein": True} if r["mode"] == "protein" else {r["mode"]: True}
        k = {"protein": 7, "dayhoff": 12, "hp": 30}[r["mode"]]

        def sketch(s):
            h = sourmash.MinHash(n=0, ksize=k, scaled=int(r["scaled"]), seed=42, **opts)
            # Fragment and proteome sequences were checked before sketch comparison.
            import re

            for part in re.findall("[ACDEFGHIKLMNPQRSTVWY]+", s.upper()):
                h.add_protein(part)
            return set(h.hashes)

        q = sketch(peptides[r["peptide_id"]])
        t = sketch(str(refs[r["target_id"]].seq))
        assert len(q & t) == int(r["shared_hashes"])
        assert abs(len(q & t) / len(q) - float(r["containment"])) < 1e-9
        checked += 1
    qc = list(rows(result / "read_qc.tsv"))
    observed = dict(
        anchors_tested=tested,
        significant_anchors=len(significant),
        selected_for_extension=len(selected),
        libraries=len(qc),
        input_pairs=sum(int(r["input_pairs"]) for r in qc),
        retained_pairs=sum(int(r["retained_pairs"]) for r in qc),
        extension_records=len(extensions),
        selected_anchors_with_extensions=len({r["anchor"] for r in extensions}),
        terminal_extensions=len(terminal),
        unique_peptides=len(peptides),
    )
    for key, value in observed.items():
        assert value == declared[key], (key, value, declared[key])
    report = dict(
        observed=observed,
        proteome_records=len(refs),
        translation_occurrences=occurrences,
        recomputed_real_botryllus_hits=checked,
        limits=[
            "Raw pilot reads absent: trimming counts reconciled to supplied logs only.",
            "Original discovery and protein search not rerun.",
            "No treatment-associated abundance inference.",
        ],
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pilot", type=Path)
    p.add_argument("proteome", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.pilot, a.proteome, a.output)
