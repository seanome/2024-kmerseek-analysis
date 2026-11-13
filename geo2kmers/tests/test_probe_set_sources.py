"""Sorting probe sets by sequence source, on GPL rows shaped like U133 Plus 2 and HTA 2.0."""

import csv
import subprocess
import sys
from pathlib import Path

FIG4 = Path(__file__).parents[1] / "fig4"


def sources(tmp_path, header, rows):
    table = tmp_path / "gpl.txt"
    table.write_text("!Platform_title = x\n" + "\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows))
    subprocess.run(
        [sys.executable, str(FIG4 / "probe_set_sources.py"), str(table), str(tmp_path / "out.tsv")], check=True
    )
    with open(tmp_path / "out.tsv") as f:
        return {r["probe_set"]: (r["source"], r["accessions"]) for r in csv.DictReader(f, delimiter="\t")}


def test_u133_rules_in_order(tmp_path):
    header = ["ID", "GB_ACC", "Sequence Type", "RefSeq Transcript ID"]
    got = sources(
        tmp_path,
        header,
        [
            ["a_at", "U48705", "Exemplar sequence", "NM_1 /// NR_2 /// XM_3"],
            ["b_at", "AI1", "Consensus sequence", "XM_4"],
            ["c_at", "AI2", "Consensus sequence", ""],
            ["d_at", "M8", "Exemplar sequence", ""],
            ["e_at", "", "", "NR_5"],
        ],
    )
    assert got == {
        "a_at": ("refseq_mrna", "NM_1"),
        "b_at": ("refseq_model", "XM_4"),
        "c_at": ("est", "AI2"),
        "d_at": ("genbank_mrna", "M8"),
        "e_at": ("other_only", "NR_5"),
    }


def test_hta_prefers_refseq_then_ensembl_coding(tmp_path):
    coding = "ENST1 // ENSEMBL // cdna gene_biotype:protein_coding transcript_biotype:protein_coding // chr1"
    lnc = "ENST2 // ENSEMBL // havana:lincRNA transcript_biotype:lincRNA // chr1"
    got = sources(
        tmp_path,
        ["ID", "mrna_assignment"],
        [
            ["TC1", f"{coding} /// NM_9 // RefSeq // x // chr1"],
            ["TC2", f"{lnc} /// {coding}"],
            ["TC3", f"{lnc} /// BC1 // GenBank // x // chr1"],
            ["TC4", lnc],
            ["TC5", "---"],
        ],
    )
    assert got == {
        "TC1": ("refseq_mrna", "NM_9"),
        "TC2": ("ensembl_coding", "ENST1"),
        "TC3": ("genbank_mrna", "BC1"),
        "TC4": ("other_only", "ENST2"),
        "TC5": ("none", ""),
    }


def test_coverage_counts_cdna_proteins_and_genomic_clones_apart(tmp_path):
    (tmp_path / "s.tsv").write_text(
        "probe_set\tsource\taccessions\tcount\n"
        "a\trefseq_mrna\tNM_1,NM_2\t2\nb\test\tBC1\t1\nc\test\tAI1\t1\nd\test\tAC1\t1\ne\test\tAI2\t1\n"
    )
    (tmp_path / "f.tsv").write_text(
        "accession\tasked_for\tsequence\tlength\n"
        "NM_1\tprotein\tNP_1\t400\nNM_2\tprotein\tNP_2\t20\nBC1\tprotein\tAAH1\t293\nBC1\test\test\t1763\n"
        "AI1\test\test\t500\nAC1\test\test\t200000\nAI2\test\test\t60\n"
    )
    script = FIG4 / "protein_coverage.py"
    subprocess.run(
        [sys.executable, str(script), str(tmp_path / "f.tsv"), str(tmp_path / "c.tsv"), f"A={tmp_path / 's.tsv'}"],
        check=True,
    )
    with open(tmp_path / "c.tsv") as f:
        got = {r["source"]: (r["probe_sets"], r["long_enough"]) for r in csv.DictReader(f, delimiter="\t")}
    assert got == {
        "refseq_mrna": ("1", "1"),
        "cdna_with_protein": ("1", "1"),
        "est": ("2", "1"),
        "genomic_clone": ("1", "1"),
    }
    with open(tmp_path / "probe_set_proteins.tsv") as f:
        pairs = {(r["probe_set"], r["sequence"]) for r in csv.DictReader(f, delimiter="\t")}
    assert pairs == {("a", "NP_1"), ("b", "AAH1"), ("c", "AI1"), ("d", "AC1")}
