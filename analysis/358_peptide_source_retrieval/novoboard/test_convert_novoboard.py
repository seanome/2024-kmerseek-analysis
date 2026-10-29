"""Tests for convert_novoboard.py on small zips with NovoBoard's real column headers."""

import gzip
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import convert_novoboard as cn  # noqa: E402
import peptide_source as ps  # noqa: E402

CASANOVO_HEADER = ("sequence,PSM_ID,accession,unique,database,database_version,search_engine,"
                   "search_engine_score[1],modifications,retention_time,charge,exp_mass_to_charge,"
                   "calc_mass_to_charge,spectra_ref,pre,post,start,end,opt_ms_run[1]_aa_scores,"
                   "Source File,Scan,Peptide,Casanovo score,m/z,z,RT,ppm")
DB_HEADER = ('"Peptide","-10LgP","Mass","Length","ppm","m/z","z","RT","Scan","Area","Feature Id",'
             '"Source File","Accession","PTM","AScore"')
REAL = "Human_Benchmark_DDA_1hr.mgf"


def casanovo_row(seq, scan, scores, source=REAL, score="0.9"):
    return ('%s,1,null,null,null,null,"[MS, MS:1003281, Casanovo, 4.1.0]",%s,null,null,2.0,500.0,'
            '500.0,ms_run[1]:index=0,null,null,null,null,"%s",%s,%d,%s,%s,500.0,2.0,0,0.1'
            % (seq, score, ",".join(scores), source, scan, seq, score))


def db_row(pep, scan, lgp, source=REAL):
    return '%s,%s,1000.0,%d,0.1,500.0,2,10.0,%d,,0,"%s","P1|X_HUMAN","",' % (pep, lgp, len(pep), scan, source)


def s(n):
    return ["0.9"] * n


@pytest.fixture
def zip_path(tmp_path):
    cas = [CASANOVO_HEADER,
           casanovo_row("PEPTIDEKR", 1, s(9)),                       # kept
           casanovo_row("+43.006PEPTM+15.995DEK", 2, s(9)),           # N-term tag: 8 residues + 1
           casanovo_row("PEPTIDEKR", 3, s(5)),                        # score count wrong
           casanovo_row("PEPTIDEKR", 4, s(9), source=REAL + ".decoy_0.10.mgf"),
           casanovo_row("PEPTK", 5, s(5)),                            # too short
           casanovo_row("AAAAPEPTK", 6, s(9)),                        # no database PSM
           casanovo_row("AAAAPEPTK", 7, s(9)), casanovo_row("CCCCPEPTK", 7, s(9)),  # two calls
           casanovo_row("LLLLPEPTK", 8, s(9)),                        # database tie
           casanovo_row("VVVVC+57.021PEK", 9, s(8))]                  # kept, C carbamidomethyl
    db = [DB_HEADER,
          db_row("PEPTLDEKR", 1, "50.0"), db_row("PEPTLDEKK", 1, "40.0"),
          db_row("PEPTM(+15.99)DEK", 2, "60.0"),
          db_row("PEPTIDEKR", 3, "60.0"), db_row("PEPTIDEKR", 4, "60.0"),
          db_row("PEPTK", 5, "60.0"), db_row("AAAAPEPTK", 7, "60.0"),
          db_row("LLLLPEPTK", 8, "60.0"), db_row("IIIIPEPTK", 8, "60.0"),
          db_row("VVVVC(+57.02)PEK", 9, "70.0"),
          db_row("VVVVCPEK", 9, "70.0", source=REAL + ".decoy_0.10.mgf")]
    path = tmp_path / "eval_human.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(cn.CASANOVO, "\n".join(cas) + "\n")
        z.writestr(cn.DB_PSMS, "\n".join(db) + "\n")
        z.writestr("eval_human/" + REAL, "BEGIN IONS\nEND IONS\nBEGIN IONS\nEND IONS\n")
        z.writestr("eval_human/decoy_spectra_X/x.mgf", "BEGIN IONS\nEND IONS\n")
    return path


def test_tokens():
    assert cn.casanovo_tokens("+43.006-17.027LM+15.995K") == [None, "L", "M", "K"]
    assert cn.casanovo_tokens("PEPTIDE") == list("PEPTIDE")
    with pytest.raises(ValueError):
        cn.casanovo_tokens("PEP(ox)K")


def test_bare_peptide():
    assert cn.bare_peptide("AC(+57.02)M(Oxidation)K") == "ACMK"
    assert cn.bare_peptide("K.PEPTIDE.R") == "PEPTIDE"


def test_convert_applies_every_rule(zip_path, tmp_path):
    out = tmp_path / "out"
    rep = cn.convert(zip_path, out)
    calls = (out / "calls.tsv").read_text().splitlines()
    acc = (out / "accepted.tsv").read_text().splitlines()
    assert [line.split("\t")[0] for line in calls[1:]] == ["scan1", "scan2", "scan9"]
    by_id = {line.split("\t")[0]: line.split("\t") for line in calls[1:]}
    assert by_id["scan2"][3] == "PEPTMDEK" and len(by_id["scan2"][5].split(",")) == 8
    assert by_id["scan9"][3] == "VVVVCPEK"
    accepted = {line.split("\t")[0]: line.split("\t")[1] for line in acc[1:]}
    assert accepted == {"scan1": "PEPTLDEKR", "scan2": "PEPTMDEK", "scan9": "VVVVCPEK"}
    assert rep["casanovo_score_count_mismatch"] == 1
    assert rep["casanovo_rows_other_source_file"] == 1
    assert rep["casanovo_length_outside_7_30"] == 1
    assert rep["casanovo_no_db_psm"] == 2  # scan 6, and scan 8 after its tie is dropped
    assert rep["casanovo_duplicate_scan_rows"] == 2  # both rows of the repeated scan
    assert rep["db_scans_dropped_tied_psms"] == 1
    assert rep["db_rows_other_source_file"] == 1
    assert rep["chosen_call_equals_accepted_IL"] == 3


def test_inspect_counts_real_scans_and_spectra(zip_path):
    info = cn.inspect(zip_path)
    members = {m["name"]: m for m in info["members"]}
    assert members["eval_human/" + REAL]["spectra"] == 2
    assert "spectra" not in members["eval_human/decoy_spectra_X/x.mgf"]
    assert members[cn.CASANOVO]["source_files"][REAL + ".decoy_0.10.mgf"] == 1
    assert info["casanovo_scans_with_db_psm"] == 7  # scans 1, 2, 3, 5, 7, 8, 9


def test_collection_and_validation(zip_path, tmp_path):
    fasta = (">sp|P11111|A_HUMAN a OS=Homo sapiens OX=9606 GN=A\nMMPEPTLDEKRAAPEPTMDEKRGGVVVVCPEKW\n"
             ">sp|P22222|B_MOUSE b OS=Mus musculus OX=10090 GN=B\nMMMMMM\n")
    buf = io.BytesIO(gzip.compress(fasta.encode()))
    tar_path = tmp_path / "sprot.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("uniprot_sprot.fasta.gz")
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    crap = tmp_path / "crap.fasta"
    crap.write_text(">sp|P11111|A_HUMAN\nMMPEPTLDEKR\n>sp|P00761|TRYP_PIG\nIVGGYTCAANSIPYQVSLNSGSHFCGGSLINSQWVVSAAHCYK\n")
    out = tmp_path / "out"
    rep = cn.collection(tar_path, crap, out)
    assert rep == {"human_entries": 1, "crap_entries_kept": 1,
                   "crap_entries_dropped_already_human": ["P11111"]}
    cn.convert(zip_path, out)
    inp = ps.load_inputs(out / "inputs.json")
    assert inp.data_origin == "experimental" and len(inp.proteins) == 2
    srcs = ps.valid_sources(list(inp.accepted["accepted_peptide"]), [(a, q) for a, q, _ in inp.proteins])
    assert srcs == [["P11111"]] * 3
