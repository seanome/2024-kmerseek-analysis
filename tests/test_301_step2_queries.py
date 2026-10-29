"""Tests for scripts/build_301_step2_queries.py on four real Swiss-Prot entries.

tests/data/301/sprot_sample.dat holds the UniProt flat-file text of P01308 (insulin,
human), P02768 (albumin, human), P0AEX9 (MalE, E. coli) and P0A910 (OmpA, E. coli),
fetched from rest.uniprot.org on 2026-10-20.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_301_step2_queries", ROOT / "scripts" / "build_301_step2_queries.py"
)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b  # dataclasses look the module up by name
spec.loader.exec_module(b)

SAMPLE = ROOT / "tests" / "data" / "301" / "sprot_sample.dat"


def entries():
    with open(SAMPLE) as fh:
        return {e.accession: e for e in b.parse_sprot_dat(fh)}


def test_parses_four_entries_with_groups():
    e = entries()
    assert sorted(e) == ["P01308", "P02768", "P0A910", "P0AEX9"]
    assert e["P01308"].group == "human" and e["P0AEX9"].group == "microbial"
    assert len(e["P01308"].sequence) == 110


def test_insulin_signal_peptide_is_experimental():
    f = b.sprot_features(entries()["P01308"])
    assert [(x.feature_type, x.start, x.end) for x in f] == [("SIGNAL", 1, 24)]


def test_albumin_signal_without_evidence_is_dropped():
    e = entries()["P02768"]
    assert "SIGNAL" in e.all_types
    assert all(x.feature_type != "SIGNAL" for x in b.sprot_features(e))


def test_wrapped_evidence_and_beta_strand_transmem_are_kept():
    f = b.sprot_features(entries()["P0A910"])
    types = [x.feature_type for x in f]
    assert "SIGNAL" in types and types.count("TRANSMEM") >= 4
    tm = [x for x in f if x.feature_type == "TRANSMEM"][0]
    assert (tm.start, tm.end) == (27, 37)


def test_elm_instance_kept_only_when_residues_match():
    e = entries()
    by_acc = {a: x for x in e.values() for a in x.accessions}
    ins = e["P01308"].sequence
    rows = [
        {"instance_id": "ELMI1", "elm_class": "DEG_test", "protein": "P01308",
         "start": "30", "end": "35", "feature_residues": ins[29:35]},
        {"instance_id": "ELMI2", "elm_class": "LIG_test", "protein": "P01308",
         "start": "30", "end": "35", "feature_residues": "AAAAAA"},
        {"instance_id": "ELMI3", "elm_class": "LIG_test", "protein": "Q99999",
         "start": "1", "end": "5", "feature_residues": "MKKKK"},
    ]  # fmt: skip
    counts = b.Counter()
    kept = b.elm_features(rows, by_acc, counts)
    assert [k.evidence for k in kept] == ["ELM ELMI1"]
    assert counts["ELM instance residues differ from Swiss-Prot sequence"] == 1
    assert counts["ELM instance not on a Swiss-Prot entry"] == 1


def test_splice_coordinates_point_at_the_insert():
    host = "M" + "A" * 99
    chimera, s, t = b.splice(host, 40, "KDEL")
    assert chimera[s - 1 : t] == "KDEL"
    assert chimera[40 : 40 + len(b.LINKER)] == b.LINKER
    assert chimera[:40] == host[:40] and chimera[t:] == host[40:]
    assert len(chimera) == len(host) + len(b.LINKER) + 4
