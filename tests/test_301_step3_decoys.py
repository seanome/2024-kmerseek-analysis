"""Tests for scripts/build_301_step3_decoys.py."""

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_301_step3_decoys", ROOT / "scripts" / "build_301_step3_decoys.py"
)
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)

# Insulin's signal peptide, P01308 residues 1-24.
INSULIN_SIGNAL = "MALWMRLLPLLALLALWGPDPAAA"


def test_shuffle_moves_pairs_and_keeps_them_whole():
    rng = random.Random(1)
    new = d.decoy_window(INSULIN_SIGNAL, "dishuffle", rng)
    assert new != INSULIN_SIGNAL and sorted(new) == sorted(INSULIN_SIGNAL)
    pairs = sorted(INSULIN_SIGNAL[i : i + 2] for i in range(0, 24, 2))
    assert sorted(new[i : i + 2] for i in range(0, 24, 2)) == pairs


def test_odd_last_residue_stays_last():
    rng = random.Random(2)
    assert d.di_residue_shuffle("ABCDEFG", rng)[-1] == "G"


def test_reversed_and_unchangeable_windows():
    rng = random.Random(3)
    assert d.decoy_window("KDEL", "reversed", rng) == "LEDK"
    # one pair plus a tail: nothing can move
    assert d.decoy_window("RRA", "dishuffle", rng) == "RRA"
    assert d.decoy_window("ABA", "reversed", rng) == "ABA"


def test_main_writes_decoys_spliced_in_place(tmp_path):
    q = tmp_path / "q.fasta"
    host = "MKT" + INSULIN_SIGNAL + "GGGK"
    q.write_text(f">P1\n{host}\n")
    t = tmp_path / "truth.tsv"
    t.write_text(
        "query_id\tfeature_type\tstart\tend\tresidues\n"
        f"P1\tSIGNAL\t4\t27\t{INSULIN_SIGNAL}\n"
    )
    sys.argv = ["x", "--queries", str(q), "--truth", str(t), "--out", str(tmp_path)]
    d.main()
    seqs = d.read_fasta(tmp_path / "decoys.fasta")
    assert set(seqs) == {"decoy_00000_dishuffle", "decoy_00000_reversed"}
    for s in seqs.values():
        assert s[:3] == "MKT" and s[27:] == "GGGK" and len(s) == len(host)
    assert seqs["decoy_00000_reversed"][3:27] == INSULIN_SIGNAL[::-1]
