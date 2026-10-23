"""Residues replaced by X must not create evidence of sequence identity."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "construct_pair_strata.py"
spec = importlib.util.spec_from_file_location("pair_strata_identity", path)
strata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strata)


def test_different_noncanonical_residues_do_not_become_identical():
    result = strata.nw_identity(("human", "target", "AAAAU", "AAAAO"))
    assert result[2] == 4 / 5


def test_known_identical_and_missing_sequences():
    assert strata.nw_identity(("human", "target", "AAAA", "AAAA"))[2] == 1
    assert strata.nw_identity(("human", "target", "", "AAAA"))[2] is None
