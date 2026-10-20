"""Checks on the Step 0 code of notebook 300."""

import gzip
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))

from alphabets import CLUSTERS, class_of  # noqa: E402
from evalue_floor import karlin_altschul_lambda  # noqa: E402

KMERSEEK = Path(os.environ.get("KMERSEEK_DIR", Path.home() / "code" / "kmerseek"))


def test_massspec18_merges_only_il_and_qk():
    classes = class_of("massspec18")
    assert len(CLUSTERS["massspec18"]) == 18
    assert classes["I"] == classes["L"]
    assert classes["Q"] == classes["K"]
    others = [aa for aa in classes if aa not in "ILQK"]
    assert len({classes[aa] for aa in others}) == 16


@pytest.mark.skipif(not (KMERSEEK / "src/rust/alphabets.rs").exists(), reason="no kmerseek")
def test_cluster_lists_match_kmerseek_source():
    """Every multi-letter cluster list equals the one in kmerseek's alphabets.rs."""
    source = (KMERSEEK / "src/rust/alphabets.rs").read_text()
    for const, body in re.findall(r"const (\w+)_CLUSTERS: &\[&str\] =\s*&\[(.*?)\];", source,
                                  flags=re.S):
        name = const.lower()
        if name == "funcgroups8" or name in CLUSTERS:
            assert re.findall(r'"([A-Z]+)"', body) == CLUSTERS[name], name
    for name, hydrophobic, polar in re.findall(
            r'static (\w+)_HP: LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| build_hp\('
            r'b"([A-Z]+)", b"([A-Z]+)"\)\);', source):
        matches = [a for a, c in CLUSTERS.items() if c == [hydrophobic, polar]]
        assert matches, f"{name}_HP {hydrophobic}/{polar} has no copy in alphabets.py"


def test_lambda_matches_closed_form_and_kmerseek_log():
    # u = 1/4, penalty 1: 0.25 e^x + 0.75 e^-x = 1 has root ln 3.
    assert karlin_altschul_lambda(np.array([0.25]), 1.0)[0] == pytest.approx(math.log(3))
    # kmerseek printed "closed form 0.339 at the database's match probability 0.513" for
    # hp_thomas_dill2 at penalty 1.63 (dark-set calibrate log, 2026-09-21).
    assert karlin_altschul_lambda(np.array([0.513]), 1.63)[0] == pytest.approx(0.339, abs=2e-3)
    # No positive root once u >= C / (1 + C).
    assert karlin_altschul_lambda(np.array([0.5, 0.9]), 1.0).tolist() == [0.0, 0.0]


def test_db_composition_counts(tmp_path):
    fasta = tmp_path / "t.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(">a x\nACDE\nFGX\n>b\nAAA\n")
    out = tmp_path / "t.json"
    subprocess.run([sys.executable, str(BIN / "db_composition.py"), str(fasta), "t", str(out)],
                   check=True)
    result = json.loads(out.read_text())
    assert result["n_sequences"] == 2
    assert result["n_residues"] == 10
    assert result["n_canonical"] == 9
    assert result["letters"]["A"] == 4
