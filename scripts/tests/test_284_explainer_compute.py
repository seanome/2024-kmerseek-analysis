"""Run the explainer page's sums against notebook 284's CSVs (see 284_explainer_compute_check.mjs)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None or not (ROOT / "notebooks/284_explainer_files").exists(),
                    reason="needs node and the tables from scripts/284_explainer_data.py")
def test_explainer_sums_match_csvs():
    r = subprocess.run(["node", str(ROOT / "scripts/tests/284_explainer_compute_check.mjs")],
                       capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
