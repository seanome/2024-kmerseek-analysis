"""AlphaFold fragment offsets must come from the fragment suffix, not the accession.

TrEMBL accessions can start with F and a digit (F6SXM4). An unanchored -F<n> match read
AF-F6SXM4-F1.cif as fragment 6 and moved every Reseek position on it by 1000; on the ELM
cover run that was 67% of Ciona Reseek rows. Folddisco's converter had the same pattern.
"""

import subprocess
import sys
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
from folddisco_to_regions import accession_and_offset  # noqa: E402

CASES = [
    # leaf name, accession, offset
    ("AF-F6SXM4-F1.cif", "F6SXM4", 0),
    ("AF-P12345-F3.cif", "P12345", 400),
    ("AF-F6SXM4-F2.cif", "F6SXM4", 200),
    ("AF-F7B1X4-F1", "F7B1X4", 0),              # no extension
    ("AF-F8VPJ6-F1.cif.gz", "F8VPJ6", 0),
    ("AF-A0A087WUL8-F2.cif", "A0A087WUL8", 200),
    # Reseek's own labels: extension dropped, chain appended (seen in reseek -search output)
    ("AF-Q03001-F21_A", "Q03001", 4000),
    ("AF-F6SXM4-F1_A", "F6SXM4", 0),
    ("AF-F6SXM4-F2_A", "F6SXM4", 200),
    ("AF-M9PCH5-F1-model_v6_A", "M9PCH5", 0),
]


def reseek_normalize(rows: list[str]) -> list[list[str]]:
    proc = subprocess.run(
        ["awk", "-f", str(BIN / "normalize_reseek.awk")],
        input="\n".join(rows) + "\n", capture_output=True, text=True, check=True,
    )
    return [line.split("\t") for line in proc.stdout.splitlines()]


@pytest.mark.parametrize("leaf,acc,offset", CASES)
def test_reseek_target_offset(leaf, acc, offset):
    # query is a plain F1 model; only the target side carries the case under test
    row = f"/s/human/AF-Q00001-F1.cif\t/s/ciona/{leaf}\t10\t20\t1\t50\t0.7\t1e-5"
    (out,) = reseek_normalize([row])
    assert out[:6] == ["Q00001", acc, "10", "20", str(1 + offset), str(50 + offset)]


@pytest.mark.parametrize("leaf,acc,offset", CASES)
def test_reseek_query_offset(leaf, acc, offset):
    row = f"{leaf}\tAF-Q00001-F1.cif\t1\t50\t10\t20\t0.7\t1e-5"
    (out,) = reseek_normalize([row])
    assert out[:6] == [acc, "Q00001", str(1 + offset), str(50 + offset), "10", "20"]


@pytest.mark.parametrize("leaf,acc,offset", CASES)
def test_folddisco_offset(leaf, acc, offset):
    assert accession_and_offset(f"/s/ciona/{leaf}") == (acc, offset)
