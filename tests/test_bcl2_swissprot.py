import gzip
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bcl2", Path(__file__).parents[1] / "scripts/bcl2_family_from_swissprot.py"
)
bcl2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcl2)


def test_no_family_matches_writes_header(tmp_path):
    source = tmp_path / "input.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("ID   OTHER\nAC   A12345;\nSQ   SEQUENCE 2 AA;\n     AA\n//\n")
    output = tmp_path / "out.tsv"
    bcl2.main(source, output)
    assert output.read_text().splitlines() == [
        "accession\tentry_name\tspecies\ttaxid\tlineage\tpfam\tlength\tsequence"
    ]


def test_truncated_entry_is_rejected(tmp_path):
    source = tmp_path / "input.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(
            "ID   BCL2\nAC   A12345;\nDR   Pfam; PF00452; Bcl-2; 1.\nSQ   SEQUENCE 2 AA;\n     AA\n"
        )
    with pytest.raises(ValueError, match="Incomplete"):
        list(bcl2.parse_entries(source))


def test_missing_separator_between_entries_is_rejected(tmp_path):
    source = tmp_path / "input.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("ID   FIRST\nAC   A12345;\nID   SECOND\nAC   B12345;\n//\n")
    with pytest.raises(ValueError, match="Incomplete"):
        list(bcl2.parse_entries(source))
