"""A repeated API page must not duplicate benchmark truth."""

import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_disprot", Path(__file__).parents[1] / "fetch_disprot.py"
)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def test_distinct_ids_are_accepted():
    f.check_entry_ids([{"disprot_id": "DP1"}, {"disprot_id": "DP2"}])


@pytest.mark.parametrize("entries", [[{}], [{"disprot_id": "DP1"}] * 2])
def test_missing_and_repeated_ids_are_rejected(entries):
    with pytest.raises(ValueError):
        f.check_entry_ids(entries)
