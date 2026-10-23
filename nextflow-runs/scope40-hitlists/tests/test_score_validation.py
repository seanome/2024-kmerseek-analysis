import importlib.util
from pathlib import Path
import polars as pl
import pytest

spec = importlib.util.spec_from_file_location("score_hitlist", Path(__file__).parents[1] / "bin" / "score_hitlist.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

@pytest.mark.parametrize("bad", [float("nan"), None])
@pytest.mark.parametrize("higher", [True, False])
def test_missing_score_cannot_earn_recovery(bad, higher):
    dom = pl.DataFrame({"domain_id": ["q", "tp", "fp"], "family": ["a.1.1.1", "a.1.1.1", "b.1.1.1"], "superfamily": ["a.1.1", "a.1.1", "b.1.1"], "fold": ["a.1", "a.1", "b.1"]})
    hits = pl.DataFrame({"query": ["q", "q"], "target": ["tp", "fp"], "score": [bad, 10.]})
    with pytest.raises(ValueError, match="null or NaN"):
        s.label_hits(hits.lazy(), dom, higher)
