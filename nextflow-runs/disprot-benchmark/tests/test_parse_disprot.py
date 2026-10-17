"""parse_disprot.py: pages count from 0, the total is payload["size"], and a disordered
region is a consensus "Structural state" region of type "D", not any annotation term."""
import importlib.util
import io
import json
import urllib.parse
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parent.parent / "bin"
spec = importlib.util.spec_from_file_location("parse_disprot", BIN / "parse_disprot.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


def entry(i, ss=None, regions=None):
    return {"disprot_id": f"DP{i:05d}", "acc": f"P{i:05d}", "ncbi_taxon_id": 9606,
            "length": 100, "genes": [{"name": {"value": f"G{i}"}}],
            "regions": regions or [],
            "disprot_consensus": {"full": [], "Structural state": ss or []}}


def serve(pages, size, monkeypatch):
    """Fake DisProt API: pages[n] is the data list for page=n; records requested pages."""
    asked = []

    def urlopen(url, timeout=None):
        page = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["page"][0])
        asked.append(page)
        data = pages[page] if page < len(pages) else []
        return io.BytesIO(json.dumps({"data": data, "size": size}).encode())

    monkeypatch.setattr(pd.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pd.time, "sleep", lambda s: None)
    return asked


def test_fetch_starts_at_page_0_and_reads_size(monkeypatch):
    pages = [[entry(i) for i in range(3)], [entry(i) for i in range(3, 5)]]
    asked = serve(pages, size=5, monkeypatch=monkeypatch)
    got = pd.fetch_disprot_json()
    assert asked == [0, 1]
    assert [e["disprot_id"] for e in got] == [f"DP{i:05d}" for i in range(5)]


def test_fetch_stops_on_short_download(monkeypatch):
    serve([[entry(0)]], size=2, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit, match="1 entries but reports size 2"):
        pd.fetch_disprot_json()


def test_regions_are_consensus_disorder_only():
    e = entry(1,
              ss=[{"start": 10, "end": 20, "type": "D"}, {"start": 30, "end": 40, "type": "S"}],
              # a function term outside the disordered stretch: must not become a region
              regions=[{"start": 10, "end": 20, "term_namespace": "Structural state"},
                       {"start": 60, "end": 90, "term_namespace": "Disorder function"}])
    rec = pd.parse_entry(e)
    assert json.loads(rec["regions_json"]) == [{"start": 10, "end": 20, "length": 11}]
    assert rec["total_disordered_residues"] == 11


def test_local_one_page_file_is_refused(tmp_path):
    f = tmp_path / "page1.json"
    f.write_text(json.dumps({"data": [entry(0)], "size": 2}))
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(BIN / "parse_disprot.py"), "--local", str(f),
                        str(tmp_path / "o.tsv")], capture_output=True, text=True)
    assert r.returncode != 0 and "one page of a paged download" in r.stderr
