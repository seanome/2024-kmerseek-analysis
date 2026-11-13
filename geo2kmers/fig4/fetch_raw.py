"""Download and unpack one series' raw array files from GEO.

Usage: fetch_raw.py <GSE of the raw files> <out folder>
GEO keeps the raw files as one tar per series. GSE111555's PBMC samples are in its part, GSE111552.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

gse, out = sys.argv[1:]
url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/suppl/{gse}_RAW.tar"
Path(out).mkdir(parents=True, exist_ok=True)
path = Path(out) / f"{gse}_RAW.tar"
urllib.request.urlretrieve(url, path)
with tarfile.open(path) as tar:
    tar.extractall(out, filter="data")
path.unlink()
