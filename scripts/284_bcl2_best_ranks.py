"""Notebook 284: BCL2's best rank for Ced9 under every setting and combination tried so far.

Rows computed here from notebook 241's ranks.csv (one alphabet-ksize setting, one metric) and
notebook 242's subset_scan*.parquet (several alphabets combined), with the tie count where the
table has one. Subset rows give Ced9 -> CD47, which has no known link, as a control: when CD47
gets the same rank, the rank is a tie, not a signal.

Three numbers exist only in notebook text (241, 242, 243 are on open PR branches). They are read
from the notebook at a pinned commit with a regex, so a changed notebook fails here instead of
leaving an old number on the page.

Inputs: ~/data/botryllus/alphabet-ranking-three-cases/{ranks.csv, subset_scan.parquet,
subset_scan.best_k.mean_idf.parquet}; notebooks at the commits in NOTEBOOK_TEXT.
Output: notebooks/284_bcl2_best_ranks.csv
"""
import re
import subprocess
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
OUT = ROOT / "notebooks/284_bcl2_best_ranks.csv"
# (label, notebook, commit, regex with named groups, which group is the value)
NOTEBOOK_TEXT = [
    ("random protein from the same hit lists does as well or better (% of draws)", "241",
     "bd9cc3b", "notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb",
     r"does as well or better than BCL2 in (?P<v>\d+)% of (?P<n>[\d_]+) draws"),
    ("logistic regression on Pfam labels, BCL2 rank", "243", "4e49c05",
     "notebooks/243_alphabet_logistic_regression.ipynb", r"ranks BCL2 (?P<v>[\d_]+) for Ced9"),
    ("logistic regression on Pfam labels, Ced9 -> CD47 rank", "243", "4e49c05",
     "notebooks/243_alphabet_logistic_regression.ipynb", r"no known link to Ced9, (?P<v>[\d_]+)\."),
    ("random queries given the same picks rank BCL2 as high, highest p", "242", "4e49c05",
     "notebooks/242_alphabet_ensemble.ipynb", r"p = (?P<v>0\.\d+), 0\.\d+, 0\.\d+ and 0\.\d+ for"),
    ("random queries given the same picks rank BCL2 as high, lowest p", "242", "4e49c05",
     "notebooks/242_alphabet_ensemble.ipynb", r"p = 0\.\d+, 0\.\d+, 0\.\d+ and (?P<v>0\.\d+) for"),
]

ARMS = {"lowest": "each at its lowest k", "all": "all k-sizes", "best": "each at BCL2's best k"}
rows = []
r = (pl.read_csv(DATA / "ranks.csv", infer_schema_length=0)
     .filter((pl.col("query") == "Ced9") & (pl.col("partner") == "BCL2") & (pl.col("partner_found") == "true"))
     .with_columns(rank=pl.col("rank").cast(pl.Float64).cast(pl.Int64), n_tied=pl.col("n_tied").cast(pl.Float64).cast(pl.Int64),
                   n_targets=pl.col("n_targets").cast(pl.Int64), ksize=pl.col("ksize").cast(pl.Int64)))
for g in r.sort("rank", "alphabet", "ksize").group_by("metric", maintain_order=True).first().iter_rows(named=True):
    rows.append(dict(approach="one setting", metric=g["metric"], setting=f"{g['alphabet']} k{g['ksize']}",
                     bcl2_rank=g["rank"], n_tied=g["n_tied"], n_ranked=g["n_targets"], ced9_cd47_rank=None,
                     source="nb 241 ranks.csv: best setting for this metric"))
for f, label in (("subset_scan.parquet", "alphabets combined"), ("subset_scan.best_k.mean_idf.parquet", "alphabets combined, each at BCL2's best k")):
    s = pl.read_parquet(DATA / f)
    best = s.sort("Ced9->BCL2", "subset").group_by("metric", "arms", "size", maintain_order=True).first().sort("metric", "arms", "size")
    for g in best.iter_rows(named=True):
        rows.append(dict(approach=label, metric=g["metric"], setting=f"{g['subset']} ({ARMS[g['arms']]})", n_alphabets=g["size"],
                         bcl2_rank=g["Ced9->BCL2"], n_tied=None, n_ranked=None, ced9_cd47_rank=g["Ced9->CD47"],
                         source=f"nb 242 {f}: best subset of this size"))
for label, nb, sha, path, pat in NOTEBOOK_TEXT:
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"], check=True, capture_output=True, text=True).stdout
    text = re.sub(r'\\n",\s*"', " ", text)  # join a markdown cell's source lines
    m = re.search(pat, text)
    assert m, (label, pat)
    rows.append(dict(approach="notebook text", metric=label, setting=m.group(0), bcl2_rank=None, n_tied=None, n_ranked=None,
                     ced9_cd47_rank=None, source=f"nb {nb} at {sha}", value=float(m.group("v").replace("_", ""))))
out = pl.DataFrame(rows, infer_schema_length=None)
out.write_csv(OUT)
print(out)
