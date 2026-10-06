"""scripts/elm_choose_arms.py: one arm per alphabet, most motifs on position, ties to the
cheaper search (larger scaled), then mask off, then name. Unfinished arms are not eligible."""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "elm_choose_arms.py"


def summary(d, arm, n, status="ok"):
    (d / f"{arm}.summary.json").write_text(json.dumps(
        {"arm": arm, "status": status, "n_ortholog_on_position": n, "n_with_projection": 540}))


def test_choice_and_tie_breaks(tmp_path):
    cover = tmp_path / "cover"
    cover.mkdir()
    summary(cover, "wwmj5.k8.lcfalse", 300)
    summary(cover, "wwmj5.k9.s5.lcfalse", 397)     # tied best; scaled 5 beats scaled 2
    summary(cover, "wwmj5.k9.s2.lcfalse", 397)
    summary(cover, "gbmr7.k13.s5.lctrue", 222)     # tied; mask off wins
    summary(cover, "gbmr7.k13.s5.lcfalse", 222)
    summary(cover, "funcgroups8.k7.lcfalse", 999, status="empty")  # not eligible
    summary(cover, "funcgroups8.k7.s2.lcfalse", 321)
    summary(cover, "hmmer3_phmmer", 529)           # a comparison tool, not an arm
    out = tmp_path / "chosen.tsv"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--cover", str(cover), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    t = pl.read_csv(out, separator="\t")
    assert dict(zip(t["alphabet"], t["arm"])) == {
        "wwmj5": "wwmj5.k9.s5.lcfalse",
        "gbmr7": "gbmr7.k13.s5.lcfalse",
        "funcgroups8": "funcgroups8.k7.s2.lcfalse",
    }
