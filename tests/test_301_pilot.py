"""Tests for scripts/pilot_301_kmerseek.py check: 0-based, end-exclusive region coordinates."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pilot_301_kmerseek", ROOT / "scripts" / "pilot_301_kmerseek.py"
)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)

INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"


def test_check_counts_finite_evalues_and_coordinates(tmp_path, capsys):
    q = tmp_path / "q.fasta"
    q.write_text(f">P01308\n{INSULIN}\n>decoy_00000_reversed\n{INSULIN}\n")
    c = tmp_path / "s.csv"
    c.write_text(
        "query_name,region_start,region_end,region_subseq,region_evalue\n"
        f"P01308,0,24,{INSULIN[0:24]},1e-5\n"  # residues 1-24, 1-based
        f"P01308,1,25,{INSULIN[0:24]},inf\n"  # read 1-based by mistake: no match
        f"decoy_00000_reversed,30,40,{INSULIN[30:40]},0.5\n"
    )
    p.check(Namespace(queries=q, search_csv=c))
    out = capsys.readouterr().out.splitlines()
    real = dict(zip(out[0].split("\t"), out[1].split("\t")))
    decoy = dict(zip(out[0].split("\t"), out[3].split("\t")))
    assert real["rows"] == "2" and real["rows_finite_region_evalue"] == "1"
    assert real["rows_region_subseq_matches_query"] == "1"
    assert real["min_region_evalue"] == "1e-05"
    assert decoy["rows_region_subseq_matches_query"] == "1"
