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


spec_f = importlib.util.spec_from_file_location(
    "filter_301_regions", ROOT / "scripts" / "filter_301_regions.py"
)
f = importlib.util.module_from_spec(spec_f)
sys.modules[spec_f.name] = f
spec_f.loader.exec_module(f)


def test_filter_keeps_regions_on_scored_windows_only(tmp_path, monkeypatch):
    import gzip

    truth = tmp_path / "truth.tsv"
    truth.write_text("query_id\tstart\tend\nP01308\t1\t24\n")
    decoys = tmp_path / "decoys.tsv"
    decoys.write_text(
        "decoy_id\tstart\tend\tunchanged\n"
        "decoy_00000_reversed\t1\t24\tFalse\n"
        "decoy_00000_dishuffle\t1\t24\tTrue\n"
    )
    head = "query_name,target_name,region_start,region_end,region_length,region_evalue,region_ka_bits,target_start,target_end\n"
    rows = [
        "P01308,T1,23,40,17,1e-3,30,0,17",  # residues 24-40: touches residue 24, kept
        "P01308,T2,24,40,16,1e-3,30,0,16",  # residues 25-40: misses the feature
        "P01308,T3,0,10,10,50,5,0,10",  # on the feature but E above 10
        "decoy_00000_reversed,T4,5,15,10,2,9,0,10",  # on the decoy window, kept
        "decoy_00000_dishuffle,T5,5,15,10,2,9,0,10",  # unchanged decoy: not scored
    ]
    csvp = tmp_path / "s.csv"
    csvp.write_text(head + "\n".join(rows) + "\n")
    out = tmp_path / "o.tsv.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--truth",
            str(truth),
            "--decoys",
            str(decoys),
            "--search-csv",
            str(csvp),
            "--out",
            str(out),
        ],
    )
    f.main()
    lines = gzip.open(out, "rt").read().splitlines()
    assert [l.split("\t")[:4] for l in lines[1:]] == [
        ["P01308", "T1", "24", "40"],
        ["decoy_00000_reversed", "T4", "6", "15"],
    ]
