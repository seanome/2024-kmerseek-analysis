"""What load_regions does with a region table that is not entirely well formed.

Imported rather than driven through the CLI, unlike test_pipeline.py: the behaviour under
test is one function's handling of one file, and a subprocess would only be able to observe
it through a whole scoring run.

The case is real. hhblitsSearch builds its TSV with an awk that reads fields by position
out of hhsearch's -blasttab output, so a record whose name carries extra whitespace shifts
every later field left. The row still has eight tab-separated columns and still looks like
a hit; what it holds in qstart is hhsearch's matched/targetLen fraction. One such row 105 KB
into a 1.2 GB file failed the whole file inside polars' CSV parser, which killed a batched
task on arm 829 of 830 with a message naming a byte offset and no tool.
"""
import gzip
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from evaluate_domain_calls import load_regions  # noqa: E402

GOOD = ["sp|P1|A_HUMAN\tsp|T1|B_CIOIN\t10\t120\t5\t115\t99.5\t1e-30",
        "sp|P2|C_HUMAN\tsp|T2|D_CIOIN\t30\t200\t20\t190\t88.0\t1e-20"]
# qstart holds 0.172 -- the fraction from column 3 of the untruncated record
SHIFTED = "sp|P3|E_HUMAN\tsp|T3|F_CIOIN\t0.172\t476\t1\t468\t70.1\t1e-10"


def write(tmp_path, lines, name="arm.tsv", gz=False):
    path = tmp_path / (name + (".gz" if gz else ""))
    body = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(body)
    else:
        path.write_text(body)
    return path


def test_a_clean_table_loads_every_row(tmp_path):
    lf = load_regions(write(tmp_path, GOOD), direct=False)
    assert lf.collect().height == 2


def test_a_shifted_row_is_dropped_and_the_rest_survive(tmp_path, capsys):
    lf = load_regions(write(tmp_path, GOOD + [SHIFTED]), direct=False)
    df = lf.collect()
    assert df.height == 2, "the two well-formed rows must still be scored"
    assert sorted(df["query_acc"]) == ["P1", "P2"]
    assert "dropped 1 of 3 rows" in capsys.readouterr().err


def test_a_wholly_shifted_table_raises_instead_of_scoring_nothing(tmp_path):
    # Every row unusable is a broken writer, not a tool that found little. Scoring it as an
    # empty result would publish a real-looking zero.
    with pytest.raises(SystemExit, match="column layout is wrong"):
        load_regions(write(tmp_path, [SHIFTED] * 3), direct=False)


def test_gzip_input_gives_the_same_rows_as_plain(tmp_path, monkeypatch):
    # chdir because inflate_for_scan writes its plain copy into the working directory --
    # under Nextflow that is the task's own work dir, and here it would be the repo.
    monkeypatch.chdir(tmp_path)
    plain = load_regions(write(tmp_path, GOOD, name="plain.tsv"), direct=False).collect()
    gzipped = load_regions(write(tmp_path, GOOD, name="gz.tsv", gz=True),
                           direct=False).collect()
    assert plain.equals(gzipped)


def test_a_nine_column_motif_table_keeps_its_extra_field(tmp_path):
    rows = [g + "\t42" for g in GOOD]
    df = load_regions(write(tmp_path, rows), direct=False).collect()
    assert df["n_matched_residues"].to_list() == [42, 42]


def test_the_shifted_row_is_caught_past_the_schema_inference_window(tmp_path, capsys):
    """The real file's shape: hundreds of clean rows first, one bad row much later.

    This is the case that actually crashed. polars reads the first 100 rows to pick a
    dtype, so a bad row inside that window makes qstart a float column and the old
    `cast(pl.Int64)` TRUNCATED 0.172 to 0 -- a hit silently starting at residue zero.
    Outside the window the same row failed the whole file instead. Both are wrong, and one
    of them is quiet, which is why the coordinates are parsed leniently and checked rather
    than trusted to inference.
    """
    rows = [f"sp|P{i}|A_HUMAN\tsp|T{i}|B_CIOIN\t{i + 1}\t{i + 120}\t5\t115\t9.5\t1e-30"
            for i in range(150)]
    df = load_regions(write(tmp_path, rows + [SHIFTED]), direct=False).collect()
    assert df.height == 150
    assert "dropped 1 of 151 rows" in capsys.readouterr().err


def test_a_backwards_interval_is_dropped_too(tmp_path, capsys):
    """The shift that lands an integer in qstart.

    The observed hhblits row shifted by four fields, so qstart held a fraction and failed to
    parse. A shift of one or two puts targetLen or mismatch there, which are integers and
    parse fine. end < start catches those; no aligner here reports a hit backwards.
    """
    backwards = "sp|P4|G_HUMAN\tsp|T4|H_CIOIN\t559\t120\t5\t115\t70.1\t1e-10"
    df = load_regions(write(tmp_path, GOOD + [backwards]), direct=False).collect()
    assert sorted(df["query_acc"]) == ["P1", "P2"]
    assert "dropped 1 of 3 rows" in capsys.readouterr().err


def test_a_backwards_target_interval_is_dropped_too(tmp_path):
    backwards = "sp|P5|I_HUMAN\tsp|T5|J_CIOIN\t10\t120\t468\t1\t70.1\t1e-10"
    df = load_regions(write(tmp_path, GOOD + [backwards]), direct=False).collect()
    assert df.height == 2
