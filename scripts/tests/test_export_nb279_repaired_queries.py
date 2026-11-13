"""Repaired summaries must preserve finite minima and reject invalid batches."""

import csv
import gzip
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "export_nb279_repaired_queries.py"
spec = importlib.util.spec_from_file_location("repaired_queries", MODULE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_finite_minimum_wins_over_infinity(tmp_path):
    source = tmp_path / "query.tsv"
    source.write_text("query_name\tmin_region_evalue\nA\tinf\nA\t0.25\nB\tinf\n")
    values = exporter.read_queries(source, {"A", "B"})
    assert values["A"] == 0.25
    assert values["B"] == float("inf")


@pytest.mark.parametrize(
    "content",
    [
        "<html>outage</html>\n",
        "query_name\tmin_region_evalue\nunknown\t0.1\n",
        "query_name\tmin_region_evalue\nA\tnan\n",
        "query_name\tmin_region_evalue\nA\t-1\n",
    ],
)
def test_invalid_query_data_cannot_look_like_a_hit(tmp_path, content):
    source = tmp_path / "query.tsv"
    source.write_text(content)
    with pytest.raises(ValueError):
        exporter.read_queries(source, {"A"})


def test_export_reduces_chunks_and_preserves_comparator_file(tmp_path):
    queries, conventional, output = (
        tmp_path / name for name in ("queries", "conventional", "out")
    )
    (queries / "human").mkdir(parents=True)
    conventional.mkdir()
    output.mkdir()
    tool = conventional / "human_protein_tool_min_evalue.csv.gz"
    with gzip.open(tool, "wt") as handle:
        handle.write("accession,phmmer,jackhmmer,mmseqs2\nA,,,\nB,0.5,,\n")
    for chunk, text in [(0, "A\tinf\nB\t0.5\n"), (1, "A\t0.25\n")]:
        path = (
            queries
            / "human"
            / f"chunk_{chunk}.hp_lehninger2.k17.s1.lcfalse.extend-c1.51.queries.tsv"
        )
        path.write_text("query_name\tmin_region_evalue\n" + text)
    record = exporter.export_species("human", queries, conventional, output)
    with gzip.open(output / "human_protein_kmerseek_min_evalue.csv.gz", "rt") as handle:
        values = {
            row["accession"]: float(row["min_region_evalue"])
            for row in csv.DictReader(handle)
        }
    assert values == {"A": 0.25, "B": 0.5}
    assert (output / tool.name).read_bytes() == tool.read_bytes()
    assert record["n_source_files"] == 2
    assert record["n_settings_searched"] == 1


def test_refused_and_searched_chunks_cannot_share_a_setting(tmp_path):
    queries, conventional, output = (
        tmp_path / name for name in ("queries", "conventional", "out")
    )
    (queries / "human").mkdir(parents=True)
    conventional.mkdir()
    output.mkdir()
    with gzip.open(
        conventional / "human_protein_tool_min_evalue.csv.gz", "wt"
    ) as handle:
        handle.write("accession,phmmer,jackhmmer,mmseqs2\nA,,,\n")
    for chunk, text in [
        (0, "#nofit\trefused\n"),
        (1, "query_name\tmin_region_evalue\nA\t0.1\n"),
    ]:
        path = (
            queries
            / "human"
            / f"chunk_{chunk}.hp_lehninger2.k17.s1.lcfalse.extend-c1.51.queries.tsv"
        )
        path.write_text(text)
    with pytest.raises(ValueError, match="both refused and searched"):
        exporter.export_species("human", queries, conventional, output)
