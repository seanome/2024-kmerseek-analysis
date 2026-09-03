"""The k-mer spectra panel reads what kmerseekIndex already writes.

kmerseekIndex has always emitted one spectrum per (species, alphabet, ksize, lc) combo via
--kmer-stats-out, and main.nf publishes them to ${outdir}/spectra under a comment saying
they are "published for plotting". Nothing plotted them until now, so these tests pin the
file format the panel depends on -- a change to kmerseek's --kmer-stats-out output should
fail here rather than silently empty the section.
"""
import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

HEADER = ("# total_kmers=54996 unique_kmers=49176 mean_seqs_per_kmer=1.1184 "
          "median_seqs_per_kmer=1.0 mode_seqs_per_kmer=1 moltype={alpha} ksize={k}\n")
BODY = "moltype,ksize,occurrences,n_kmers\n{alpha},{k},1,43785\n{alpha},{k},2,4992\n"


def write(d: Path, species="ecoli", alpha="hp_pbotc_1st_ed2", k=18, lc="false"):
    p = d / f"spectrum.{species}.{alpha}.k{k}.lc{lc}.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(HEADER.format(alpha=alpha, k=k) + BODY.format(alpha=alpha, k=k))
    return p


def test_filename_carries_the_coordinates_the_csv_body_lacks(tmp_path):
    """The body has only moltype and ksize. Species and the lc arm exist only in the name,
    so a parser that trusts the body alone silently collapses every proteome together."""
    write(tmp_path, species="yeast", lc="true")
    spectra, summary = bmi.load_spectra(tmp_path)
    assert spectra["species"].to_list() == ["yeast", "yeast"]
    assert spectra["lowcomp"].to_list() == [True, True]
    assert summary["ksize"][0] == 18


def test_totals_come_from_the_header_not_from_re_summing_the_body(tmp_path):
    """A truncated body must show up as a disagreement, not be reconstructed into a
    plausible-looking total."""
    _, summary = bmi.load_spectra(tmp_path if write(tmp_path) else tmp_path)
    r = summary.to_dicts()[0]
    assert r["total_kmers"] == 54996          # header
    assert r["unique_kmers"] == 49176         # header
    assert r["max_occurrences"] == 2          # body, which is truncated on purpose


def test_absent_or_empty_directory_is_not_fatal(tmp_path):
    for arg in (None, tmp_path / "nope", tmp_path):
        spectra, summary = bmi.load_spectra(arg)
        assert spectra.height == 0 and summary.height == 0


def test_unparsable_filename_is_skipped_and_named(tmp_path, capsys):
    (tmp_path / "spectrum.garbage.csv.gz").write_bytes(gzip.compress(b"moltype\n"))
    write(tmp_path)
    spectra, _ = bmi.load_spectra(tmp_path)
    assert spectra.height == 2
    assert "spectrum.garbage.csv.gz" in capsys.readouterr().out


def test_section_splits_the_species_and_drops_the_low_complexity_dimension(tmp_path):
    """One panel, not two. The filter changes Fmax by under 0.001 at every alphabet's own
    best k, so drawing a second copy of this section for it doubled the report to show a
    tail difference that changes no result. The supplementary low-complexity panel is the
    whole record of the comparison."""
    src, out = tmp_path / "s", tmp_path / "o"
    src.mkdir(), out.mkdir()
    write(src, species="ecoli", lc="false")
    write(src, species="yeast", lc="false")
    write(src, species="ecoli", lc="true")
    bmi.section_kmer_spectra(out, *bmi.load_spectra(src))
    off = json.loads((out / "qfo_kmer_spectra_mqc.json").read_text())
    # One dataset per proteome: two proteomes are two different key sets and must never
    # share a panel.
    assert [d["name"] for d in off["pconfig"]["data_labels"]] == ["ecoli", "yeast"]
    assert not list(out.glob("qfo_kmer_spectra_lc*"))
    assert off["pconfig"]["xlog"] and off["pconfig"]["ylog"]


def test_the_summary_table_loses_the_low_complexity_arm_too(tmp_path):
    src, out = tmp_path / "s2", tmp_path / "o2"
    src.mkdir(), out.mkdir()
    write(src, species="ecoli", lc="false")
    write(src, species="ecoli", lc="true")
    bmi.section_kmer_spectra(out, *bmi.load_spectra(src))
    table = json.loads((out / "qfo_kmer_spectra_table_mqc.json").read_text())
    assert not any("lcT" in k or "lcF" in k for k in table["data"])
    assert len(table["data"]) == 1


def test_no_spectra_writes_no_section(tmp_path):
    import polars as pl
    bmi.section_kmer_spectra(tmp_path, pl.DataFrame(), pl.DataFrame())
    assert not list(tmp_path.glob("*_mqc.json"))


def test_duplication_agrees_with_the_header_mean(tmp_path):
    """total/unique is the same quantity kmerseek reports as mean_seqs_per_kmer. If those
    two disagree, one of them is being read wrong."""
    write(tmp_path)
    _, summary = bmi.load_spectra(tmp_path)
    r = summary.to_dicts()[0]
    assert r["total_kmers"] / r["unique_kmers"] == pytest.approx(
        r["mean_seqs_per_kmer"], abs=1e-4)
