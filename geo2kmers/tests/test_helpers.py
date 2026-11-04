"""Each helper script, on small files whose answers are worked out by hand."""

import csv
import gzip
import io
import json
import runpy
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from test_units import RANDOM_PROTEIN, STOPS, to_dna

from geo2kmers import sequences, sketch
from geo2kmers.cli import main
from geo2kmers.geo import Geo2KmersError

HERE = Path(__file__).resolve().parent.parent
FIRST = RANDOM_PROTEIN[:110]  # long enough to pass the 100 amino acid rule
OPEN_FRAME = STOPS + to_dna(FIRST) + STOPS


def run(monkeypatch, script, *args):
    monkeypatch.setattr(sys, "argv", [script, *map(str, args)])
    runpy.run_path(str(HERE / script), run_name="__main__")


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_summarise_counts_probes_and_spots(tmp_path, monkeypatch):
    write(tmp_path / "s/a.report.json", json.dumps({"sample": "GSM1", "alphabet": "dna", "k": 31, "distinct_kmers": 7}))
    fasta = write(tmp_path / "p.fasta", ">1\nACGT\n>2\nACGT\n>3\nACGTA\n>4\nACGTA\n")
    source = write(
        tmp_path / "t.tsv",
        "probe_id\tsource\tlength_nt\n1\tgenbank_fetch\t4\n2\tgenbank_fetch\t4\n3\tgpl_sequence\t5\n"
        "4\tgpl_sequence\t5\n5\trecord_too_long\t9000\n6\tmissing\t0\n",
    )
    run(monkeypatch, "summarise.py", tmp_path / "s", fasta, source, tmp_path)
    measures = {r["measure"]: r["value"] for r in rows(tmp_path / "probe_summary.tsv")}
    assert measures["probes_in_platform"] == "6" and measures["probes_record_too_long"] == "1"
    assert measures["distinct_sequences"] == "2"
    assert (measures["spots_per_sequence_min"], measures["spots_per_sequence_max"]) == ("2", "2")
    assert measures["length_nt_max"] == "5"  # the 9000 nt record has no sequence, so it does not count
    summary = rows(tmp_path / "sketch_summary.tsv")
    assert len(summary) == 1 and summary[0]["distinct_kmers"] == "7"


def test_array_tables(tmp_path, monkeypatch):
    other = sketch.reverse_complement(OPEN_FRAME)  # the open frame is on the other strand
    n_codons = STOPS + to_dna(RANDOM_PROTEIN[:20]) + "NNN" * 95 + STOPS
    for name in ("seq1211", "seq738"):
        write(
            tmp_path / f"{name}/probes.fasta",
            f">1\n{OPEN_FRAME}\n>2\n{OPEN_FRAME}\n>3\n{other}\n>4\n{n_codons}\n>5\n{'A' * 6000}\n",
        )
        write(tmp_path / f"{name}/probe_sequence_source.tsv", "probe_id\n1\n2\n3\n4\n5\n6\n")
    write(tmp_path / "real/mussel.fasta", f">1\n{OPEN_FRAME}\n")
    write(tmp_path / "GPL1211.txt", "ID\tGB_ACC\n1\tA.1\n2\tA.1\n3\tB.1\n4\tC.1\n5\tD.1\n")
    # the true frame of A is 2 and of B is 0, and the longest stretch of sequence 1 is in frame 2
    frame = sketch.longest_stretch(OPEN_FRAME)[0]
    write(tmp_path / "cds_GPL1211.tsv", f"accession\tstart\tend\tcodon_start\nA\t{frame + 1}\t9\t1\nB\t1\t9\t1\n")
    start = sketch.longest_stretch(OPEN_FRAME)[1]
    length = len(sketch.longest_stretch(OPEN_FRAME)[2])
    head = "probe_id\talphabet\tk\tframe\tstretch_start_aa\tstretch_length_aa\n"
    calls = f"1\tprotein\t10\t{frame}\t{start}\t{length}\n1\thp\t30\t3\t0\t500\n"
    n_pick = sketch.longest_stretch(n_codons)
    x_call = f"4\tprotein\t10\t{n_pick[0]}\t{n_pick[1]}\t{len(n_pick[2])}\n"  # X does not count towards 100 aa
    write(tmp_path / "calls_regress.tsv", head + calls)
    write(tmp_path / "calls_GPL738.tsv", head + calls + x_call)
    write(tmp_path / "calls_GPL1211.tsv", head + calls)
    run(monkeypatch, "array_tables.py", tmp_path, tmp_path)
    human = rows(tmp_path / "array_summary.tsv")[0]
    assert (human["probe_rows"], human["rows_with_sequence"], human["distinct_sequences"]) == ("6", "5", "4")
    assert human["rows_over_5000_nt"] == "1" and human["length_nt_max"] == "6000"
    assert human["distinct_with_a_stop_free_frame_k10"] == "1"  # the 6000 As read AAA throughout
    table = {r["min_longest_stretch_aa"]: r for r in rows(tmp_path / "longest_stretch_GPL1211.tsv")}
    assert table["0"]["sequences"] == "2" and table["0"]["longest_in_true_frame"] == "1"  # the other strand is wrong
    compared = {r["array"]: r for r in rows(tmp_path / "longest_vs_proteome.tsv")}
    assert compared["GPL738 mouse"]["sequences_with_passing_stretch_100aa"] == "1"
    assert compared["GPL738 mouse"]["longest_stretch_same_frame"] == "1"


def test_frame_test_finds_the_one_coding_sequence(tmp_path, monkeypatch):
    write(tmp_path / "proteins.fa", f">a\n{RANDOM_PROTEIN}\n")
    noncoding = "ATTAGCTTAACGTAGATAG" * 20  # no match to the protein
    fasta = write(tmp_path / "probes.fasta", f">1\n{OPEN_FRAME}\n>2\n{OPEN_FRAME}\n>3\n{noncoding}\n")
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "frame_test.py", fasta, tmp_path, tmp_path / "o.tsv", "p=proteins.fa", tmp_path / "c.tsv")
    found = {(r["alphabet"], r["k"]): r for r in rows(tmp_path / "o.tsv")}
    protein = found["protein", "10"]
    assert protein["tags"] == "2"  # two distinct sequences: spots 1 and 2 are one
    assert protein["tags_stretch_ge90pct"] == "1"
    assert protein["shuffled_stretch_ge90pct_mean"] == "0.0"
    calls = [r for r in rows(tmp_path / "c.tsv") if (r["alphabet"], r["k"]) == ("protein", "10")]
    assert [(c["probe_id"], c["frame"], c["stretch_length_aa"]) for c in calls] == [("1", "0", "110")]


def test_frame_truth_scores_the_longest_call(tmp_path, monkeypatch):
    gpl = write(tmp_path / "g.txt", "1\tA.1\n2\tB.1\n3\tC.1\n")
    fasta = write(tmp_path / "p.fasta", ">1\nAAA\n>2\nCCC\n>3\nGGG\n")
    cds = write(tmp_path / "c.tsv", "accession\tstart\tend\tcodon_start\nA\t2\t9\t1\nB\t1\t9\t2\n")
    head = "probe_id\talphabet\tk\tframe\tstretch_start_aa\tstretch_length_aa\n"
    # A is frame 1 and its longest call is right; B is frame 1 and its longest call is wrong; C has no annotation
    calls = "1\tprotein\t10\t1\t0\t120\n1\tprotein\t10\t0\t0\t20\n2\tprotein\t10\t2\t0\t40\n3\tprotein\t10\t0\t0\t90\n"
    for setting in ("protein\t7", "hp\t30"):
        calls += f"1\t{setting}\t1\t0\t120\n"
    call_table = write(tmp_path / "k.tsv", head + calls)
    run(monkeypatch, "frame_truth.py", gpl, fasta, cds, call_table, tmp_path)
    truth = {(r["alphabet"], r["k"]): r for r in rows(tmp_path / "frame_truth.tsv")}
    protein = truth["protein", "10"]
    assert protein["annotated"] == "2" and protein["with_passing_stretch"] == "2"
    assert protein["longest_in_true_frame"] == "1" and protein["all_in_true_frame"] == "0"
    by_length = {
        r["min_longest_stretch_aa"]: r
        for r in rows(tmp_path / "frame_truth_by_length.tsv")
        if (r["alphabet"], r["k"]) == ("protein", "10")
    }
    assert by_length["100"]["sequences"] == "1" and by_length["100"]["longest_in_true_frame"] == "1"


def test_fetch_coding_regions_reads_ncbi_feature_tables(tmp_path, monkeypatch):
    table = write(tmp_path / "g.txt", "1\tAB123.1\n2\tNM_5.2\n3\t\nID\tGB\n")
    text = (
        ">Feature gb|AB123.1|\n<1\t>90\tCDS\n\t\t\tcodon_start\t3\n\t\t\tproduct\tx\n100\t200\tgene\n"
        ">Feature ref|NM_5.2|\n5\t50\tCDS\n\t\t\tproduct\ty\n"
    )
    asked = []

    def urlopen(url, data, timeout):
        asked.append(data.decode())
        return io.BytesIO(text.encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("NCBI_EMAIL", "test@example.org")
    run(monkeypatch, "fetch_coding_regions.py", table, tmp_path / "o.tsv")
    assert "AB123%2CNM_5" in asked[0] and len(asked) == 1
    assert [tuple(r.values()) for r in rows(tmp_path / "o.tsv")] == [
        ("AB123", "1", "90", "3"),
        ("NM_5", "5", "50", "1"),
    ]


def test_the_figure_script_runs_on_the_results(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    names = (
        "longest_stretch_GPL1211 frame_truth_by_length mussel_vs_same_species mussel_vs_oyster mussel_vs_human_mouse"
    )
    for name in names.split():
        write(tmp_path / f"{name}.tsv", (HERE / f"results/frame_test/{name}.tsv").read_text())
    run(monkeypatch, "figure_frames.py", tmp_path)
    assert (tmp_path / "frame_choice_human_mussel.png").stat().st_size > 0


def test_a_probe_id_with_a_space_joins_through_sequences_and_sketch(tmp_path, capsys):
    gpl = write(
        tmp_path / "GPL1.txt",
        f"!platform_table_begin\nID\tSEQUENCE\nspot 1\t{'ACGTTGCA' * 10}\nspot 2\t{'TTGACCAG' * 10}\n!platform_table_end\n",
    )
    fasta = tmp_path / "probes.fasta"
    sequences.make_probe_fasta(gpl, fasta, tmp_path / "t.tsv")
    gsm = write(
        tmp_path / "GSM1.txt",
        "!Sample_channel_count = 1\n!sample_table_begin\nID_REF\tVALUE\nspot 1\t100\nspot 2\t200\n!sample_table_end\n",
    )
    out = tmp_path / "x.tsv"
    args = f"sketch --fasta {fasta} --intensities {gsm} --sample GSM1 --k 21 --alphabet dna --mode weighted --out {out}"
    assert main(args.split()) == 0
    assert json.loads(Path(f"{out}.report.json").read_text())["probes_used"] == 2


def test_a_fasta_header_with_a_description_still_joins_by_first_word(tmp_path):
    fasta = write(tmp_path / "p.fasta", f">a some words\n{'ACGTTGCA' * 10}\n>b more\n{'TTGACCAG' * 10}\n")
    matched = sketch.match_names(sketch.read_fasta(fasta, whole_name=True), {"a": 5.0, "b": 6.0})
    assert sorted(matched) == ["a", "b"]
    with pytest.raises(Geo2KmersError, match="first word"):
        sketch.match_names({"spot 1": "A", "spot 2": "C"}, {"x": 1.0})


def test_a_folder_of_gzipped_sample_tables_is_read(tmp_path):
    text = "!Sample_channel_count = 1\n!sample_table_begin\nID_REF\tVALUE\na\t100\n!sample_table_end\n"
    with gzip.open(tmp_path / "GSM1.txt.gz", "wt") as f:
        f.write(text)
    assert sketch.load_intensities(tmp_path, "GSM1")[0] == {"a": 100.0}


def test_a_lost_connection_is_retried_then_stops_with_a_message():
    tries = []

    def post(batch, email, api_key):
        tries.append(batch)
        if len(tries) < 3:
            raise urllib.error.URLError("no route to host")
        return ">A1234\nACGT\n"

    assert sequences.fetch_accessions(["A1234"], "e", post=post, sleep=lambda s: None) == {"A1234": "ACGT"}

    def down(batch, email, api_key):
        raise TimeoutError("timed out")

    with pytest.raises(Geo2KmersError, match="could not reach NCBI"):
        sequences.fetch_accessions(["A1234"], "e", post=down, sleep=lambda s: None)


def test_a_connection_error_is_a_clean_stop_at_the_command_line(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("geo2kmers.geo.fetch", no_network)
    assert main(["fetch", "GSE1", "--outdir", str(tmp_path)]) == 1
    assert "no route to host" in capsys.readouterr().err
