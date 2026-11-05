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

import numpy as np
import pytest
from test_units import RANDOM_PROTEIN, STOPS, _Closing, to_dna

from geo2kmers import landing, scoring, sequences, sketch
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


def test_land_stops_on_long_probes_with_or_without_k(tmp_path):
    proteome = write(tmp_path / "p.fasta", f">GENE\n{RANDOM_PROTEIN[:60]}\n")
    for k in (None, 7):
        with pytest.raises(Geo2KmersError, match="sketch --proteome"):
            landing.land_probes({"a": "ACGT" * 30}, [proteome], k)


def test_land_stops_on_an_empty_probe_file(tmp_path):
    proteome = write(tmp_path / "p.fasta", f">GENE\n{RANDOM_PROTEIN[:60]}\n")
    with pytest.raises(Geo2KmersError, match="no probes"):
        landing.land_probes({}, [proteome])


def test_land_stops_above_the_largest_k(tmp_path):
    proteome = write(tmp_path / "p.fasta", f">GENE\n{RANDOM_PROTEIN[:60]}\n")
    with pytest.raises(Geo2KmersError, match="up to k = 14"):
        landing.land_probes({"a": to_dna(RANDOM_PROTEIN[:20])}, [proteome], 15)


def test_a_probe_set_lands_when_3_probes_land_on_one_protein(tmp_path):
    protein = RANDOM_PROTEIN[:60]
    proteome = write(tmp_path / "p.fasta", f">GENE\n{protein}\n")
    # 8 residues in frame 0 of each 25-letter probe, from three places in the protein
    probes = {f"set1/{i}": to_dna(protein[i * 10 : i * 10 + 8]) + "A" for i in range(3)}
    probes |= {"set2/0": probes["set1/0"], "set2/1": probes["set1/1"], "set2/2": "ATTAGCTTAACGTAGATAGCCATAA"}
    rule, k, rows, touched = landing.land_probes(probes, [proteome])
    assert (rule, k, touched) == ("probe_set", 7, 5)
    passed = landing.passing_landings(rule, rows, lambda p: [p.split("/")[0]])
    assert set(passed) == {"set1"} and set(passed["set1"]) == {"GENE"}  # set2 has only 2 probes on the protein
    assert passed["set1"]["GENE"] == 6  # three places, two k-mers each


def test_landing_reads_both_strands_and_agrees_with_translate_frames(tmp_path):
    protein = RANDOM_PROTEIN[:60]
    proteome = write(tmp_path / "p.fasta", f">GENE\n{protein}\n")
    forward = to_dna(protein[:8]) + "A"
    backward = sketch.reverse_complement("A" + to_dna(protein[10:18]))
    _, _, rows, _ = landing.land_probes({"f": forward, "b": backward}, [proteome])
    found = {(r[0], r[2]) for r in rows}
    assert ("f", 0) in found
    frames = sketch.translate_frames(backward)
    assert ("b", next(i for i, f in enumerate(frames) if protein[10:18] in f)) in found


def test_a_short_oligo_landing_needs_half_its_kmers(tmp_path):
    protein = RANDOM_PROTEIN[:40]
    proteome = write(tmp_path / "p.fasta", f">GENE\n{protein}\n")
    whole = to_dna(protein[:20])
    a_little = to_dna(protein[:8] + "W" * 12)
    rule, _, rows, touched = landing.land_probes({"a": whole, "b": a_little}, [proteome])
    assert rule == "short_oligo" and touched == 2
    assert set(landing.passing_landings(rule, rows)) == {"a"}


def fig4_files(tmp_path, values):
    """A one-series test: gene A is M1, gene B is M2, and post samples score higher."""
    samples = ["GSM1", "GSM2", "GSM3", "GSM4"]
    write(
        tmp_path / "sig.tsv",
        "gene\tcluster\tgroup\nA\t13\tM1\nB\t14\tM2\n",
    )
    write(
        tmp_path / "groups.tsv",
        "dataset\tcomparison\tgroup\tsample\n"
        + "".join(f"GSE1\tmain\t{g}\t{s}\n" for g, s in zip([0, 0, 1, 1], samples, strict=True)),
    )
    write(
        tmp_path / "pub.tsv",
        "dataset\tpanel\tcomparison\tgroup0\tgroup1\tn0\tn1\tauc\tp\tmodel\n"
        "GSE1\t4D\tmain\tPre\tPost\t2\t2\t0.90\t0.1\tfull\nGSE2\t4C\tmain\tPre\tPost\t2\t2\t0.10\t0.1\tfull\n",
    )
    write(tmp_path / "plat.tsv", "gse\ttype\tlanding_share_of_sequenced_probes\nGSE1\tarray\t0.5\nGSE2\tRNA-seq\t\n")
    write(tmp_path / "land/GSE1.landing.tsv", "feature\tprotein\tdistinct_kmers\na\tA\t5\nb\tB\t5\n")
    (tmp_path / "series/GSE1").mkdir(parents=True)
    rows = "".join(f'"{name}"\t' + "\t".join(map(str, row)) + "\n" for name, row in values.items())
    header = '"ID_REF"\t' + "\t".join(f'"{s}"' for s in samples) + "\n"
    with gzip.open(tmp_path / "series/GSE1/GSE1_series_matrix.txt.gz", "wt") as f:
        f.write("!Series_x\n!series_matrix_table_begin\n" + header + rows + "!series_matrix_table_end\n")


def test_fig4_score_marks_a_dataset_reproduced_and_an_rna_seq_one_not_covered(tmp_path, monkeypatch):
    values = {"a": [9, 8, 2, 1], "b": [1, 2, 8, 9], "c": [0, 0, 0, 0]}  # M1 falls and M2 rises after exercise
    fig4_files(tmp_path, values)
    args = ["sig.tsv", "groups.tsv", "pub.tsv", "plat.tsv", "series", "land", "."]
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "fig4/score.py", *args)
    first, second = rows(tmp_path / "auc.tsv")
    assert first["reproduced_auc"] == "1.000" and first["verdict"] == "reproduced"
    assert second["verdict"] == "not covered by geo2kmers" and second["reproduced_auc"] == ""


def test_fig4_score_says_not_reproduced_on_the_wrong_side_of_a_half(tmp_path, monkeypatch):
    values = {"a": [1, 2, 8, 9], "b": [9, 8, 2, 1], "c": [0, 0, 0, 0]}  # the opposite of the published direction
    fig4_files(tmp_path, values)
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "fig4/score.py", "sig.tsv", "groups.tsv", "pub.tsv", "plat.tsv", "series", "land", ".")
    assert rows(tmp_path / "auc.tsv")[0]["verdict"] == "not reproduced"


def test_fig4_land_keeps_the_reading_with_more_probes_on_the_proteome(tmp_path, monkeypatch):
    protein = RANDOM_PROTEIN[:60]
    proteins = write(tmp_path / "p.fasta", f">GENE\n{protein}\n")
    complement = str.maketrans("ACGT", "TGCA")
    probes = [to_dna(protein[i * 10 : i * 10 + 8]).translate(complement) + "A" for i in range(3)]
    with gzip.open(tmp_path / "t.probes.tsv.gz", "wt") as f:
        f.write("probe_set\tsequence\n" + "".join(f"set1\t{p}\n" for p in probes))
    run(monkeypatch, "fig4/land.py", "GSE1", "GPL1", tmp_path / "t.probes.tsv.gz", proteins, tmp_path)
    share = rows(tmp_path / "GSE1.landing_share.tsv")[0]
    assert share["orientation"] == "complement" and share["probes_with_a_kmer_complemented"] == "3"
    assert [(r["feature"], r["protein"]) for r in rows(tmp_path / "GSE1.landing.tsv")] == [("set1", "GENE")]


def test_fig4_annotation_control_counts_agreement_only_among_annotated_landings(tmp_path, monkeypatch):
    write(
        tmp_path / "series/GSE1/GPL570.txt",
        "!platform_table_begin\nID\tGene Symbol\na\tGENE\nb\tOTHER /// MORE\nc\t\nd\t---\n!platform_table_end\n",
    )
    landing_table = "feature\tprotein\tdistinct_kmers\na\tGene\t5\nb\tGENE\t2\nc\tGENE\t9\nd\tGENE\t9\n"
    write(tmp_path / "land/GSE1.landing.tsv", landing_table)
    run(monkeypatch, "fig4/annotation_control.py", tmp_path / "series", tmp_path / "land", tmp_path / "o.tsv")
    row = rows(tmp_path / "o.tsv")[0]
    assert (row["landings"], row["landings_with_annotation"], row["agree"]) == ("4", "2", "1")  # c and d have no gene
    assert (row["annotated_5_or_more_kmers"], row["agree_5_or_more_kmers"]) == ("1", "1")
    assert (row["annotated_under_5_kmers"], row["agree_under_5_kmers"]) == ("1", "0")


def test_a_short_oligo_lands_only_through_its_longest_stop_free_frame(tmp_path):
    protein = RANDOM_PROTEIN[:40]
    proteome = write(tmp_path / "p.fasta", f">GENE\n{protein}\n")
    # frame 0 is the protein but holds a stop, and frame 1 is stop-free, so frame 0 cannot give a landing
    probe = to_dna(protein[:9]) + "TAA" + to_dna(protein[9:18])
    _, _, rows, touched = landing.land_probes({"a": probe + "A" * 3}, [proteome], k=7)
    assert touched == 1
    assert all(r[2] != 0 for r in rows)


def test_no_kmer_crosses_a_stop_in_landing(tmp_path):
    proteome = write(tmp_path / "p.fasta", f">GENE\n{'M' * 4}{'A' * 3}\n")
    _, _, rows, touched = landing.land_probes(
        {"a": to_dna("MMMM") + "TAA" + to_dna("AAA") + "AAAAAAAA"}, [proteome], k=7
    )
    assert touched == 0 and not rows


def test_a_kmer_owned_by_two_proteins_lands_on_both(tmp_path):
    shared = RANDOM_PROTEIN[:20]
    proteome = write(tmp_path / "p.fasta", f">ONE\n{shared}\n>TWO\n{shared}W\n")
    rule, _, rows, _ = landing.land_probes({"a": to_dna(shared)}, [proteome])
    assert set(landing.passing_landings(rule, rows)["a"]) == {"ONE", "TWO"}


def test_exactly_half_the_kmers_is_enough_for_a_short_oligo():
    rows = [("a", "GENE", 0, 7, 14, frozenset(range(7)))]
    assert set(landing.passing_landings("short_oligo", rows)) == {"a"}
    assert not landing.passing_landings("short_oligo", [("a", "GENE", 0, 6, 14, frozenset(range(6)))])


def test_a_probe_in_two_probe_sets_counts_in_both():
    rows = [(str(i), "GENE", 0, 1, 2, frozenset({i})) for i in range(3)]
    passed = landing.passing_landings("probe_set", rows, lambda p: ["x", "y"])
    assert set(passed) == {"x", "y"} and passed["x"]["GENE"] == 3


def test_stepminer_threshold_agrees_with_a_search_of_every_split():
    rng = np.random.default_rng(1)
    for _ in range(50):
        values = rng.normal(size=int(rng.integers(3, 30)))
        a = np.sort(values)
        errors = [((a[:i] - a[:i].mean()) ** 2).sum() + ((a[i:] - a[i:].mean()) ** 2).sum() for i in range(1, len(a))]
        best = int(np.argmin(errors)) + 1
        assert scoring.stepminer_threshold(values) == pytest.approx((a[:best].mean() + a[best:].mean()) / 2)


def test_standardise_by_hand():
    values = np.array([0.0, 0.0, 4.0, 4.0])
    # threshold 2, sample sd 2.3094...
    assert scoring.standardise(values)[2] == pytest.approx((4 - 2.5) / (3 * 2.3094010767585))
    assert scoring.standardise(np.ones(4)) is None


def test_scores_weigh_clusters_and_keep_m1_alone_when_asked():
    z = {"A": np.array([1.0]), "B": np.array([2.0]), "C": np.array([4.0]), "D": np.array([8.0])}
    clusters = {"A": {13}, "B": {14}, "C": {3}, "D": {14, 3}}
    planned, weighted = scoring.scores(z, clusters, False)
    assert planned[0] == pytest.approx(-1 + (2 + 4 + 8) / 3)
    assert weighted[0] == pytest.approx(-1 + (2 + 8) + 2 * (4 + 8))
    planned, weighted = scoring.scores(z, clusters, True)
    assert (planned[0], weighted[0]) == (-1.0, -1.0)


@pytest.mark.parametrize(
    "auc, published, expected",
    [
        (0.3, 0.17, True),
        (0.33, 0.17, False),
        (0.6, 0.54, True),
        (0.45, 0.54, False),
        (0.65, 0.8, True),
        (0.64, 0.8, False),
    ],
)
def test_the_rule_needs_the_same_side_and_a_distance_of_at_most_0_15(auc, published, expected):
    assert scoring.passes(auc, published) is expected


def test_platforms_labels_rna_seq_as_not_covered(tmp_path, monkeypatch):
    share = write(tmp_path / "s.tsv", "gse\tgpl\trule\nGSE1\tGPL1\tprobe_set\n")
    run(monkeypatch, "fig4/platforms.py", share, tmp_path / "o.tsv")
    found = {r["gse"]: r["type"] for r in rows(tmp_path / "o.tsv")}
    assert found["GSE1"] == "array" and found["GSE99963"] == "RNA-seq, not covered by geo2kmers"


def test_fig4_chance_writes_one_row_per_model_and_score(tmp_path, monkeypatch):
    fig4_files(tmp_path, {"a": [9, 8, 2, 1], "b": [1, 2, 8, 9], "c": [5, 1, 7, 3], "d": [2, 6, 4, 8]})
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "fig4/score.py", "sig.tsv", "groups.tsv", "pub.tsv", "plat.tsv", "series", "land", ".")
    run(monkeypatch, "fig4/chance.py", "sig.tsv", "groups.tsv", "pub.tsv", "auc.tsv", "series", "land", ".")
    table = rows(tmp_path / "chance_test.tsv")
    assert {(r["model"], r["score"]) for r in table} == {
        (m, s) for m in ("random features", "shuffled labels") for s in ("planned", "ghosh_weights")
    }
    assert all(0 <= float(r["share_of_draws_at_least_observed"]) <= 1 for r in table)
    assert {r["observed_passing"] for r in table if r["score"] == "planned"} == {"1"}


def test_signature_picks_the_entry_whose_main_name_is_the_gene(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "C13-14-3"
    sheet.append(["Gene Signature"])
    sheet.append(["Weight:-1", "Weight:1", "Weight:2"])
    sheet.append(["AAA", "BBB", "BBB"])
    sheet.append(["LOC1", None, None])
    book.save(tmp_path / "s.xlsx")
    answers = {
        "AAA": "Entry\tGene Names (Primary)\tSequence\nP2\tOTHER\tMMMMMMMM\nP1\tAAA\tMK\n",
        "BBB": "Entry\tGene Names (Primary)\tSequence\nP3\tBBB\tMKV\n",
        "LOC1": "Entry\tGene Names (Primary)\tSequence\n",
    }

    def urlopen(request, timeout):
        symbol = request.full_url.split("gene_exact%3A")[1].split("%29")[0]
        return io.BytesIO(answers[symbol].encode())

    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout: _Closing(urlopen(r, timeout).read()))
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("NCBI_EMAIL", "test@example.org")
    run(monkeypatch, "fig4/signature.py", tmp_path / "s.xlsx", tmp_path)
    found = rows(tmp_path / "signature_proteins.tsv")
    assert [(r["gene"], r["cluster"], r["accession"]) for r in found[:2]] == [("AAA", "13", "P1"), ("BBB", "14", "P3")]
    assert found[-1]["resolved"] == "False"
    assert (tmp_path / "signature_proteins.fasta").read_text().count(">BBB") == 1  # listed twice, written once
