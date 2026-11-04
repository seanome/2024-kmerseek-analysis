"""Checks on each rule the README states."""

import io
import json
import urllib.error
from pathlib import Path

import pytest

from geo2kmers import geo, sequences, sketch
from geo2kmers.cli import main
from geo2kmers.geo import Geo2KmersError

sourmash = pytest.importorskip("sourmash")


def write(path, text):
    Path(path).write_text(text)
    return path


def test_translate_matches_biopython():
    seq_mod = pytest.importorskip("Bio.Seq")
    seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGCAT"
    rc = str(seq_mod.Seq(seq).reverse_complement())
    expected = [str(seq_mod.Seq(s[i : i + (len(s) - i) // 3 * 3]).translate()) for s in (seq, rc) for i in range(3)]
    assert sketch.translate_frames(seq) == expected


def test_frames_with_a_stop_are_dropped():
    assert sketch.protein_kmers("TAATAATAA", 2, "protein")[1] == 5  # only + frame 1 has a stop
    assert "*" not in "".join(sketch.protein_kmers("ATGTAAATG", 2, "protein")[0])


def test_probe_with_no_usable_frame_is_reported():
    # every frame on both strands holds a stop codon
    seq = "TAATTAATAATTAGCTAACTAGTTAGTCTAGACTAAGTGA" * 2
    assert sketch.protein_kmers(seq, 3, "protein") == ([], 0)
    probes = {"p": seq, "open": "GCTGCTGCTGCTGCTGCT"}
    table, report = sketch.build_sketch(probes, {"p": 10.0, "open": 10.0, "q": 1.0}, 3, "protein", "weighted")
    assert report["probes_with_zero_usable_frames"] == 1 and table


@pytest.mark.parametrize("alphabet,k", [("dna", 15), ("protein", 5), ("hp", 9)])
def test_signature_hashes_equal_sourmash_own_hashes(tmp_path, alphabet, k):
    seq = "ATGGCCATTGTAATGGGCCGCTGCAAAGGGTGCCCGACAGCATGGAGTTTCGCTCCGGAACTGTACAAA"
    table, _ = sketch.build_sketch({"p": seq}, {"p": 100.0, "q": 0.0}, k, alphabet, "weighted")
    assert sketch.write_signature(table, tmp_path / "x.sig", k, alphabet, "weighted", "x")
    got = next(iter(sourmash.load_file_as_signatures(str(tmp_path / "x.sig")))).minhash
    args = {"n": 0, "ksize": k, "scaled": 1}
    if alphabet != "dna":
        args[{"protein": "is_protein", "hp": "hp"}[alphabet]] = True
    own = sourmash.MinHash(**args)
    if alphabet == "dna":
        own.add_sequence(seq)
    else:
        for frame in sketch.translate_frames(seq):
            if "*" not in frame:
                own.add_protein(frame)
    assert set(got.hashes) == set(own.hashes)


def test_kmer_seen_twice_in_a_probe_counts_twice():
    table, report = sketch.build_sketch({"p": "AAAAAA"}, {"p": 10.0, "q": 0.0}, 3, "dna", "weighted")
    background = 0.5  # 5th percentile of 0 and 10
    assert table["AAA"][0] == pytest.approx(4 * (10 - background))
    assert table["AAA"][2] == 1
    assert report["background"] == background


def test_probe_with_n_skips_those_windows():
    assert sketch.dna_kmers("AAANAAA", 3) == ["AAA", "AAA"]


def test_rounding():
    assert [sketch.rounded_abundance(w) for w in (0.2, 0.5, 1.5, 2.49)] == [1, 1, 2, 2]


def test_weight_column_is_never_called_count(tmp_path):
    table, report = sketch.build_sketch({"p": "ACGTACGT"}, {"p": 10.0, "q": 0.0}, 3, "dna", "weighted")
    sketch.write_sketch(table, report, tmp_path / "x.tsv", "GSM1", None)
    header = (tmp_path / "x.tsv").read_text().splitlines()[0]
    assert header == "kmer\tintensity_weight\tn_probes"


def sample_table(path, rows, columns, channels=1):
    lines = [f"!Sample_channel_count = {channels}", "!sample_table_begin", "\t".join(columns)]
    lines += ["\t".join(map(str, r)) for r in rows] + ["!sample_table_end"]
    return write(path, "\n".join(lines) + "\n")


def test_detection_call_is_used_when_present(tmp_path, capsys):
    t = sample_table(
        tmp_path / "GSM1.txt", [("a", 1, "P"), ("b", 9, "A"), ("c", 9, "M")], ["ID_REF", "VALUE", "ABS_CALL"]
    )
    fasta = write(tmp_path / "p.fasta", ">a\nACGTACGT\n>b\nTTTTGGGG\n>c\nCCCCAAAA\n")
    out = tmp_path / "x.tsv"
    args = ["sketch", "--fasta", str(fasta), "--intensities", str(t), "--sample", "GSM1"]
    assert main(args + ["--k", "4", "--alphabet", "dna", "--mode", "present", "--out", str(out)]) == 0
    assert "rule: GEO detection call" in capsys.readouterr().out
    assert json.loads(Path(f"{out}.report.json").read_text())["probes_used"] == 1


def test_two_colour_needs_a_channel(tmp_path):
    columns = ["ID_REF", "ch1 Intensity", "ch1 Back", "ch2 Intensity", "ch2 Back", "VALUE"]
    t = sample_table(tmp_path / "GSM1.txt", [("a", 100, 5, 200, 6, 1.0)], columns, channels=2)
    with pytest.raises(Geo2KmersError, match="--channel"):
        sketch.load_intensities(t, "GSM1")
    values, _, source = sketch.load_intensities(t, "GSM1", "ch2")
    assert values == {"a": 200.0} and "ch2 Intensity" in source


def test_negative_channel_values_are_refused_too(tmp_path):
    t = sample_table(tmp_path / "GSM1.txt", [("a", -1.2, 3.0)], ["ID_REF", "ch1 Intensity", "VALUE"], channels=2)
    with pytest.raises(Geo2KmersError, match="log ratio"):
        sketch.load_intensities(t, "GSM1", "ch1")
    t = sample_table(tmp_path / "GSM2.txt", [("a", 3.0, -0.4)], ["ID_REF", "ch1 Intensity", "VALUE"], channels=2)
    with pytest.raises(Geo2KmersError, match="log ratio"):
        sketch.load_intensities(t, "GSM2", column="VALUE")


def test_channel_on_a_one_colour_matrix_is_refused(tmp_path):
    text = (
        '!Sample_channel_count\t"1"\n!series_matrix_table_begin\n"ID_REF"\t"GSM1"\n"a"\t300\n!series_matrix_table_end\n'
    )
    m = write(tmp_path / "GSE1_series_matrix.txt", text)
    with pytest.raises(Geo2KmersError, match="one-colour"):
        sketch.load_intensities(m, "GSM1", "ch1")
    assert sketch.load_intensities(m, "GSM1")[0] == {"a": 300.0}


def test_values_that_look_like_logarithms_print_a_warning(tmp_path, capsys):
    t = sample_table(tmp_path / "GSM1.txt", [("a", 7.1), ("b", 12.0)], ["ID_REF", "VALUE"])
    sketch.load_intensities(t, "GSM1")
    assert "logarithms" in capsys.readouterr().out


def test_numeric_detection_column_is_not_read_as_a_call(tmp_path):
    t = sample_table(tmp_path / "GSM1.txt", [("a", 900, 0.01), ("b", 5, 0.9)], ["ID_REF", "VALUE", "Detection"])
    assert sketch.load_intensities(t, "GSM1")[1] is None


def test_detection_p_value_limit_is_0_05(tmp_path):
    rows = [("a", 1, 0.05), ("b", 1, 0.051)]
    t = sample_table(tmp_path / "GSM1.txt", rows, ["ID_REF", "VALUE", "Detection Pval"])
    assert sketch.load_intensities(t, "GSM1")[1] == {"a": True, "b": False}


def test_k_too_long_for_every_probe_stops(tmp_path):
    with pytest.raises(Geo2KmersError, match="no probe gave a k-mer"):
        sketch.build_sketch({"p": "ACGTACGTACGTACGTACGTACGTA"}, {"p": 10.0, "q": 0.0}, 10, "protein", "weighted")


def test_zero_weight_kmers_stay_in_the_table_and_leave_the_signature(tmp_path):
    probes = {"low": "ACGTTGCAAC", "high": "GGGATTCCTA"}
    table, report = sketch.build_sketch(probes, {"low": 0.0, "high": 100.0}, 5, "dna", "weighted")
    sketch.write_sketch(table, report, tmp_path / "x.tsv", "GSM1", None)
    rows = [r.split("\t") for r in (tmp_path / "x.tsv").read_text().splitlines()[1:]]
    zero = {r[0] for r in rows if float(r[1]) == 0}
    assert zero and len(zero) < len(rows)
    minhash = next(iter(sourmash.load_file_as_signatures(str(tmp_path / "x.tsv.sig")))).minhash
    assert len(minhash) == len(rows) - len(zero)


def test_default_batch_is_200_and_the_key_shortens_the_pause():
    sleeps, sizes = [], []

    def post(ids, email, api_key):
        sizes.append(len(ids))
        return ""

    sequences.fetch_accessions([f"A{i}" for i in range(450)], "e", api_key="k", post=post, sleep=sleeps.append)
    assert sizes == [200, 200, 50]
    assert sleeps == [0.11, 0.11, 0.11]


def test_two_colour_series_matrix_is_refused(tmp_path):
    text = (
        '!Sample_channel_count\t"2"\n!series_matrix_table_begin\n"ID_REF"\t"GSM1"\n"a"\t0.3\n!series_matrix_table_end\n'
    )
    m = write(tmp_path / "GSE1_series_matrix.txt", text)
    with pytest.raises(Geo2KmersError, match="ratio"):
        sketch.load_intensities(m, "GSM1")


def test_negative_values_are_refused(tmp_path):
    t = sample_table(tmp_path / "GSM1.txt", [("a", -0.5), ("b", 1.0)], ["ID_REF", "VALUE"])
    with pytest.raises(Geo2KmersError, match="log ratio"):
        sketch.load_intensities(t, "GSM1")


def test_affymetrix_with_no_sequences_names_the_package(tmp_path):
    t = write(
        tmp_path / "GPL570.txt",
        "!Platform_title = [HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array\n"
        "!platform_table_begin\nID\tGene Symbol\n1007_s_at\tDDR1\n!platform_table_end\n",
    )
    with pytest.raises(Geo2KmersError, match="hgu133plus2probe"):
        sequences.make_probe_fasta(t, tmp_path / "o.fasta", tmp_path / "o.tsv")


def test_affymetrix_with_a_genbank_column_still_stops(tmp_path):
    t = write(
        tmp_path / "GPL570.txt",
        "!Platform_title = [HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array\n"
        "!platform_table_begin\nID\tGB_ACC\n1007_s_at\tU48705\n!platform_table_end\n",
    )
    with pytest.raises(Geo2KmersError, match="hgu133plus2probe"):
        sequences.make_probe_fasta(t, tmp_path / "o.fasta", tmp_path / "o.tsv")


def gpl(path):
    return write(
        path,
        "!platform_table_begin\nID\tGB_ACC\tSEQUENCE\n"
        "1\t\tacgtacgt\n2\tAJ000001.1\t\n3\tAJ000002\t\n4\t\t\n!platform_table_end\n",
    )


def test_sources_are_recorded_and_email_comes_from_the_environment(tmp_path, monkeypatch):
    t = gpl(tmp_path / "GPL1.txt")
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    with pytest.raises(Geo2KmersError, match="NCBI_EMAIL"):
        sequences.make_probe_fasta(t, tmp_path / "o.fasta", tmp_path / "o.tsv")
    monkeypatch.setenv("NCBI_EMAIL", "someone@example.org")
    seen = []

    def post(ids, email, api_key):
        seen.append(email)
        return ">AJ000001.1 a mussel EST\nACGTTGCA\n"

    counts = sequences.make_probe_fasta(t, tmp_path / "o.fasta", tmp_path / "o.tsv", post=post)
    assert counts == {"gpl_sequence": 1, "genbank_fetch": 1, "missing": 2}
    assert seen == ["someone@example.org"]
    assert (tmp_path / "o.tsv").read_text().splitlines()[1:] == [
        "1\tgpl_sequence\t8",
        "2\tgenbank_fetch\t8",
        "3\tmissing\t0",
        "4\tmissing\t0",
    ]


def test_batches_respect_the_rate_limit_and_a_bad_accession_loses_only_itself():
    sleeps, batches = [], []

    def post(ids, email, api_key):
        batches.append(list(ids))
        if "BAD" in ids:
            raise urllib.error.HTTPError("u", 400, "bad", None, io.BytesIO())
        return "".join(f">{i}.1 x\nACGT\n" for i in ids)

    ids = ["A", "B", "BAD", "C", "D"]
    found = sequences.fetch_accessions(ids, "e", post=post, sleep=sleeps.append, batch_size=4)
    assert set(found) == {"A", "B", "C", "D"}
    assert all(s >= 0.34 for s in sleeps) and len(sleeps) == len(batches)
    assert max(map(len, batches)) == 4


def test_fetch_records_provenance_and_uses_the_disk_cache(tmp_path):
    pages = {
        "acc=GSE1&": b"!Series_platform_id = GPL1\n!Series_sample_id = GSM1\n",
        "acc=GPL1&": b"!platform_table_begin\nID\tSEQUENCE\n1\tACGT\n!platform_table_end\n",
        "GSE1_series_matrix": None,
        "acc=GSM1&": b"!Sample_channel_count = 1\n!sample_table_begin\nID_REF\tVALUE\n1\t5\n!sample_table_end\n",
    }
    calls = []

    def opener(url, timeout):
        calls.append(url)
        for key, body in pages.items():
            if key in url:
                if body is None:
                    raise OSError("no such file")
                return _Closing(body)
        raise AssertionError(url)

    provenance = geo.fetch("GSE1", tmp_path, opener=opener)
    assert provenance["accession"] == "GSE1" and len(provenance["files"]) == 3
    assert all(len(f["sha256"]) == 64 for f in provenance["files"])
    assert json.loads((tmp_path / "provenance.json").read_text()) == provenance
    assert (tmp_path / "samples.tsv").read_text() == "gsm\tchannel_count\nGSM1\t1\n"
    first = len(calls)
    geo.fetch("GSE1", tmp_path, opener=opener)
    assert len(calls) == first + 1  # only the missing matrix is asked for again


class _Closing(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
