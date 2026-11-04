"""Checks on each rule the README states."""

import gzip
import io
import json
import random
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from geo2kmers import geo, schema, sequences, sketch
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


def test_longest_stretch_is_taken_from_any_frame_even_one_with_stops():
    protein = RANDOM_PROTEIN[:150]
    both = STOPS + sketch.reverse_complement(STOPS)  # a stop in all six frames
    dna = to_dna("MA") + both + to_dna(protein) + both
    pick = sketch.longest_stretch(dna)
    assert pick[2] == "L" + protein and pick[:2] == (0, 11)  # the last codon of the flank reads L
    assert "*" in sketch.translate_frames(dna)[pick[0]]


def test_longest_stretch_ties_go_to_the_lower_frame_then_the_earlier_stretch(monkeypatch):
    monkeypatch.setattr(sketch, "translate_frames", lambda seq: ["TT*VV", "WWW", "YYY"])
    assert sketch.longest_stretch("x")[:3] == (1, 0, "WWW")
    monkeypatch.setattr(sketch, "translate_frames", lambda seq: ["A*CC*GG", "TT"])
    assert sketch.longest_stretch("x")[:3] == (0, 2, "CC")


def test_only_the_longest_stretch_adds_kmers():
    protein = RANDOM_PROTEIN[:150]
    both = STOPS + sketch.reverse_complement(STOPS)  # a stop in all six frames
    dna = to_dna("MA") + both + to_dna(protein) + both
    table, report = sketch.build_sketch({"p": dna}, {"p": 10.0, "q": 1.0}, 3, "protein", "weighted")
    stretch = "L" + protein
    assert set(table) == {stretch[i : i + 3] for i in range(149)}
    assert report["frame_rule"] == "longest stretch between stops"


def test_probe_with_no_stretch_is_reported():
    probes = {"p": "AC", "open": "GCTGCTGCTGCTGCTGCT"}
    table, report = sketch.build_sketch(probes, {"p": 10.0, "open": 10.0, "q": 1.0}, 3, "protein", "weighted")
    assert report["probes_with_no_stretch"] == 1 and table


def test_a_k_longer_than_every_stretch_stops():
    with pytest.raises(Geo2KmersError, match="no probe gave a k-mer"):
        sketch.build_sketch({"p": "ACG"}, {"p": 10.0, "q": 1.0}, 3, "protein", "weighted")


@pytest.mark.parametrize("alphabet,k", [("dna", 15), ("protein", 5), ("hp", 9), ("dayhoff", 7)])
def test_signature_hashes_equal_sourmash_own_hashes(tmp_path, alphabet, k):
    seq = "ATGGCCATTGTAATGGGCCGCTGCAAAGGGTGCCCGACAGCATGGAGTTTCGCTCCGGAACTGTACAAA"
    table, _ = sketch.build_sketch({"p": seq}, {"p": 100.0, "q": 0.0}, k, alphabet, "weighted")
    assert sketch.write_signature(table, tmp_path / "x.sig", k, alphabet, "weighted", "x")
    got = next(iter(sourmash.load_file_as_signatures(str(tmp_path / "x.sig")))).minhash
    args = {"n": 0, "ksize": k, "scaled": 1}
    if alphabet != "dna":
        args[{"protein": "is_protein", "hp": "hp", "dayhoff": "dayhoff"}[alphabet]] = True
    own = sourmash.MinHash(**args)
    if alphabet == "dna":
        own.add_sequence(seq)
    else:
        own.add_protein(sketch.longest_stretch(seq)[2])
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
    assert [sketch.rounded_abundance(w) for w in (0.2, 0.5, 1.5, 2.49, 2.5, 4.5)] == [1, 1, 2, 2, 3, 5]


def test_weight_column_is_never_called_count(tmp_path):
    table, report = sketch.build_sketch({"p": "ACGTACGT"}, {"p": 10.0, "q": 0.0}, 3, "dna", "weighted")
    sketch.write_sketch(table, report, tmp_path / "x.tsv", "GSM1", None)
    header = (tmp_path / "x.tsv").read_text().splitlines()[0]
    assert header == "kmer\talphabet\tk\tsetting\tintensity_weight\tn_probes"


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
    zero = {r[0] for r in rows if float(r[4]) == 0}
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
    assert counts == {"gpl_sequence": 1, "genbank_fetch": 1, "record_too_long": 0, "missing": 2}
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
                    raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
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


def test_a_genbank_record_over_the_limit_is_not_a_probe(tmp_path, monkeypatch):
    t = tmp_path / "GPL.txt"
    t.write_text("!platform_table_begin\nID\tGB_ACC\n1\tAJ000001\n2\tAJ000002\n!platform_table_end\n")
    monkeypatch.setenv("NCBI_EMAIL", "someone@example.org")

    def post(ids, email, api_key):
        return ">AJ000001\nACGTACGT\n>AJ000002\nACGTACGTACGT\n"

    out, table = tmp_path / "o.fasta", tmp_path / "o.tsv"
    counts = sequences.make_probe_fasta(t, out, table, post=post, max_genbank_nt=8)
    assert counts == {"gpl_sequence": 0, "genbank_fetch": 1, "record_too_long": 1, "missing": 0}
    assert out.read_text() == ">1\nACGTACGT\n"
    assert "2\trecord_too_long\t12" in table.read_text()


# a fixed sequence of amino acids with no repeated 5-mer, and one codon per amino acid
RANDOM_PROTEIN = "".join(random.Random(7).choices(sketch.AMINO_ACIDS, k=400))
CODON_OF = {"A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT", "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA",
            "L": "CTT", "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAA", "R": "CGT", "S": "TCT", "T": "ACT", "V": "GTT",
            "W": "TGG", "Y": "TAT"}  # fmt: skip
STOPS = "TAACTAGTTAACTAG"  # a stop in each of the three forward frames


def to_dna(protein):
    return "".join(CODON_OF[a] for a in protein)


def write_proteome(tmp_path, *proteins):
    path = tmp_path / "proteome.fasta"
    path.write_text("".join(f">p{i}\n{p}\n" for i, p in enumerate(proteins)))
    return [path]


def test_random_protein_has_no_repeated_5mer():
    kmers = [RANDOM_PROTEIN[i : i + 5] for i in range(len(RANDOM_PROTEIN) - 4)]
    assert len(set(kmers)) == len(kmers)


@pytest.mark.parametrize(("length", "passes"), [(99, False), (100, True)])
def test_a_stretch_needs_100_amino_acids(tmp_path, length, passes):
    protein = RANDOM_PROTEIN[:length]
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, protein), 5, "protein")
    pick = picker.pick(STOPS + to_dna(protein) + STOPS)
    assert (pick is not None and pick[2] == protein) == passes


@pytest.mark.parametrize(("matching", "passes"), [(89, False), (90, True)])
def test_a_stretch_needs_90_percent_of_its_kmers(tmp_path, matching, passes):
    protein = RANDOM_PROTEIN[:104]  # 100 k-mers at k = 5
    assert len({protein[i : i + 5] for i in range(100)}) == 100
    known = protein[: matching + 4]
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, known), 5, "protein")
    assert (picker.pick(STOPS + to_dna(protein) + STOPS) is not None) == passes


def test_the_longest_passing_stretch_wins_and_shorter_ones_add_nothing(tmp_path):
    short, long = RANDOM_PROTEIN[:110], RANDOM_PROTEIN[120:300]
    dna = STOPS + to_dna(short) + STOPS + "A" + to_dna(long) + STOPS  # the long stretch is one frame over
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, short, long), 5, "protein")
    pick = picker.pick(dna)
    assert pick[2].endswith(long)  # the frame shift leaves a few amino acids before the long part
    frames = sketch.translate_frames(dna)
    assert frames[pick[0]][pick[1] : pick[1] + len(pick[2])] == pick[2]


def test_a_frame_with_a_stop_can_be_picked_and_the_reverse_strand_is_read(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    dna = sketch.reverse_complement(STOPS + to_dna(protein) + STOPS)
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, protein), 5, "protein")
    pick = picker.pick(dna)
    assert pick[2] == protein and pick[0] >= 3
    assert "*" in sketch.translate_frames(dna)[pick[0]]


def test_a_stretch_is_never_longer_than_the_part_between_stops(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    dna = to_dna(protein[:75]) + "TAA" + to_dna(protein[75:])
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, protein), 5, "protein")
    assert picker.pick(dna) is None


@pytest.mark.parametrize("alphabet", ["protein", "hp", "dayhoff"])
def test_sketch_holds_exactly_the_picked_stretch_and_the_sum_holds(tmp_path, alphabet):
    k = {"protein": 5, "hp": 20, "dayhoff": 8}[alphabet]
    coding, other = RANDOM_PROTEIN[:150], RANDOM_PROTEIN[200:350]
    probes = {"a": STOPS + to_dna(coding) + STOPS, "b": STOPS + to_dna(other) + STOPS, "c": "ACGT" * 3}
    table, report = sketch.build_sketch(
        probes, {"a": 900.0, "b": 50.0, "c": 70.0, "d": 90.0}, k, alphabet, "weighted",
        proteome=write_proteome(tmp_path, coding),
    )  # fmt: skip
    text = coding if alphabet == "protein" else "".join(sketch.REDUCED[alphabet][a] for a in coding)
    assert set(table) == {text[i : i + k] for i in range(len(text) - k + 1)}
    assert report["probes_with_no_stretch"] == 2
    assert report["sum_of_weight"] == pytest.approx(report["sum_of_probe_weight_times_kmers"])
    assert report["sum_of_weight"] == pytest.approx((900.0 - report["background"]) * (len(text) - k + 1))


def test_hp_mode_decides_with_hp_kmers(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    swapped = "".join({"A": "V", "D": "E"}.get(a, a) for a in protein)
    proteome = write_proteome(tmp_path, swapped)
    assert sketch.ProteomeFrames(proteome, 20, "hp").pick(STOPS + to_dna(protein) + STOPS)[2] == protein
    assert sketch.ProteomeFrames(proteome, 5, "protein").pick(STOPS + to_dna(protein) + STOPS) is None


def test_proteome_reads_gzip_and_several_files(tmp_path):
    protein = RANDOM_PROTEIN[:120]
    first = tmp_path / "a.fasta.gz"
    with gzip.open(first, "wt") as f:
        f.write(f">a\n{protein[:30]}\n{protein[30:60]}\n")
    second = tmp_path / "b.fasta"
    second.write_text(f">b\n{protein[60:]}\n")
    assert sketch.ProteomeFrames([first, second], 5, "protein").pick(STOPS + to_dna(protein) + STOPS)[2] == protein
    assert sketch.ProteomeFrames([first], 5, "protein").pick(STOPS + to_dna(protein) + STOPS) is None


def test_no_kmer_crosses_a_joint_between_proteins_or_an_unusual_letter(tmp_path):
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, "AAAAC", "DDDDD", "AAXAAUCC"), 5, "protein")
    numbers = sketch.kmer_numbers("AAAACDDDDD", 5, "protein", picker.table)
    assert picker.share_in_proteome("AAAAC") == 1.0
    assert (picker.known == np.unique(sketch.kmer_numbers("AAAAC#DDDDD", 5, "protein", picker.table))).all()
    assert len(numbers) == 6 and len(picker.known) == 2


@pytest.mark.parametrize(
    ("k", "alphabet", "phrase"),
    [
        (31, "dna", "does not apply"),
        (15, "protein", "up to k = 14"),
        (64, "hp", "up to k = 63"),
        (25, "dayhoff", "up to k = 24"),
    ],
)
def test_proteome_refuses_what_it_cannot_do(tmp_path, k, alphabet, phrase):
    with pytest.raises(Geo2KmersError, match=phrase):
        sketch.build_sketch(
            {"a": "ACGT"}, {"a": 1.0}, k, alphabet, "weighted", proteome=write_proteome(tmp_path, "MKV")
        )


def test_a_proteome_with_no_kmer_stops(tmp_path):
    with pytest.raises(Geo2KmersError, match="no k-mer"):
        sketch.ProteomeFrames(write_proteome(tmp_path, "MKV"), 5, "protein")


def test_frames_file_names_the_pick(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    table, report = sketch.build_sketch(
        {"a": STOPS + to_dna(protein) + STOPS}, {"a": 30.0, "b": 10.0}, 5, "protein", "weighted",
        proteome=write_proteome(tmp_path, protein),
    )  # fmt: skip
    out = tmp_path / "s.tsv"
    sketch.write_sketch(table, report, out, "GSM1", None)
    lines = (tmp_path / "s.tsv.frames.tsv").read_text().splitlines()
    assert lines[1] == "a\t0\t5\t150\t1.000"
    assert "picked_stretches" not in json.loads((tmp_path / "s.tsv.report.json").read_text())
    assert "proteome.fasta" in json.loads((tmp_path / "s.tsv.report.json").read_text())["proteome"]


def test_dayhoff_mode_decides_with_dayhoff_kmers(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    swapped = "".join({"A": "G", "D": "E"}.get(a, a) for a in protein)  # same group each time
    proteome = write_proteome(tmp_path, swapped)
    assert sketch.ProteomeFrames(proteome, 15, "dayhoff").pick(STOPS + to_dna(protein) + STOPS)[2] == protein
    assert sketch.ProteomeFrames(proteome, 5, "protein").pick(STOPS + to_dna(protein) + STOPS) is None


def test_frames_file_leaves_the_share_empty_with_no_proteome(tmp_path):
    protein = RANDOM_PROTEIN[:150]
    table, report = sketch.build_sketch({"a": to_dna(protein)}, {"a": 30.0, "b": 10.0}, 5, "dayhoff", "weighted")
    out = tmp_path / "s.tsv"
    sketch.write_sketch(table, report, out, "GSM1", None)
    assert (tmp_path / "s.tsv.frames.tsv").read_text().splitlines()[1] == "a\t0\t0\t150\t"
    assert next(iter(sourmash.load_file_as_signatures(f"{out}.sig"))).minhash.dayhoff


def test_dayhoff_groups_are_complete():
    assert sorted(sketch.DAYHOFF) == sorted(sketch.AMINO_ACIDS)
    assert sketch.letters("dayhoff") == 6


def test_the_largest_k_keeps_every_letter_of_a_kmer():
    table = sketch.letter_codes("hp")
    one, other = "A" * 63, "C" + "A" * 62  # they differ only in the first letter
    assert sketch.kmer_numbers(one, 63, "hp", table)[0] != sketch.kmer_numbers(other, 63, "hp", table)[0]


def test_fetch_stops_if_the_matrix_fails_for_any_reason_but_404(tmp_path):
    pages = {
        "acc=GSE1&": b"!Series_platform_id = GPL1\n!Series_sample_id = GSM1\n",
        "acc=GPL1&": b"!platform_table_begin\nID\tSEQUENCE\n1\tACGT\n!platform_table_end\n",
    }

    def opener(url, timeout):
        for key, body in pages.items():
            if key in url:
                return _Closing(body)
        raise urllib.error.HTTPError(url, 503, "Busy", None, None)

    with pytest.raises(Geo2KmersError, match="could not download"):
        geo.fetch("GSE1", tmp_path, opener=opener)


def test_the_default_genbank_limit_is_5000():
    assert sequences.MAX_GENBANK_NT == 5000


def test_a_value_just_under_the_log_limit_warns_and_one_at_it_does_not(capsys):
    sketch.check_not_ratio({"a": 31.9}, "GSM1")
    assert "logarithms" in capsys.readouterr().out
    sketch.check_not_ratio({"a": 32.0}, "GSM1")
    assert capsys.readouterr().out == ""


def test_the_negative_value_message_names_both_causes():
    with pytest.raises(Geo2KmersError, match="log ratio.*Average Difference"):
        sketch.check_not_ratio({"a": -1.0}, "GSM1")


def test_affymetrix_message_says_the_package_name_is_not_checked(tmp_path):
    t = write(
        tmp_path / "GPL6244.txt",
        "!Platform_title = [HuGene-1_0-st] Affymetrix Human Gene 1.0 ST Array\n"
        "!platform_table_begin\nID\tGene Symbol\n1\tDDR1\n!platform_table_end\n",
    )
    with pytest.raises(Geo2KmersError, match="not checked"):
        sequences.make_probe_fasta(t, tmp_path / "o.fasta", tmp_path / "o.tsv")


def test_x_does_not_count_towards_the_100_amino_acids(tmp_path):
    protein = RANDOM_PROTEIN[:20]
    picker = sketch.ProteomeFrames(write_proteome(tmp_path, protein), 10, "protein")
    assert picker.pick(STOPS + to_dna(protein) + "NNN" * 80 + STOPS) is None


def test_no_kmer_of_the_longest_stretch_crosses_an_x():
    protein = RANDOM_PROTEIN[:40]
    dna = STOPS + to_dna(protein[:20]) + "NNN" + to_dna(protein[20:]) + STOPS
    table, _ = sketch.build_sketch({"p": dna}, {"p": 10.0, "q": 0.0}, 5, "protein", "weighted")
    assert not any("X" in kmer for kmer in table)
    assert protein[18:23] not in table


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_1_stops(k):
    with pytest.raises(Geo2KmersError, match="at least 1"):
        sketch.build_sketch({"p": "ACGTACGT"}, {"p": 10.0}, k, "dna", "weighted")


def test_a_name_twice_in_a_fasta_stops_and_a_blank_line_is_skipped(tmp_path):
    assert sketch.read_fasta(write(tmp_path / "a.fasta", "\n>a\nAC\n\nGT\n")) == {"a": "ACGT"}
    with pytest.raises(Geo2KmersError, match="twice"):
        sketch.read_fasta(write(tmp_path / "b.fasta", ">a\nAC\n>a\nGT\n"))


def test_an_all_blank_detection_column_is_no_detection_column():
    rows = [{"ID_REF": "a", "ABS_CALL": ""}, {"ID_REF": "b", "ABS_CALL": " "}]
    assert sketch.read_detection(["ID_REF", "ABS_CALL"], rows) is None


GOOD_TABLE = "kmer\talphabet\tk\tsetting\tintensity_weight\tn_probes\nACD\tprotein\t3\tprotein20_k3\t1.5\t2\n"


def test_a_written_table_follows_the_schema(tmp_path):
    table, report = sketch.build_sketch({"p": "ACGTACGT"}, {"p": 10.0, "q": 0.0}, 3, "dna", "weighted")
    sketch.write_sketch(table, report, tmp_path / "x.tsv", "GSM1", None)
    assert schema.validate_table(tmp_path / "x.tsv") == ("dna_k3", len(table))
    table, report = sketch.build_sketch({"p": "ACGTACGT"}, {"p": 10.0, "q": 0.0}, 3, "dna", "present")
    sketch.write_sketch(table, report, tmp_path / "y.tsv", "GSM1", None)
    assert schema.validate_table(tmp_path / "y.tsv")[0] == "dna_k3"


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("intensity_weight", "weight", "does not allow"),
        ("kmer\t", "sequence\t", "does not allow"),
        ("setting", "name", "does not allow"),
        ("\tintensity_weight\t", "\tintensity_weight\tcount\t", "mixes the value columns"),
        ("protein20_k3", "protein20_k4", "does not match"),
        ("ACD", "ACB", "not a protein k-mer"),
        ("ACD", "ACDE", "not a protein k-mer"),
        ("\t1.5\t", "\t-1.5\t", "not allowed"),
    ],
)
def test_the_schema_refuses_a_changed_table(tmp_path, old, new, message):
    path = write(tmp_path / "x.tsv", GOOD_TABLE.replace(old, new))
    with pytest.raises(Geo2KmersError, match=message):
        schema.validate_table(path)


def test_the_schema_refuses_two_settings_in_one_table(tmp_path):
    extra = "ACDE\tprotein\t4\tprotein20_k4\t1.5\t2\n"
    with pytest.raises(Geo2KmersError, match="mixes the settings"):
        schema.validate_table(write(tmp_path / "x.tsv", GOOD_TABLE + extra))


def test_a_read_sketch_uses_count_and_needs_whole_numbers(tmp_path):
    good = "kmer\talphabet\tk\tsetting\tcount\tn_probes\nACD\tprotein\t3\tprotein20_k3\t4\t1\n"
    assert schema.validate_table(write(tmp_path / "a.tsv", good)) == ("protein20_k3", 1)
    with pytest.raises(Geo2KmersError, match="not allowed"):
        schema.validate_table(write(tmp_path / "b.tsv", good.replace("\t4\t", "\t4.5\t")))


def test_a_saved_proteome_gives_the_same_picks_and_checks_its_setting(tmp_path):
    protein = RANDOM_PROTEIN[:120]
    paths = write_proteome(tmp_path, protein)
    picker = sketch.ProteomeFrames(paths, 5, "protein")
    picker.save(tmp_path / "p.npz")
    loaded = sketch.ProteomeFrames([tmp_path / "p.npz"], 5, "protein")
    assert loaded.sources == ["proteome.fasta"]
    dna = STOPS + to_dna(protein) + STOPS
    assert loaded.pick(dna) == picker.pick(dna)
    with pytest.raises(Geo2KmersError, match="holds protein k = 5"):
        sketch.ProteomeFrames([tmp_path / "p.npz"], 6, "protein")
    with pytest.raises(Geo2KmersError, match="holds protein k = 5"):
        sketch.ProteomeFrames([tmp_path / "p.npz"], 5, "hp")
