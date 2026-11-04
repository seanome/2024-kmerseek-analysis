"""Known-answer checks on real mussel sequences from GenBank (see tests/README.md)."""

import gzip
import shutil
from pathlib import Path

import pytest

from geo2kmers.cli import main
from geo2kmers.sketch import read_fasta

DATA = Path(__file__).parent / "data"
K = 21


def hand_canonical_kmers(seq, k):
    """Written apart from the package on purpose, so a bug there cannot hide here."""
    pairs = {"A": "T", "C": "G", "G": "C", "T": "A"}
    out = set()
    for i in range(len(seq) - k + 1):
        word = seq[i : i + k]
        back = "".join(pairs[c] for c in reversed(word))
        out.add(min(word, back))
    return out


def write_sample(path, values):
    with open(path, "w") as f:
        f.write("!Sample_channel_count = 1\n!sample_table_begin\nID_REF\tVALUE\n")
        f.writelines(f"{probe}\t{value}\n" for probe, value in values.items())
        f.write("!sample_table_end\n")


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return header, [line.rstrip("\n").split("\t") for line in f]


@pytest.fixture
def known(tmp_path):
    """50 real probes. In this made-up sample the first 20 are bright and the other 30 are dim."""
    fasta = DATA / "known_answer_probes.fasta"
    probes = read_fasta(fasta)
    names = list(probes)
    assert len(names) == 50
    bright, dim = names[:20], names[20:]
    sample = tmp_path / "GSM1.txt"
    write_sample(sample, {**{p: 5000 for p in bright}, **{p: 100 for p in dim}})
    return fasta, sample, probes, bright, dim


def test_sketch_holds_exactly_the_kmers_of_the_20_bright_probes(known, tmp_path, capsys):
    fasta, sample, probes, bright, dim = known
    out = tmp_path / "s.tsv"
    args = ["sketch", "--fasta", str(fasta), "--intensities", str(sample), "--sample", "GSM1"]
    assert main(args + ["--k", str(K), "--alphabet", "dna", "--mode", "present", "--out", str(out)]) == 0
    assert "intensity above the median" in capsys.readouterr().out
    header, rows = read_tsv(out)
    assert header == ["kmer", "presence", "n_probes"]
    expected = set().union(*(hand_canonical_kmers(probes[p], K) for p in bright))
    assert {r[0] for r in rows} == expected
    # the test only means something if the dim probes hold k-mers the bright ones lack
    assert set().union(*(hand_canonical_kmers(probes[p], K) for p in dim)) - expected


def test_weight_sum_equals_probe_weight_times_kmers_on_a_real_array(tmp_path):
    """Counted from the written table and from the raw GEO table, not from the code's report."""
    with gzip.open(DATA / "GSE2183" / "probes.fasta.gz", "rb") as a, open(tmp_path / "p.fasta", "wb") as b:
        shutil.copyfileobj(a, b)
    with gzip.open(DATA / "GSE2183" / "GSM39421.txt.gz", "rb") as a, open(tmp_path / "GSM39421.txt", "wb") as b:
        shutil.copyfileobj(a, b)
    out = tmp_path / "s.tsv"
    args = ["sketch", "--fasta", str(tmp_path / "p.fasta"), "--intensities", str(tmp_path / "GSM39421.txt")]
    args += ["--sample", "GSM39421", "--channel", "ch2", "--k", str(K), "--alphabet", "dna", "--mode", "weighted"]
    assert main(args + ["--out", str(out)]) == 0
    header, rows = read_tsv(out)
    assert header == ["kmer", "intensity_weight", "n_probes"]
    observed = sum(float(r[1]) for r in rows)

    intensity = {}
    with open(tmp_path / "GSM39421.txt") as f:
        for line in f:
            if line.startswith("!sample_table_begin"):
                columns = next(f).rstrip("\n").split("\t")
                i = columns.index("ch2 Intensity")
                for row in f:
                    if row.startswith("!"):
                        break
                    cells = row.rstrip("\n").split("\t")
                    intensity[cells[0]] = float(cells[i])
    ordered = sorted(intensity.values())
    position = 0.05 * (len(ordered) - 1)
    low = int(position)
    background = ordered[low] + (position - low) * (ordered[low + 1] - ordered[low])

    probes = read_fasta(tmp_path / "p.fasta")
    expected = 0.0
    for probe, seq in probes.items():
        weight = max(0.0, intensity[probe] - background)
        windows = sum(max(0, len(piece) - K + 1) for piece in "".join(c if c in "ACGT" else " " for c in seq).split())
        expected += weight * windows
    assert observed == pytest.approx(expected, rel=1e-6)
    assert expected > 0
