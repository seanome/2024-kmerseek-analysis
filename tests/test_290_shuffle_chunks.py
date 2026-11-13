"""Tests for scripts/290_shuffle_control/shuffle_chunks.py (notebook 290)."""

import hashlib
import importlib.util
import random
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shuffle_chunks", ROOT / "scripts/290_shuffle_control/shuffle_chunks.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

# Realistic test proteins: the first five dipeptide copies of BHF from notebook 279, plus a
# poly-Q run, a hydrophobic stretch, an X, and the shortest lengths.
BHF_COPIES = [s for _, s in sc.read_fasta(ROOT / "data/279_bhf_dipeptide_shuffles.fasta")][:5]
EDGE = ["M", "MK", "MKV", "MQQQQQQQQQQQQQQQQQQQQQQRS", "MALLVLLAILVAFLLGIVSAKRDE", "MKXXRAXE" * 4]
PROTEINS = BHF_COPIES + EDGE


def pair_counts(s):
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


@pytest.mark.parametrize("seq", PROTEINS)
def test_dipeptide_keeps_pair_counts_and_ends(seq):
    out = sc.shuffle_one("dipeptide", "P1", seq)
    assert pair_counts(out) == pair_counts(seq)
    assert Counter(out) == Counter(seq)
    assert out[:1] == seq[:1] and out[-1:] == seq[-1:]


@pytest.mark.parametrize("seq", PROTEINS)
def test_window_keeps_each_window(seq):
    out = sc.shuffle_one("window10", "P1", seq)
    assert len(out) == len(seq)
    for i in range(0, len(seq), 10):
        assert Counter(out[i : i + 10]) == Counter(seq[i : i + 10])


def test_long_proteins_change():
    for seq in BHF_COPIES:
        assert sc.shuffle_one("dipeptide", "P1", seq) != seq
        assert sc.shuffle_one("window10", "P1", seq) != seq


def test_seed_depends_on_name_and_kind_only():
    seq = BHF_COPIES[0]
    assert sc.shuffle_one("dipeptide", "A", seq) == sc.shuffle_one("dipeptide", "A", seq)
    assert sc.shuffle_one("dipeptide", "A", seq) != sc.shuffle_one("dipeptide", "B", seq)


def test_dipeptide_on_random_sequences():
    rng = random.Random(1)
    for n in range(3, 400, 7):
        seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n))
        out = sc.shuffle_one("dipeptide", "R%d" % n, seq)
        assert pair_counts(out) == pair_counts(seq)


def test_chunk_keeps_order_names_and_refuses_overwrite(tmp_path):
    chunk = tmp_path / "chunk_0000.fasta"
    chunk.write_text("".join(">P%d some description\n%s\n" % (i, s) for i, s in enumerate(PROTEINS)))
    md5 = hashlib.md5(chunk.read_bytes()).hexdigest()
    tsv = tmp_path / "chunks.tsv"
    tsv.write_text("species\tchunk\tpath\tmd5\nhuman\tchunk_0000\t%s\t%s\n" % (chunk, md5))
    out = tmp_path / "out"
    sc.main(str(tsv), str(out))
    for kind, suffix in sc.KINDS.items():
        recs = sc.read_fasta(out / "human" / ("chunk_0000.%s.fasta" % kind))
        assert [n for n, _ in recs] == ["P%d%s" % (i, suffix) for i in range(len(PROTEINS))]
    report = (out / "shuffles.tsv").read_text().splitlines()
    assert len(report) == 3
    # Run again: same content, accepted. Different content: refused.
    sc.main(str(tsv), str(out))
    (out / "human" / "chunk_0000.dipeptide.fasta").write_text(">x\nM\n")
    with pytest.raises(SystemExit):
        sc.main(str(tsv), str(out))
    # A changed input chunk is refused.
    chunk.write_text(">P0\nMKV\n")
    with pytest.raises(SystemExit):
        sc.main(str(tsv), str(out))
