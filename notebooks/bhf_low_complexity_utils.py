"""BHF's low-complexity stretches, found with kmerseek's own mask rule, and the copies of
BHF that keep or remove them (notebook 283).

kmerseek's low-complexity mask (`--remove-low-complexity`, on for every index of notebooks
241, 242 and 256) is not SEG. At kmerseek 982a055, the build those notebooks used, it drops
one k-mer at a time (`src/rust/sketch.rs`, `src/rust/kmer.rs::is_homopolymer_kmer`):

1. if its k residues are all the same amino acid (a raw run such as KKKKK), for every
   alphabet; or
2. if its k letters, after encoding with the alphabet, are all the same class, for every
   alphabet that has a class table in `src/rust/alphabets.rs` (all but protein20 and
   dayhoff6), and for hp_lehninger2.

A stretch, for one alphabet and k, is a run of residues covered by masked k-mers: the union
of every masked window, with windows that touch or overlap merged. So the stretches depend
on the setting (alphabet and k), and this module computes them per setting.

The class tables are read from the kmerseek source file itself, not typed in here, and
`check_against_binary` compares the number of masked windows with the count the kmerseek
binary prints when it indexes BHF alone.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

# kmerseek at 982a055: the source the class tables are read from, and the binary built from it.
KMERSEEK_SRC = Path(os.environ.get("KMERSEEK_SRC", "/Users/olga/code/kmerseek-ka-lambda-region"))
KMERSEEK = KMERSEEK_SRC / "target/release/kmerseek"

# Alphabet name in kmerseek -> the name of its table in alphabets.rs (Alphabet::partition).
_TABLE_NAME = {
    "hp_lehninger2": "LEHNINGER_HP",
    "hp_thomas_dill2": "THOMAS_DILL_HP",
    "hp_kyte_doolittle2": "KYTE_DOOLITTLE_HP",
    "hp_thomas_dill_no_c2": "THOMAS_DILL_NO_C_HP",
    "hp_lehninger_c_nonpolar2": "LEHNINGER_C_NONPOLAR_HP",
    "hp_lehninger_hpc3": "LEHNINGER_HPC",
    "hp_pbotc_1st_ed2": "PBOTC_1ST_ED_HP",
    "gbmr4": "GBMR4",
    "polarity4": "POLARITY4",
    "wwmj5": "WWMJ5",
    "gbmr7": "GBMR7",
    "funcgroups8": "FUNCGROUPS8",
    "sdm12": "SDM12",
    "mmseqs12": "MMSEQS12",
    "wass14": "WASS14",
    "hsdm17": "HSDM17",
    "uniprot18": "UNIPROT18",
}
# No class check: kmerseek hands these to sourmash's own encoder and only checks raw runs.
RAW_ONLY = {"protein20", "dayhoff6"}


def alphabet_classes(src: Path = KMERSEEK_SRC / "src/rust/alphabets.rs") -> dict[str, dict[str, int]]:
    """Residue -> class number for every alphabet with a class table, read from kmerseek's
    alphabets.rs. Two residues share a number when the alphabet puts them in one class."""
    text = src.read_text()
    out = {}
    for alphabet, name in _TABLE_NAME.items():
        m = re.search(rf"static {name}: LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| (\w+)\(([^;]*?)\)\);", text)
        if not m:
            raise ValueError(f"table {name} not found in {src}")
        builder, args = m.group(1), m.group(2)
        if builder in ("build_hp", "build_hpc"):
            groups = re.findall(r'b"([A-Z]+)"', args)
        elif builder == "build_reduced":
            consts = re.search(rf"const {args.strip()}: &\[&str\] =\s*&\[(.*?)\];", text, re.S)
            groups = re.findall(r'"([A-Z]+)"', consts.group(1))
        else:
            raise ValueError(f"unknown table builder {builder} for {name}")
        residues = "".join(groups)
        if sorted(residues) != sorted("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"{name}: classes do not cover the 20 amino acids once each: {groups}")
        out[alphabet] = {aa: i for i, g in enumerate(groups) for aa in g}
    return out


CLASSES = alphabet_classes()


def masked_windows(seq: str, alphabet: str, k: int) -> list[int]:
    """0-based start of every k-mer of `seq` that kmerseek's mask drops for this setting."""
    table = None if alphabet in RAW_ONLY else CLASSES[alphabet]
    starts = []
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        if len(set(w)) == 1 or (table is not None and len({table[c] for c in w}) == 1):
            starts.append(i)
    return starts


def stretches(seq: str, alphabet: str, k: int) -> list[tuple[int, int]]:
    """The residues covered by masked k-mers, as merged (start, end) intervals, 0-based and
    end-exclusive. Two windows that overlap or touch end to end make one stretch."""
    out: list[list[int]] = []
    for s in masked_windows(seq, alphabet, k):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s + k)
        else:
            out.append([s, s + k])
    return [(a, b) for a, b in out]


def check_against_binary(seq: str, alphabet: str, k: int) -> tuple[int, int]:
    """(masked windows here, k-mers the kmerseek binary reports removed) for `seq` alone."""
    with tempfile.TemporaryDirectory() as d:
        fa = Path(d) / "q.fa"
        fa.write_text(f">q\n{seq}\n")
        r = subprocess.run([str(KMERSEEK), "index", "-i", str(fa), "-o", str(Path(d) / "q.rocksdb"),
                            "-k", str(k), "-s", "1", "-a", alphabet, "--remove-low-complexity",
                            "--extend-mismatch-penalty", "0"],
                           capture_output=True, text=True)
        m = re.search(r"Removed (\d+) of (\d+) k-mer windows as low-complexity", r.stderr + r.stdout)
        if r.returncode != 0 or not m:
            raise RuntimeError(f"{alphabet} k={k}: rc={r.returncode}\n{r.stderr[-2000:]}")
        removed = int(m.group(1).replace(",", ""))
    return len(masked_windows(seq, alphabet, k)), removed


def _segments(n: int, keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The intervals of [0, n) between the kept stretches."""
    out, pos = [], 0
    for a, b in keep:
        if a > pos:
            out.append((pos, a))
        pos = b
    if pos < n:
        out.append((pos, n))
    return out


def stretches_kept(seq: str, keep: list[tuple[int, int]], rng: random.Random, shuffle) -> str:
    """A copy of `seq` with every stretch in `keep` verbatim at its own position and each
    segment between stretches dipeptide-shuffled on its own (`shuffle(segment, rng)`)."""
    pieces = sorted([(a, b, True) for a, b in keep] + [(a, b, False) for a, b in _segments(len(seq), keep)])
    out = "".join(seq[a:b] if kept else shuffle(seq[a:b], rng) for a, b, kept in pieces)
    assert len(out) == len(seq), (len(out), len(seq))
    for a, b in keep:
        assert out[a:b] == seq[a:b]
    return out


def stretches_removed(seq: str, keep: list[tuple[int, int]]) -> str:
    """`seq` with every stretch deleted and the pieces between them joined."""
    return "".join(seq[a:b] for a, b in _segments(len(seq), keep))
