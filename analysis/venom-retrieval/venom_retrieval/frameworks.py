"""Cysteine frameworks, inter-cysteine loops, and disulfide connectivity.

A cysteine framework is the pattern of cysteine positions in a peptide, written
``C-C-CC-C-C``: one ``C`` per cysteine, runs of adjacent cysteines kept together, and a
dash for one or more other residues in between. Venom-peptide families are named by it.

An inter-cysteine loop is the stretch of residues between one cysteine and the next. The
loop lengths are the framework in numbers: ``[6, 6, 0, 3, 4]`` for ``C-C-CC-C-C``.

Disulfide connectivity is which cysteine is bonded to which, written by cysteine index
rather than residue position: ``1-4,2-5,3-6`` for the inhibitor cystine knot. Connectivity
is a measurement, not something the cysteine order implies, so nothing here derives it
from a framework. `connectivity_from_pairs` only translates measured residue positions
into cysteine indices.

The framework string and loop-length functions come from `notebooks/cys_framework_utils.py`,
which the cys-frameworks pipeline already uses, so the two cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .shared import cys_framework_utils

framework_string = cys_framework_utils.framework_string
loop_lengths = cys_framework_utils.cysteine_spacers
framework_regex = cys_framework_utils.framework_regex


def cysteine_positions(seq: str) -> list[int]:
    """The 0-based index of every cysteine in `seq`.

    >>> cysteine_positions("ACDCE")
    [1, 3]
    """
    return [i for i, residue in enumerate(seq) if residue == "C"]


@dataclass(frozen=True)
class LoopLayout:
    """A peptide cut at its cysteines.

    `nterm` is everything before the first cysteine and `cterm` everything after the last
    one; both can be empty. `loops` holds the stretch between each pair of consecutive
    cysteines, in order, so a peptide with n cysteines has n - 1 of them, and a loop
    between two adjacent cysteines is the empty string.

    The cysteines themselves are not in any of these strings: they are the anchors, and
    every method that uses this layout treats them as matched rather than scoring them.
    """

    nterm: str
    loops: tuple[str, ...]
    cterm: str
    n_cysteines: int

    @property
    def loop_lengths(self) -> tuple[int, ...]:
        return tuple(len(loop) for loop in self.loops)


def loop_layout(seq: str) -> LoopLayout:
    """Cut `seq` at its cysteines into terminal segments and inter-cysteine loops.

    >>> layout = loop_layout("AACGGCTT")
    >>> layout.nterm, layout.loops, layout.cterm
    ('AA', ('GG',), 'TT')
    >>> loop_layout("CC").loops
    ('',)
    """
    positions = cysteine_positions(seq)
    if not positions:
        return LoopLayout(nterm=seq, loops=(), cterm="", n_cysteines=0)
    loops = tuple(
        seq[a + 1 : b] for a, b in zip(positions, positions[1:])
    )
    return LoopLayout(
        nterm=seq[: positions[0]],
        loops=loops,
        cterm=seq[positions[-1] + 1 :],
        n_cysteines=len(positions),
    )


def frameworks_compatible(
    query: str, target: str, max_loop_difference: int | None = None
) -> bool:
    """Whether two peptides can be compared loop by loop.

    True when both hold the same number of cysteines, so the cysteines line up one to one
    in order and loop i of one corresponds to loop i of the other. With
    `max_loop_difference` set, every corresponding pair of loops must also be within that
    many residues of each other.

    Equal cysteine counts and an ordered correspondence are all this asks for. It says
    nothing about which cysteines are bonded: two peptides with the same framework can
    have different connectivity, and this returns True for them.

    >>> frameworks_compatible("CACAC", "CAACAAC")
    True
    >>> frameworks_compatible("CACAC", "CAACAAC", max_loop_difference=0)
    False
    >>> frameworks_compatible("CAC", "CACAC")
    False
    """
    a, b = loop_layout(query), loop_layout(target)
    if a.n_cysteines != b.n_cysteines or a.n_cysteines == 0:
        return False
    if max_loop_difference is None:
        return True
    return all(
        abs(x - y) <= max_loop_difference
        for x, y in zip(a.loop_lengths, b.loop_lengths)
    )


_DISULFID = re.compile(r"DISULFID\s+(\d+)\.\.(\d+)")

#: ECO codes that mean a human read a measurement: experiment (269), a solved structure
#: (312 where the source is a PDB entry) and large-scale experimental data (7744.)
#: Everything else on a UniProt disulfide feature is an inference: 250 is "by similarity",
#: 255 is sequence analysis, 305 is curator reasoning, 303 is a non-traceable statement.
EXPERIMENTAL_ECO = ("ECO:0000269", "ECO:0007744", "ECO:0000312")


def disulfide_evidence(feature_text: str) -> str:
    """Whether a UniProt disulfide-bond column was measured, inferred, or is absent.

    Returns ``"experimental"`` if any bond in the column carries an experimental ECO code,
    ``"inferred"`` if bonds are present but all of their evidence is inference, and
    ``"none"`` when the column holds no disulfide feature at all.

    One verdict for the whole column rather than per bond: a connectivity pattern is only
    as good as its weakest bond, and mixing a measured bond with an inferred one into one
    pattern would make the pattern look measured.

    >>> disulfide_evidence('DISULFID 1..4; /evidence="ECO:0000269|PubMed:1"')
    'experimental'
    >>> disulfide_evidence('DISULFID 1..4; /evidence="ECO:0000250|UniProtKB:P1"')
    'inferred'
    >>> disulfide_evidence("")
    'none'
    """
    if not feature_text or "DISULFID" not in feature_text:
        return "none"
    if any(code in feature_text for code in EXPERIMENTAL_ECO):
        return "experimental"
    return "inferred"


def connectivity_from_pairs(
    feature_text: str, precursor: str, offset: int = 0, length: int | None = None
) -> str | None:
    """Translate measured disulfide positions into a connectivity string.

    `feature_text` is UniProt's disulfide-bond column, whose positions are 1-based in the
    precursor. `offset` is the 0-based start of the mature peptide inside that precursor
    and `length` its length; bonds with either end outside the mature peptide are dropped,
    because a bond to a cysteine that processing removes is not part of this peptide's
    connectivity.

    The result names cysteines by their order within the mature peptide, 1-based, each
    bond as ``i-j`` with i < j, bonds sorted: ``1-4,2-5,3-6``. Returns None when no bond
    survives, and raises nothing for a position that is not a cysteine -- that entry is
    reported by the manifest as a mismatch rather than silently renumbered.

    The cysteines of ``CARCDTCK`` are at 1-based 1, 4 and 7, so a bond 1..7 joins the
    first and third, and a bond 2..5 touches no cysteine and is dropped.

    >>> connectivity_from_pairs('DISULFID 1..7; DISULFID 2..5', "CARCDTCK", 0, 8)
    '1-3'
    """
    pairs = _DISULFID.findall(feature_text or "")
    if not pairs:
        return None
    end = offset + (length if length is not None else len(precursor) - offset)
    mature = precursor[offset:end]
    index_of = {position: i for i, position in enumerate(cysteine_positions(mature), 1)}
    bonds = set()
    for low, high in pairs:
        a, b = int(low) - 1 - offset, int(high) - 1 - offset
        if a in index_of and b in index_of:
            bonds.add(tuple(sorted((index_of[a], index_of[b]))))
    if not bonds:
        return None
    return ",".join(f"{a}-{b}" for a, b in sorted(bonds))


def connectivity_is_complete(connectivity: str | None, n_cysteines: int) -> bool:
    """Whether every cysteine in the peptide appears in exactly one bond.

    A partial pattern is not a fold label: half the bonds of a cystine knot are also half
    the bonds of several other arrangements.

    >>> connectivity_is_complete("1-4,2-5,3-6", 6)
    True
    >>> connectivity_is_complete("1-4,2-5", 6)
    False
    """
    if not connectivity or n_cysteines == 0 or n_cysteines % 2:
        return False
    seen = [int(part) for bond in connectivity.split(",") for part in bond.split("-")]
    return sorted(seen) == list(range(1, n_cysteines + 1))
