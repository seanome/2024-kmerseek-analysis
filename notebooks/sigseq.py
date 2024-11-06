"""
File for managing SourmashSignatures and their associated sequences
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rich.console import Console
from rich.theme import Theme
from rich.highlighter import RegexHighlighter
from sourmash import SourmashSignature
from sig2kmer import degenerate_protein_chatgpt


class FastaHeaderHighlighter(RegexHighlighter):
    """Apply style to FASTA headers, ID and description"""

    base_style = "fasta."
    highlights = [r"^(?P<id>\S+)\s+(?P<description>.+)$"]


custom_theme = Theme(
    {
        "overlap": "bold white on blue",
        "fasta.id": "bold magenta",
        "fasta.description": "white",
    }
)


def get_to_add(encoded, protein, use_encoded):
    if use_encoded:
        to_add = encoded
    else:
        to_add = protein
    return to_add


def add_to_stitched(stitched, prev_i, to_add):
    if prev_i == None:
        stitched = to_add
    else:
        stitched += to_add[-1]
    return stitched


def stitch_kmers(overlap, use_encoded=False):
    """
    use_encoded: whether or not to use the encoded protein sequences, e.g. the HP version and not the original protein
    """

    prev_i = None

    stitched = ""
    for i, encoded, protein, hashval in overlap:
        if prev_i == None or i == prev_i + 1:
            # if first or sequential
            to_add = get_to_add(encoded, protein, use_encoded)
            stitched = add_to_stitched(stitched, prev_i, to_add)
            prev_i = i
        else:
            raise ValueError(
                f"Non-sequential indices -- Previous index: {prev_i}, this index: {i}"
            )
    return stitched


class KmerStitcher:
    @staticmethod
    def get_to_add(encoded: str, protein: str, use_encoded: bool) -> str:
        return encoded if use_encoded else protein

    @staticmethod
    def add_to_stitched(stitched: str, prev_i: Optional[int], to_add: str) -> str:
        return to_add if prev_i is None else stitched + to_add[-1]

    @classmethod
    def stitch_kmers(cls, overlap: List[Tuple], use_encoded: bool = False) -> str:
        """Stitches kmers together based on overlap information"""
        prev_i = None
        stitched = ""

        for i, encoded, protein, _ in overlap:
            if prev_i is None or i == prev_i + 1:
                to_add = cls.get_to_add(encoded, protein, use_encoded)
                stitched = cls.add_to_stitched(stitched, prev_i, to_add)
                prev_i = i
            else:
                raise ValueError(
                    f"Non-sequential indices -- Previous: {prev_i}, current: {i}"
                )

        return stitched


def get_is_protein(sig):
    is_protein = False
    if sig.minhash.moltype != "DNA":
        is_protein = True
    return is_protein


def add_sequence(minhash, sequence, is_protein):
    if is_protein:
        minhash.add_protein(sequence)
    else:
        minhash.add_sequence(sequence)


def sequence_kmers_in_sig(sig, sequence):
    """Return k-mers from sequence present in signature k-mers"""
    # Make a minhash just for looking through the sequence
    query_mh = sig.minhash.flatten()
    seq_mh = query_mh.copy_and_clear()

    # figure out protein vs dna
    is_protein = get_is_protein(sig)
    add_sequence(seq_mh, sequence, is_protein)

    intersecting = query_mh.intersection(seq_mh)
    if intersecting:
        # Found matching k-mers!

        kmers_hashes = intersecting.kmers_and_hashes(
            sequence, force=False, is_protein=is_protein
        )
        found_kmers_hashes = [
            (i, degenerate_protein_chatgpt(kmer, sig.minhash.moltype), kmer, hashval)
            for i, (kmer, hashval) in enumerate(kmers_hashes)
            if hashval in intersecting.hashes
        ]
        return found_kmers_hashes

    else:
        return []


@dataclass
class OverlapInfo:
    """Stores information about sequence overlap"""

    index: int
    seq: str
    encoded: str
    length: int


class SigSeq:
    """Sourmash Signature with Sequence"""

    def __init__(self, sig: SourmashSignature, seq: str):
        self.sig = sig
        self.seq = seq

        # Returns None if no encoding
        self.seq_encoded = degenerate_protein_chatgpt(seq, sig.minhash.moltype)

    def get_overlapping_kmers(self, other: "SigSeq") -> tuple:
        # Type hinting in quotes
        return sequence_kmers_in_sig(self.sig, other.seq)

    @staticmethod
    def _get_display_padding(self_overlap, other_overlap):
        if self_overlap[0][0] > other_overlap[0][0]:
            self_left_pad = (self_overlap[0][0] - other_overlap[0][0]) * " "
            other_left_pad = ""
            add_to_match_string = " " * other_overlap[0][0]
        else:
            self_left_pad = ""
            other_left_pad = (other_overlap[0][0] - self_overlap[0][0]) * " "
            add_to_match_string = " " * self_overlap[0][0]
        return self_left_pad, other_left_pad, add_to_match_string

    def _verify_overlap(self) -> None:
        """Verifies that overlap sequences match within the main sequence"""
        start = self.overlap.index
        end = start + self.overlap.length
        assert self.overlap.seq == self.seq[start:end]
        assert self.overlap.encoded == self.seq_encoded[start:end]

    def compute_overlap(self, other: "SigSeq") -> None:
        """Computes overlap information between two sequences"""
        overlap = self.get_overlapping_kmers(other)
        overlap_seq = KmerStitcher.stitch_kmers(overlap, use_encoded=False)
        overlap_encoded = KmerStitcher.stitch_kmers(overlap, use_encoded=True)

        index = other.seq.index(overlap_seq)
        overlap_length = overlap[-1][0] - overlap[0][0] + self.sig.minhash.ksize

        # Since the overlapping k-mers were originally in 'other', not 'self' -> assign to 'other'
        # If we had used 'self', then would have returned ALL k-mers since they are all present in self
        other.overlap = OverlapInfo(index, overlap_seq, overlap_encoded, overlap_length)

    @staticmethod
    def calculate_padding(
        self,
        other: "SigSeq",
    ) -> Tuple[str, str, str]:
        """Calculates display padding for alignment visualization"""
        if self.overlap.index > other.overlap.index:
            other_pad = " " * (self.overlap.index - other.overlap.index)
            self_pad = ""
            match_pad = " " * self.overlap.index
        else:
            other_pad = ""
            self_pad = " " * (other.overlap.index - self.overlap.index)
            match_pad = " " * other.overlap.index

        self.pad = self_pad
        other.pad = other_pad
        return match_pad

    def _get_display_sequence(self, encoded=False) -> str:
        """Creates the display sequence with overlap highlighting"""
        start = self.overlap.index
        end = start + self.overlap.length

        seq = self.seq_encoded if encoded else self.seq
        return (
            f"{self.pad}{seq[:start]}"
            f"[overlap]{seq[start:end]}[/overlap]"
            f"{seq[end:]}"
        )

    def display_alignment(self, other):
        """Displays the alignment between two sequences"""
        # Compute overlaps
        self.compute_overlap(other)
        other.compute_overlap(self)

        # Verify overlaps
        self._verify_overlap()
        other._verify_overlap()

        # Calculate padding
        match_pad = self.calculate_padding(self, other)

        # Create vertical lines for where k-mer matches happen
        match_string = match_pad + "|" * self.overlap.length

        console = Console(theme=custom_theme, highlighter=FastaHeaderHighlighter())
        console.print(self.sig.name)
        console.print(self._get_display_sequence())
        console.print(self._get_display_sequence(encoded=True))
        console.print(match_string)
        console.print(other._get_display_sequence(encoded=True))
        console.print(other._get_display_sequence())
        console.print(other.sig.name)
