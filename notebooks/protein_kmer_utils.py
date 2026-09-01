"""Utility functions for protein k-mer analysis."""

from typing import Dict, List, Set, Tuple
import pandas as pd
from sig2kmer import degenerate_protein_chatgpt


class Sequence:
    """Represents a protein sequence with HP encoding."""

    def __init__(self, sequence: str, name: str, moltype: str = "hp"):
        self.sequence = sequence
        self.name = name
        self.moltype = moltype
        self.processed_seq = self._process_sequence()

    def _process_sequence(self) -> str:
        return degenerate_protein_chatgpt(self.sequence, self.moltype)

    def get_kmers(self, k: int) -> List[str]:
        return [
            self.processed_seq[i : i + k]
            for i in range(len(self.processed_seq) - k + 1)
        ]


class KmerAnalyzer:
    """Analyzes k-mer overlap between two sequences."""

    def __init__(self, seq1: Sequence, seq2: Sequence):
        self.seq1 = seq1
        self.seq2 = seq2

    def calculate_jaccard(self, kmer_set1: Set[str], kmer_set2: Set[str]) -> float:
        intersection = len(kmer_set1.intersection(kmer_set2))
        union = len(kmer_set1.union(kmer_set2))
        return intersection / union if union > 0 else 0.0

    def analyze_kmer_range(self, start_k: int, end_k: int) -> pd.DataFrame:
        results = []
        for k in range(start_k, end_k + 1):
            results.append(self._analyze_single_k(k))
        return pd.DataFrame(results, columns=self._get_column_names())

    def get_intersecting_kmer_positions(self, kmers1, kmers2, kmer_set1, kmer_set2):
        intersection = kmer_set1.intersection(kmer_set2)

        positions1 = []
        positions2 = []
        for kmer in intersection:
            positions1.append([kmers1.index(kmer), kmer])
            positions2.append([kmers2.index(kmer), kmer])

        positions1 = sorted(positions1, key=lambda x: x[0])
        positions2 = sorted(positions2, key=lambda x: x[0])
        return positions1, positions2

    def _analyze_single_k(self, k: int) -> List:
        kmers1 = self.seq1.get_kmers(k)
        kmers2 = self.seq2.get_kmers(k)
        set1 = set(kmers1)
        set2 = set(kmers2)

        pos1, pos2 = self.get_intersecting_kmer_positions(kmers1, kmers2, set1, set2)

        return [
            self.seq1.name,
            self.seq2.name,
            self.seq1.moltype,
            k,
            self.calculate_jaccard(set1, set2),
            len(kmers1),
            len(set1),
            pos1,
            len(kmers2),
            len(set2),
            pos2,
        ]

    def _get_column_names(self) -> List[str]:
        return [
            "query",
            "match",
            "moltype",
            "ksize",
            "jaccard",
            "query_n_kmers",
            "query_n_unique_kmers",
            "query_intersection_positions",
            "match_n_kmers",
            "match_n_unique_kmers",
            "match_intersection_positions",
        ]


def compare_sequences(
    seq1_str: str,
    seq2_str: str,
    seq1_name: str,
    seq2_name: str,
    start_k: int = 5,
    end_k: int = 30,
) -> pd.DataFrame:
    """Compare two protein sequences and return k-mer analysis."""
    seq1 = Sequence(seq1_str, seq1_name)
    seq2 = Sequence(seq2_str, seq2_name)
    analyzer = KmerAnalyzer(seq1, seq2)
    return analyzer.analyze_kmer_range(start_k, end_k)


def find_all_occurrences(sequence: str, pattern: str) -> List[int]:
    """Find all starting positions of pattern in sequence (including overlaps)."""
    positions = []
    start = 0
    while True:
        pos = sequence.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1  # Move to next position to find overlapping matches
    return positions


def display_kmer_alignment(
    query_seq: str,
    match_seq: str,
    query_pos: int,
    match_pos: int,
    kmer: str,
    context: int = 5,
    query_label: str = "Query",
    match_label: str = "Match",
) -> None:
    """Display alignment visualization for a k-mer match between two sequences."""
    # Extract context around the k-mer
    query_start = max(0, query_pos - context)
    query_end = min(len(query_seq), query_pos + len(kmer) + context)
    match_start = max(0, match_pos - context)
    match_end = min(len(match_seq), match_pos + len(kmer) + context)

    query_context = query_seq[query_start:query_end]
    match_context = match_seq[match_start:match_end]

    # Create the visualization
    padding = '.' * context
    alignment_line = f'{padding}{kmer}{padding}'

    print(f"{query_label:>5}: {query_context} {query_start}-{query_end}")
    print(f"       {alignment_line}")
    print(f"{match_label:>5}: {match_context} {match_start}-{match_end}")
    print()
