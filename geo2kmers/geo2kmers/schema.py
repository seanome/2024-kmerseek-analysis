"""Check a k-mer table against sketch_schema.md, the format array and read sketches share."""

import csv
import re

from geo2kmers.geo import Geo2KmersError

ID_COLUMNS = ["kmer", "alphabet", "k", "setting"]
VALUE_COLUMNS = ["count", "intensity_weight"]
COMPLEMENT = str.maketrans("ACGT", "TGCA")
WHOLE = re.compile(r"[0-9]+")
WHOLE_ABOVE_0 = re.compile(r"[1-9][0-9]*")
DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")
KMER_LETTERS = {
    "dna": "ACGT",
    "protein": "ACDEFGHIKLMNPQRSTVWY",
    "dayhoff": "abcdef",
    "hp": "hp",
}
# kmerseek's own names for the protein alphabets (canonical_moltype in its src/rust/alphabets.rs). It has no
# DNA alphabet, so dna is this format's own name.
KMERSEEK_NAMES = {"dna": "dna", "protein": "protein20", "dayhoff": "dayhoff6", "hp": "hp_lehninger2"}


def setting_name(alphabet, k):
    """The one string that names an alphabet at a k, with kmerseek's name for the alphabet, as in protein20_k10."""
    return f"{KMERSEEK_NAMES[alphabet]}_k{k}"


def header(value_column, n_probes=True):
    """The columns of a table, in order. value_column is count, intensity_weight, or None for a plain set."""
    return [*ID_COLUMNS, *([value_column] if value_column else []), *(["n_probes"] if n_probes else [])]


ALLOWED_HEADERS = [header(None), header(None, False), header("intensity_weight"), header("intensity_weight", False)]
ALLOWED_HEADERS.append(header("count", False))


def validate_table(path):
    """Stop with an error unless the TSV at path follows sketch_schema.md. Returns its setting and row count."""
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        columns = next(reader, [])
        values = [c for c in columns if c in VALUE_COLUMNS]
        if len(values) > 1:
            raise Geo2KmersError(f"{path} mixes the value columns {' and '.join(values)}")
        if columns not in ALLOWED_HEADERS:
            raise Geo2KmersError(f"{path} has columns {columns}, which sketch_schema.md does not allow")
        where = {name: i for i, name in enumerate(columns)}
        settings, seen, rows = set(), set(), 0
        for row in reader:
            rows += 1
            if len(row) != len(columns):
                raise Geo2KmersError(f"{path} row {rows}: {len(row)} cells, not {len(columns)}")
            kmer, alphabet, k, setting = row[:4]
            if alphabet not in KMER_LETTERS or not WHOLE_ABOVE_0.fullmatch(k):
                raise Geo2KmersError(f"{path} row {rows}: bad alphabet {alphabet!r} or k {k!r}")
            if setting != setting_name(alphabet, k):
                raise Geo2KmersError(f"{path} row {rows}: setting {setting} does not match {alphabet} and k {k}")
            settings.add(setting)
            if len(settings) > 1:
                raise Geo2KmersError(f"{path} mixes the settings {sorted(settings)}")
            if len(kmer) != int(k) or not re.fullmatch(f"[{KMER_LETTERS[alphabet]}]+", kmer):
                raise Geo2KmersError(f"{path} row {rows}: {kmer!r} is not a {alphabet} k-mer of length {k}")
            if alphabet == "dna" and kmer > kmer.translate(COMPLEMENT)[::-1]:
                raise Geo2KmersError(f"{path} row {rows}: {kmer} is not in canonical form")
            if kmer in seen:
                raise Geo2KmersError(f"{path} row {rows}: {kmer} appears twice")
            seen.add(kmer)
            for value in values:
                cell = row[where[value]]
                allowed = WHOLE.fullmatch(cell) if value == "count" else DECIMAL.fullmatch(cell)
                if not allowed:
                    raise Geo2KmersError(f"{path} row {rows}: {value} {cell} is not allowed")
            if "n_probes" in where and not WHOLE_ABOVE_0.fullmatch(row[where["n_probes"]]):
                raise Geo2KmersError(f"{path} row {rows}: n_probes {row[where['n_probes']]} is not allowed")
    return next(iter(settings), None), rows
