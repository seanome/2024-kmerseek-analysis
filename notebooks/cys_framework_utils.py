"""Disulfide-rich peptides grouped by their cysteine framework, and the data behind them.

A disulfide-rich peptide is a short secreted peptide, 10 to 40 residues in most venoms,
held in shape by two or more disulfide bonds rather than by a hydrophobic core. Its
**cysteine framework** is the pattern of cysteine positions along the sequence, written by
convention as runs of cysteine separated by dashes, so `C-C-CC-C-C` means six cysteines in
five runs: one, one, two adjacent, one, one. Conotoxin families are named by that pattern
(framework VI/VII is `C-C-CC-C-C`), and the residues between the cysteines change fast
between species while the pattern holds.

This module builds the query and reference sets for that question, and scores a search
against them. It covers:

- ConoServer protein entries, their framework labels and their mature-peptide spans
- Swiss-Prot entries for the other venomous lineages, trimmed to the mature peptide
- held-out references: Swiss-Prot minus the query's genus, or minus its phylum
- the regular expression that the cysteine spacing alone implies, which is the control
  that decides whether k-mers add anything
- precision and recall per query, with a bootstrap confidence interval over queries

The shuffle controls live in `nextflow-runs/shared/decoys.py`, next to the other decoy
code, rather than here.
"""

from __future__ import annotations

import csv
import gzip
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# A long UniProt sequence column overruns Python's default CSV field limit.
csv.field_size_limit(10**7)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

#: ConoServer writes the framework in its own field, misspelled as `cysteineFramewrok`.
CONOSERVER_FRAMEWORK_TAG = "cysteineFramewrok"

#: The longest ConoServer entry that is taken to be a mature peptide when the entry
#: carries no mature-peptide region of its own. ConoServer's framework-labelled entries
#: have a median length of 29 residues; anything this long or longer without an explicit
#: region could be a precursor, so it is dropped rather than guessed at.
MATURE_LENGTH_CEILING = 50

#: UniProt's keyword for a toxin, which is what a true positive carries.
TOXIN_KEYWORD = "Toxin"

#: What counts as a short disulfide-rich peptide here: a mature peptide of this many
#: residues holding at least `MIN_CYSTEINES` cysteines.
#:
#: "Disulfide-rich" means at least two disulfide bonds, so at least 4 cysteines. The length
#: window is the one the venom-peptide literature uses, 10 to 40 residues, widened to 60 so
#: that the longer sea-anemone and spider peptides are not cut off at an arbitrary line.
#: Without an upper bound the query set also picks up latrotoxins, which carry UniProt's
#: Toxin keyword and run to about 1_180 residues: real toxins, but proteins rather than
#: peptides, and nothing a k-mer of 12 to 28 residues is being asked about.
MIN_PEPTIDE_LENGTH = 10
MAX_PEPTIDE_LENGTH = 60
MIN_CYSTEINES = 4


def is_short_drp(peptide: str) -> bool:
    """Whether `peptide` is a short disulfide-rich peptide by the rule above."""
    return (
        MIN_PEPTIDE_LENGTH <= len(peptide) <= MAX_PEPTIDE_LENGTH
        and peptide.count("C") >= MIN_CYSTEINES
    )


# ---------------------------------------------------------------------------------------
# Cysteine frameworks
# ---------------------------------------------------------------------------------------


def framework_string(seq: str) -> str:
    """The cysteine framework of `seq`, e.g. `C-C-CC-C-C`.

    Runs of adjacent cysteines stay together and a dash stands for one or more
    non-cysteine residues between them. Residues before the first cysteine and after the
    last one are not part of the pattern. Returns an empty string for a sequence with no
    cysteine.

    >>> framework_string("CKGKGAKCSRLMYDCCTGSCRSGKC")
    'C-C-CC-C-C'
    """
    runs = re.findall(r"C+", seq)
    return "-".join(runs)


def cysteine_spacers(seq: str) -> list[int]:
    """The number of non-cysteine residues between one cysteine and the next.

    One entry per gap between consecutive cysteines, so a sequence with n cysteines gives
    n - 1 numbers, and 0 means the two cysteines are adjacent.

    >>> cysteine_spacers("CKGKGAKCSRLMYDCCTGSCRSGKC")
    [6, 6, 0, 3, 4]
    """
    positions = [i for i, residue in enumerate(seq) if residue == "C"]
    return [b - a - 1 for a, b in zip(positions, positions[1:])]


def framework_regex(seq: str, slack: int = 0) -> str:
    """The regular expression the cysteine spacing of `seq` implies, and nothing else.

    Each cysteine is fixed and each gap becomes `[^C]{a,b}`, where a and b are the real gap
    length minus and plus `slack`, with a floor of 0. The pattern runs from the first
    cysteine to the last, so it can match anywhere inside a longer protein.

    This is the falsification test for the whole experiment. If scanning a reference with
    this recovers the same homologs as a kmerseek setting, then the spacing explains the
    performance and the k-mers add nothing.

    >>> framework_regex("CCGC")
    'CC[^C]{1,1}C'
    >>> framework_regex("CCGC", slack=2)
    'C[^C]{0,2}C[^C]{0,3}C'
    """
    if "C" not in seq:
        raise ValueError("a framework regex needs at least one cysteine")
    parts = ["C"]
    for gap in cysteine_spacers(seq):
        low, high = max(0, gap - slack), gap + slack
        parts.append(f"[^C]{{{low},{high}}}" if high else "")
        parts.append("C")
    return "".join(parts)


def regex_hits(pattern: str, sequences: dict[str, str]) -> list[str]:
    """Every name in `sequences` whose sequence contains `pattern` at least once.

    One hit per sequence, however many times the pattern occurs, so the count is
    comparable with a search that reports a target once.
    """
    compiled = re.compile(pattern)
    return [name for name, seq in sequences.items() if compiled.search(seq)]


# ---------------------------------------------------------------------------------------
# ConoServer
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One annotated span of a sequence, 1-based and inclusive, as both sources write it."""

    start: int
    end: int
    type: str


def _conoserver_root(xml_path: Path):
    """The parsed ConoServer XML.

    ConoServer's dump holds HTML entities (`&alpha;`), bare ampersands (`&muM`) and stray
    control bytes inside reference titles, so a strict XML parser stops partway through a
    file that is otherwise fine. lxml's recovering parser reads all of it; the damage is
    confined to reference text, and the fields used here (id, sequence, framework,
    organism, regions) are plain ASCII. The caller checks the entry count against the
    FASTA, which is the real guard.
    """
    import lxml.etree as ET

    parser = ET.XMLParser(recover=True, huge_tree=True)
    return ET.parse(str(xml_path), parser).getroot()


def read_conoserver(xml_path: Path) -> pl.DataFrame:
    """Every ConoServer protein entry, one row each.

    Columns: `id`, `name`, `organism`, `genus`, `framework`, `sequence`, `length`,
    `is_synthetic`, `mature_start`, `mature_end` (null where the entry annotates no mature
    peptide) and `n_cysteines`.
    """
    rows = []
    for entry in _conoserver_root(xml_path).findall("entry"):
        sequence = (entry.findtext("sequence") or "").strip()
        organism = (entry.findtext("organismLatin") or "").strip()
        mature = None
        regions = entry.find("sequenceRegions")
        if regions is not None:
            for region in regions.findall("region"):
                if (region.findtext("type") or "").strip() == "mature peptide":
                    mature = Region(
                        int(region.findtext("start")),
                        int(region.findtext("end")),
                        "mature peptide",
                    )
                    break
        rows.append(
            {
                "id": entry.findtext("id"),
                "name": (entry.findtext("name") or "").strip(),
                "organism": organism,
                "genus": organism.split()[0] if organism else "",
                "framework": entry.findtext(CONOSERVER_FRAMEWORK_TAG),
                "sequence": sequence,
                "length": len(sequence),
                "is_synthetic": organism == "synthetic construct",
                "mature_start": mature.start if mature else None,
                "mature_end": mature.end if mature else None,
                "n_cysteines": sequence.count("C"),
            }
        )
    return pl.DataFrame(rows)


def conoserver_queries(entries: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The ConoServer query peptides, and a table of what was dropped and why.

    Keeps an entry when it is natural rather than a synthetic construct, carries a
    framework label, and its mature peptide can be located: either the entry annotates a
    mature-peptide region, or the whole entry is short enough
    (`MATURE_LENGTH_CEILING`) to be a mature peptide itself. The peptide is then cut to
    that span.

    Sequences holding a letter outside the 20 canonical residues are dropped. ConoServer
    writes `X` for a modified or unknown residue and `O` for hydroxyproline, and the
    three tools compared here read those differently: a kmerseek alphabet has no class for
    them, HMMER treats X as any residue, and the framework regex would count one as a
    spacer residue. Keeping them would mean three different sequences under one name.

    Returns `(queries, dropped)`. `queries` has one row per peptide with `query_id`,
    `framework`, `organism`, `genus`, `source`, `peptide` and `peptide_length`.
    """
    dropped = []

    def drop(frame: pl.DataFrame, reason: str) -> None:
        if frame.height:
            dropped.append({"reason": reason, "n": frame.height})

    natural = entries.filter(~pl.col("is_synthetic"))
    drop(entries.filter(pl.col("is_synthetic")), "synthetic construct")

    labelled = natural.filter(pl.col("framework").is_not_null())
    drop(natural.filter(pl.col("framework").is_null()), "no cysteine framework label")

    locatable = labelled.filter(
        pl.col("mature_start").is_not_null()
        | (pl.col("length") <= MATURE_LENGTH_CEILING)
    )
    drop(
        labelled.filter(
            pl.col("mature_start").is_null()
            & (pl.col("length") > MATURE_LENGTH_CEILING)
        ),
        f"no mature-peptide region and longer than {MATURE_LENGTH_CEILING} aa",
    )

    cut = locatable.with_columns(
        peptide=pl.when(pl.col("mature_start").is_not_null())
        .then(
            pl.col("sequence").str.slice(
                pl.col("mature_start") - 1,
                pl.col("mature_end") - pl.col("mature_start") + 1,
            )
        )
        .otherwise(pl.col("sequence"))
    )

    canonical = cut.filter(
        pl.col("peptide").str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
        & (pl.col("peptide").str.count_matches("C") >= 2)
    )
    drop(
        cut.filter(~pl.col("peptide").str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$")),
        "a residue outside the 20 canonical amino acids",
    )
    drop(
        cut.filter(
            pl.col("peptide").str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
            & (pl.col("peptide").str.count_matches("C") < 2)
        ),
        "fewer than 2 cysteines in the mature peptide",
    )

    queries = canonical.select(
        query_id=pl.col("id"),
        framework=pl.col("framework"),
        organism=pl.col("organism"),
        genus=pl.col("genus"),
        source=pl.lit("ConoServer"),
        peptide=pl.col("peptide"),
        peptide_length=pl.col("peptide").str.len_chars(),
    )
    return queries, (
        pl.DataFrame(dropped)
        if dropped
        else pl.DataFrame(
            {"reason": [], "n": []}, schema={"reason": pl.Utf8, "n": pl.Int64}
        )
    )


# ---------------------------------------------------------------------------------------
# Swiss-Prot
# ---------------------------------------------------------------------------------------

_FEATURE = re.compile(r"\b(PEPTIDE|CHAIN|SIGNAL|PROPEP)\s+(\d+)\.\.(\d+)")


def parse_features(value: str) -> list[Region]:
    """The spans in one UniProt feature column.

    A column holds features of one kind run together, each written `PEPTIDE 46..70;
    /note="..."`, so the spans are read off the type-and-range prefixes and the qualifiers
    ignored. A feature with an uncertain endpoint (`?`, `<`, `>`) does not match and is
    skipped, which is what we want: a span we cannot place is a span we cannot cut.
    """
    return [
        Region(int(a), int(b), kind) for kind, a, b in _FEATURE.findall(value or "")
    ]


def read_swissprot_table(tsv_path: Path) -> pl.DataFrame:
    """A UniProt TSV download as a DataFrame, with the feature columns left as text."""
    with open(tsv_path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return pl.DataFrame(rows)


def swissprot_queries(
    table: pl.DataFrame, taxonomy: dict[int, tuple[str | None, str | None]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The Swiss-Prot query peptides, and a table of what was dropped and why.

    Each entry is cut to its mature peptide: the `PEPTIDE` feature if there is one, else
    the `CHAIN` feature. An entry with neither is kept whole only when it annotates no
    signal peptide and no propeptide and is short enough to be a mature peptide; otherwise
    it is dropped, because keeping it would put signal or propeptide residues into a query
    whose whole point is the mature framework. An entry with several mature features
    contributes the longest one, so one entry gives one query.

    The same canonical-residue and two-cysteine rules as `conoserver_queries` apply.
    """
    dropped: list[dict] = []
    queries: list[dict] = []
    counts: Counter[str] = Counter()

    for row in table.iter_rows(named=True):
        sequence = row["Sequence"]
        features = parse_features(row.get("Peptide", "")) + parse_features(
            row.get("Chain", "")
        )
        mature = [f for f in features if f.type in ("PEPTIDE", "CHAIN")]
        if mature:
            best = max(mature, key=lambda f: f.end - f.start)
            peptide = sequence[best.start - 1 : best.end]
            span = f"{best.type} {best.start}..{best.end}"
        else:
            has_leader = parse_features(
                row.get("Signal peptide", "")
            ) or parse_features(row.get("Propeptide", ""))
            if has_leader or len(sequence) > MATURE_LENGTH_CEILING:
                counts["no mature feature, and a leader or too long to keep whole"] += 1
                continue
            peptide, span = sequence, "whole entry"

        if not set(peptide) <= CANONICAL_AA:
            counts["a residue outside the 20 canonical amino acids"] += 1
            continue
        if peptide.count("C") < 2:
            counts["fewer than 2 cysteines in the mature peptide"] += 1
            continue

        taxid = int(row["Organism (ID)"])
        genus, phylum = taxonomy.get(taxid, (None, None))
        queries.append(
            {
                "query_id": row["Entry"],
                "framework": framework_string(peptide),
                "organism": row["Organism"],
                "genus": genus or "",
                "phylum": phylum or "",
                "taxid": taxid,
                "source": "Swiss-Prot",
                "mature_span": span,
                "peptide": peptide,
                "peptide_length": len(peptide),
                "is_toxin": TOXIN_KEYWORD in row["Keywords"].split(";"),
            }
        )

    dropped = [{"reason": reason, "n": n} for reason, n in counts.most_common()]
    return pl.DataFrame(queries), (
        pl.DataFrame(dropped)
        if dropped
        else pl.DataFrame(
            {"reason": [], "n": []}, schema={"reason": pl.Utf8, "n": pl.Int64}
        )
    )


def read_taxonomy(tsv_path: Path) -> dict[int, tuple[str | None, str | None]]:
    """taxid to (genus, phylum), from the table built off the NCBI taxonomy dump.

    An empty cell becomes None. A taxid with no genus or no phylum rank in its lineage
    (viruses, and entries placed only at a higher rank) keeps None there, and
    `held_out_reference` treats None as "cannot be excluded", so such an entry stays in
    every reference.
    """
    out = {}
    with open(tsv_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[int(row["taxid"])] = (row["genus"] or None, row["phylum"] or None)
    return out


def read_fasta(path: Path) -> dict[str, str]:
    """A FASTA file as {header without '>': sequence}, reading .gz transparently."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sequences: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name, chunks = line[1:].rstrip("\n"), []
            else:
                chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


_SPROT_HEADER = re.compile(r"^(?:sp|tr)\|(?P<acc>[^|]+)\|(?P<name>\S+)")
_OX = re.compile(r"\bOX=(\d+)")


def swissprot_header_fields(header: str) -> tuple[str, int | None]:
    """The accession and the taxid (`OX=`) of a Swiss-Prot FASTA header.

    Taking the accession from the header rather than from a tool's output is deliberate:
    mmseqs, HMMER and kmerseek each report a different part of a UniProt header
    (`mmseqs_uniprot_header_accession_trap`), so every tool's hit is mapped back to an
    accession here, by the same rule.
    """
    match = _SPROT_HEADER.match(header)
    accession = match.group("acc") if match else header.split()[0]
    taxid = _OX.search(header)
    return accession, int(taxid.group(1)) if taxid else None


# ---------------------------------------------------------------------------------------
# Held-out references
# ---------------------------------------------------------------------------------------


def held_out_reference(
    sequences: dict[str, str],
    accession_taxid: dict[str, int],
    taxonomy: dict[int, tuple[str | None, str | None]],
    exclude: str,
    level: str,
) -> dict[str, str]:
    """`sequences` with every entry from `exclude` at `level` ('genus' or 'phylum') removed.

    An entry whose taxid has no value at that level stays in, since nothing says it
    belongs to the excluded group. That is the cautious direction for this benchmark: a
    kept entry can only ever be scored as a hit or a miss, whereas dropping entries we
    cannot place would quietly shrink the search space and inflate recall.
    """
    if level not in ("genus", "phylum"):
        raise ValueError(f"level must be 'genus' or 'phylum', not {level!r}")
    index = 0 if level == "genus" else 1
    kept = {}
    for header, seq in sequences.items():
        taxid = accession_taxid.get(header)
        value = taxonomy.get(taxid, (None, None))[index] if taxid is not None else None
        if value != exclude:
            kept[header] = seq
    return kept


# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------

#: One row per kmerseek setting this experiment searches with. A setting is one
#: combination of alphabet, k-size, scaled, extension penalty and mask; `arm` is kept as a
#: column name only where older pipeline code requires it.
#:
#: The k-sizes follow the peptide lengths. A mature disulfide-rich peptide is 10 to 40
#: residues, so a k-mer longer than the peptide cannot be drawn at all: at k=28 only
#: peptides of 28 residues or more are searchable. cys2 carries 1 bit per position and
#: cys_hp3 about 1.58, so the cys2 k-sizes run higher to reach comparable bits per k-mer.
#:
#: The low-complexity mask (kmerseek --remove-low-complexity) is on for cys2 and off for
#: cys_hp3. Under cys2 it drops only the k-mer that is one letter throughout: in practice
#: the one with no cysteine, which is 70-83% of a cys2 reference's k-mers. Every k-mer
#: holding a cysteine, and so every k-mer that can carry a framework, is kept. Without the
#: mask, kmerseek's lambda-K fit on 200 reference sequences met about 2-3 x 10^11 seed hits
#: and was killed at every memory up to 1.4 TB (Sherlock pilot, 2026-10-15). A query of at
#: least k residues always keeps a k-mer, since every query has 4 or more cysteines.
SETTINGS = [
    {"alphabet": "cys2", "ksize": k, "scaled": 1, "extend": False, "mask": True}
    for k in (12, 16, 20, 24, 28)
] + [
    {"alphabet": "cys_hp3", "ksize": k, "scaled": 1, "extend": False, "mask": False}
    for k in (9, 11, 13, 15)
]


def settings_table(queries: pl.DataFrame) -> pl.DataFrame:
    """`SETTINGS` as a table, with how many queries each setting can search.

    A query shorter than k has no k-mer, so it is dropped for that setting and counted
    here rather than silently missing from the results.
    """
    rows = []
    lengths = queries["peptide_length"].to_list()
    for setting in SETTINGS:
        searchable = sum(1 for length in lengths if length >= setting["ksize"])
        rows.append(
            {
                **setting,
                "moltype": (
                    "cys2" if setting["alphabet"] == "cys2" else "hp_pbotc_1st_ed_hpc3"
                ),
                "n_queries_searchable": searchable,
                "n_queries_dropped": len(lengths) - searchable,
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------


def score_one_query(hits: set[str], truth: set[str], reference: set[str]) -> dict:
    """Precision and recall for one query against one held-out reference.

    `truth` is every accession in `reference` that carries UniProt's Toxin keyword, so a
    true positive is a toxin-annotated reference entry and the reference has already had
    the query's genus or phylum removed. A hit outside `reference` is not scored; that
    would mean the hit and the reference disagree about the search space.

    Recall is undefined when the reference holds no toxin at all, and precision when the
    search returned nothing; both come back as None rather than 0, so an undefined case
    cannot be averaged in as a zero.
    """
    hits = hits & reference
    true_positives = len(hits & truth)
    precision = true_positives / len(hits) if hits else None
    recall = true_positives / len(truth) if truth else None
    return {
        "n_hits": len(hits),
        "n_true_positives": true_positives,
        "n_truth": len(truth),
        "precision": precision,
        "recall": recall,
    }


def bootstrap_ci(
    values: list[float], n_draws: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float | None, float | None, float | None]:
    """Mean of `values` and a percentile bootstrap interval over them, resampling queries.

    The queries are the unit, so a draw is a sample of queries with replacement, and the
    interval says how much the mean would move with a different set of queries of the same
    size. Returns (mean, low, high), all None for an empty list.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, None
    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(n_draws):
        means.append(sum(rng.choice(clean) for _ in range(n)) / n)
    means.sort()
    tail = (1 - level) / 2
    return (
        sum(clean) / n,
        means[int(tail * n_draws)],
        means[min(n_draws - 1, int((1 - tail) * n_draws))],
    )


if __name__ == "__main__":
    import doctest

    failures, tests = doctest.testmod()
    print(f"{tests - failures} of {tests} doctests pass")
    sys.exit(1 if failures else 0)
