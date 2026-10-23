"""Build one table of peptides with everything the three endpoints need.

The benchmark asks three questions that are easy to mistake for one:

  1. does a method retrieve peptides with a similar cysteine framework?
  2. does it retrieve peptides with a similar experimentally supported fold?
  3. does it retrieve peptides acting on the same experimentally supported target?

A manifest row therefore carries three separate groups of columns, each with its own
evidence column, and a method is never credited with one endpoint on the strength of
another. Where a label is absent the row says so in a `*_missing` flag: an absent target
is unknown, not a measured absence of activity.

Nothing here infers a mature-peptide boundary, a disulfide connectivity or a target
without writing down that it did. `mature_boundary_source` names which annotation the
boundary came from and `mature_boundary_inferred` is true when the pipeline chose it
rather than read it; `connectivity_evidence` separates a measured bond from one carried
over by similarity; `target_label_source` names the database field each activity label
came from, with no translation between vocabularies.

Modifications are kept as metadata and never written into the sequence. A C-terminally
amidated peptide and the same chain without the amide are one string to every method
compared here, and `n_modifications` is what tells a reader which comparisons that
affects.

Two namespaces are kept apart on purpose. A ConoServer identifier looks like a UniProt
accession (`P06751`) and is not one, so ConoServer rows are `cono:` and Swiss-Prot rows
are `sp:`, and the two are joined only where ConoServer states a UniProt cross-reference
of its own, or where the mature peptides are character-for-character equal.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import frameworks
from .shared import cys_framework_utils

csv.field_size_limit(10**7)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

#: What counts as a short disulfide-rich peptide. Taken from the cys-frameworks pipeline
#: so the two experiments select the same peptides.
MIN_PEPTIDE_LENGTH = cys_framework_utils.MIN_PEPTIDE_LENGTH
MAX_PEPTIDE_LENGTH = cys_framework_utils.MAX_PEPTIDE_LENGTH
MIN_CYSTEINES = cys_framework_utils.MIN_CYSTEINES
MATURE_LENGTH_CEILING = cys_framework_utils.MATURE_LENGTH_CEILING

#: UniProt keywords that name what a toxin acts on, grouped so that two keywords meaning
#: the same target class give the same label. Every label is prefixed `uniprot:` and is
#: only ever compared with another `uniprot:` label: this file does no translation between
#: UniProt's vocabulary and ConoServer's.
#:
#: KW-0872 "Ion channel impairing toxin" is the parent of the four channel keywords below
#: it, so it is used only when no specific channel keyword is present, and then it says
#: only that a channel is involved.
TARGET_KEYWORDS = {
    "KW-0738": "sodium_channel",  # Voltage-gated sodium channel impairing toxin
    "KW-1220": "potassium_channel",  # Voltage-gated potassium channel impairing toxin
    "KW-0632": "potassium_channel",  # Potassium channel impairing toxin
    "KW-1218": "calcium_channel",  # Voltage-gated calcium channel impairing toxin
    "KW-0108": "calcium_channel",  # Calcium channel impairing toxin
    "KW-0008": "acetylcholine_receptor",  # Acetylcholine receptor inhibiting toxin
    "KW-1213": "gpcr",  # G protein-coupled receptor impairing toxin
    "KW-0646": "protease_inhibition",  # Protease inhibitor
    "KW-0929": "antimicrobial",  # Antimicrobial
    "KW-0044": "antimicrobial",  # Antibiotic
    "KW-0354": "hemolysis",  # Hemolysis
    "KW-1201": "platelet_aggregation",  # Platelet aggregation inhibiting toxin
}
ION_CHANNEL_PARENT = "KW-0872"
CHANNEL_LABELS = {"sodium_channel", "potassium_channel", "calcium_channel"}

#: The fold keyword UniProt states for short disulfide-rich peptides. Knottin is the
#: inhibitor cystine knot; it is the only named fold in UniProt's keyword list that these
#: peptides carry, and it is a curator's assignment, not a measurement, which is why the
#: fold endpoint also needs `connectivity_evidence`.
KNOTTIN_KEYWORD = "KW-0960"
STRUCTURE_KEYWORD = "KW-0002"

#: Where UniProt writes an experimental claim. Reused from frameworks so the fold and the
#: activity columns judge evidence the same way.
EXPERIMENTAL_ECO = frameworks.EXPERIMENTAL_ECO

_FEATURE_SPAN = re.compile(
    r"\b(PEPTIDE|CHAIN|SIGNAL|PROPEP)\s+(\d+)\.\.(\d+)((?:(?!\b(?:PEPTIDE|CHAIN|SIGNAL|PROPEP)\b).)*)",
    re.S,
)
_MOD_RES = re.compile(
    r"\b(MOD_RES|LIPID|CARBOHYD|CROSSLNK)\s+(\d+)(?:\.\.(\d+))?;\s*/note=\"([^\"]*)\""
)
_PHYLUM = re.compile(r"([A-Za-z ]+) \(phylum\)")
_CLASS = re.compile(r"([A-Za-z ]+) \(class\)")


@dataclass
class Span:
    """One annotated stretch of a precursor, 1-based and inclusive, with its evidence."""

    kind: str
    start: int
    end: int
    evidence: str = ""

    @property
    def is_experimental(self) -> bool:
        return any(code in self.evidence for code in EXPERIMENTAL_ECO)


@dataclass
class DropLog:
    """Why peptides were left out, counted by reason."""

    counts: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, n: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + n

    def frame(self) -> pl.DataFrame:
        if not self.counts:
            return pl.DataFrame({"reason": [], "n": []}, schema={"reason": pl.Utf8, "n": pl.Int64})
        rows = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return pl.DataFrame({"reason": [r for r, _ in rows], "n": [n for _, n in rows]})


def parse_spans(value: str) -> list[Span]:
    """Every PEPTIDE, CHAIN, SIGNAL and PROPEP span in a UniProt feature column.

    >>> spans = parse_spans('PEPTIDE 1..31; /evidence="ECO:0000269|PubMed:1"')
    >>> spans[0].kind, spans[0].start, spans[0].end, spans[0].is_experimental
    ('PEPTIDE', 1, 31, True)
    """
    return [
        Span(kind=kind, start=int(start), end=int(end), evidence=rest)
        for kind, start, end, rest in _FEATURE_SPAN.findall(value or "")
    ]


#: What separates one modification from the next in the `modifications` column. A UniProt
#: note can hold a semicolon of its own ("Phosphoserine; alternate"), which split a joined
#: list into pieces with no position on them, so the separator is a character UniProt's
#: controlled vocabulary does not use.
MODIFICATION_SEPARATOR = "|"


def parse_modifications(row: dict) -> list[str]:
    """Every modification UniProt states, as `position:note` strings.

    Covers modified residues, lipidation, glycosylation and cross-links, which is every
    modification column downloaded. The position is the precursor's 1-based one; the
    caller shifts it to the mature peptide.

    >>> parse_modifications({"Modified residue": 'MOD_RES 35; /note="Pyrrolidone carboxylic acid"'})
    ['35:Pyrrolidone carboxylic acid']
    """
    out = []
    for column in ("Modified residue", "Lipidation", "Glycosylation", "Cross-link"):
        for _, start, _, note in _MOD_RES.findall(row.get(column) or ""):
            out.append(f"{start}:{note.replace(MODIFICATION_SEPARATOR, '/')}")
    return out


def _uniprot_targets(row: dict) -> tuple[list[str], str]:
    """The target labels a Swiss-Prot row states, and where they came from."""
    ids = {part.strip() for part in (row.get("Keyword ID") or "").split(";")}
    labels = {TARGET_KEYWORDS[k] for k in ids if k in TARGET_KEYWORDS}
    if ION_CHANNEL_PARENT in ids and not (labels & CHANNEL_LABELS):
        labels.add("ion_channel_unspecified")
    if not labels:
        return [], ""
    return sorted(f"uniprot:{label}" for label in labels), "UniProt keyword"


def _mature_span(spans: list[Span], precursor: str) -> tuple[Span | None, str, str]:
    """Choose the mature peptide's span, and say how the choice was made.

    Returns `(span, source, inference)`. `span` is None when no boundary can be chosen.
    `inference` is empty when the boundary was read from an annotation and holds a
    sentence when it was derived, so that every derived boundary is on the record.

    The order is strict, most direct annotation first:
      PEPTIDE      UniProt's own mature-peptide feature.
      CHAIN        a processed chain, used when there is no PEPTIDE.
      after SIGNAL and PROPEP, derived: the precursor with the parts processing removes
                   cut off. This is a derivation, not an annotation.
      whole entry, derived: a short entry with no processing features at all.
    Where several PEPTIDE or CHAIN features qualify, the longest one with at least
    `MIN_CYSTEINES` cysteines is taken, and ties go to the earliest start.
    """

    def best(kind: str) -> Span | None:
        candidates = [
            span
            for span in spans
            if span.kind == kind
            and precursor[span.start - 1 : span.end].count("C") >= MIN_CYSTEINES
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.end - s.start, -s.start))

    for kind, source in (("PEPTIDE", "annotated_peptide"), ("CHAIN", "annotated_chain")):
        span = best(kind)
        if span is not None:
            return span, source, ""

    cut = [s.end for s in spans if s.kind in ("SIGNAL", "PROPEP")]
    if cut:
        start = max(cut) + 1
        if start <= len(precursor):
            return (
                Span("DERIVED", start, len(precursor)),
                "after_signal_propeptide",
                f"no mature-peptide feature; took residues {start} to {len(precursor)}, "
                f"the precursor after the last annotated signal or propeptide",
            )
    if len(precursor) <= MATURE_LENGTH_CEILING:
        return (
            Span("DERIVED", 1, len(precursor)),
            "whole_entry_short",
            f"no processing features and only {len(precursor)} residues, so the whole "
            f"entry was taken as the mature peptide",
        )
    return None, "", ""


def _taxon(lineage: str) -> tuple[str | None, str | None]:
    """The phylum and class named in UniProt's taxonomic lineage column."""
    phylum = _PHYLUM.search(lineage or "")
    klass = _CLASS.search(lineage or "")
    return (
        phylum.group(1).strip() if phylum else None,
        klass.group(1).strip() if klass else None,
    )


def read_swissprot(
    tsv_path: Path, is_toxin: bool, log: DropLog
) -> list[dict]:
    """Manifest rows from one Swiss-Prot search result."""
    rows = []
    prefix = "toxin" if is_toxin else "nontoxin"
    for row in csv.DictReader(open(tsv_path), delimiter="\t"):
        precursor = (row["Sequence"] or "").strip()
        spans = parse_spans(row.get("Peptide", "")) + parse_spans(row.get("Chain", ""))
        spans += parse_spans(row.get("Signal peptide", "")) + parse_spans(
            row.get("Propeptide", "")
        )
        span, boundary_source, inference = _mature_span(spans, precursor)
        if span is None:
            log.drop(f"{prefix}: no mature-peptide boundary could be chosen")
            continue
        written = precursor[span.start - 1 : span.end]
        noncanonical = sorted(set(written) - CANONICAL_AA)
        if noncanonical:
            # Swiss-Prot writes X for an unknown residue. Dropped rather than guessed at,
            # because the framework, the k-mers and the alignment would each read it
            # differently and the three methods would not be comparing one sequence.
            log.drop(f"{prefix}: a residue outside the 20 canonical amino acids")
            continue
        if not (
            MIN_PEPTIDE_LENGTH <= len(written) <= MAX_PEPTIDE_LENGTH
        ):
            log.drop(
                f"{prefix}: mature peptide outside {MIN_PEPTIDE_LENGTH}-"
                f"{MAX_PEPTIDE_LENGTH} residues"
            )
            continue

        offset = span.start - 1
        connectivity = frameworks.connectivity_from_pairs(
            row.get("Disulfide bond", ""), precursor, offset, len(written)
        )
        n_cysteines = written.count("C")
        targets, target_source = _uniprot_targets(row)
        keyword_ids = {p.strip() for p in (row.get("Keyword ID") or "").split(";")}
        modifications = [
            f"{int(position) - offset}:{note}"
            for position, note in (
                m.split(":", 1) for m in parse_modifications(row)
            )
            if 1 <= int(position) - offset <= len(written)
        ]
        phylum, taxon_class = _taxon(row.get("Taxonomic lineage", ""))
        function = (row.get("Function [CC]") or "").strip()
        rows.append(
            {
                "peptide_id": f"sp:{row['Entry']}",
                "source": "Swiss-Prot",
                "source_id": row["Entry"],
                "uniprot_xref": row["Entry"],
                "entry_name": row.get("Entry Name"),
                "protein_name": row.get("Protein names"),
                "mature_peptide": written,
                "mature_peptide_as_written": written,
                "precursor": precursor,
                "precursor_length": len(precursor),
                "mature_start": span.start,
                "mature_end": span.end,
                "mature_boundary_source": boundary_source,
                "mature_boundary_inferred": bool(inference),
                "mature_boundary_inference": inference,
                "mature_boundary_experimental": span.is_experimental,
                "noncanonical_residues": "",
                "species": row.get("Organism"),
                "taxid": int(row["Organism (ID)"]) if row.get("Organism (ID)") else None,
                "genus": (row.get("Organism") or "").split()[0] or None,
                "phylum": phylum,
                "taxon_class": taxon_class,
                "gene_family": (row.get("Protein families") or "").strip() or None,
                "gene_family_source": "UniProt protein families"
                if (row.get("Protein families") or "").strip()
                else "",
                "framework_label": None,
                "connectivity": connectivity,
                "connectivity_evidence": frameworks.disulfide_evidence(
                    row.get("Disulfide bond", "")
                ),
                "disulfide_raw": (row.get("Disulfide bond") or "").strip(),
                "knottin": KNOTTIN_KEYWORD in keyword_ids,
                "has_structure": bool((row.get("PDB") or "").strip())
                or STRUCTURE_KEYWORD in keyword_ids,
                "pdb_ids": (row.get("PDB") or "").strip(),
                "target_labels": ";".join(targets),
                "target_label_source": target_source,
                "activity_text": function[:600],
                "activity_has_experimental_evidence": any(
                    code in function for code in EXPERIMENTAL_ECO
                ),
                "modifications": MODIFICATION_SEPARATOR.join(modifications),
                "modification_source": "UniProt features" if modifications else "",
                "is_toxin": is_toxin,
            }
        )
    return rows


def read_conoserver(xml_path: Path, log: DropLog) -> list[dict]:
    """Manifest rows from ConoServer's protein dump.

    ConoServer states a cysteine framework label and a pharmacological family that
    Swiss-Prot does not, and writes modifications with their positions. It states no
    disulfide connectivity, so every ConoServer row has `connectivity_evidence` of
    `"none"` and relies on its Swiss-Prot counterpart, where one exists, for the fold
    endpoint.

    Modifications are read and kept, and the residue they sit on is kept as written.
    ConoServer writes `X` for a modified or unknown residue and `O` for hydroxyproline;
    those rows are dropped from the comparison, because the three methods read a
    non-standard letter three different ways, but the drop is counted and the entry's
    modifications are what says why it was dropped.
    """
    root = cys_framework_utils._conoserver_root(xml_path)
    rows = []
    for entry in root.findall("entry"):
        sequence = (entry.findtext("sequence") or "").strip()
        organism = (entry.findtext("organismLatin") or "").strip()
        if organism == "synthetic construct":
            log.drop("ConoServer: synthetic construct")
            continue
        framework_label = entry.findtext(cys_framework_utils.CONOSERVER_FRAMEWORK_TAG)

        mature_start = mature_end = None
        regions = entry.find("sequenceRegions")
        if regions is not None:
            for region in regions.findall("region"):
                if (region.findtext("type") or "").strip() == "mature peptide":
                    mature_start = int(region.findtext("start"))
                    mature_end = int(region.findtext("end"))
                    break
        if mature_start is not None:
            boundary_source, inference = "annotated_region", ""
        elif len(sequence) <= MATURE_LENGTH_CEILING:
            mature_start, mature_end = 1, len(sequence)
            boundary_source = "whole_entry_short"
            inference = (
                f"no mature-peptide region and only {len(sequence)} residues, so the "
                f"whole entry was taken as the mature peptide"
            )
        else:
            log.drop(
                f"ConoServer: no mature-peptide region and longer than "
                f"{MATURE_LENGTH_CEILING} residues"
            )
            continue

        offset = mature_start - 1
        written = sequence[offset:mature_end]
        modifications = []
        mods = entry.find("sequenceModifications")
        if mods is not None:
            for modification in mods.findall("modification"):
                position = int(modification.get("position", 0)) - offset
                name = (
                    modification.get("name") or modification.get("symbol") or "?"
                ).replace(MODIFICATION_SEPARATOR, "/")
                if 1 <= position <= len(written):
                    modifications.append(f"{position}:{name}")

        noncanonical = sorted(set(written) - CANONICAL_AA)
        if noncanonical:
            log.drop(
                "ConoServer: a residue outside the 20 canonical amino acids "
                "(a modified residue written in the sequence)"
            )
            continue
        if not (MIN_PEPTIDE_LENGTH <= len(written) <= MAX_PEPTIDE_LENGTH):
            log.drop(
                f"ConoServer: mature peptide outside {MIN_PEPTIDE_LENGTH}-"
                f"{MAX_PEPTIDE_LENGTH} residues"
            )
            continue

        pharmacological = entry.findtext("pharmacologicalFamily")
        xref = None
        for link in entry.findall("linkOut"):
            if link.get("database") == "UniProt" and (link.text or "").strip():
                xref = link.text.strip()
                break
        rows.append(
            {
                "peptide_id": f"cono:{entry.findtext('id')}",
                "source": "ConoServer",
                "source_id": entry.findtext("id"),
                "uniprot_xref": xref,
                "entry_name": (entry.findtext("name") or "").strip(),
                "protein_name": (entry.findtext("name") or "").strip(),
                "mature_peptide": written,
                "mature_peptide_as_written": written,
                "precursor": sequence,
                "precursor_length": len(sequence),
                "mature_start": mature_start,
                "mature_end": mature_end,
                "mature_boundary_source": boundary_source,
                "mature_boundary_inferred": bool(inference),
                "mature_boundary_inference": inference,
                # ConoServer states whether the sequence itself was seen at protein level,
                # which is the nearest thing it has to evidence for the boundary.
                "mature_boundary_experimental": (
                    (entry.findtext("sequenceEvidence") or "").strip() == "protein level"
                ),
                "noncanonical_residues": "",
                "species": organism or None,
                "taxid": None,
                "genus": organism.split()[0] if organism else None,
                "phylum": "Mollusca",
                "taxon_class": "Gastropoda",
                "gene_family": (entry.findtext("geneSuperfamily") or "").strip() or None,
                "gene_family_source": "ConoServer geneSuperfamily"
                if (entry.findtext("geneSuperfamily") or "").strip()
                else "",
                "framework_label": framework_label,
                "connectivity": None,
                "connectivity_evidence": "none",
                "disulfide_raw": "",
                "knottin": False,
                "has_structure": False,
                "pdb_ids": "",
                "target_labels": f"conoserver:{pharmacological.strip()}"
                if pharmacological and pharmacological.strip()
                else "",
                "target_label_source": "ConoServer pharmacologicalFamily"
                if pharmacological and pharmacological.strip()
                else "",
                "activity_text": (entry.findtext("note") or "").strip()[:600],
                # ConoServer's note is free text with no evidence code, so no activity
                # claim from ConoServer is marked experimental here.
                "activity_has_experimental_evidence": False,
                "modifications": MODIFICATION_SEPARATOR.join(modifications),
                "modification_source": "ConoServer sequenceModifications"
                if modifications
                else "",
                "is_toxin": True,
            }
        )
    return rows


def add_derived(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the framework, loop, control-role and missing-label columns."""
    peptides = frame["mature_peptide"].to_list()
    layouts = [frameworks.loop_layout(peptide) for peptide in peptides]
    connectivity = frame["connectivity"].to_list()
    n_cysteines = [layout.n_cysteines for layout in layouts]
    frame = frame.with_columns(
        framework=pl.Series([frameworks.framework_string(p) for p in peptides]),
        n_cysteines=pl.Series(n_cysteines, dtype=pl.Int64),
        loop_lengths=pl.Series(
            [",".join(str(n) for n in layout.loop_lengths) for layout in layouts]
        ),
        nterm_length=pl.Series([len(layout.nterm) for layout in layouts], dtype=pl.Int64),
        cterm_length=pl.Series([len(layout.cterm) for layout in layouts], dtype=pl.Int64),
        peptide_length=pl.col("mature_peptide").str.len_chars(),
        connectivity_complete=pl.Series(
            [
                frameworks.connectivity_is_complete(c, n)
                for c, n in zip(connectivity, n_cysteines)
            ]
        ),
        n_modifications=pl.when(pl.col("modifications") == "")
        .then(0)
        .otherwise(
            pl.col("modifications").str.count_matches(
                re.escape(MODIFICATION_SEPARATOR)
            )
            + 1
        ),
    )
    return frame.with_columns(
        has_cterm_amidation=pl.col("modifications")
        .str.to_lowercase()
        .str.contains("amide|amidation|amidated|nh2"),
        control_class=pl.when(pl.col("is_toxin"))
        .then(pl.lit("toxin"))
        .when(pl.col("n_cysteines") >= MIN_CYSTEINES)
        .then(pl.lit("nontoxin_cysrich"))
        .otherwise(pl.lit("nontoxin_other")),
        framework_missing=pl.col("n_cysteines") < MIN_CYSTEINES,
        connectivity_missing=~pl.col("connectivity_complete"),
        # The fold endpoint needs either a measured complete connectivity or a curated
        # fold name. Neither is a fold classification, and the evaluation keeps them apart.
        fold_missing=~(
            (pl.col("connectivity_complete") & (pl.col("connectivity_evidence") == "experimental"))
            | pl.col("knottin")
        ),
        target_missing=pl.col("target_labels") == "",
    )


def build(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The manifest, and a table of what was dropped and why."""
    log = DropLog()
    rows = read_conoserver(data_dir / "conoserver_raw" / "protein.xml", log)
    rows += read_swissprot(data_dir / "uniprot" / "toxins.tsv", True, log)
    rows += read_swissprot(data_dir / "uniprot" / "metazoa_nontoxin.tsv", False, log)
    # infer_schema_length=None reads every row before fixing a column's type. The default
    # reads 100, and the first 100 rows are ConoServer, whose taxid is always null, so a
    # column polars had typed as null met an integer 6_871 rows later and failed.
    frame = add_derived(pl.DataFrame(rows, infer_schema_length=None))
    # One row per peptide identifier, ordered so two runs give byte-identical files.
    frame = frame.unique(subset=["peptide_id"], keep="first").sort("peptide_id")
    return frame, log.frame()
