"""Find isoform pairs that differ by one internal exon (step 1 of notebook 341).

An "exon event" here is one internal exon together with the two introns around it: the
including isoform splices upstream exon -> exon -> downstream exon, and the skipping isoform
splices upstream exon -> downstream exon directly. Matching on introns rather than on whole
exon lists means isoforms that differ only in where transcription starts or ends (common in
long reads) still pair up.

For each pair the module reads the two proteins from the isoforms themselves: each isoform's
mRNA is assembled from the genome, and its protein is its longest ORF (ATG to stop, forward
strand of the mRNA). Set A (Patowary et al. 2024) ships no ORF predictions, so the longest ORF
is the stated fallback. The longest ORF starts at the first ATG that gives the longest
reading frame, so for isoforms that Ensembl starts at a later ATG (the shorter p53 forms, for
example) it gives a longer protein than the annotation; a pair whose two proteins do not line
up this way is flagged by `orf_otherwise_identical`.

Coordinates are 1-based and inclusive, as in GTF.
"""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from dataclasses import dataclass, field

CODON_TABLE = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        ((x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
ATTR = re.compile(r'(\S+) "([^"]*)"')


@dataclass
class Isoform:
    """One transcript from a GTF: its gene, strand and exons in genome order."""

    transcript_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    exons: list[tuple[int, int]] = field(default_factory=list)

    def introns(self) -> list[tuple[int, int]]:
        """Introns as (first intronic base, last intronic base), in genome order."""
        ex = sorted(self.exons)
        return [(ex[i][1] + 1, ex[i + 1][0] - 1) for i in range(len(ex) - 1)]


def read_gtf_isoforms(path: str) -> dict[str, Isoform]:
    """Every transcript with its exons, from a GTF (plain or gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    isoforms: dict[str, Isoform] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = dict(ATTR.findall(fields[8]))
            tid = attrs["transcript_id"]
            iso = isoforms.get(tid)
            if iso is None:
                iso = Isoform(
                    tid,
                    attrs["gene_id"],
                    attrs.get("gene_name", ""),
                    fields[0],
                    fields[6],
                )
                isoforms[tid] = iso
            iso.exons.append((int(fields[3]), int(fields[4])))
    for iso in isoforms.values():
        iso.exons.sort()
    return isoforms


def read_fasta_chromosomes(path: str, wanted: set[str]):
    """Yield (name, sequence) for each wanted chromosome of a FASTA, one at a time."""
    opener = gzip.open if path.endswith(".gz") else open
    name, chunks = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name in wanted:
                    yield name, "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            elif name in wanted:
                chunks.append(line.strip())
    if name in wanted:
        yield name, "".join(chunks).upper()


def mrna(iso: Isoform, chrom_seq: str, offset: int = 0) -> str:
    """The isoform's spliced sequence, 5' to 3'. `offset` is the genome position of chrom_seq[0] minus 1."""
    seq = "".join(chrom_seq[s - 1 - offset : e - offset] for s, e in iso.exons)
    if iso.strand == "-":
        seq = seq.translate(COMPLEMENT)[::-1]
    return seq


def longest_orf(seq: str) -> tuple[int, int] | None:
    """(start, end) of the longest ATG...stop ORF in seq, end exclusive and including the stop.

    Ties go to the ORF that starts first. None when there is no complete ORF.
    """
    best = None
    for frame in range(3):
        start = None
        for i in range(frame, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if start is None and codon == "ATG":
                start = i
            elif start is not None and CODON_TABLE.get(codon) == "*":
                if (
                    best is None
                    or (i + 3 - start) > (best[1] - best[0])
                    or ((i + 3 - start) == (best[1] - best[0]) and start < best[0])
                ):
                    best = (start, i + 3)
                start = None
    return best


def translate(seq: str) -> str:
    """Translate codon by codon; 'X' for a codon with N; the stop is dropped."""
    protein = "".join(
        CODON_TABLE.get(seq[i : i + 3], "X") for i in range(0, len(seq) - 2, 3)
    )
    return protein[:-1] if protein.endswith("*") else protein


def exon_mrna_span(iso: Isoform, exon_start: int, exon_end: int) -> tuple[int, int]:
    """0-based, inclusive (first, last) position of one of the isoform's exons in its 5'->3' mRNA."""
    exons = iso.exons if iso.strand == "+" else iso.exons[::-1]
    before = 0
    for s, e in exons:
        if (s, e) == (exon_start, exon_end):
            return before, before + (e - s)
        before += e - s + 1
    raise ValueError(f"{iso.transcript_id} has no exon {exon_start}-{exon_end}")


def mrna_and_orf(
    iso: Isoform, chrom_seq: str, offset: int = 0, cache: dict | None = None
):
    """(mRNA, longest ORF span) of an isoform, memoised in `cache` by transcript id."""
    if cache is not None and iso.transcript_id in cache:
        return cache[iso.transcript_id]
    m = mrna(iso, chrom_seq, offset)
    result = (m, longest_orf(m))
    if cache is not None:
        cache[iso.transcript_id] = result
    return result


def find_exon_events(isoforms: dict[str, Isoform]) -> list[dict]:
    """One row per (including isoform, skipping isoform, exon): every one-internal-exon pair.

    Including isoform: has introns (up_end+1 .. exon_start-1) and (exon_end+1 .. down_start-1).
    Skipping isoform: has the intron (up_end+1 .. down_start-1), and every other intron of
    the two isoforms is the same.
    """
    by_gene = defaultdict(list)
    for iso in isoforms.values():
        if len(iso.exons) >= 2:
            by_gene[(iso.gene_id, iso.chrom, iso.strand)].append(iso)
    rows = []
    for (gene_id, chrom, strand), isos in by_gene.items():
        intron_sets = {iso.transcript_id: frozenset(iso.introns()) for iso in isos}
        by_intron = defaultdict(list)
        for iso in isos:
            for intron in iso.introns():
                by_intron[intron].append(iso)
        for inc in isos:
            ex = inc.exons
            inc_introns = intron_sets[inc.transcript_id]
            for k in range(1, len(ex) - 1):
                up_intron = (ex[k - 1][1] + 1, ex[k][0] - 1)
                down_intron = (ex[k][1] + 1, ex[k + 1][0] - 1)
                joined = (up_intron[0], down_intron[1])
                expected = (inc_introns - {up_intron, down_intron}) | {joined}
                for skip in by_intron.get(joined, []):
                    if intron_sets[skip.transcript_id] == expected:
                        rows.append(
                            {
                                "gene_id": gene_id,
                                "gene_name": inc.gene_name,
                                "chrom": chrom,
                                "strand": strand,
                                "exon_start": ex[k][0],
                                "exon_end": ex[k][1],
                                "upstream_exon_end": ex[k - 1][1],
                                "downstream_exon_start": ex[k + 1][0],
                                "including_isoform": inc.transcript_id,
                                "skipping_isoform": skip.transcript_id,
                            }
                        )
    return rows


def pair_proteins(
    row: dict,
    inc: Isoform,
    skip: Isoform,
    chrom_seq: str,
    offset: int = 0,
    cache: dict | None = None,
) -> dict:
    """Proteins of both isoforms (longest ORF each) and where the exon sits in the including one.

    Adds: protein_including, protein_skipping, exon_nt, frame_preserving (exon length a
    multiple of 3), exon_in_orf (the whole exon lies inside the including isoform's ORF),
    exon_first_residue / exon_last_residue (1-based, residues touched by the exon's bases),
    exon_residues, and orf_otherwise_identical (the skipping protein is the including protein
    with exactly the exon's residues removed; for an exon that starts mid-codon, one merged
    residue at the join may differ).
    """
    out = dict(row)
    m_inc, orf_inc = mrna_and_orf(inc, chrom_seq, offset, cache)
    m_skip, orf_skip = mrna_and_orf(skip, chrom_seq, offset, cache)
    exon_nt = row["exon_end"] - row["exon_start"] + 1
    out.update(
        exon_nt=exon_nt,
        frame_preserving=exon_nt % 3 == 0,
        protein_including=None,
        protein_skipping=None,
        exon_in_orf=False,
        exon_first_residue=None,
        exon_last_residue=None,
        exon_residues=None,
        orf_otherwise_identical=False,
    )
    if orf_inc is None or orf_skip is None:
        return out
    p_inc = translate(m_inc[orf_inc[0] : orf_inc[1]])
    p_skip = translate(m_skip[orf_skip[0] : orf_skip[1]])
    out.update(protein_including=p_inc, protein_skipping=p_skip)
    a, b = exon_mrna_span(
        inc, row["exon_start"], row["exon_end"]
    )  # exon's mRNA span, inclusive
    if not (orf_inc[0] <= a and b < orf_inc[1] - 3):
        return out
    first, last = (a - orf_inc[0]) // 3, (
        b - orf_inc[0]
    ) // 3  # 0-based residues touched
    out.update(
        exon_in_orf=True,
        exon_first_residue=first + 1,
        exon_last_residue=last + 1,
        exon_residues=p_inc[first : last + 1],
    )
    if out["frame_preserving"]:
        n_removed = exon_nt // 3
        if (a - orf_inc[0]) % 3 == 0:
            same = p_inc[:first] + p_inc[first + n_removed :] == p_skip
        else:
            # The exon's bases are split across codons at both ends; one merged residue.
            same = (
                len(p_skip) == len(p_inc) - n_removed
                and p_inc[:first] == p_skip[:first]
                and p_inc[last + 1 :] == p_skip[first + 1 :]
            )
        out["orf_otherwise_identical"] = bool(same)
    return out


def event_key(row: dict) -> tuple:
    """The exon event a pair belongs to: gene, chromosome, strand, exon and its two neighbours."""
    return (
        row["gene_id"],
        row["chrom"],
        row["strand"],
        row["exon_start"],
        row["exon_end"],
        row["upstream_exon_end"],
        row["downstream_exon_start"],
    )


def event_read_counts(
    key: tuple,
    gene_isoforms: list[Isoform],
    counts: dict[str, list[int]],
    n_samples: int,
) -> tuple[list[int], list[int]]:
    """Per-sample reads of isoforms that include the exon and of isoforms that skip it.

    Including: every isoform of the gene with both introns around the exon. Skipping: every
    isoform of the gene with the intron that joins the two neighbouring exons directly. Other
    differences between isoforms are ignored, so Ψ is the exon's inclusion at this junction.
    """
    gene_id, chrom, strand, ex_s, ex_e, up_end, dn_start = key
    up_intron, down_intron, joined = (
        (up_end + 1, ex_s - 1),
        (ex_e + 1, dn_start - 1),
        (up_end + 1, dn_start - 1),
    )
    inc, skip = [0] * n_samples, [0] * n_samples
    for iso in gene_isoforms:
        if (
            iso.gene_id != gene_id
            or iso.chrom != chrom
            or iso.transcript_id not in counts
        ):
            continue
        introns = set(iso.introns())
        target = (
            inc
            if (up_intron in introns and down_intron in introns)
            else skip if joined in introns else None
        )
        if target is None:
            continue
        for i, c in enumerate(counts[iso.transcript_id]):
            target[i] += c
    return inc, skip
