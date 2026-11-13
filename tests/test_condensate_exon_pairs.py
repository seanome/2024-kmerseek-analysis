"""Tests for notebooks/condensate_exon_pairs.py on ZNF207 (BuGZ), from cached set A data.

The ZNF207 isoforms are the Patowary et al. 2024 Bulk.gtf lines for the gene (GRCh37), and the
genome is GRCh37 chr17:30,670,000-30,720,000 from the Ensembl GRCh37 REST server. The expected
proteins are Ensembl 116's ZNF207-203 and ZNF207-204 translations, which the longest ORF of the
matching set A isoforms must reproduce exactly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
import condensate_exon_pairs as cp  # noqa: E402

DATA = ROOT / "data" / "341_condensate_exons"
SLICE_START = 30_670_000
EXON9 = (30_693_684, 30_693_776)


def load():
    seq = (DATA / "ensembl_grch37_chr17_30670000_30720000.txt").read_text().strip()
    isoforms = cp.read_gtf_isoforms(str(DATA / "patowary2024_dataS1_bulk_ZNF207.gtf"))
    return seq, isoforms


def protein(iso, seq):
    m, orf = cp.mrna_and_orf(iso, seq, SLICE_START - 1)
    return cp.translate(m[orf[0] : orf[1]])


def test_longest_orf_reproduces_ensembl_proteins():
    seq, isoforms = load()
    for tid, name in [
        ("ENST00000394670.8_1", "ZNF207-203"),
        ("ENST00000394673.6_3", "ZNF207-204"),
    ]:
        expected = (DATA / f"ensembl_protein_{name}.txt").read_text().strip()
        assert protein(isoforms[tid], seq) == expected, name


def test_exon9_pair_found_with_residues_277_to_307():
    seq, isoforms = load()
    rows = [
        r
        for r in cp.find_exon_events(isoforms)
        if (r["exon_start"], r["exon_end"]) == EXON9
    ]
    pairs = {(r["including_isoform"], r["skipping_isoform"]) for r in rows}
    assert ("ENST00000394670.8_1", "ENST00000394673.6_3") in pairs
    row = next(r for r in rows if r["including_isoform"] == "ENST00000394670.8_1")
    out = cp.pair_proteins(
        row,
        isoforms[row["including_isoform"]],
        isoforms[row["skipping_isoform"]],
        seq,
        SLICE_START - 1,
    )
    assert (
        out["frame_preserving"]
        and out["exon_in_orf"]
        and out["orf_otherwise_identical"]
    )
    assert (out["exon_first_residue"], out["exon_last_residue"]) == (277, 307)
    assert out["exon_residues"] == "MGTPVTSSSTASSNSESLSASSKALFPSTAQ"


def test_skipping_isoform_has_no_extra_exon_differences():
    """Every reported pair differs by exactly the one exon: same introns otherwise."""
    _, isoforms = load()
    for r in cp.find_exon_events(isoforms):
        inc = set(isoforms[r["including_isoform"]].introns())
        skip = set(isoforms[r["skipping_isoform"]].introns())
        up = (r["upstream_exon_end"] + 1, r["exon_start"] - 1)
        down = (r["exon_end"] + 1, r["downstream_exon_start"] - 1)
        assert inc - skip == {up, down}
        assert skip - inc == {(up[0], down[1])}


def test_event_read_counts_match_notebook_for_donor_334():
    """Exon 9 reads in donor 334: 1_168 including and 116 skipping in the germinal zone (nb 341, 3b)."""
    import csv

    _, isoforms = load()
    rows = list(
        csv.reader(
            open(DATA / "patowary2024_dataS2B_ZNF207_counts.tsv"), delimiter="\t"
        )
    )
    samples = rows[0][3:]
    counts = {r[2]: [int(x) for x in r[3:]] for r in rows[1:]}
    row = next(
        r
        for r in cp.find_exon_events(isoforms)
        if (r["exon_start"], r["exon_end"]) == EXON9
    )
    inc, skip = cp.event_read_counts(
        cp.event_key(row), list(isoforms.values()), counts, len(samples)
    )
    gz_334 = [
        i for i, s in enumerate(samples) if s.startswith("334_") and s.endswith("_GZ")
    ]
    assert sum(inc[i] for i in gz_334) == 1_168
    assert sum(skip[i] for i in gz_334) == 116
