#!/usr/bin/env python3
"""Check notebook 359's exon-to-protein mapping on SQANTI3's own test data (real SQANTI3
output, chr22), before any run: for every coding isoform,

1. translating the genome through our exon order and SQANTI3's CDS_start gives SQANTI3's
   ORF_seq exactly;
2. the per-exon protein segments tile 0..len(ORF) with no gap or overlap;
3. our genomic span of the codons matches SQANTI3's CDS_genomic_start/end (the stop codon
   included);
4. each exon's segment, translated from that exon's own codons (two-of-three rule), is
   the matching piece of ORF_seq.

    check_codon_mapping_on_sqanti3_test_data.py <dir with test_isoforms_classification.tsv,
                                                 test_isoforms.gtf, genome_test.fasta>

The three files are SQANTI3 v6.0.1's test/test_data (isoforms/ and genome/).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isoform_blank_exons_359 as ib  # noqa: E402

BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = {
    a + b + c: AA[16 * i + 4 * j + k]
    for i, a in enumerate(BASES)
    for j, b in enumerate(BASES)
    for k, c in enumerate(BASES)
}
COMP = str.maketrans("ACGTN", "TGCAN")


def read_genome(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip().upper())
    seqs[name] = "".join(buf)
    return seqs


def main(d):
    d = Path(d)
    genome = read_genome(d / "genome_test.fasta")
    tx = {}
    for line in open(d / "test_isoforms.gtf"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        tid = f[8].split('transcript_id "')[1].split('"')[0]
        tx.setdefault(tid, {"chrom": f[0], "strand": f[6], "exons": []})[
            "exons"
        ].append(ib.Exon(int(f[3]), int(f[4])))
    n = ok = 0
    for r in csv.DictReader(
        open(d / "test_isoforms_classification.tsv"), delimiter="\t"
    ):
        if r["coding"] != "coding" or r["isoform"] not in tx:
            continue
        n += 1
        t = tx[r["isoform"]]
        prot = r["ORF_seq"].rstrip("*")
        cs = int(float(r["CDS_start"]))
        mrna = "".join(
            genome[t["chrom"]][e.start - 1 : e.end]
            for e in sorted(t["exons"], key=lambda e: e.start)
        )
        if t["strand"] == "-":
            mrna = mrna.translate(COMP)[::-1]
        cds = mrna[cs - 1 : cs - 1 + 3 * len(prot)]
        trans = "".join(CODE.get(cds[i : i + 3], "X") for i in range(0, len(cds), 3))
        segs = ib.exon_protein_intervals(t["exons"], t["strand"], cs, len(prot))
        tiles = [s for s in segs if s]
        tiled = (
            tiles[0][0] == 0
            and tiles[-1][1] == len(prot)
            and all(a[1] == b[0] for a, b in zip(tiles, tiles[1:]))
        )
        g = ib.protein_to_genomic((0, len(prot)), t["exons"], t["strand"], cs)
        lo, hi = g[0][0], g[-1][1]
        # SQANTI3 reports the CDS with its stop codon
        if t["strand"] == "+":
            hi += 3
        else:
            lo -= 3
        span_ok = (lo, hi) == (
            int(float(r["CDS_genomic_start"])),
            int(float(r["CDS_genomic_end"])),
        ) or (hi, lo) == (
            int(float(r["CDS_genomic_start"])),
            int(float(r["CDS_genomic_end"])),
        )
        pieces_ok = all(trans[s[0] : s[1]] == prot[s[0] : s[1]] for s in tiles)
        good = trans == prot and tiled and pieces_ok
        ok += good
        print(
            f"{r['isoform']}\t{t['strand']}\texons={len(t['exons'])}\taa={len(prot)}\t"
            f"translation={'same' if trans == prot else 'DIFFERENT'}\ttiles={'yes' if tiled else 'NO'}\t"
            f"cds_genomic_span={'same' if span_ok else (lo, hi, r['CDS_genomic_start'], r['CDS_genomic_end'])}"
        )
    print(
        f"{ok} of {n} coding isoforms: translation, tiling and per-exon pieces all match"
    )
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
