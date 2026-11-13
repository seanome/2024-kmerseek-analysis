"""Check 2b: for the 101 targets whose chosen transcript's translation id is not the one in
_DNA.fasta, (a) does the chosen transcript's exon nt sum equal 3 x protein length, and
(b) does the DNA FASTA's CDS (another translation id) translate to the protein, and if so do
the exon slices still translate to protein[prot_start:prot_end] using the chosen transcript's
exon lengths?"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *
from check2_exon_coords import (
    read_fasta,
    translate,
    D,
    SRC,
)  # noqa (re-runs check 2 on import)

ex = exons()
tot = ex.group_by("species", "target_acc").agg(
    pl.col("n_nt").sum(), pl.col("target_len").first()
)
print(
    "targets where sum(exon nt) == 3*len:",
    (tot["n_nt"] == 3 * tot["target_len"]).sum(),
    "of",
    tot.height,
    "; == 3*len+3:",
    (tot["n_nt"] == 3 * tot["target_len"] + 3).sum(),
)
vv = verdict_exons()
ok = bad = nocds = 0
for sp, stem in SRC.items():
    prot = read_fasta(f"{D}/{stem}.fasta", lambda h: h.split("|")[1])
    dna = read_fasta(f"{D}/{stem}_DNA.fasta", lambda h: h.split("|")[1])
    for acc, g in ex.filter(pl.col("species") == sp).group_by("target_acc"):
        acc = acc[0]
        g = g.sort("exon_rank")
        cds = dna.get(acc)
        if cds is None or translate(cds).rstrip("*") != prot[acc]:
            nocds += g.height
            continue
        o = 0
        for r in g.iter_rows(named=True):
            n = r["n_nt"]
            tr = translate(cds[(o // 3) * 3 : ((o + n + 2) // 3) * 3]).replace("*", "")
            if tr == prot[acc][r["prot_start"] : r["prot_end"]]:
                ok += 1
            else:
                bad += 1
            o += n
print(
    f"using the DNA FASTA CDS for every target (keyed by accession only): {ok} exons ok, {bad} differ, "
    f"{nocds} with no CDS that translates to the protein"
)
