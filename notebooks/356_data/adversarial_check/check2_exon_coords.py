"""Check 2: rebuild every exon's protein interval from the gcoord XML genomic exon lengths,
and confirm by translating the CDS (from _DNA.fasta, header tr|ACC|ENSxxxP) that the
residues in [prot_start, prot_end) are the ones that exon codes, against the protein FASTA.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xml.etree.ElementTree as ET
import polars as pl
from common import *

D = "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota"
SRC = {"chicken": "UP000000539_9031", "zebrafish": "UP000000437_7955"}
NS = "{http://www.ebi.ac.uk/proteins/api/doc/xsd/feature}"

bases = "TCAG"
codons = [a + b + c for a in bases for b in bases for c in bases]
aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = dict(zip(codons, aas))


def translate(s):
    return "".join(CODE.get(s[i : i + 3].upper(), "X") for i in range(0, len(s) - 2, 3))


def read_fasta(path, key):
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    out[name] = "".join(buf)
                name = key(line[1:].strip())
                buf = []
            else:
                buf.append(line.strip())
    if name:
        out[name] = "".join(buf)
    return out


ex = exons()
rows = []
for sp, stem in SRC.items():
    want = dict(
        ex.filter(pl.col("species") == sp)
        .select("target_acc", "transcript")
        .unique()
        .iter_rows()
    )
    prot = read_fasta(f"{D}/{stem}.fasta", lambda h: h.split("|")[1])
    dna = read_fasta(
        f"{D}/{stem}_DNA.fasta", lambda h: (h.split("|")[1], h.split("|")[2].split()[0])
    )
    for _, el in ET.iterparse(f"{D}/{stem}_gcoord.xml", events=("end",)):
        if el.tag != "gnEntry":
            continue
        acc = el.findtext("accession")
        if acc in want:
            for gc in el.findall("gnCoordinate"):
                if gc.get("ensembl_transcript_id") != want[acc]:
                    continue
                tl = gc.get("ensembl_translation_id")
                exs = gc.find("genomicLocation").findall("exon")
                cds = dna.get((acc, tl))
                o = 0
                for k, x in enumerate(exs, 1):
                    gl = x.find("genomeLocation")
                    b, e = gl.find(f"{NS}begin"), gl.find(f"{NS}end")
                    if b is None or e is None:
                        n = None
                    else:
                        n = abs(int(e.get("position")) - int(b.get("position"))) + 1
                    rows.append(
                        dict(
                            species=sp,
                            target_acc=acc,
                            transcript=want[acc],
                            translation=tl,
                            exon_id=x.get("id"),
                            my_rank=k,
                            my_n=len(exs),
                            my_nt=n,
                            my_off=o,
                            xml_seq_eq_fasta=el.findtext("sequence") == prot.get(acc),
                            cds_found=cds is not None,
                            cds_translates=(
                                cds is not None
                                and translate(cds).rstrip("*") == prot.get(acc)
                            ),
                            cds_len=None if cds is None else len(cds),
                        )
                    )
                    if n is None:
                        o = None
                    elif o is not None:
                        o += n
        el.clear()
    # keep sequences for the slice test
    globals()[f"prot_{sp}"] = prot
    globals()[f"dna_{sp}"] = dna

mine = pl.DataFrame(rows, infer_schema_length=None)
j = ex.join(mine, on=["species", "target_acc", "transcript", "exon_id"], how="left")
print(
    "exon rows:",
    ex.height,
    " matched to XML:",
    j.filter(pl.col("my_nt").is_not_null()).height,
)
j = j.with_columns(
    (pl.col("my_off") // 3).alias("my_ps"),
    ((pl.col("my_off") + pl.col("my_nt") + 2) // 3).alias("my_pe"),
    (pl.col("my_off") % 3).alias("my_phase"),
)
chk = j.select(
    (pl.col("my_rank") == pl.col("exon_rank")).alias("rank"),
    (pl.col("my_n") == pl.col("n_exons")).alias("n_exons"),
    (pl.col("my_nt") == pl.col("n_nt")).alias("n_nt"),
    (pl.col("my_ps") == pl.col("prot_start")).alias("prot_start"),
    (pl.col("my_pe") == pl.col("prot_end")).alias("prot_end"),
    (pl.col("my_phase") == pl.col("start_phase")).alias("phase"),
    pl.col("xml_seq_eq_fasta"),
    pl.col("cds_translates"),
)
print("fraction agreeing per field:")
print(chk.mean())

# Translate each exon's own CDS stretch, padded to whole codons, and compare with the
# protein FASTA slice [prot_start, prot_end).
ok = bad = 0
badrows = []
for r in j.iter_rows(named=True):
    prot = globals()[f"prot_{r['species']}"][r["target_acc"]]
    cds = globals()[f"dna_{r['species']}"].get((r["target_acc"], r["translation"]))
    if cds is None or r["my_off"] is None:
        bad += 1
        badrows.append((r["target_acc"], r["exon_rank"], "no cds"))
        continue
    o, n = r["my_off"], r["my_nt"]
    cs, ce = (o // 3) * 3, ((o + n + 2) // 3) * 3
    tr = translate(cds[cs:ce]).replace("*", "")
    sl = prot[r["prot_start"] : r["prot_end"]]
    if tr == sl and len(sl) == r["exon_len"]:
        ok += 1
    else:
        bad += 1
        badrows.append((r["target_acc"], r["exon_rank"], tr[:15], sl[:15]))
print(f"translated exon stretch == protein[prot_start:prot_end]: {ok} ok, {bad} not")
from collections import Counter

print(Counter(b[2] if b[2] == "no cds" else "mismatch" for b in badrows))
vb = set((a, r) for a, r, *_ in badrows)
vv = verdict_exons()
nv = sum((a, r) in vb for a, r in vv.select("target_acc", "exon_rank").iter_rows())
print("verdict exons without a translation check:", nv, "of", vv.height)
miss = sorted({b[0] for b in badrows if b[2] == "no cds"})
for sp in ["chicken", "zebrafish"]:
    d = globals()[f"dna_{sp}"]
    accs = {k[0] for k in d}
    spacc = set(ex.filter(pl.col("species") == sp)["target_acc"])
    m = [a for a in miss if a in spacc]
    print(
        sp,
        "targets with no CDS keyed by (acc, translation):",
        len(m),
        " acc present under another translation id:",
        sum(a in accs for a in m),
    )
print("mismatch examples:", [b for b in badrows if b[2] != "no cds"][:5])

# The 20 picked exons, printed residue by residue
v = verdict_exons().select("species", "target_acc", "exon_id")
pick = []
for sp in ["chicken", "zebrafish"]:
    s = j.filter(pl.col("species") == sp)
    for ph in [0, 1, 2]:
        pick.append(
            s.filter(
                pl.col("internal")
                & (pl.col("start_phase") == ph)
                & (pl.col("exon_len") <= 60)
            )
            .sort("target_acc", "exon_rank")
            .head(2)
        )
    pick.append(s.filter(pl.col("exon_rank") == 1).sort("target_acc").slice(5, 2))
    pick.append(
        s.filter(pl.col("exon_rank") == pl.col("n_exons"))
        .sort("target_acc")
        .slice(9, 2)
    )
pick = pl.concat(pick).head(20)
for r in pick.iter_rows(named=True):
    prot = globals()[f"prot_{r['species']}"][r["target_acc"]]
    cds = globals()[f"dna_{r['species']}"][(r["target_acc"], r["translation"])]
    o, n = r["my_off"], r["my_nt"]
    cs, ce = (o // 3) * 3, ((o + n + 2) // 3) * 3
    tr = translate(cds[cs:ce])
    print(
        f"{r['species']:9s} {r['target_acc']:11s} exon {r['exon_rank']:2d}/{r['n_exons']:2d} phase {r['start_phase']} "
        f"[{r['prot_start']},{r['prot_end']}) nt={n} translated={tr} protein={prot[r['prot_start']:r['prot_end']]} "
        f"{'OK' if tr.rstrip('*') == prot[r['prot_start']:r['prot_end']] else 'DIFF'}"
    )
